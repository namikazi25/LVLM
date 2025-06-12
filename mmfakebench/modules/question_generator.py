"""Question generation module for MMFakeBench.

This module contains the QAGenerationTool class that generates search-ready questions
to gather evidence for fact-checking claims.
"""

import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from core.base import BasePipelineModule

# Patterns that indicate unhelpful answers
UNHELPFUL_ANSWER_PATTERNS = [
    "not found", "no information", "cannot be answered", "unable to find",
    "no results", "not available", "insufficient information", "unclear",
    "cannot determine", "no clear answer", "inconclusive", "no evidence"
]


class QAGenerationTool(BasePipelineModule):
    """Pipeline module for generating questions and answers from news content.
    
    This module processes news images and headlines to generate:
    1. Relevant questions about the content
    2. Corresponding answers based on the visual and textual information
    """
    
    def __init__(self, 
                 name: str = "qa_generation",
                 llm_provider: str = "openai", 
                 model_name: str = "gpt-4-vision-preview",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: int = 30,
                 **kwargs):
        """Initialize the QA generation tool.
        
        Args:
            name: Name of the module
            llm_provider: The LLM provider to use
            model_name: The specific model to use
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
            **kwargs: Additional configuration parameters
        """
        config = {
            'llm_provider': llm_provider,
            'model_name': model_name,
            'max_retries': max_retries,
            'retry_delay': retry_delay,
            'timeout': timeout,
            **kwargs
        }
        super().__init__(name=name, config=config)
        
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.model_router: Optional[ModelRouter] = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Load prompt template
        from core.prompts import get_prompt_manager
        self.prompt_manager = get_prompt_manager()
        self.template_name = "question_generation"
        
    def initialize(self) -> None:
        """Initialize the module with its configuration."""
        try:
            if not self.model_router:
                from models.router import ModelRouter
                import os
                
                # Get API key from config or environment
                api_key = (
                    self.config.get('api_key') or
                    self.config.get('openai_api_key') or
                    os.getenv('OPENAI_API_KEY') or
                    os.getenv('GEMINI_API_KEY') or
                    'test-key-for-demo'  # Fallback for testing
                )
                
                self.model_router = ModelRouter(
                    model_name=self.model_name,
                    api_key=api_key,
                    temperature=self.config.get('temperature', 0.2),
                    max_retries=self.max_retries,
                    provider_name=self.llm_provider
                )
            self.logger.info(f"Initialized {self.name} with model router")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            raise
        
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data for question generation.
        
        Args:
            data: Input data containing enriched_claim, image_context, and optional previous_qa
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        required_fields = ['enriched_claim', 'image_context']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
            if not isinstance(data[field], str) or not data[field].strip():
                self.logger.error(f"Field {field} must be a non-empty string")
                return False
                
        # Validate previous_qa if provided
        if 'previous_qa' in data and data['previous_qa'] is not None:
            if not isinstance(data['previous_qa'], list):
                self.logger.error("previous_qa must be a list")
                return False
            for i, qa_pair in enumerate(data['previous_qa']):
                if not isinstance(qa_pair, dict):
                    self.logger.error(f"previous_qa[{i}] must be a dictionary")
                    return False
                if 'question' not in qa_pair or 'answer' not in qa_pair:
                    self.logger.error(f"previous_qa[{i}] must contain 'question' and 'answer' keys")
                    return False
                    
        return True
        
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the output schema for this module.
        
        Returns:
            Dict describing the output schema
        """
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Generated search-ready question"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether question generation was successful"
                }
            },
            "required": ["question", "success"]
        }
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data to generate a search-ready question.
        
        Args:
            data: Input data containing enriched_claim, image_context, and optional previous_qa
            
        Returns:
            Dict containing the generated question and success status
        """
        enriched_claim = data['enriched_claim']
        image_context = data['image_context']
        previous_qa = data.get('previous_qa', None)
        
        question, success = self.run(enriched_claim, image_context, previous_qa)
        
        return {
            **data,
            "question": question,
            "qa_generation_success": success
        }
        
    def run(self, enriched_claim: str, image_context: str, previous_qa: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, bool]:
        """Generate a search-ready question using retry logic.
        
        Args:
            enriched_claim: The claim to generate questions for
            image_context: Context from the image
            previous_qa: Previous question-answer pairs
            
        Returns:
            Tuple of (generated_question, success_flag)
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Question generation attempt {attempt + 1}/{self.max_retries}")
                
                # System message with detailed instructions
                # Render prompt template
                system_message_content = self.prompt_manager.render_template(
                    self.template_name, 
                    {
                        'enriched_claim': enriched_claim,
                        'image_context': image_context
                    }
                )

                # Prepare previous Q&A formatted for the human message
                if previous_qa:
                    prev_qa_list_str = []
                    for i, qa_pair in enumerate(previous_qa):
                        question_text = qa_pair.get('question', 'N/A')
                        answer_text = qa_pair.get('answer', 'N/A')
                        # Check if the answer from the previous QA pair was unhelpful
                        answer_is_unhelpful = any(pattern in answer_text.lower() for pattern in UNHELPFUL_ANSWER_PATTERNS)
                        indicator = " [Unhelpful Answer]" if answer_is_unhelpful else ""
                        prev_qa_list_str.append(f"Q{i+1}: {question_text}\nA{i+1}: {answer_text}{indicator}")

                    if prev_qa_list_str:
                        previous_qa_formatted = "Previous Q-A pairs (analyze carefully, especially unhelpful ones, to generate a NEW and DIFFERENT question):\n" + "\n---\n".join(prev_qa_list_str)
                    else:
                        previous_qa_formatted = "No previous Q-A pairs to display, but previous Q-A pairs were provided (possibly empty)."
                else:
                    previous_qa_formatted = "No previous Q-A pairs. This is the first question for this line of inquiry."

                # Create prompt template and inputs
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_message_content),
                    ("human", "{previous_qa_formatted_for_human_message}"),
                ])

                inputs = {
                    "previous_qa_formatted_for_human_message": previous_qa_formatted,
                }

                # LLM call using model router
                llm = self.model_router.get_llm(self.llm_provider, self.model_name)
                chain = LLMChain(llm=llm, prompt=prompt_template)
                response = chain.invoke(inputs)

                raw_text = ""
                if isinstance(response, dict):
                    raw_text = response.get('text', "").strip()
                elif isinstance(response, str):
                    raw_text = response.strip()
                else:
                    if hasattr(response, 'content'):
                        raw_text = str(response.content).strip()
                    else:
                        raw_text = str(response).strip()

                # Parse "QUESTION:" line
                match = re.search(r"QUESTION:\s*(.+)", raw_text, re.IGNORECASE | re.DOTALL)
                generated_question = match.group(1).strip() if match else raw_text.strip()

                if not generated_question or generated_question.lower() == "question:":
                    self.logger.error(f"QAGenerationTool produced an empty or malformed question. Raw response: '{raw_text}' from inputs: {inputs}")
                    raise ValueError("Empty question generated")

                # Check for repetition with previous questions
                if previous_qa:
                    for prev_q_pair in previous_qa:
                        if prev_q_pair.get('question', '').strip().lower() == generated_question.strip().lower():
                            self.logger.warning(
                                f"QAGenerationTool generated a question very similar or identical to a previous one: '{generated_question}'. "
                                f"This may indicate the LLM is not fully adhering to the 'avoid repetition' instruction for unhelpful answers."
                            )
                            break

                self.logger.info(f"Question generation successful on attempt {attempt + 1}: '{generated_question}'")
                return generated_question, True

            except Exception as e:
                self.logger.warning(f"Question generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    self.logger.debug(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed for question generation")
        
        return f"API Error: Failed after {self.max_retries} attempts", False