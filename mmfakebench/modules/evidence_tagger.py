"""Evidence tagging module for MMFakeBench.

This module contains the EvidenceTagger class that classifies how Q&A pairs
relate to claims being fact-checked.
"""

import logging
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from core.base import BasePipelineModule


class EvidenceTagger(BasePipelineModule):
    """Tool for tagging evidence based on how Q&A pairs relate to claims.
    
    This module classifies the relationship between a question-answer pair
    and an enriched claim into one of four categories: supports, refutes,
    background, or irrelevant.
    """
    
    def __init__(self, 
                 name: str = "evidence_tagger",
                 llm_provider: str = "openai", 
                 model_name: str = "gpt-4",
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: int = 30,
                 **kwargs):
        """Initialize the evidence tagger.
        
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
        self.model_router: Optional['ModelRouter'] = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Load prompt template
        from core.prompts import get_prompt_manager
        self.prompt_manager = get_prompt_manager()
        self.template_name = "evidence_tagging"
        
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
        """Validate input data for evidence tagging.
        
        Args:
            data: Input data containing question, answer, and enriched_claim
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        required_fields = ['question', 'answer', 'enriched_claim']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
            if not isinstance(data[field], str):
                self.logger.error(f"Field {field} must be a string")
                return False
            if not data[field].strip():
                self.logger.error(f"Field {field} cannot be empty")
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
                "tag": {
                    "type": "string",
                    "description": "Evidence tag: supports, refutes, background, or irrelevant",
                    "enum": ["supports", "refutes", "background", "irrelevant"]
                }
            },
            "required": ["tag"]
        }
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data to tag the evidence relationship.
        
        Args:
            data: Input data containing question, answer, and enriched_claim
            
        Returns:
            Dict containing the evidence tag
        """
        question = data['question']
        answer = data['answer']
        enriched_claim = data['enriched_claim']
        
        tag, success = self.run(question, answer, enriched_claim)
        
        return {
            **data,
            "evidence_tag": tag,
            "evidence_tagging_success": success
        }
        
    def run(self, question: str, answer: str, enriched_claim: str) -> tuple[str, bool]:
        """Tag the evidence relationship using retry logic.
        
        Args:
            question: The question text
            answer: The answer text
            enriched_claim: The claim to evaluate against
            
        Returns:
            tuple: (evidence_tag, success_flag)
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Evidence tagging attempt {attempt + 1}/{self.max_retries}")
                
                # Render prompt template
                system_prompt = self.prompt_manager.render_template(
                    self.template_name, 
                    {}
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Claim: {claim}\nQ: {question}\nA: {answer}")
                ])

                llm = self.model_router.get_llm(self.llm_provider, self.model_name)
                chain = LLMChain(llm=llm, prompt=prompt)
                response = chain.invoke({
                    "claim": enriched_claim,
                    "question": question,
                    "answer": answer
                })

                # Extract tag from response
                if isinstance(response, dict):
                    tag = response.get('text', '').strip().lower()
                elif isinstance(response, str):
                    tag = response.strip().lower()
                else:
                    if hasattr(response, 'content'):
                        tag = str(response.content).strip().lower()
                    else:
                        tag = str(response).strip().lower()
                
                valid_tags = {"supports", "refutes", "background", "irrelevant"}
                if tag in valid_tags:
                    self.logger.info(f"Evidence tagging successful on attempt {attempt + 1}: {tag}")
                    return tag, True
                else:
                    self.logger.warning(f"Invalid tag '{tag}' received, defaulting to 'background'")
                    return "background", True

            except Exception as e:
                self.logger.warning(f"Evidence tagging attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    self.logger.debug(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed for evidence tagging")
        
        return "background", False