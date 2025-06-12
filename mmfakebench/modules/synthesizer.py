"""Decision Synthesis Module for MMFakeBench.

This module contains the Synthesizer class that combines evidence from multiple
sources to generate final verdicts on misinformation claims.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from core.base import BasePipelineModule
from models.router import ModelRouter


class VerdictType(Enum):
    """Enumeration of possible verdict types."""
    REAL = "real"
    FAKE = "fake"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Synthesizer(BasePipelineModule):
    """Pipeline module for synthesizing final decisions.
    
    This module combines evidence from multiple sources (relevance checking,
    claim enrichment, Q&A generation, evidence tagging, web search) to
    generate a final verdict on misinformation claims.
    """
    
    def __init__(self, 
                 name: str = "synthesizer",
                 llm_provider: str = "openai",
                 model_name: str = "gpt-4",
                 confidence_threshold: float = 0.7,
                 evidence_weight_config: Optional[Dict[str, float]] = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 timeout: Optional[float] = None,
                 **kwargs):
        """Initialize the synthesizer.
        
        Args:
            name: Name of the module
            llm_provider: The LLM provider to use
            model_name: The specific model to use
            confidence_threshold: Minimum confidence for high-confidence verdicts
            evidence_weight_config: Weights for different types of evidence
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
            timeout: Timeout for operations in seconds
            **kwargs: Additional configuration parameters
        """
        config = {
            'llm_provider': llm_provider,
            'model_name': model_name,
            'confidence_threshold': confidence_threshold,
            'evidence_weight_config': evidence_weight_config,
            'max_retries': max_retries,
            'retry_delay': retry_delay,
            'timeout': timeout,
            **kwargs
        }
        super().__init__(name=name, config=config)
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.model_router: Optional[ModelRouter] = None
        self.logger = logging.getLogger(__name__)
        
        # Default evidence weights
        self.evidence_weights = evidence_weight_config or {
            'relevance_score': 0.15,
            'enriched_claim': 0.20,
            'qa_evidence': 0.25,
            'web_search_evidence': 0.30,
            'image_context': 0.10
        }
        
        # Load prompt template
        from core.prompts import get_prompt_manager
        self.prompt_manager = get_prompt_manager()
        self.template_name = "evidence_synthesis"
    
    def initialize(self) -> None:
        """Initialize the module with its configuration."""
        try:
            if not self.model_router:
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
            self.logger.info(f"Synthesizer initialized with provider: {self.llm_provider}, model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Synthesizer: {e}")
            raise
        
        # Validate evidence weights sum to 1.0
        total_weight = sum(self.evidence_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Evidence weights sum to {total_weight}, normalizing to 1.0")
            self.evidence_weights = {k: v/total_weight for k, v in self.evidence_weights.items()}
        
        self.logger.info(f"Synthesizer initialized with model {self.llm_provider}/{self.model_name}")
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        required_fields = ['original_claim']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Check if we have at least some evidence to work with
        evidence_fields = [
            'relevance_analysis',
            'enriched_claim',
            'qa_evidence',
            'search_results',
            'image_context'
        ]
        
        has_evidence = any(field in data for field in evidence_fields)
        if not has_evidence:
            self.logger.error("No evidence fields found in input data")
            return False
        
        return True
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema for this module.
        
        Returns:
            Dictionary describing the output schema
        """
        return {
            'original_claim': 'str',
            'final_verdict': {
                'verdict': 'str',  # real|fake|uncertain|insufficient_evidence
                'confidence': 'str',  # high|medium|low
                'confidence_score': 'float',  # 0.0-1.0
                'reasoning': 'str',
                'key_evidence': 'List[str]',
                'contradictions': 'List[str]',
                'limitations': 'str'
            },
            'evidence_summary': {
                'total_evidence_points': 'int',
                'supporting_evidence': 'int',
                'contradicting_evidence': 'int',
                'neutral_evidence': 'int'
            },
            'synthesis_metadata': {
                'evidence_weights_used': 'Dict[str, float]',
                'processing_time': 'float'
            },
            'module_status': 'str'
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through decision synthesis.
        
        Args:
            data: Input data dictionary containing evidence from previous modules
            
        Returns:
            Processed data dictionary with final verdict
        """
        import time
        start_time = time.time()
        
        original_claim = data['original_claim']
        self.logger.info(f"Synthesizing evidence for claim: {original_claim[:100]}...")
        
        # Extract and structure evidence
        evidence_summary = self._extract_evidence(data)
        
        # Generate synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(original_claim, evidence_summary)
        
        # Get LLM synthesis
        try:
            response = self.model_router.generate(
                prompt=synthesis_prompt,
                model_provider=self.llm_provider,
                model_name=self.model_name,
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=1000
            )
            
            # Parse the response
            verdict_data = self._parse_synthesis_response(response)
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {e}")
            # Fallback to rule-based synthesis
            verdict_data = self._fallback_synthesis(evidence_summary)
        
        processing_time = time.time() - start_time
        
        return {
            **data,
            'final_verdict': verdict_data,
            'evidence_summary': self._calculate_evidence_stats(evidence_summary),
            'synthesis_metadata': {
                'evidence_weights_used': self.evidence_weights,
                'processing_time': processing_time
            }
        }
    
    def _extract_evidence(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and structure evidence from input data.
        
        Args:
            data: Input data containing evidence from various modules
            
        Returns:
            Structured evidence dictionary
        """
        evidence = {
            'relevance_analysis': data.get('relevance_analysis', {}),
            'enriched_claim': data.get('enriched_claim', ''),
            'image_context': data.get('image_context', ''),
            'qa_evidence': data.get('qa_pairs', []),
            'evidence_tags': data.get('evidence_tags', []),
            'search_results': data.get('search_results', {}),
            'web_evidence': []
        }
        
        # Extract web evidence snippets
        if 'search_results' in data and 'combined_results' in data['search_results']:
            for result in data['search_results']['combined_results'][:5]:  # Top 5 results
                evidence['web_evidence'].append({
                    'title': result.get('title', ''),
                    'snippet': result.get('snippet', ''),
                    'url': result.get('url', ''),
                    'source': result.get('source', '')
                })
        
        return evidence
    
    def _build_synthesis_prompt(self, claim: str, evidence: Dict[str, Any]) -> str:
        """Build the synthesis prompt for the LLM.
        
        Args:
            claim: Original claim to analyze
            evidence: Structured evidence dictionary
            
        Returns:
            Complete synthesis prompt
        """
        # Render prompt template
        system_prompt = self.prompt_manager.render_template(
            self.template_name, 
            {}
        )
        prompt_parts = [system_prompt]
        
        prompt_parts.append(f"\n\nCLAIM TO ANALYZE: {claim}\n")
        
        # Add relevance analysis
        if evidence['relevance_analysis']:
            relevance = evidence['relevance_analysis']
            prompt_parts.append(f"\nRELEVANCE ANALYSIS:")
            prompt_parts.append(f"- Relevance Score: {relevance.get('relevance_score', 'N/A')}")
            prompt_parts.append(f"- Analysis: {relevance.get('relevance_reasoning', 'N/A')}")
        
        # Add enriched claim and image context
        if evidence['enriched_claim']:
            prompt_parts.append(f"\nENRICHED CLAIM: {evidence['enriched_claim']}")
        
        if evidence['image_context']:
            prompt_parts.append(f"\nIMAGE CONTEXT: {evidence['image_context']}")
        
        # Add Q&A evidence
        if evidence['qa_evidence']:
            prompt_parts.append(f"\nQUESTION-ANSWER EVIDENCE:")
            for i, qa in enumerate(evidence['qa_evidence'][:3], 1):  # Top 3 Q&A pairs
                prompt_parts.append(f"{i}. Q: {qa.get('question', 'N/A')}")
                prompt_parts.append(f"   A: {qa.get('answer', 'N/A')}")
                if 'evidence_tag' in qa:
                    prompt_parts.append(f"   Tag: {qa['evidence_tag']}")
        
        # Add web search evidence
        if evidence['web_evidence']:
            prompt_parts.append(f"\nWEB SEARCH EVIDENCE:")
            for i, result in enumerate(evidence['web_evidence'], 1):
                prompt_parts.append(f"{i}. {result['title']}")
                prompt_parts.append(f"   {result['snippet']}")
                prompt_parts.append(f"   Source: {result['source']}")
        
        prompt_parts.append("\nProvide your synthesis in the specified JSON format:")
        
        return "\n".join(prompt_parts)
    
    def _parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM synthesis response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed verdict dictionary
        """
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                verdict_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['verdict', 'confidence', 'reasoning']
                for field in required_fields:
                    if field not in verdict_data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Ensure confidence_score is present
                if 'confidence_score' not in verdict_data:
                    confidence_map = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
                    verdict_data['confidence_score'] = confidence_map.get(
                        verdict_data.get('confidence', 'low'), 0.4
                    )
                
                # Ensure lists are present
                verdict_data.setdefault('key_evidence', [])
                verdict_data.setdefault('contradictions', [])
                verdict_data.setdefault('limitations', '')
                
                return verdict_data
            
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Failed to parse synthesis response: {e}")
            # Return a fallback response
            return {
                'verdict': 'uncertain',
                'confidence': 'low',
                'confidence_score': 0.3,
                'reasoning': f"Failed to parse LLM response: {str(e)}",
                'key_evidence': [],
                'contradictions': [],
                'limitations': 'Response parsing failed'
            }
    
    def _fallback_synthesis(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback rule-based synthesis when LLM fails.
        
        Args:
            evidence: Structured evidence dictionary
            
        Returns:
            Fallback verdict dictionary
        """
        self.logger.info("Using fallback rule-based synthesis")
        
        # Simple rule-based logic
        supporting_count = 0
        contradicting_count = 0
        total_evidence = 0
        
        # Count evidence tags
        for qa in evidence.get('qa_evidence', []):
            if 'evidence_tag' in qa:
                total_evidence += 1
                if qa['evidence_tag'] in ['supports', 'background']:
                    supporting_count += 1
                elif qa['evidence_tag'] == 'refutes':
                    contradicting_count += 1
        
        # Simple decision logic
        if total_evidence == 0:
            verdict = 'insufficient_evidence'
            confidence = 'low'
            confidence_score = 0.2
        elif contradicting_count > supporting_count:
            verdict = 'fake'
            confidence = 'medium' if contradicting_count >= 2 else 'low'
            confidence_score = min(0.7, 0.4 + (contradicting_count * 0.1))
        elif supporting_count > contradicting_count:
            verdict = 'real'
            confidence = 'medium' if supporting_count >= 2 else 'low'
            confidence_score = min(0.7, 0.4 + (supporting_count * 0.1))
        else:
            verdict = 'uncertain'
            confidence = 'low'
            confidence_score = 0.4
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'confidence_score': confidence_score,
            'reasoning': f"Rule-based analysis: {supporting_count} supporting, {contradicting_count} contradicting evidence points",
            'key_evidence': [f"Total evidence points analyzed: {total_evidence}"],
            'contradictions': [],
            'limitations': 'Fallback rule-based analysis used due to LLM synthesis failure'
        }
    
    def _calculate_evidence_stats(self, evidence: Dict[str, Any]) -> Dict[str, int]:
        """Calculate statistics about the evidence.
        
        Args:
            evidence: Structured evidence dictionary
            
        Returns:
            Evidence statistics
        """
        total_points = 0
        supporting = 0
        contradicting = 0
        neutral = 0
        
        # Count Q&A evidence
        for qa in evidence.get('qa_evidence', []):
            if 'evidence_tag' in qa:
                total_points += 1
                tag = qa['evidence_tag']
                if tag in ['supports', 'background']:
                    supporting += 1
                elif tag == 'refutes':
                    contradicting += 1
                else:
                    neutral += 1
        
        # Count web evidence
        web_evidence_count = len(evidence.get('web_evidence', []))
        total_points += web_evidence_count
        neutral += web_evidence_count  # Web evidence is considered neutral until tagged
        
        return {
            'total_evidence_points': total_points,
            'supporting_evidence': supporting,
            'contradicting_evidence': contradicting,
            'neutral_evidence': neutral
        }