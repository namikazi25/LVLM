#!/usr/bin/env python3
"""
Agentic Orchestrator Module

This module orchestrates the 5-stage agentic workflow for misinformation detection,
coordinating between relevance checking, claim enrichment, iterative Q&A, evidence
tagging, and final synthesis.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from core.base import BasePipelineModule
from modules.relevance_checker import ImageHeadlineRelevancyChecker
from modules.claim_enrichment import ClaimEnrichmentTool
from modules.question_generator import QAGenerationTool
from modules.web_searcher import WebSearcher
from modules.evidence_tagger import EvidenceTagger
from modules.synthesizer import Synthesizer


class AgenticOrchestrator(BasePipelineModule):
    """Orchestrates the 5-stage agentic misinformation detection workflow.
    
    This module coordinates the execution of:
    1. Relevance checking between image and headline
    2. Claim enrichment and context gathering
    3. Iterative Q&A generation and web search
    4. Evidence tagging and classification
    5. Final synthesis and verdict generation
    """
    
    def __init__(self, 
                 name: str = "agentic_orchestrator",
                 config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """Initialize the agentic orchestrator.
        
        Args:
            name: Name of the module
            config: Module configuration dictionary
            **kwargs: Additional configuration parameters
        """
        # Set default configuration
        default_config = {
            'llm_provider': 'openai',
            'model_name': 'gpt-4o',
            'max_qa_iterations': 3,
            'confidence_threshold': 0.7,
            'enable_web_search': True,
            'max_retries': 3,
            'retry_delay': 1.0,
            'timeout': 30,
            'show_workflow': True
        }
        
        # Merge configurations
        final_config = {**default_config, **(config or {}), **kwargs}
        super().__init__(name, final_config)
        
        # Initialize stage modules
        self.relevance_checker = None
        self.claim_enricher = None
        self.qa_generator = None
        self.web_searcher = None
        self.evidence_tagger = None
        self.synthesizer = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
    
    def initialize(self) -> None:
        """Initialize all stage modules."""
        try:
            # Stage 1: Relevance Checker
            self.relevance_checker = ImageHeadlineRelevancyChecker(
                name="relevance_checker",
                llm_provider=self.config['llm_provider'],
                model_name=self.config['model_name'],
                show_workflow=self.config['show_workflow'],
                max_retries=self.config['max_retries'],
                retry_delay=self.config['retry_delay'],
                timeout=self.config['timeout']
            )
            self.relevance_checker.initialize()
            
            # Stage 2: Claim Enrichment
            self.claim_enricher = ClaimEnrichmentTool(
                name="claim_enricher",
                llm_provider=self.config['llm_provider'],
                model_name=self.config['model_name'],
                show_workflow=self.config['show_workflow'],
                max_retries=self.config['max_retries']
            )
            self.claim_enricher.initialize()
            
            # Stage 3: Q&A Generation
            self.qa_generator = QAGenerationTool(
                name="qa_generator",
                llm_provider=self.config['llm_provider'],
                model_name=self.config['model_name'],
                show_workflow=self.config['show_workflow'],
                max_retries=self.config['max_retries']
            )
            self.qa_generator.initialize()
            
            # Web Searcher for Q&A
            if self.config['enable_web_search']:
                self.web_searcher = WebSearcher(
                    name="web_searcher",
                    max_retries=self.config['max_retries']
                )
                self.web_searcher.initialize()
            
            # Stage 4: Evidence Tagger
            self.evidence_tagger = EvidenceTagger(
                name="evidence_tagger",
                llm_provider=self.config['llm_provider'],
                model_name=self.config['model_name'],
                show_workflow=self.config['show_workflow']
            )
            self.evidence_tagger.initialize()
            
            # Stage 5: Synthesizer
            self.synthesizer = Synthesizer(
                name="synthesizer",
                llm_provider=self.config['llm_provider'],
                model_name=self.config['model_name'],
                show_workflow=self.config['show_workflow'],
                confidence_threshold=self.config['confidence_threshold']
            )
            self.synthesizer.initialize()
            
            self.logger.info(f"Initialized {self.name} with 5-stage agentic workflow")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            raise RuntimeError(f"Agentic orchestrator initialization failed: {e}")
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format.
        
        Args:
            data: Input data dictionary
            
        Returns:
            True if input contains required fields, False otherwise
        """
        required_fields = ['image_path', 'headline']
        return all(field in data for field in required_fields)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the 5-stage agentic workflow.
        
        Args:
            data: Input data containing image_path and headline
            
        Returns:
            Dictionary containing results from all stages
        """
        workflow_start_time = time.time()
        result = data.copy()
        result['agentic_workflow'] = {
            'stages_completed': [],
            'stage_results': {},
            'errors': [],
            'status': 'running'
        }
        
        try:
            # Stage 0: Relevance Check
            self.logger.info("Stage 0: Starting relevance check")
            stage_result = self.relevance_checker.safe_process(data)
            result.update(stage_result)
            result['agentic_workflow']['stages_completed'].append('relevance_check')
            result['agentic_workflow']['stage_results']['relevance_check'] = {
                'success': stage_result.get('relevancy_check_success', False),
                'confidence': stage_result.get('relevancy_confidence', 0.0),
                'potential_mismatch': stage_result.get('potential_mismatch', False)
            }
            
            # Stage 1: Claim Enrichment
            self.logger.info("Stage 1: Starting claim enrichment")
            stage_result = self.claim_enricher.safe_process(result)
            result.update(stage_result)
            result['agentic_workflow']['stages_completed'].append('claim_enrichment')
            result['agentic_workflow']['stage_results']['claim_enrichment'] = {
                'success': stage_result.get('enrichment_success', False),
                'enriched_claim': stage_result.get('enriched_claim', '')
            }
            
            # Stage 2: Iterative Q&A Generation and Web Search
            self.logger.info("Stage 2: Starting iterative Q&A generation")
            qa_pairs = []
            enriched_claim = result.get('enriched_claim', result.get('headline', ''))
            
            for iteration in range(self.config['max_qa_iterations']):
                self.logger.info(f"Q&A Iteration {iteration + 1}/{self.config['max_qa_iterations']}")
                
                # Generate questions
                qa_data = {
                    'enriched_claim': enriched_claim,
                    'iteration': iteration + 1,
                    'previous_qa_pairs': qa_pairs
                }
                qa_result = self.qa_generator.safe_process(qa_data)
                
                if qa_result.get('qa_generation_success', False):
                    questions = qa_result.get('generated_questions', [])
                    
                    # Search for answers if web search is enabled
                    if self.web_searcher and questions:
                        for question in questions[:2]:  # Limit to 2 questions per iteration
                            search_data = {'query': question}
                            search_result = self.web_searcher.safe_process(search_data)
                            
                            if search_result.get('search_success', False):
                                qa_pairs.append({
                                    'question': question,
                                    'answer': search_result.get('search_results', ''),
                                    'iteration': iteration + 1
                                })
                
                # Break if we have enough evidence
                if len(qa_pairs) >= 4:  # Reasonable amount of evidence
                    break
            
            result['qa_pairs'] = qa_pairs
            result['agentic_workflow']['stages_completed'].append('qa_generation')
            result['agentic_workflow']['stage_results']['qa_generation'] = {
                'success': len(qa_pairs) > 0,
                'qa_count': len(qa_pairs),
                'iterations_completed': min(iteration + 1, self.config['max_qa_iterations'])
            }
            
            # Stage 3: Evidence Tagging
            self.logger.info("Stage 3: Starting evidence tagging")
            if qa_pairs:
                evidence_data = {
                    'qa_pairs': qa_pairs,
                    'enriched_claim': enriched_claim
                }
                evidence_result = self.evidence_tagger.safe_process(evidence_data)
                result.update(evidence_result)
                result['agentic_workflow']['stages_completed'].append('evidence_tagging')
                result['agentic_workflow']['stage_results']['evidence_tagging'] = {
                    'success': evidence_result.get('tagging_success', False),
                    'tagged_evidence': evidence_result.get('tagged_evidence', [])
                }
            
            # Stage 4: Final Synthesis
            self.logger.info("Stage 4: Starting final synthesis")
            synthesis_data = {
                'headline': result.get('headline', ''),
                'enriched_claim': enriched_claim,
                'qa_pairs': qa_pairs,
                'relevancy_response': result.get('relevancy_response', ''),
                'tagged_evidence': result.get('tagged_evidence', []),
                'potential_mismatch': result.get('potential_mismatch', False)
            }
            
            synthesis_result = self.synthesizer.safe_process(synthesis_data)
            result.update(synthesis_result)
            result['agentic_workflow']['stages_completed'].append('synthesis')
            result['agentic_workflow']['stage_results']['synthesis'] = {
                'success': synthesis_result.get('synthesis_success', False),
                'final_verdict': synthesis_result.get('final_verdict', 'uncertain'),
                'confidence': synthesis_result.get('final_confidence', 0.0),
                'explanation': synthesis_result.get('final_explanation', '')
            }
            
            # Set final workflow status
            result['agentic_workflow']['status'] = 'completed'
            result['agentic_workflow']['total_time'] = time.time() - workflow_start_time
            
            # Set final detection results for compatibility
            result['prediction'] = synthesis_result.get('final_verdict', 'uncertain')
            result['confidence'] = synthesis_result.get('final_confidence', 0.0)
            result['explanation'] = synthesis_result.get('final_explanation', '')
            
            self.logger.info(f"Agentic workflow completed successfully in {result['agentic_workflow']['total_time']:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Agentic workflow failed: {e}")
            result['agentic_workflow']['status'] = 'failed'
            result['agentic_workflow']['errors'].append(str(e))
            result['agentic_workflow']['total_time'] = time.time() - workflow_start_time
            
            # Set fallback results
            result['prediction'] = 'uncertain'
            result['confidence'] = 0.0
            result['explanation'] = f'Agentic workflow failed: {str(e)}'
        
        return result
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema for this module.
        
        Returns:
            Dictionary describing the output schema
        """
        return {
            'prediction': 'str - Final misinformation verdict (real/fake/uncertain)',
            'confidence': 'float - Confidence score (0.0-1.0)',
            'explanation': 'str - Detailed explanation of the verdict',
            'agentic_workflow': 'dict - Detailed workflow execution information',
            'relevancy_response': 'str - Relevance check results',
            'enriched_claim': 'str - Enriched claim with context',
            'qa_pairs': 'list - Generated Q&A pairs with web search results',
            'tagged_evidence': 'list - Evidence tagged by type and strength',
            'final_verdict': 'str - Final synthesis verdict',
            'final_confidence': 'float - Final confidence score',
            'final_explanation': 'str - Final detailed explanation'
        }