#!/usr/bin/env python3
"""
Test Phase 5.2: Prompt Engineering Implementation

This test suite validates the enhanced prompt templates with:
- Chain-of-Thought (CoT) reasoning
- Few-shot learning examples
- Optimized prompt structures
- Improved clarity and guidance
"""

import unittest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.prompts import PromptManager, get_prompt_manager


class TestPromptEngineering(unittest.TestCase):
    """Test enhanced prompt templates and engineering improvements."""
    
    def setUp(self):
        """Set up test environment."""
        self.manager = get_prompt_manager()
        self.test_data = {
            'headline': 'Local mayor announces new park construction project',
            'image_context': 'Person in business suit standing at podium in council chambers',
            'enriched_claim': 'The mayor of the city announced a new park construction project during a city council meeting',
            'claim': 'Local mayor announces new park construction project',
            'question': 'What new projects has the mayor announced recently?',
            'answer': 'The mayor announced a $2.5 million park construction project at yesterday\'s council meeting',
            'evidence_summary': 'Visual evidence shows council meeting setting. Q&A confirms project announcement. Web search validates official announcement.',
            'previous_qa': 'Q: What happened at the council meeting? A: Mayor discussed budget items and announced park project.'
        }
    
    def test_cot_templates_exist(self):
        """Test that all Chain-of-Thought templates exist and are loadable."""
        cot_templates = [
            'relevance_check_cot',
            'claim_enrichment_cot', 
            'question_generation_cot',
            'evidence_tagging_cot',
            'evidence_synthesis_cot'
        ]
        
        for template_name in cot_templates:
            with self.subTest(template=template_name):
                template = self.manager.load_template(template_name)
                self.assertIsNotNone(template)
                
                # Test rendering with sample data
                result = self.manager.render_template(template_name, self.test_data)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 100)  # Should be substantial content
                
                # Check for CoT indicators
                self.assertIn('Chain-of-Thought', result)
                self.assertIn('reasoning', result.lower())
    
    def test_optimized_templates_exist(self):
        """Test that all optimized templates exist and are loadable."""
        optimized_templates = [
            'relevance_check_optimized',
            'claim_enrichment_optimized',
            'question_generation_optimized', 
            'evidence_tagging_optimized',
            'evidence_synthesis_optimized'
        ]
        
        for template_name in optimized_templates:
            with self.subTest(template=template_name):
                template = self.manager.load_template(template_name)
                self.assertIsNotNone(template)
                
                # Test rendering with sample data
                result = self.manager.render_template(template_name, self.test_data)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 50)  # Should have meaningful content
    
    def test_cot_few_shot_examples(self):
        """Test that CoT templates contain few-shot examples."""
        cot_templates = [
            'relevance_check_cot',
            'claim_enrichment_cot',
            'question_generation_cot', 
            'evidence_tagging_cot',
            'evidence_synthesis_cot'
        ]
        
        for template_name in cot_templates:
            with self.subTest(template=template_name):
                result = self.manager.render_template(template_name, self.test_data)
                
                # Check for few-shot example indicators
                self.assertTrue(
                    'Example' in result or 'example' in result,
                    f"Template {template_name} should contain examples"
                )
    
    def test_template_variable_substitution(self):
        """Test that templates properly substitute variables."""
        # Test relevance check CoT
        result = self.manager.render_template('relevance_check_cot', self.test_data)
        self.assertIn(self.test_data['headline'], result)
        self.assertIn(self.test_data['image_context'], result)
        
        # Test claim enrichment CoT
        result = self.manager.render_template('claim_enrichment_cot', self.test_data)
        self.assertIn(self.test_data['headline'], result)
        
        # Test question generation CoT
        result = self.manager.render_template('question_generation_cot', self.test_data)
        self.assertIn(self.test_data['enriched_claim'], result)
        self.assertIn(self.test_data['image_context'], result)
        
        # Test evidence tagging CoT
        result = self.manager.render_template('evidence_tagging_cot', self.test_data)
        self.assertIn(self.test_data['claim'], result)
        self.assertIn(self.test_data['question'], result)
        self.assertIn(self.test_data['answer'], result)
        
        # Test evidence synthesis CoT
        result = self.manager.render_template('evidence_synthesis_cot', self.test_data)
        self.assertIn(self.test_data['claim'], result)
        self.assertIn(self.test_data['evidence_summary'], result)
    
    def test_prompt_structure_improvements(self):
        """Test that optimized prompts have improved structure."""
        optimized_templates = [
            'relevance_check_optimized',
            'claim_enrichment_optimized',
            'question_generation_optimized',
            'evidence_tagging_optimized', 
            'evidence_synthesis_optimized'
        ]
        
        for template_name in optimized_templates:
            with self.subTest(template=template_name):
                result = self.manager.render_template(template_name, self.test_data)
                
                # Check for structural improvements
                self.assertTrue(
                    any(indicator in result for indicator in ['**', '•', '1.', 'Task:', 'Guidelines:']),
                    f"Template {template_name} should have clear structure"
                )
    
    def test_cot_reasoning_steps(self):
        """Test that CoT templates include reasoning steps."""
        cot_templates = [
            'relevance_check_cot',
            'claim_enrichment_cot',
            'question_generation_cot',
            'evidence_tagging_cot',
            'evidence_synthesis_cot'
        ]
        
        for template_name in cot_templates:
            with self.subTest(template=template_name):
                result = self.manager.render_template(template_name, self.test_data)
                
                # Check for step-by-step reasoning indicators
                step_indicators = ['1.', '2.', '3.', 'Step', 'step', 'Process:', 'reasoning:']
                self.assertTrue(
                    any(indicator in result for indicator in step_indicators),
                    f"Template {template_name} should include reasoning steps"
                )
    
    def test_template_completeness(self):
        """Test that all templates are complete and functional."""
        all_templates = [
            'relevance_check_cot', 'relevance_check_optimized',
            'claim_enrichment_cot', 'claim_enrichment_optimized',
            'question_generation_cot', 'question_generation_optimized',
            'evidence_tagging_cot', 'evidence_tagging_optimized',
            'evidence_synthesis_cot', 'evidence_synthesis_optimized'
        ]
        
        for template_name in all_templates:
            with self.subTest(template=template_name):
                # Test template loading
                template = self.manager.load_template(template_name)
                self.assertIsNotNone(template)
                
                # Test rendering
                result = self.manager.render_template(template_name, self.test_data)
                self.assertIsInstance(result, str)
                self.assertGreater(len(result.strip()), 0)
                
                # Test that no template variables remain unsubstituted
                self.assertNotIn('{{', result, f"Template {template_name} has unsubstituted variables")
                self.assertNotIn('}}', result, f"Template {template_name} has unsubstituted variables")
    
    def test_backward_compatibility(self):
        """Test that original templates still work alongside new ones."""
        original_templates = [
            'relevance_check',
            'claim_enrichment', 
            'question_generation',
            'evidence_tagging',
            'evidence_synthesis'
        ]
        
        for template_name in original_templates:
            with self.subTest(template=template_name):
                template = self.manager.load_template(template_name)
                self.assertIsNotNone(template)
                
                # Test rendering with appropriate data
                if template_name == 'question_generation':
                    # This template expects specific variables
                    result = self.manager.render_template(template_name, {
                        'enriched_claim': self.test_data['enriched_claim'],
                        'image_context': self.test_data['image_context']
                    })
                else:
                    result = self.manager.render_template(template_name, self.test_data)
                
                self.assertIsInstance(result, str)
                self.assertGreater(len(result.strip()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)