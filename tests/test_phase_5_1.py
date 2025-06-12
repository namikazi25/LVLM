#!/usr/bin/env python3
"""
Test Phase 5.1: Template System

This test file validates the implementation of the template system including:
- PromptManager functionality
- Template loading and rendering
- Integration with existing modules
- Template versioning and caching
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mmfakebench.core.prompts import PromptManager, get_prompt_manager
from mmfakebench.modules.relevance_checker import ImageHeadlineRelevancyChecker
from mmfakebench.modules.claim_enrichment import ClaimEnrichmentTool
from mmfakebench.modules.question_generator import QAGenerationTool
from mmfakebench.modules.evidence_tagger import EvidenceTagger
from mmfakebench.modules.synthesizer import Synthesizer
from mmfakebench.models.router import ModelRouter


class TestPromptManager(unittest.TestCase):
    """Test the PromptManager class functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.prompts_dir = os.path.join(self.test_dir, 'prompts')
        os.makedirs(self.prompts_dir)
        
        # Create test template
        self.test_template_path = os.path.join(self.prompts_dir, 'test_template.txt')
        with open(self.test_template_path, 'w') as f:
            f.write('Hello {{ name }}! You are {{ age }} years old.')
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_prompt_manager_initialization(self):
        """Test PromptManager initialization."""
        manager = PromptManager(prompts_dir=self.prompts_dir)
        self.assertEqual(str(manager.prompts_dir), str(Path(self.prompts_dir)))
        self.assertIsNotNone(manager.jinja_env)
    
    def test_template_loading(self):
        """Test template loading functionality."""
        manager = PromptManager(prompts_dir=self.prompts_dir)
        template = manager.load_template('test_template')
        self.assertIsNotNone(template)
    
    def test_template_rendering(self):
        """Test template rendering with variables."""
        manager = PromptManager(prompts_dir=self.prompts_dir)
        result = manager.render_template('test_template', {
            'name': 'Alice',
            'age': 25
        })
        expected = 'Hello Alice! You are 25 years old.'
        self.assertEqual(result, expected)
    
    def test_template_not_found(self):
        """Test handling of non-existent templates."""
        manager = PromptManager(prompts_dir=self.prompts_dir)
        with self.assertRaises(Exception):
            manager.load_template('non_existent_template')
    
    def test_get_prompt_manager_singleton(self):
        """Test the global prompt manager singleton."""
        # This test assumes the default prompts directory exists
        manager1 = get_prompt_manager()
        manager2 = get_prompt_manager()
        self.assertIs(manager1, manager2)


class TestTemplateIntegration(unittest.TestCase):
    """Test template integration with existing modules."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock router for testing
        self.mock_router = MockModelRouter()
    
    def test_relevance_checker_template_integration(self):
        """Test relevance checker uses template system."""
        checker = ImageHeadlineRelevancyChecker(
            name="test_relevance_checker",
            model_router=self.mock_router
        )
        
        # Check that template system is initialized
        self.assertIsNotNone(checker.prompt_manager)
        self.assertEqual(checker.template_name, "relevance_check")
    
    def test_claim_enrichment_template_integration(self):
        """Test claim enrichment uses template system."""
        enricher = ClaimEnrichmentTool(
            name="test_claim_enrichment",
            model_router=self.mock_router
        )
        
        # Check that template system is initialized
        self.assertIsNotNone(enricher.prompt_manager)
        self.assertEqual(enricher.template_name, "claim_enrichment")
    
    def test_question_generator_template_integration(self):
        """Test question generator uses template system."""
        generator = QAGenerationTool(
            name="test_question_generator",
            model_router=self.mock_router
        )
        
        # Check that template system is initialized
        self.assertIsNotNone(generator.prompt_manager)
        self.assertEqual(generator.template_name, "question_generation")
    
    def test_evidence_tagger_template_integration(self):
        """Test evidence tagger uses template system."""
        tagger = EvidenceTagger(
            name="test_evidence_tagger",
            model_router=self.mock_router
        )
        
        # Check that template system is initialized
        self.assertIsNotNone(tagger.prompt_manager)
        self.assertEqual(tagger.template_name, "evidence_tagging")
    
    def test_synthesizer_template_integration(self):
        """Test synthesizer uses template system."""
        synthesizer = Synthesizer(
            name="test_synthesizer",
            model_router=self.mock_router
        )
        
        # Check that template system is initialized
        self.assertIsNotNone(synthesizer.prompt_manager)
        self.assertEqual(synthesizer.template_name, "evidence_synthesis")


class TestTemplateFiles(unittest.TestCase):
    """Test that all required template files exist and are valid."""
    
    def setUp(self):
        """Set up test environment."""
        self.prompts_dir = os.path.join(project_root, 'prompts')
        self.manager = PromptManager(prompts_dir=self.prompts_dir)
    
    def test_relevance_check_template_exists(self):
        """Test relevance check template exists and is loadable."""
        template = self.manager.load_template('relevance_check')
        self.assertIsNotNone(template)
        
        # Test rendering with empty context
        result = self.manager.render_template('relevance_check', {})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_claim_enrichment_template_exists(self):
        """Test claim enrichment template exists and is loadable."""
        template = self.manager.load_template('claim_enrichment')
        self.assertIsNotNone(template)
        
        # Test rendering with empty context
        result = self.manager.render_template('claim_enrichment', {})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_question_generation_template_exists(self):
        """Test question generation template exists and is loadable."""
        template = self.manager.load_template('question_generation')
        self.assertIsNotNone(template)
        
        # Test rendering with required variables
        result = self.manager.render_template('question_generation', {
            'enriched_claim': 'Test claim',
            'image_context': 'Test context'
        })
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('Test claim', result)
        self.assertIn('Test context', result)
    
    def test_evidence_tagging_template_exists(self):
        """Test evidence tagging template exists and is loadable."""
        template = self.manager.load_template('evidence_tagging')
        self.assertIsNotNone(template)
        
        # Test rendering with empty context
        result = self.manager.render_template('evidence_tagging', {})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_evidence_synthesis_template_exists(self):
        """Test evidence synthesis template exists and is loadable."""
        template = self.manager.load_template('evidence_synthesis')
        self.assertIsNotNone(template)
        
        # Test rendering with empty context
        result = self.manager.render_template('evidence_synthesis', {})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class MockModelRouter:
    """Mock model router for testing."""
    
    def __init__(self):
        self.name = "mock_router"
    
    def llm_multimodal(self, system_prompt, user_message, image_path):
        """Mock multimodal LLM call."""
        return MockResponse("Mock response")
    
    def get_llm(self, provider, model_name):
        """Mock LLM getter."""
        return MockLLM()


class MockResponse:
    """Mock response object."""
    
    def __init__(self, content):
        self.content = content


class MockLLM:
    """Mock LLM object."""
    
    def __init__(self):
        self.name = "mock_llm"


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)