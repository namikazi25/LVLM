# MMFakeBench Implementation To-Do List

Based on the analysis of the current Python implementation and the proposed architecture in `instruction.md`, here's a comprehensive to-do list to transform the existing Colab notebook into a modular, production-ready system:

## 🏗️ **Phase 1: Core Infrastructure Refactoring**

### 1.1 Project Structure Setup
- [x] **Create modular directory structure** as outlined in `instruction.md`:
  ```
  mmfakebench/
  ├── __main__.py
  ├── core/
  ├── models/
  ├── modules/
  ├── datasets/
  ├── prompts/
  ├── configs/
  ├── utils/
  └── results/
  ```
- [x] **Create `requirements.txt`** with all dependencies from the notebook
- [x] **Setup proper Python package structure** with `__init__.py` files
- [x] **Create `.env.example`** file for API key configuration

**🧪 Phase 1.1 Testing:**
- [ ] **Verify directory structure**: `ls -la mmfakebench/` shows all directories
- [ ] **Test package imports**: `python -c "import mmfakebench; print('✅ Package structure OK')"`
- [ ] **Validate requirements**: `pip install -r mmfakebench/requirements.txt --dry-run`
- [ ] **Check CLI entry point**: `python -m mmfakebench --help` shows usage
- [ ] **Verify environment template**: `.env.example` contains all required API keys

### 1.2 Core Module Development
- [x] **Extract and refactor `ModelRouter`** → `models/router.py`
- [x] **Create base classes** in `core/base.py` for:
  - `BaseClient` (model interface)
  - `BaseDataset` (dataset interface)
  - `BasePipelineModule` (pipeline component interface)
- [x] **Build `core/runner.py`** - main benchmark orchestrator
- [x] **Build `core/pipeline.py`** - pipeline workflow manager
- [x] **Build `core/io.py`** - CSV output and logging utilities

**🧪 Phase 1.2 Testing:**
- [x] **Test base class imports**: `python -c "from mmfakebench.core.base import BaseClient, BaseDataset, BasePipelineModule; print('✅ Base classes OK')"`
- [x] **Test ModelRouter**: `python -c "from mmfakebench.models.router import ModelRouter; print('✅ ModelRouter OK')"`
- [x] **Test core modules**: `python -c "from mmfakebench.core import runner, pipeline, io; print('✅ Core modules OK')"`
- [x] **Validate abstract methods**: Ensure base classes raise NotImplementedError for abstract methods
- [x] **Test runner initialization**: Create BenchmarkRunner instance without errors
- [x] **Test I/O utilities**: Create sample CSV output and verify logging setup

### 1.3 Configuration System
- [x] **Create YAML configuration schema** using Pydantic models
- [x] **Build configuration loader** in `core/config.py`
- [x] **Create sample config files**:
  - `configs/mmfakebench_baseline.yml`
  - `configs/mocheg_evaluation.yml`

**🧪 Phase 1.3 Testing:**
- [x] **Test config loading**: `python -c "from mmfakebench.core.config import ConfigManager; cm = ConfigManager(); print('✅ Config manager OK')"`
- [x] **Validate YAML parsing**: Load sample config files without errors
- [x] **Test Pydantic validation**: Invalid configs should raise validation errors
- [x] **Test CLI config validation**: `python -m mmfakebench validate --config configs/mmfakebench_baseline.yml`
- [ ] **Test config override**: CLI parameters should override config file values
- [ ] **Test environment variable substitution**: Config should support ${ENV_VAR} syntax

## 🔧 **Phase 2: Pipeline Module Extraction**

### 2.1 Extract Existing Components
- [x] **Extract `ImageHeadlineRelevancyChecker`** → `modules/relevance_checker.py`
- [x] **Extract `ClaimEnrichmentTool`** → `modules/claim_enrichment.py`
- [x] **Extract `QAGenerationTool`** → `modules/question_generator.py`
- [x] **Extract `EvidenceTagger`** → `modules/evidence_tagger.py`
- [x] **Extract image processing functions** → `utils/image_processor.py`

**🧪 Phase 2.1 Testing:**
- [x] **Test module imports**: `python -c "from mmfakebench.modules import relevance_checker, claim_enrichment, question_generator, evidence_tagger; print('✅ Modules OK')"`
- [x] **Test image processing**: `python -c "from mmfakebench.utils.image_processor import encode_image; print('✅ Image utils OK')"`
- [x] **Test module initialization**: All modules can be instantiated without errors
- [x] **Test base class inheritance**: All modules should inherit from BasePipelineModule
- [ ] **Test relevance checker**: Initialize with mock ModelRouter and test with sample image/headline
- [ ] **Test claim enrichment**: Verify claim extraction and image context description
- [ ] **Test question generator**: Generate questions from sample claims and verify format
- [ ] **Test evidence tagger**: Classify sample Q&A pairs with expected outputs

### 2.2 Build Missing Components
- [x] **Create `modules/web_searcher.py`** - integrate Brave Search and DuckDuckGo
- [x] **Create `modules/synthesizer.py`** - final decision synthesis module
- [x] **Create `utils/metrics.py`** - evaluation metrics and confusion matrix utilities
- [x] **Create `utils/logging.py`** - structured logging setup

**🧪 Phase 2.2 Testing:**
- [x] **Test web searcher imports**: `python -c "from mmfakebench.modules.web_searcher import WebSearcher; print('✅ WebSearcher OK')"`
- [x] **Test search functionality**: Perform sample searches with both Brave and DuckDuckGo APIs
- [x] **Test decision synthesizer**: `python -c "from mmfakebench.modules.synthesizer import Synthesizer; print('✅ Synthesizer OK')"`
- [x] **Test verdict generation**: Generate final verdicts from sample evidence and verify format
- [x] **Test metrics calculator**: `python -c "from mmfakebench.utils.metrics import MetricsCalculator; print('✅ MetricsCalculator OK')"`
- [x] **Test metrics computation**: Calculate accuracy, precision, recall from sample predictions
- [x] **Test API error handling**: Verify graceful handling of API failures and rate limits

### 2.3 Standardize Module Interfaces
- [x] **Implement consistent input/output schemas** for all modules
- [x] **Add error handling and retry logic** to all components
- [x] **Add configuration parameters** for each module

**🧪 Phase 2.3 Testing:**
- [x] **Test interface consistency**: All modules should have standardized `process()` method signatures
- [x] **Test error handling**: Modules should handle invalid inputs gracefully with proper error messages
- [x] **Test logging integration**: Verify all modules log operations with consistent format
- [x] **Test module registry**: `python -c "from mmfakebench.core.registry import ModuleRegistry; print('✅ Registry OK')"`
- [x] **Test dynamic loading**: Load modules by name through registry system
- [x] **Test pipeline integration**: Chain multiple modules together and verify data flow
- [x] **Test schema validation**: Invalid inputs should raise appropriate validation errors

## 📊 **Phase 3: Dataset Management**

### 3.1 Dataset Loaders
- [x] **Extract `MMFakeBenchDataset`** → `datasets/mmfakebench.py`
- [x] **Create `datasets/mocheg.py`** - MOCHEG dataset loader
- [x] **Create `datasets/base.py`** - base dataset interface
- [x] **Add dataset validation and preprocessing** utilities

**🧪 Phase 3.1 Testing:**
- [x] **Test base dataset**: `python -c "from mmfakebench.datasets.base import BaseDataset; print('✅ BaseDataset OK')"`
- [x] **Test MMFakeBench dataset**: `python -c "from mmfakebench.datasets.mmfakebench import MMFakeBenchDataset; print('✅ MMFakeBench OK')"`
- [x] **Test MOCHEG dataset**: `python -c "from mmfakebench.datasets.mocheg import MOCHEGDataset; print('✅ MOCHEG OK')"`
- [x] **Test dataset loading**: Load sample data files and verify structure
- [x] **Test data validation**: Invalid data should raise appropriate errors
- [x] **Test preprocessing**: Verify image encoding and text normalization
- [x] **Test iteration**: Datasets should be iterable with consistent item format

### 3.2 Data Pipeline
- [x] **Implement data sampling strategies** (random, stratified, etc.)
- [x] **Add data augmentation options** (if needed)
- [x] **Create data statistics and preview functions**

**🧪 Phase 3.2 Testing:**
- [x] **Test sampling strategies**: `python -c "from mmfakebench.datasets.sampling import RandomSampler, StratifiedSampler; print('✅ Samplers OK')"`
- [x] **Test random sampling**: Verify random selection produces different subsets
- [x] **Test stratified sampling**: Ensure balanced representation across categories
- [x] **Test data augmentation**: Verify augmented samples maintain label consistency
- [x] **Test statistics generation**: Compute dataset statistics and verify accuracy
- [x] **Test preview functions**: Generate data previews with correct formatting
- [x] **Test memory efficiency**: Large datasets should not cause memory issues

## 🤖 **Phase 4: Model Provider System**

### 4.1 Refactor Existing Providers
- [x] **Extract OpenAI client** → `models/openai/client.py`
- [x] **Extract Gemini client** → `models/gemini/client.py`
- [x] **Create model registry system** in `models/__init__.py`

**🧪 Phase 4.1 Testing:**
- [x] **Test model router**: `python -c "from mmfakebench.models.router import ModelRouter; print('✅ ModelRouter OK')"`
- [x] **Test OpenAI provider**: `python -c "from mmfakebench.models.openai.client import OpenAIClient; print('✅ OpenAI OK')"`
- [x] **Test Gemini provider**: `python -c "from mmfakebench.models.gemini.client import GeminiClient; print('✅ Gemini OK')"`
- [x] **Test model registry**: `python -c "from mmfakebench.models import ModelRegistry; print('✅ Registry OK')"`
- [x] **Test unified interface**: All clients should have consistent `generate()` method
- [x] **Test model switching**: Router should seamlessly switch between providers
- [x] **Test error handling**: Invalid API keys should raise appropriate errors

### 4.2 Add New Providers
- [ ] **Add Claude support** → `models/anthropic/client.py`
- [ ] **Add Qwen support** → `models/qwen/client.py`
- [ ] **Implement provider-agnostic cost tracking**

**🧪 Phase 4.2 Testing:**
- [ ] **Test Claude integration**: `python -c "from mmfakebench.models.anthropic.client import AnthropicClient; print('✅ Claude OK')"`
- [ ] **Test Qwen integration**: `python -c "from mmfakebench.models.qwen.client import QwenClient; print('✅ Qwen OK')"`
- [ ] **Test cost tracking**: Verify accurate cost calculation across all providers
- [ ] **Test provider registration**: New providers should auto-register with ModelRouter
- [ ] **Test API compatibility**: All providers should implement consistent interface
- [ ] **Test error handling**: Provider-specific errors should be handled gracefully
- [ ] **Test cost limits**: System should respect configured spending limits

### 4.3 Model Management
- [x] **Add model switching capabilities**
- [x] **Implement rate limiting and quota management**
- [x] **Add model performance monitoring**

**🧪 Phase 4.3 Testing:**
- [x] **Test model switching**: `python -c "from mmfakebench.models.router import ModelRouter; r = ModelRouter(); r.switch_model('gpt-4'); print('✅ Switching OK')"`
- [x] **Test performance monitoring**: Verify latency and token usage tracking
- [x] **Test comparison utilities**: Generate model performance comparisons
- [x] **Test model validation**: Invalid model names should raise appropriate errors
- [x] **Test performance metrics**: Accuracy, speed, and cost metrics should be tracked
- [x] **Test model health checks**: Verify model availability before use

## 📝 **Phase 5: Prompt Management**

### 5.1 Template System ✅ COMPLETED
- [x] **Extract prompts to separate files**:
  - `prompts/relevance_check.txt` ✅
  - `prompts/claim_enrichment.txt` ✅
  - `prompts/question_generation.txt` ✅
  - `prompts/evidence_tagging.txt` ✅
  - `prompts/synthesis.txt` ✅
- [x] **Implement Jinja2 templating support** ✅
- [x] **Create centralized PromptManager class** ✅
- [x] **Integrate template system with all modules** ✅
- [x] **Add dynamic variable substitution** ✅

**🧪 Phase 5.1 Testing:** ✅ ALL TESTS PASSED
- [x] **Test prompt loading**: `python -m mmfakebench.test_phase_5_1` ✅
- [x] **Test template rendering**: Verified Jinja2 templates render with sample data ✅
- [x] **Test module integration**: All 5 core modules successfully integrated ✅
- [x] **Test error handling**: Missing template errors handled gracefully ✅

**📋 Implementation Summary:**
- Created `core/prompts.py` with PromptManager class
- Extracted all hardcoded prompts to template files in `prompts/` directory
- Updated all modules: relevance_checker, claim_enrichment, question_generator, evidence_tagger, synthesizer
- Added Jinja2 variable support (e.g., `{{ claim }}`, `{{ image_context }}`)
- Implemented singleton pattern for centralized prompt management
- Created comprehensive test suite with 15 test cases (all passing)
- [ ] **Test prompt organization**: All module prompts should be properly categorized
- [ ] **Test template validation**: Invalid templates should raise syntax errors
- [ ] **Test variable substitution**: Templates should correctly substitute variables
- [ ] **Test prompt versioning**: Different prompt versions should be supported
- [ ] **Test missing template handling**: Missing prompts should raise appropriate errors

### 5.2 Prompt Engineering ✅ COMPLETED
- [x] **Optimize existing prompts** based on current implementation ✅
- [x] **Add Chain-of-Thought (CoT) prompt variants** ✅
- [x] **Create few-shot learning examples** ✅

**🧪 Phase 5.2 Testing:** ✅ ALL TESTS PASSED
- [x] **Test prompt optimization**: Created optimized versions of all 5 core prompts ✅
- [x] **Test CoT variants**: Implemented Chain-of-Thought reasoning for all templates ✅
- [x] **Test few-shot examples**: Added examples to improve model performance ✅
- [x] **Test template completeness**: All 10 new templates (5 CoT + 5 optimized) validated ✅
- [x] **Test backward compatibility**: Original templates still functional ✅

**📋 Implementation Summary:**
- Created 5 Chain-of-Thought (CoT) prompt variants with step-by-step reasoning
- Created 5 optimized prompt variants with improved structure and clarity
- Added few-shot learning examples to all CoT templates
- Enhanced prompt structure with clear guidelines and formatting
- Maintained backward compatibility with original templates
- Created comprehensive test suite with 8 test cases (all passing)
- Total of 15 prompt templates now available (5 original + 5 CoT + 5 optimized)
- [ ] **Test A/B testing**: Compare multiple prompt versions systematically
- [ ] **Test prompt regression**: Ensure prompt changes don't break existing functionality

## 🖥️ **Phase 6: CLI and Interface**

### 6.1 Command Line Interface ✅
- [x] **Create `__main__.py`** with CLI commands:
  - `run` - execute benchmark with config
  - `preview` - preview dataset samples
  - `validate` - validate configuration
- [x] **Add parameter override capabilities** via CLI flags
- [x] **Implement progress indicators** and user-friendly output

**🧪 Phase 6.1 Testing:** ✅ **COMPLETED**
- [x] **Test CLI commands**: Enhanced CLI with comprehensive argument parsing and validation
- [x] **Test parameter overrides**: Implemented `parse_override_params()` and `apply_overrides()` methods
- [x] **Test progress indicators**: Added verbose/quiet logging options and dry-run mode
- [x] **Test error handling**: Comprehensive error handling with specific exit codes
- [x] **Test help system**: Complete help documentation with usage examples
- [x] **All 12 test cases passing** in `test_phase_6_1.py`

**✨ Key Enhancements Implemented:**
- Enhanced argument parser with global flags (--verbose, --quiet)
- Parameter override system supporting dot notation (model.temperature=0.5)
- Dry-run mode for configuration validation
- Resume functionality with checkpoint support
- Sample limiting capabilities
- Comprehensive error handling with specific exit codes
- User-friendly help system with examples

### 6.2 Output Management
- [ ] **Implement structured CSV output** with all metrics
- [ ] **Add detailed logging with different verbosity levels**
- [ ] **Create result visualization utilities**

**🧪 Phase 6.2 Testing:**
- [ ] **Test CSV output**: `python -c "from mmfakebench.core.io import CSVExporter; print('✅ CSV export OK')"`
- [ ] **Test JSON export**: Verify JSON output contains all required fields
- [ ] **Test result structure**: Output should have consistent schema across runs
- [ ] **Test visualization**: `python -c "from mmfakebench.utils.visualization import ResultVisualizer; print('✅ Visualization OK')"`
- [ ] **Test comparison utilities**: Compare results from different runs
- [ ] **Test data integrity**: Exported data should match internal results
- [ ] **Test large result handling**: System should handle large result sets efficiently

## 🧪 **Phase 7: Testing and Quality**

### 7.1 Test Suite
- [ ] **Create unit tests** for all modules
- [ ] **Add integration tests** for full pipeline
- [ ] **Create mock data and API responses** for testing
- [ ] **Add performance benchmarks**

**🧪 Phase 7.1 Testing:**
- [ ] **Test suite setup**: `python -m pytest tests/ --collect-only` shows all tests
- [ ] **Test coverage**: `python -m pytest tests/ --cov=mmfakebench --cov-report=html`
- [ ] **Test core modules**: All core classes should have >90% test coverage
- [ ] **Test integration**: End-to-end pipeline tests should pass
- [ ] **Test mock providers**: Tests should run without real API calls
- [ ] **Test error scenarios**: Edge cases and error conditions should be tested
- [ ] **Test performance**: Critical paths should have performance benchmarks

### 7.2 Code Quality
- [ ] **Add type hints** throughout codebase
- [ ] **Implement code formatting** (Black, isort)
- [ ] **Add linting** (flake8, pylint)
- [ ] **Create pre-commit hooks**

**🧪 Phase 7.2 Testing:**
- [ ] **Test type checking**: `python -m mypy mmfakebench/` should pass without errors
- [ ] **Test code formatting**: `python -m black --check mmfakebench/` should pass
- [ ] **Test linting**: `python -m flake8 mmfakebench/` should pass with minimal warnings
- [ ] **Test docstring coverage**: `python -m pydocstyle mmfakebench/` should show good coverage
- [ ] **Test documentation build**: `sphinx-build -b html docs/ docs/_build/` should succeed
- [ ] **Test import sorting**: `python -m isort --check-only mmfakebench/` should pass
- [ ] **Test security**: `python -m bandit -r mmfakebench/` should show no high-risk issues

## 🚀 **Phase 8: Advanced Features**

### 8.1 Performance Optimization
- [ ] **Add async/await support** for concurrent API calls
- [ ] **Implement caching** for repeated API calls
- [ ] **Add batch processing capabilities**
- [ ] **Optimize image processing pipeline**

**🧪 Phase 8.1 Testing:**
- [ ] **Test async processing**: `python -c "import asyncio; from mmfakebench.core.async_runner import AsyncRunner; print('✅ Async OK')"`
- [ ] **Test performance improvement**: Async should be faster than sync for multiple API calls
- [ ] **Test caching effectiveness**: Cached operations should be significantly faster
- [ ] **Test memory efficiency**: Large datasets should not cause memory leaks
- [ ] **Test parallel processing**: Multiple workers should improve throughput
- [ ] **Test resource limits**: System should respect CPU and memory constraints
- [ ] **Test scalability**: Performance should scale reasonably with dataset size

### 8.2 Analysis and Reporting
- [ ] **Create HTML dashboard** for multi-run analysis
- [ ] **Add statistical significance testing**
- [ ] **Implement cost-accuracy trade-off analysis**
- [ ] **Create comparative model reports**

**🧪 Phase 8.2 Testing:**
- [ ] **Test dashboard generation**: `python -c "from mmfakebench.reporting.dashboard import DashboardGenerator; print('✅ Dashboard OK')"`
- [ ] **Test HTML output**: Dashboard should render correctly in browsers
- [ ] **Test statistical analysis**: Statistical tests should produce valid results
- [ ] **Test result aggregation**: Multiple runs should aggregate correctly
- [ ] **Test plot generation**: Plots should be publication-quality and properly formatted
- [ ] **Test interactive features**: Dashboard should have working interactive elements
- [ ] **Test export functionality**: Reports should export to PDF/PNG formats

### 8.3 Extensibility
- [ ] **Create plugin architecture** for custom modules
- [ ] **Add custom evaluator support**
- [ ] **Implement experiment tracking** (MLflow, Weights & Biases)

**🧪 Phase 8.3 Testing:**
- [ ] **Test plugin loading**: `python -c "from mmfakebench.core.plugins import PluginManager; pm = PluginManager(); print('✅ Plugins OK')"`
- [ ] **Test custom modules**: Custom pipeline modules should integrate seamlessly
- [ ] **Test hooks system**: Pipeline hooks should execute at correct points
- [ ] **Test custom datasets**: Custom dataset loaders should work with existing pipeline
- [ ] **Test custom metrics**: User-defined metrics should be calculated correctly
- [ ] **Test plugin discovery**: System should auto-discover installed plugins
- [ ] **Test plugin isolation**: Plugin errors should not crash the main system

## 🐳 **Phase 9: Production Readiness**

### 9.1 Containerization
- [ ] **Create Dockerfile** for the application
- [ ] **Add docker-compose.yml** for development
- [ ] **Create deployment scripts**

**🧪 Phase 9.1 Testing:**
- [ ] **Test Docker build**: `docker build -t mmfakebench .` should succeed
- [ ] **Test Docker run**: `docker run mmfakebench --help` should work
- [ ] **Test environment configs**: Different environments should load correct settings
- [ ] **Test health checks**: `python -m mmfakebench health` should report system status
- [ ] **Test monitoring**: System should expose metrics for monitoring tools
- [ ] **Test deployment scripts**: Automated deployment should work end-to-end
- [ ] **Test resource requirements**: Container should run within specified resource limits

### 9.2 CI/CD Pipeline
- [ ] **Setup GitHub Actions** or similar CI/CD
- [ ] **Add automated testing** on multiple Python versions
- [ ] **Create release automation**

**🧪 Phase 9.2 Testing:**
- [ ] **Test CI pipeline**: GitHub Actions should run on push/PR
- [ ] **Test multi-version compatibility**: Tests should pass on Python 3.8, 3.9, 3.10, 3.11
- [ ] **Test security scanning**: Security tools should run without critical issues
- [ ] **Test automated releases**: Version tags should trigger automated releases
- [ ] **Test deployment pipeline**: Successful tests should trigger deployment
- [ ] **Test rollback capability**: Failed deployments should rollback automatically
- [ ] **Test notification system**: CI/CD failures should notify maintainers

### 9.3 Documentation
- [ ] **Create comprehensive README.md**
- [ ] **Add API documentation** (Sphinx)
- [ ] **Create usage examples and tutorials**
- [ ] **Add troubleshooting guide**

**🧪 Phase 9.3 Testing:**
- [ ] **Test README accuracy**: All installation and usage instructions should work
- [ ] **Test API documentation**: `sphinx-build docs/ docs/_build/` should generate complete API docs
- [ ] **Test code examples**: All code examples in documentation should execute successfully
- [ ] **Test tutorial completeness**: Users should be able to follow tutorials end-to-end
- [ ] **Test troubleshooting guides**: Common issues should have working solutions
- [ ] **Test documentation links**: All internal and external links should be valid
- [ ] **Test documentation currency**: Documentation should reflect current codebase

## 📋 **Priority Implementation Order**

### **High Priority (Weeks 1-2)**
1. Project structure setup (1.1)
2. Core module development (1.2)
3. Extract existing pipeline components (2.1)
4. Basic CLI interface (6.1)

### **Medium Priority (Weeks 3-4)**
1. Configuration system (1.3)
2. Dataset management (3.1-3.2)
3. Model provider refactoring (4.1)
4. Prompt management (5.1)

### **Lower Priority (Weeks 5-6)**
1. Testing suite (7.1-7.2)
2. Advanced features (8.1-8.2)
3. Documentation (9.3)

## 🎯 **Success Metrics**

- [ ] **Functional parity** with current notebook implementation
- [ ] **Modular architecture** allowing easy component swapping
- [ ] **Configuration-driven** execution without code changes
- [ ] **Comprehensive test coverage** (>80%)
- [ ] **Production-ready** deployment capabilities
- [ ] **Extensible design** for future enhancements

## 📝 **Implementation Notes**

### Current State Analysis
The existing implementation in `agentic_workflow_test_16(mmfakebench_implementation) (3).py` contains:

**✅ Already Implemented:**
- `ModelRouter` class with OpenAI/Gemini support
- `MMFakeBenchDataset` loader
- `ImageHeadlineRelevancyChecker` module
- `ClaimEnrichmentTool` for claim extraction
- `QAGenerationTool` for question generation
- `EvidenceTagger` for evidence classification
- Image processing utilities
- Basic evaluation metrics

**❌ Missing Components:**
- Modular project structure
- Configuration management system
- CLI interface
- Proper error handling and logging
- Test suite
- Documentation
- Production deployment setup

### Key Architectural Changes
1. **Separation of Concerns**: Split monolithic notebook into focused modules
2. **Configuration-Driven**: Replace hardcoded parameters with YAML configs
3. **Provider Abstraction**: Standardize model interfaces across providers
4. **Pipeline Modularity**: Make each step independently configurable and testable
5. **Error Resilience**: Add comprehensive error handling and retry logic

This transformation will convert the research prototype into a production-ready benchmarking toolkit suitable for academic research and industry applications.