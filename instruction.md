# MMFakeBench: Multimodal Misinformation Detection Benchmarking Toolkit

A modular Python framework for running repeatable, data-driven benchmarks across multiple Large Language Models (LLMs) for multimodal misinformation detection tasks.

## 1. Why this project?

- **Compare multimodal models fairly** – Measure latency, cost, and detection accuracy for OpenAI GPT-4o, Google Gemini, and other vision-language models
- **Rapid experimentation** – Swap models, prompts, and evaluation datasets with minimal code changes
- **Auditability** – Persist all run metadata, raw generations, and evidence chains to CSV for later inspection
- **Agentic workflow** – Multi-step pipeline with image-headline relevance checking, web search, and RAG-style evidence synthesis

## 2. High-level architecture

```
mmfakebench/
├── __main__.py              ← CLI entry-point
├── core/                    ← Model-agnostic orchestration
│   ├── runner.py            ← Runs a single benchmark job
│   ├── pipeline.py          ← Advanced RAG pipeline orchestrator
│   └── io.py                ← CSV + logging helpers
├── models/                  ← One sub-package per provider
│   ├── router.py            ← Unified model router (OpenAI/Gemini)
│   ├── openai/
│   │   └── client.py
│   ├── gemini/
│   │   └── client.py
│   └── …/                   ← (add Claude, Qwen, etc.)
├── modules/                 ← Specialized pipeline components
│   ├── relevance_checker.py ← Image-headline relevance analysis
│   ├── claim_enrichment.py  ← Claim extraction and image context
│   ├── question_generator.py ← Fact-checking question generation
│   ├── web_searcher.py      ← Web search and evidence gathering
│   ├── evidence_tagger.py   ← Evidence classification
│   └── synthesizer.py       ← Final decision synthesis
├── datasets/                ← Dataset loaders and processors
│   ├── mmfakebench.py       ← MMFakeBench dataset loader
│   ├── mocheg.py            ← MOCHEG dataset loader
│   └── base.py              ← Base dataset interface
├── prompts/                 ← Pre-prompt templates (plain text / Jinja2)
│   ├── relevance_check.txt
│   ├── claim_enrichment.txt
│   ├── question_generation.txt
│   ├── evidence_synthesis.txt
│   └── …
├── configs/                 ← YAML files describing each benchmark run
│   ├── mmfakebench_baseline.yml
│   ├── mocheg_evaluation.yml
│   └── …
├── utils/                   ← Utility functions
│   ├── image_processor.py   ← Image encoding and processing
│   ├── metrics.py           ← Evaluation metrics
│   └── logging.py           ← Structured logging
└── results/                 ← Auto-generated CSV & logs
```

**Design rule:** Nothing inside `core/` should import provider-specific code.

## 3. Installation

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional provider SDKs
pip install openai google-generativeai anthropic langchain langchain-openai langchain-google-genai
pip install duckduckgo-search brave-search-api
```

Add your API keys as environment variables or in a `.env` file:
```bash
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
SEARCH_API=...  # Brave Search API key
```

## 4. Configuration files (configs/*.yml)

```yaml
name: mmfakebench-baseline
model: gemini-2.0-flash  # or openai/gpt-4o-mini
dataset:
  type: mmfakebench
  json_path: data/MMFakeBench_test.json
  images_dir: data/MMFakeBench_test/
  limit: 100
pipeline:
  num_claims_per_branch: 3
  num_chains: 3
  enable_web_search: true
model_params:
  temperature: 0.2
  max_retries: 5
output:
  csv_path: results/mmfakebench-baseline.csv
  log_level: INFO
  verbose: false
```

All keys map 1-to-1 to pipeline and model parameters.

## 5. Prompt templates (prompts/*.txt)

Plain text or Jinja2 placeholders:

```text
# prompts/relevance_check.txt
You are a news fact-checking assistant. Analyze the relationship between a news image and headline.
Determine if the image could reasonably illustrate the headline content. Consider:
- People/objects shown vs described
- Time indicators (clothing, technology, weather)
- Location clues
- Event specificity

Respond ONLY in one of these two formats:
"No, the image is appropriately used."
OR
"Yes, potential mismatch: [concise reason]."

Headline: {{ headline }}
```

Select templates in your config; variables are filled at runtime from each input.

## 6. Adding a new model provider

1. Create a sub-package under `models/` (e.g., `models/anthropic`)
2. Implement a Client class:

```python
from models.base import BaseClient

class AnthropicClient(BaseClient):
    name = "claude-3-sonnet"
    
    def __init__(self, api_key, **kwargs):
        self.client = anthropic.Anthropic(api_key=api_key)
        
    def generate(self, messages, **params) -> str:
        # Implement provider-specific generation logic
        pass
        
    def generate_multimodal(self, system_prompt, text, image_path, **params) -> str:
        # Implement multimodal generation
        pass
```

3. Register it in `models/__init__.py`:

```python
from .anthropic.client import AnthropicClient
MODEL_REGISTRY["claude-3-sonnet"] = AnthropicClient
```

## 7. Running benchmarks

```bash
# Single config
python -m mmfakebench --config configs/mmfakebench_baseline.yml

# All configs in folder
python -m mmfakebench --config-dir configs/ --all

# Override specific parameters
python -m mmfakebench -c configs/baseline.yml --model gpt-4o-mini --limit 50
```

CLI flags override YAML values. Run `python -m mmfakebench --help` for full options.

## 8. Outputs

Each run produces artifacts in `results/`:

| File | Description |
|------|-------------|
| `<name>.csv` | One row per input with: image_path, headline, prediction, confidence, evidence_chain, latency, cost, etc. |
| `<name>.log` | Structured logs (JSON) for debugging/re-runs |
| `<name>_detailed.csv` | Detailed pipeline step outputs for analysis |

**Tip:** Load the CSV into a notebook for custom scoring or qualitative inspection.

## 9. Pipeline Architecture

The agentic workflow consists of these modules:

### ☑️ 9.1 ImageHeadlineRelevancyChecker
- **Purpose:** Analyze if image appropriately illustrates the headline
- **Input:** Image path, headline text
- **Output:** Relevance assessment, confidence score
- **Implementation:** `modules/relevance_checker.py`

### ☑️ 9.2 ClaimEnrichmentTool
- **Purpose:** Extract core claims and describe image context
- **Input:** Image, headline
- **Output:** Restated claim, detailed image description
- **Implementation:** `modules/claim_enrichment.py`

### ☑️ 9.3 QuestionGeneratorTool
- **Purpose:** Generate fact-checking questions based on claims
- **Input:** Enriched claim, image context
- **Output:** List of targeted questions
- **Implementation:** `modules/question_generator.py`

### ☑️ 9.4 WebSearchTool
- **Purpose:** Search web for evidence using generated questions
- **Input:** Questions list
- **Output:** Search results with snippets
- **Implementation:** `modules/web_searcher.py`

### ☑️ 9.5 EvidenceTagger
- **Purpose:** Classify evidence as supporting/refuting/background
- **Input:** Question-answer pairs, claim
- **Output:** Evidence tags and relevance scores
- **Implementation:** `modules/evidence_tagger.py`

### ☑️ 9.6 EvidenceSynthesizer
- **Purpose:** Make final misinformation determination
- **Input:** All evidence, tags, confidence scores
- **Output:** Binary classification + reasoning
- **Implementation:** `modules/synthesizer.py`

## 10. Dataset Support

### ☑️ 10.1 MMFakeBench Dataset
- **Format:** JSON with image paths and labels
- **Labels:** Binary (True/Fake) + Multiclass (original/mismatch/etc.)
- **Loader:** `datasets/mmfakebench.py`

### ☑️ 10.2 MOCHEG Dataset
- **Format:** CSV with topic-based organization
- **Features:** Image-evidence relationships
- **Loader:** `datasets/mocheg.py`

## 11. Extensibility tips

- **Keep providers thin** – Only translate between internal schema ↔ SDK calls
- **Use Pydantic models** for type-safe configs & records
- **Modular prompts** – Create templates for different strategies (CoT, RAG, etc.)
- **Extend evaluations** by adding columns in `io.write_row()` – backward compatible
- **Plugin architecture** – New modules should inherit from base classes

## 12. Troubleshooting

| Symptom | Possible Cause |
|---------|----------------|
| `KeyError: my-model` | Forgot to register client in `MODEL_REGISTRY` |
| Empty CSV output | Check `name` & `output_csv` paths – duplicates are overwritten |
| Slow Gemini uploads | Reduce `batch_size` or check quota limits |
| Image encoding errors | Check image format support and file permissions |
| Rate limit errors | Implement exponential backoff in model router |

## 13. Development Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Model router with OpenAI/Gemini support
- [x] Basic pipeline orchestration
- [x] MMFakeBench dataset loader
- [x] Image processing utilities

### Phase 2: Pipeline Modules
- [ ] Refactor relevance checker into standalone module
- [ ] Implement claim enrichment module
- [ ] Build question generator with templates
- [ ] Add web search integration (Brave/DuckDuckGo)
- [ ] Create evidence tagging system
- [ ] Develop evidence synthesizer

### Phase 3: Advanced Features
- [ ] Add Claude, Qwen model support
- [ ] Implement MOCHEG dataset loader
- [ ] Add automatic statistical tests (paired bootstrap)
- [ ] Create HTML dashboard for multi-run analysis
- [ ] Plugin evaluators (BLEU, Rouge, GPT-based scoring)
- [ ] Parallel execution with Ray/asyncio

### Phase 4: Production Ready
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Comprehensive test suite
- [ ] Documentation website
- [ ] Performance optimization
- [ ] Cost tracking and optimization

## 14. Contributing

When implementing new modules:

1. **Follow the interface contracts** defined in `core/base.py`
2. **Add comprehensive tests** in `tests/`
3. **Update configuration schemas** in `configs/`
4. **Document new prompt templates** in `prompts/`
5. **Add usage examples** in `examples/`

## 15. Research Applications

- **Model comparison studies** across vision-language models
- **Prompt engineering** for multimodal fact-checking
- **Evidence quality analysis** in automated fact-checking
- **Cost-accuracy trade-offs** in production systems
- **Robustness testing** against adversarial examples

---

*This toolkit is designed for academic/research use. Always secure your API keys and consult provider documentation for up-to-date rate limits and pricing.*