# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.1.0-rc.1] — 2025-05-30

### Added
- **Core memory architecture**: `TemporalSemanticTree` + `KnowledgeGraph` dual structure for evolving short-term conversations into long-term structured knowledge.
- **Indexing pipeline**: `TreeBuilder`, `GraphBuilder`, and `Indexer` for offline memory construction.
- **Retrieval engine**: `Planner` that decomposes and scopes queries; `TreeSearcher` + `GraphSearcher` for hybrid retrieval; `Synthesizer` for final answer generation.
- **LLM adapters**: OpenRouter (cheapest production access), OpenAI, Anthropic, and local (Ollama/SGLang) with fallback chaining.
- **Evaluation suite**: F1, Exact Match, and LLM-as-Judge metrics; `EvalHarness` for LoCoMo, LongMemEvalS, and REALTALK.
- **Benchmark datasets**: Loaders for LoCoMo, LongMemEvalS, and REALTALK with download scripts.
- **Smoke tests**: No-LLM validation via `DummyEmbedder` and `DummyLLM` for CI and onboarding.
- **CLI**: `hmem index`, `hmem query`, `hmem benchmark`.
- **Documentation**: Architecture docs, quickstart, install guide, contribution guidelines.
- **License**: MIT.
