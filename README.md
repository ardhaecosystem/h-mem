# H-Mem

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://img.shields.io/badge/arXiv-2605.15701-red)](https://arxiv.org/abs/2605.15701)

A memory system for LLM agents that **learns** — not just stores.

## What It Is

H-Mem is a hybrid memory mechanism for long-lived LLM agents. Instead of treating memory as flat chunks, it models how memory **evolves** over time and **retrieves** adaptively based on query complexity.

### Core Innovation

| Component | What It Does |
|-----------|-------------|
| **Temporal-Semantic Tree** | Short-term conversations progressly consolidate into long-term summaries |
| **Knowledge Graph** | Entities and relationships extracted for multi-hop reasoning |
| **Adaptive Retrieval Planner** | Per-query: decompose, select scope (SHORT/LONG/MIXED), and trigger follow-up when evidence is insufficient |

This is a reference implementation of the paper:  
**"H-Mem: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure"**  
(Yu et al., 2026 — arXiv:2605.15701)

---

## Why This Exists

Most agent memory is just RAG on conversation chunks. It works, but it doesn't:
- Model how memories compress and persist over time
- Handle queries that need both recent detail and old summaries
- Do multi-hop reasoning over entity relationships
- Adapt retrieval strategy per query

H-Mem fixes all four.

---

## Quick Start

```bash
pip install hmem
```

### 1. Configure

```bash
export OPENROUTER_API_KEY="sk-or-..."
# or export OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

### 2. Index Some Conversations

```python
from hmem import HMem, MemoryFragment

hmem = HMem()

for convo_turn in conversation_history:
    hmem.index(MemoryFragment(
        text=convo_turn,
        timestamp=convo_timestamp,
        source_type=SourceType.CONVERSATION,
        metadata={"session_id": session_id}
    ))

# Consolidation happens automatically after each batch
hmem.consolidate()
```

### 3. Query

```python
answer = hmem.query(
    "What did the user say about their project timeline last month?"
)
print(answer)
```

### 4. Evaluate

```bash
# Download benchmark datasets
python scripts/download_datasets.py

# Run evaluation
python -m hmem.evaluation.harness --dataset locom
python -m hmem.evaluation.harness --dataset longmemeval
python -m hmem.evaluation.harness --dataset realtalk
```

---

## Architecture

```
hmem/
├── core/
│   ├── types.py              # Pydantic types (MemoryFragment, TreeNode, Entity, Relation)
│   ├── tree.py               # Temporal-semantic tree
│   └── graph.py              # Knowledge graph (NetworkX)
├── indexing/
│   ├── tree_builder.py       # Incremental tree construction
│   ├── graph_builder.py      # Entity/relation extraction
│   └── indexer.py            # Orchestrates offline indexing
├── retrieval/
│   ├── planner.py            # Query decomposition + scope prediction
│   ├── tree_search.py        # Hierarchical tree search
│   ├── graph_search.py       # Multi-hop entity traversal
│   ├── reranker.py           # Semantic reranking
│   ├── synthesizer.py        # Answer synthesis
│   └── engine.py             # Full retrieval orchestration
├── llm/
│   ├── adapter.py            # Base LLM adapter
│   ├── openrouter.py         # OpenRouter adapter (default)
│   ├── openai.py             # OpenAI adapter
│   └── anthropic.py          # Anthropic adapter
├── evaluation/
│   ├── datasets/             # Benchmark loaders
│   ├── harness.py            # Evaluation runner
│   └── metrics.py            # F1, LLM-Judge, etc.
└── cli.py                    # Typer CLI
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design spec.

---

## LLM Backends

| Provider | Env Var | Status |
|----------|---------|--------|
| OpenRouter | `OPENROUTER_API_KEY` | ✅ Default |
| OpenAI | `OPENAI_API_KEY` | ✅ Direct |
| Anthropic | `ANTHROPIC_API_KEY` | ✅ Direct |
| Local (vLLM/Ollama) | `LOCAL_LLM_URL` | ✅ Self-hosted |

---

## Benchmarks

This repo includes evaluation harnesses for all three benchmarks from the paper:

| Benchmark | Description | Status |
|-----------|-------------|--------|
| **LoCoMo** | Very long-term conversational memory (up to 35 sessions) | 🔄 Loader |
| **LongMemEvalS** | Long-term interactive memory (115K tokens/session) | 🔄 Loader |
| **REALTALK** | 21-day real-world human conversations | 🔄 Loader |

---

## Development

```bash
git clone https://github.com/ardhaecosystem/h-mem.git
cd h-mem
pip install -e ".[dev]"

# Run smoke test (no LLM required)
python -m tests.smoke_test

# Run full benchmarks
python -m hmem.evaluation.harness --dataset all
```

---

## Roadmap

| Phase | Target | Status |
|-------|--------|--------|
| Core structures | Tree + Graph + Types | ✅ Done |
| Indexing pipeline | Offline indexing + consolidation | ✅ Done |
| Retrieval pipeline | Planner + Search + Synthesis | ✅ Done |
| LLM abstraction | OpenRouter / OpenAI / Anthropic | ✅ Done |
| Benchmarks | Data loaders + evaluation harness | 🔄 In progress |
| Optimization | Batch processing, async, caching | ⏳ Planned |
| Release | PyPI + docs | ⏳ Planned |

---

## Citation

```bibtex
@article{yu2026hmem,
  title={H-Mem: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure},
  author={Yu, Jiawei and Fang, Yixiang and Liu, Xilin and Ma, Yuchi},
  journal={arXiv preprint arXiv:2605.15701},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

Built with 🔥 by Humanth & Veda.
