# H-Mem

A memory system for LLM agents that **learns** — not just stores.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What It Is

H-Mem is a hybrid memory mechanism for long-lived LLM agents. Instead of treating memory as flat chunks, it models how memory **evolves** over time and **retrieves** adaptively based on query complexity.

### Core Innovation

| Component | What It Does |
|-----------|-------------|
| **Temporal-Semantic Tree** | Short-term conversations progressively consolidate into long-term summaries |
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

# Add memory fragments as they happen
for convo_turn in conversation_history:
    hmem.index(MemoryFragment(
        text=convo_turn,
        timestamp=convo_timestamp,
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

### 4. Evaluate on Benchmarks

```bash
hmem benchmark --dataset locom
hmem benchmark --dataset longmemeval
hmem benchmark --dataset realtalk
```

---

## Architecture

```
hmem/
├── core/
│   ├── memory_fragment.py    # Atomic unit of memory
│   ├── tree.py               # Temporal-semantic tree
│   └── graph.py              # Knowledge graph (NetworkX)
├── indexing/
│   ├── tree_builder.py       # Incremental tree construction
│   ├── graph_builder.py      # Entity/relation extraction + graph
│   └── consolidator.py       # Bottom-up memory consolidation
├── retrieval/
│   ├── planner.py            # Query decomposition + scope prediction
│   ├── tree_search.py        # Temporal/semantic evidence search
│   ├── graph_search.py       # Multi-hop entity traversal
│   └── synthesizer.py        # Sub-answer → final answer
├── llm/
│   └── adapter.py            # Pluggable LLM backend (OpenRouter, OpenAI, Anthropic, local)
└── evaluation/
    └── harness.py            # Benchmark evaluation (LoCoMo, LongMemEval, REALTALK)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design spec.

---

## LLM Backends

H-Mem ships with adapters for:

| Provider | Env Var | Status |
|----------|---------|--------|
| **OpenRouter** | `OPENROUTER_API_KEY` | ✅ Default — cheapest access to GPT-4o-mini / 4.1-mini |
| OpenAI | `OPENAI_API_KEY` | ✅ Direct |
| Anthropic | `ANTHROPIC_API_KEY` | ✅ Claude 3.5 Sonnet / 4 |
| Local (vLLM/Ollama) | `LOCAL_LLM_URL` | ✅ Self-hosted |

The LLM is used for:
- **Entity/relation extraction** (graph construction)
- **Temporal-semantic consolidation** (tree merging)
- **Retrieval planning** (query decomposition, scope prediction)
- **Missing-information detection** (follow-up query generation)
- **Answer synthesis** (sub-answer → final)

All LLM calls are batched where possible and cached to reduce cost.

---

## Benchmarks

This repo includes evaluation harnesses for all three benchmarks from the paper:

| Benchmark | Conversations | Avg Turns | Sessions | Focus |
|-----------|---------------|-----------|----------|-------|
| **LoCoMo** | 1,200+ | 300 | Up to 35 | Long-term QA |
| **LongMemEvalS** | Multi-session | 100s | Multiple | Multi-session reasoning |
| **REALTALK** | 21-day real-world | 100s+ | Continuous | Realistic noise + persona |

```bash
# Download datasets
python scripts/download_datasets.py

# Run full evaluation
python -m hmem.evaluation.harness --dataset all --model openrouter/gpt-4o-mini
```

---

## Configuration

```python
from hmem import HMemConfig

config = HMemConfig(
    llm_provider="openrouter",           # or "openai", "anthropic", "local"
    llm_model="openai/gpt-4o-mini",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    tree_similarity_threshold=0.75,       # For consolidation
    tree_max_depth=4,                    # Leaf = short-term, root = long-term
    graph_entity_threshold=2,            # Min mentions to be salient
    retrieval_top_k=10,
    retrieval_workflows=True,           # Enable adaptive planner
    missing_info_retrieval=True,         # Enable follow-up retrieval
    cache_dir=".cache/hmem",
)

hmem = HMem(config)
```

---

## Roadmap

| Phase | Target | Status |
|-------|--------|--------|
| 1. Core structures | Tree + graph working with toy data | 🔄 In progress |
| 2. Indexing pipeline | Full offline indexing + consolidation | ⏳ Planned |
| 3. Retrieval pipeline | Planner + evidence search + synthesis | ⏳ Planned |
| 4. LLM abstraction | OpenRouter + pluggable backends | ⏳ Planned |
| 5. Benchmarks | LoCoMo, LongMemEval, REALTALK | ⏳ Planned |
| 6. Optimization | Batch processing, async, caching | ⏳ Planned |
| 7. Release | PyPI + docs + benchmark reproduction | ⏳ Planned |

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
