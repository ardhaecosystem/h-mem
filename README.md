<div align="center">

# **H-Mem** 🔥
### Memory that *learns*

[![PyPI](https://img.shields.io/badge/pip-hmem-blue)](https://pypi.org/project/hmem/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Open Issues](https://img.shields.io/github/issues/ardhaecosystem/h-mem)](https://github.com/ardhaecosystem/h-mem/issues)

**The first open-source agent memory system that evolves short-term conversations into long-term structured knowledge — and retrieves it intelligently.**

[Quickstart](#quickstart) · [Install](#install) · [Benchmarks](#benchmarks) · [Contribute](#contribute)

</div>

---

## What makes H-Mem different?

Most agent "memory" is just a vector store with chunk retrieval. H-Mem actually **models how memory works**:

| What others do | What H-Mem does |
|----------------|-----------------|
| Flat chunks, retrieved by similarity | **Temporal-semantic tree** — conversations consolidate upward over time |
| No concept of "long-term" vs "recent" | **Adaptive scope** — automatically decides if a query needs recent detail or old summaries |
| Single-hop retrieval | **Knowledge graph** — multi-hop reasoning across entities and relationships |
| Static retrieval | **Query planner** — decomposes complex questions, detects missing info, and follows up |

The result: agents that actually *remember*, not just retrieve.

---

## Quickstart

```bash
pip install hmem
```

```python
from hmem import HMem, MemoryFragment, SourceType
from datetime import datetime

# 1. Create your memory engine
hmem = HMem()

# 2. Feed it conversations as they happen
for message in conversation_history:
    hmem.index(MemoryFragment(
        text=message,
        timestamp=datetime.utcnow(),
        source_type=SourceType.CONVERSATION,
    ))

# 3. Ask anything
answer = hmem.query("What did the user say about their budget last month?")
print(answer.final_answer)
```

That's it. Consolidation, graph extraction, and retrieval happen automatically.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│            Query → Answer                     │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │ Planner  │→ │ Search   │→ │ Answer  │  │
│  │ (decomp) │  │ (tree+   │  │ (synth) │  │
│  └──────────┘  │graph)    │  └─────────┘  │
│                └──────────┘               │
├─────────────────────────────────────────────┤
│           Offline Indexing                  │
│  ┌──────────┐  ┌──────────┐               │
│  │ Temporal │  │ Knowledge│               │
│  │ Tree     │  │ Graph    │               │
│  │(semantic │  │(entities │               │
│  │compress) │  │& rels)   │               │
│  └──────────┘  └──────────┘               │
└─────────────────────────────────────────────┘
```

**Three principles that make it work:**

1. **Tree for memory evolution** — Short-term conversations naturally compress into long-term summaries through semantic similarity merging. Higher nodes = broader time windows + summarized context.

2. **Graph for relationships** — Entities (people, projects, places) and their connections are tracked independently for multi-hop questions like "Who else worked on the project the user mentioned to Bob?"

3. **Planner for intelligence** — Every query gets decomposed, scoped (short-term? long-term? both?), and checked for sufficiency. If evidence is weak, the system asks for more before answering.

---

## Benchmarks

We target three long-term memory benchmarks. Full reproduction code is included.

| Benchmark | What it tests | Our approach |
|-----------|---------------|--------------|
| **LoCoMo** | 35-session conversations, single/multi-hop + temporal QA | Hybrid tree search + graph traversal + temporal filters |
| **LongMemEvalS** | 115K-token sessions, multi-session reasoning | Adaptive scope prediction (SHORT vs LONG vs MIXED) |
| **REALTALK** | 21-day real-world human conversations | Noisy-robust graph extraction + salience ranking |

Run evaluation:

```bash
# Download datasets (one-time)
python -m hmem.scripts.download_datasets

# Run evaluation
python -m hmem.evaluation.harness --dataset locomo --output-dir results/
```

Metrics reported: **F1**, **Exact Match**, and optional **LLM-as-Judge** accuracy.

---

## Install

```bash
# PyPI (coming soon)
pip install hmem

# Development (latest)
git clone https://github.com/ardhaecosystem/h-mem.git
cd h-mem
pip install -e ".[dev]"
```

**LLM Setup** (pick one):

```bash
export OPENROUTER_API_KEY="sk-..."    # Recommended — cheapest access to GPT-4o-mini
export OPENAI_API_KEY="sk-..."        # Direct
export ANTHROPIC_API_KEY="sk-..."     # Claude
```

No keys needed for smoke testing:

```bash
python -m tests.smoke_test  # Runs tree + graph + end-to-end without LLM calls
```

---

## Structure

```
hmem/
├── core/           # TemporalSemanticTree, KnowledgeGraph
├── indexing/       # TreeBuilder, GraphBuilder, Indexer
├── retrieval/      # Planner, TreeSearcher, GraphSearcher, Synthesizer
├── llm/            # Adapters: OpenRouter, OpenAI, Anthropic, local
├── evaluation/     # Benchmark loaders + harness + metrics
└── cli.py          # hmem index, query, benchmark
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for deep design docs.

---

## Why open source?

Agent memory shouldn't be a black box. We're building this in the open because:

- **Reproducibility matters** — Every benchmark result should be independently verifiable
- **Extensibility matters** — Plug in your own LLM, embedding model, or retrieval strategy
- **Community matters** — Memory for agents is an unsolved problem. We need more minds on it.

---

## Contribute

We welcome PRs, issues, and ideas.

```bash
# Quick dev setup
pip install -e ".[dev]"
python -m pytest tests/
```

Check out [good first issues](https://github.com/ardhaecosystem/h-mem/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or propose a feature in [Discussions](https://github.com/ardhaecosystem/h-mem/discussions).

---

## Citation

If you use H-Mem in research, you can cite the underlying paper:

```bibtex
@article{yu2026hmem,
  title={H-Mem: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure},
  author={Yu, Jiawei and Fang, Yixiang and Liu, Xilin and Ma, Yuchi},
  journal={arXiv preprint arXiv:2605.15701},
  year={2026}
}
```

This open-source implementation is an independent community effort.

---

## License

MIT — see [LICENSE](LICENSE).

Built with 🔥 by [Humanth & Veda](https://github.com/ardhaecosystem).
