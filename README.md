<div align="center">

<pre align="center" style="font-family: 'SF Mono', 'JetBrains Mono', monospace;">
  ██╗  ██╗    ███╗   ███╗   ███████╗   ███╗   ███╗
  ██║  ██║    ████╗ ████║   ██╔════╝   ████╗ ████║
  ███████║    ██╔████╔██║   █████╗     ██╔████╔██║
  ██╔══██║    ██║╚██╔╝██║   ██╔══╝     ██║╚██╔╝██║
  ██║  ██║    ██║ ╚═╝ ██║   ███████╗   ██║ ╚═╝ ██║
  ╚═╝  ╚═╝    ╚═╝     ╚═╝   ╚══════╝   ╚═╝     ╚═╝
</pre>

<h2>Adaptive long-term memory for autonomous agents.</h2>

<p>
  <a href="https://pypi.org/project/hmem/"><img src="https://img.shields.io/badge/pip-hmem-blue" alt="PyPI"></a>
  <a href="https://github.com/ardhaecosystem/h-mem/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"></a>
  <a href="https://github.com/ardhaecosystem/h-mem/actions"><img src="https://img.shields.io/badge/tests-passing-brightgreen" alt="Tests"></a>
  <a href="https://github.com/ardhaecosystem/h-mem/issues"><img src="https://img.shields.io/github/issues/ardhaecosystem/h-mem" alt="Issues"></a>
</p>

<p><strong>H-Mem</strong> treats retrieval as architecture — not a database lookup.</p>

</div>

---

## Why memory, not search

Most agent memory is a vector store with chunk retrieval.
Vectors match words. Memory has *structure*.

Memory is **temporal** (what happened when), **relational** (who + why), and **contextual** (yesterday vs last year).

H-Mem models this with three structures working together:

| Structure | What it does |
|-----------|-------------|
| **Temporal-Semantic Tree** | Short-term conversations compress into long-term summaries as they age. Leaves = raw events. Higher nodes = summaries across broader windows. Semantic memory with a clock. |
| **Knowledge Graph** | Entities (people, projects, places) and their relationships persist independently of the tree. "Bob" stays linked to every project he touched, even if conversations are summarized away. |
| **Adaptive Planner** | Every query is decomposed into sub-queries, scoped to the right timescale, and checked for sufficiency. Weak evidence triggers follow-up before answering. |

---

## How it works

```
in-memory buffer  →  temporal-semantic tree  →  knowledge graph
     (today)              (this week)              (always)
        │                       │                       │
        ▼                       ▼                       ▼
   raw conversations    consolidated summaries    entities + relations
```

**Query flow**

```
    ┌─────────────────────────────────────┐
    │  "What did Alice and Bob agree on   │
    │   in the last sprint?"              │
    └──────────────┬──────────────────────┘
                   ▼
    ┌─────────────────────────────────────┐
    │  Planner → decompose + scope        │
    │  (SHORT + LONG + MIXED)             │
    └──────────────┬──────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
    ┌────▼────┐          ┌────▼────┐
    │ Tree    │          │ Graph   │
    │ Search  │          │ Search  │
    │(cos-sim,│          │(multi-  │
    │levelled)│          │ hop BFS) │
    └────┬────┘          └────┬────┘
         │                    │
         └────────┬───────────┘
                  ▼
    ┌─────────────────────────────────────┐
    │  Merge + rerank + synthesize →      │
    │  structured answer                  │
    └─────────────────────────────────────┘
```

---

## Quick start

```bash
pip install hmem
```

```python
from hmem import HMem, MemoryFragment, SourceType
from datetime import datetime

hmem = HMem()

hmem.index(MemoryFragment(
    text="Alice met Bob to go over Q3 roadmap priorities.",
    timestamp=datetime.utcnow(),
    source_type=SourceType.CONVERSATION,
))

hmem.index(MemoryFragment(
    text="Bob approved the new observability budget but wants a phased rollout.",
    timestamp=datetime.utcnow(),
    source_type=SourceType.CONVERSATION,
))

answer = hmem.query("What budget decisions did Bob approve?")
print(answer.final_answer)
```

**No LLM keys needed for smoke tests:**

```bash
git clone https://github.com/ardhaecosystem/h-mem.git
cd h-mem
python -m tests.smoke_test
```

---

## Architecture

```
hmem/
├── core/
│   ├── tree.py          TemporalSemanticTree — semantic compression + time windows
│   └── graph.py         KnowledgeGraph — entities, relations, salience
├── indexing/
│   ├── indexer.py        Orchestrates tree + graph builders
│   ├── tree_builder.py   Embedding → leaf insertion → consolidation trigger
│   └── graph_builder.py  LLM entity extraction → graph update
├── retrieval/
│   ├── engine.py         RetrievalEngine — full query pipeline
│   ├── tree_search.py    Cosine-similarity search across tree levels
│   ├── graph_search.py   Multi-hop BFS over entity-relation graph
│   ├── planner.py        Query decomposition + scope prediction
│   └── synthesizer.py    Evidence → structured answer
├── llm/
│   ├── adapter.py        Prompts: extract, consolidate, decompose, synthesize
│   ├── openrouter.py     Default: GPT-4o-mini via OpenRouter (cheapest production access)
│   ├── openai.py
│   └── anthropic.py
└── evaluation/
    ├── harness.py        Benchmark runner (LoCoMo, LongMemEvalS, REALTALK)
    └── metrics.py        F1, Exact Match, LLM-as-Judge
```

---

## Benchmarks

Full reproduction code and dataset loaders included.

| Benchmark | What it tests | F1 | EM |
|-----------|-------------|-----|-----|
| **LoCoMo** | 35-session conversations, single/multi-hop + temporal QA | — | — |
| **LongMemEvalS** | 115K-token sessions, multi-session reasoning | — | — |
| **REALTALK** | 21-day real-world human conversations | — | — |

```bash
python -m hmem.scripts.download_datasets
python -m hmem.evaluation.harness --dataset locomo --output-dir results/
```

---

## LLM setup

```bash
# Pick one
export OPENROUTER_API_KEY="sk-..."   # Recommended
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
```

OpenRouter is the default — cheap production-grade access with one key.

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

This open-source implementation is an independent community effort.

---

<div align="left">

MIT [LICENSE](LICENSE). Built by [Humanth & Veda](https://github.com/ardhaecosystem).

</div>
