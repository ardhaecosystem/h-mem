# Architecture Design Document — H-Mem

> Version: 0.1.0  
> Date: 2026-05-30  
> Authors: Humanth Shashani, Veda  
> Source paper: arXiv 2605.15701 (Yu et al., 2026)

---

## 1. Design Goals

1. **Faithful reproduction** of the paper's hybrid structure, retrieval pipeline, and evaluation.
2. **Plug-and-play LLM support** — OpenRouter by default, any model via simple adapter.
3. **Benchmark reproducibility** — LoCoMo, LongMemEval, REALTALK with identical metrics.
4. **Production-ready foundation** — Types, tests, async support, caching.
5. **Open-source from day 1** — Clean API, documented, MIT licensed.

---

## 2. Core Data Model

### 2.1 MemoryFragment

The atomic unit of memory — every conversation turn or external input.

```python
class MemoryFragment(BaseModel):
    id: str                          # UUID
    text: str                        # Raw text content
    timestamp: datetime             # When this occurred
    source_type: str                # "conversation", "tool_use", "document", etc.
    metadata: dict[str, Any]        # Arbitrary metadata
    # Ingested by, provenance, etc.
```

### 2.2 Temporal-Semantic Tree

A rooted tree where each node represents a memory event at a specific time window and semantic granularity.

```
Level 4 (Root):  Long-term consolidated memory (months/year)
Level 3:           Medium-term summaries (weeks)
Level 2:           Short-term events (days)
Level 1 (Leaves):  Raw memory fragments (individual turns)
```

Tree nodes store:
- `time_window: (start, end)`
- `text: str` — the node's content (raw or consolidated)
- `summary: str` — optional LLM-generated summary (for non-leaf nodes)
- `embedding: ndarray` — semantic vector
- `children: list[str]` — child node IDs
- `parent: str | None`

**Consolidation rule:** two sibling leaves L1 and L2 are merged into parent P if:
1. `time_window` adjacency: their time windows are contiguous or overlapping
2. Semantic similarity: `sim(embedding(L1), embedding(L2)) >= threshold`

When merged, P.text is an LLM synthesis of L1 + L2, P.time_window is the union, and P.summary is generated.

*This is the heart of memory evolution.*

### 2.3 Knowledge Graph

An entity-relationship graph built incrementally from memory fragments.

```python
class EntityNode:
    id: str
    name: str
    entity_type: str            # "person", "organization", "location", "product", etc.
    profile: str                # LLM-generated profile from all mentions
    first_seen: datetime
    last_seen: datetime
    mention_count: int
    source_fragments: list[str] # IDs
```

```python
class RelationEdge:
    id: str
    source: str                 # entity ID
    target: str                 # entity ID
    relation_type: str          # "works_for", "located_in", "likes", etc.
    evidence: list[str]         # fragment IDs supporting this relation
    confidence: float           # accumulated confidence
```

**Entity disambiguation:**
- Exact string match + type compatibility → merge
- Prefix/suffix overlap → create "overlap" edge for traversal recall (not semantic identity)
- LLM NER primary, spaCy NER fallback

**Salience:** entities appearing in ≥ `entity_threshold` fragments are "salient" and get profiles.

---

## 3. Pipeline Architecture

### 3.1 Offline Indexing (Write Path)

```
[MemoryFragment] 
    │
    ├─→ TreeBuilder ──→ TemporalSemanticTree
    │                    (incremental insertion + periodic consolidation)
    │
    ├─→ GraphBuilder ──→ KnowledgeGraph
    │                    (entity extraction + relation extraction + disambiguation)
    │
    └─→ EmbeddingIndexer ──→ VectorStore
                         (sentence-transformer embeddings for all nodes)
```

```python
class Indexer:
    def index(self, fragment: MemoryFragment) -> None:
        # 1. Add to tree as leaf
        tree_node = self.tree.add_leaf(fragment)
        
        # 2. Extract entities and relations
        entities, relations = self.llm.extract_entities_relations(fragment.text)
        
        # 3. Update graph
        self.graph.update(entities, relations, fragment.id)
        
        # 4. Recompute embeddings
        tree_node.embedding = self.embedder.encode(tree_node.text)
        
        # 5. Periodic: consolidate tree
        self.tree.consolidate()
```

### 3.2 Online Retrieval (Read Path)

```
[Query Q]
    │
    ├─→ RetrievalPlanner 
    │       Decomposes Q into {Qk}
    │       Predicts scope(SHORT/LONG/MIXED) per sub-query
    │       Identifies key entities and temporal cues
    │       Generates retrieval workflows
    │
    ├─→ EvidenceRetrieval
    │       ├─ TreeSearch ──→ temporal/semantic evidence
    │       ├─ GraphTraversal ──→ relational evidence (multi-hop)
    │       └─ Reranking ──→ semantic relevance scoring
    │
    ├─→ MissingInformationDetector
    │       Checks if evidence is insufficiently specific
    │       Generates follow-up queries, triggers additional retrieval
    │
    └─→ AnswerSynthesizer
            Generates sub-answers per Qk
            Synthesizes final answer
```

### 3.3 Retrieval Planner Detail

The planner is the most critical and novel component. It implements the "agent-assisted retrieval" from the paper.

**Step 1 — Decomposition:**
```python
sub_queries = llm.decompose(query=Q, history=[])
# Returns list of {text: str, type: str}
```

**Step 2 — Scope Prediction:**
```python
scope = llm.predict_scope(sub_query)
# Returns: "SHORT" | "LONG" | "MIXED"
```

**Step 3 — Entity/Temporal Identification:**
```python
entities = llm.identify_key_entities(sub_query)
temporal_cues = llm.identify_temporal_cues(sub_query)
```

**Step 4 — Workflow Generation:**
```python
workflow = {
    "sub_query_id": str,
    "scope": "SHORT|LONG|MIXED",
    "search_tree": bool,
    "tree_level": int,              # which tree level to search
    "search_graph": bool,
    "graph_start_entities": list,    # seed entity IDs for graph traversal
    "graph_hops": int,
    "time_filter": (start, end) | None,
    "top_k": int,
    "rerank": bool,
    "missing_info_check": bool,
}
```

**Step 5 — Missing-Information Loop:**
```python
evidence_results = self.retrieve(workflow)
if self.is_insufficient(evidence_results, sub_query):
    missing_query = self.llm.generate_missing_query(sub_query, evidence_results)
    additional_evidence = self.retrieve(missing_query)
    evidence_results.extend(additional_evidence)
```

---

## 4. LLM Abstraction Layer

```python
class LLMAdapter(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str: ...
    
    @abstractmethod
    async def extract_entities_relations(self, text: str) -> tuple[list[Entity], list[Relation]]: ...
    
    @abstractmethod
    async def decompose_query(self, query: str) -> list[SubQuery]: ...
    
    @abstractmethod
    async def predict_scope(self, sub_query: str) -> str: ...
    
    @abstractmethod
    async def consolidate(self, texts: list[str]) -> str: ...
    
    @abstractmethod
    async def synthesize(self, query: str, evidence: list[str]) -> str: ...
```

**Implementations:**
- `OpenRouterAdapter` — default, sends to openrouter.ai/api/v1/chat/completions
- `OpenAIAdapter` — direct OpenAI API
- `AnthropicAdapter` — direct Anthropic API
- `LocalLLMAdapter` — vLLM/TGI/Ollama compatible, OpenAI-compatible `/chat/completions`

---

## 5. Configuration

See `Config` class. Key tunables matching paper hyperparameters:

| Parameter | Paper Value | Description |
|-----------|-------------|-------------|
| `tree_similarity_threshold` | 0.75 | Min cosine similarity for consolidation |
| `tree_max_depth` | 4 | Levels: leaf=0, root=3 |
| `graph_entity_threshold` | 2 | Min fragment mentions for salience |
| `graph_max_hops` | 3 | Max graph traversal depth |
| `retrieval_top_k` | Varies | Top-k evidence per sub-query |
| `missing_info_threshold` | Implicit | When to trigger follow-up retrieval |

---

## 6. Evaluation Architecture

### 6.1 Benchmark Loader

```python
class BenchmarkLoader:
    def load_locmo() -> BenchmarkSet
    def load_longmemeval() -> BenchmarkSet
    def load_realtalk() -> BenchmarkSet
```

### 6.2 Metrics

| Metric | Formula | What It Measures |
|--------|---------|----------------|
| **F1** | 2PR/(P+R) | QA accuracy with partial match |
| **Exact-Match Accuracy** | Binary | Strict correctness |
| **LLM-Judge Accuracy** | Binary | LLM-as-judge (GPT-4) rates answer |
| **Indexing Time** | Seconds | Offline cost |
| **Retrieval Latency** | Milliseconds | Per-query latency |
| **Token Cost** | Tokens | LLM token consumption |

### 6.3 Baselines

Inclusion of key baselines for ablation comparison:
- MemGPT
- Zep
- MemoryBank
- RecallM
- Flat RAG (no structure)
- H-Mem (full)
- H-Mem w/o tree
- H-Mem w/o graph
- H-Mem w/o long-term memory
- H-Mem w/o adaptive planner

---

## 7. File Map

```
hmem/
├── __init__.py                # Public API: HMem, MemoryFragment, Config
├── types.py                  # Core dataclasses (Pydantic)
├── config.py                 # HMemConfig
├── core/
│   ├── __init__.py
│   ├── memory_fragment.py     # MemoryFragment type
│   ├── tree_node.py           # TreeNode type
│   ├── entity.py              # Entity type
│   └── relation.py            # Relation type
├── indexing/
│   ├── __init__.py
│   ├── tree_builder.py        # TemporalSemanticTree + consolidation
│   ├── graph_builder.py       # Entity/relation extraction + graph update
│   └── indexer.py             # Orchestrates tree + graph + embedding
├── retrieval/
│   ├── __init__.py
│   ├── planner.py             # RetrievalPlanner (decompose, scope, workflow)
│   ├── tree_search.py         # Search tree by level/time/semantic
│   ├── graph_search.py        # Multi-hop graph traversal
│   ├── reranker.py            # Semantic re-ranking
│   ├── missing_info.py        # Detect gaps + generate follow-up
│   ├── synthesizer.py         # Sub-answer → final answer
│   └── engine.py              # Orchestrates full retrieval pipeline
├── llm/
│   ├── __init__.py
│   ├── adapter.py             # LLMAdapter ABC
│   ├── openrouter.py          # OpenRouter adapter (default)
│   ├── openai.py              # OpenAI adapter
│   ├── anthropic.py           # Anthropic adapter
│   └── local.py               # Local vllm/ollama adapter
├── evaluation/
│   ├── __init__.py
│   ├── benchmark.py           # Benchmark dataclass
│   ├── loaders.py             # Dataset loading
│   ├── metrics.py             # F1, EM, LLM-Judge
│   ├── harness.py             # Main evaluation orchestrator
│   └── baselines.py           # Baseline implementations
├── cli.py                     # `hmem` command-line tool
└── utils/
    ├── __init__.py
    ├── cache.py               # Disk-based LLM response cache
    ├── embeddings.py          # Sentence-transformer wrapper
    └── logging.py             # Rich console logging
```

---

## 8. Data Flow

### Write (Indexing)

```
Conversation.text
    │
    ├─→ MemoryFragment (ingestion timestamp)
    │       │
    │       ├─→ TreeBuilder.add_leaf(fragment) → Leaf Node
    │       │       → periodic_consolidate() → Parent/Summary Nodes
    │       │
    │       ├─→ GraphBuilder.extract_entities_relations(text)
    │       │       → EntityNode (new or merge)
    │       │       → RelationEdge (new or update evidence)
    │       │
    │       └─→ Embedder.encode(text) → Vector
    │
    └─→ Persist (JSON/Parquet + NetworkX graph + FAISS/Annoy index)
```

### Read (Query)

```
Query.text
    │
    ├─→ RetrievalPlanner.decompose(query)
    │       → SubQueries + Scopes + Entities + Workflows
    │
    ├─→ For each workflow:
    │       ├─→ TreeSearch.search(level, time_filter, semantic_filter) → Candidates
    │       ├─→ GraphSearch.traverse(start_entities, hops) → Candidates
    │       └─→ Reranker.rerank(candidates, sub_query) → Evidence E
    │
    ├─→ MissingInformationDetector.check(evi, sub_query)
    │       → If insufficient: generate follow-up → additional retrieval
    │
    └─→ Synthesizer.synthesize(query, all_evidence) → Answer
```

---

## 9. Decisions (ADRs)

### ADR-001: NetworkX for Knowledge Graph

**Considered:** NetworkX, neo4j, igraph, rustworkx  
**Chosen:** NetworkX

**Rationale:**
- Zero external dependencies (pure Python)
- Easy serialization to JSON/YAML/Pickle
- Multi-hop traversal is small-graph (< 10k nodes) — performance fine
- Can migrate to neo4j later if scale demands

### ADR-002: Sentence-Transformers for Embeddings

**Considered:** ST, OpenAI embeddings, Cohere, local BERT  
**Chosen:** Sentence-Transformers (all-MiniLM-L6-v2 default)

**Rationale:**
- Offline, no API cost
- Fast enoughenough
- Configurable to any other model
- Can swap for OpenAI embeddings via config

### ADR-003: Async-First LLM Calls

**Considered:** Sync, async, batch  
**Chosen:** Async with optional batching

**Rationale:**
- Most LLM calls are independent (entity extraction per fragment)
- Parallelization reduces wall-clock time
- Benchmark evaluation is IO-bound on LLM API
- Sync wrapper provided for simple use cases

### ADR-004: OpenRouter as Default LLM Backend

**Considered:** OpenAI direct only, Anthropic direct, multiple providers  
**Chosen:** OpenRouter with pluggable adapters

**Rationale:**
- Cost efficiency — OpenRouter aggregates cheapest routing
- One key for many models
- Paper used GPT-4o-mini/4.1-mini — cheap models available through OpenRouter
- Can plug in any OpenAI-compatible endpoint

---

## 10. Future Extensions

- [ ] Streaming memory (real-time indexing during active conversation)
- [ ] Distributed consolidation (across multiple agent sessions)
- [ ] Hierarchical multi-agent memory (shared graph, per-agent trees)
- [ ] Persistent vector store (FAISS / Chroma / pgvector)
- [ ] Web UI for inspection

---

*This document is living. After each milestone, update with lessons learned.*
