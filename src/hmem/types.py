"""Core type definitions for H-Mem.

All public types are Pydantic models for validation, serialization, and
JSON Schema generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


class ScopeType(StrEnum):
    """Memory scope predicted by the retrieval planner."""
    SHORT = "SHORT"
    LONG = "LONG"
    MIXED = "MIXED"


class SourceType(StrEnum):
    """Origin of a memory fragment."""
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"
    DOCUMENT = "document"
    OBSERVATION = "observation"


class MemoryFragment(BaseModel):
    """Atomic unit of memory ingested into H-Mem."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_type: SourceType = SourceType.CONVERSATION
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)


class TreeNode(BaseModel):
    """A node in the temporal-semantic tree.

    Leaf nodes represent raw memory fragments.
    Internal nodes represent consolidated summaries.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    level: int = Field(ge=0, description="0 = leaf, increases upward")
    time_window_start: datetime
    time_window_end: datetime
    text: str
    summary: str | None = None
    # Embedding is managed externally by the Indexer; not stored on the model directly
    # to avoid serialization bloat.
    children: list[str] = Field(default_factory=list)  # node IDs
    parent: str | None = None
    source_fragments: list[str] = Field(default_factory=list)  # memory fragment IDs

    @property
    def is_leaf(self) -> bool:
        return self.level == 0 and not self.children

    @property
    def is_root(self) -> bool:
        return self.parent is None


class EntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PRODUCT = "product"
    EVENT = "event"
    CONCEPT = "concept"
    DATE = "date"
    OTHER = "other"


class Entity(BaseModel):
    """An entity node in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    entity_type: EntityType
    profile: str = ""
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    mention_count: int = 0
    source_fragments: list[str] = Field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.id)


class Relation(BaseModel):
    """An edge in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str  # entity ID
    target: str  # entity ID
    relation_type: str
    evidence: list[str] = Field(default_factory=list)  # fragment IDs
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def __hash__(self) -> int:
        return hash(self.id)


class SubQuery(BaseModel):
    """Output of the retrieval planner's query decomposition."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    predicted_scope: ScopeType = ScopeType.MIXED
    key_entities: list[str] = Field(default_factory=list)  # predicted entity IDs
    temporal_cues: list[str] = Field(default_factory=list)  # date/time strings
    workflow: RetrievalWorkflow | None = None


class RetrievalWorkflow(BaseModel):
    """The retrieval strategy generated for a single sub-query."""

    sub_query_id: str
    scope: ScopeType
    search_tree: bool = True
    tree_level: int = 0  # 0 = leaves, increases upward
    search_graph: bool = True
    graph_start_entities: list[str] = Field(default_factory=list)
    graph_hops: int = 2
    time_filter: tuple[datetime, datetime] | None = None
    top_k: int = 10
    rerank: bool = True
    missing_info_check: bool = True


class Evidence(BaseModel):
    """A single piece of evidence retrieved for a sub-query."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_type: Literal["tree", "graph"] = "tree"
    source_id: str  # tree node ID or entity ID
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Complete result of a retrieval operation."""

    query: str
    sub_queries: list[SubQuery] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    sub_answers: dict[str, str] = Field(default_factory=dict)
    final_answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
