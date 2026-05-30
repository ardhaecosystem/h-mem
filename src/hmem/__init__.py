"""H-Mem: Hybrid agent memory system.

Public API:
    from hmem import HMem, MemoryFragment, HMemConfig

    hmem = HMem()
    hmem.index(MemoryFragment(text="Hello world"))
    result = hmem.query("What was said earlier?")
"""

from __future__ import annotations

from hmem.config import HMemConfig
from hmem.types import (
    MemoryFragment,
    RetrievalResult,
    ScopeType,
    SourceType,
    TreeNode,
    Entity,
    Relation,
    SubQuery,
)

__all__ = [
    "HMem",
    "HMemConfig",
    "MemoryFragment",
    "RetrievalResult",
    "ScopeType",
    "SourceType",
    "TreeNode",
    "Entity",
    "Relation",
    "SubQuery",
]

# Lazy import to avoid heavy initialization at import time
def HMem(config: HMemConfig | None = None):
    """Factory: return a HMemEngine instance.

    Args:
        config: HMemConfig instance. If None, uses default configuration.
    """
    from hmem.engine import HMemEngine
    return HMemEngine(config or HMemConfig())
