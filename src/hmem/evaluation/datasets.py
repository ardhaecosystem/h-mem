"""H-Mem benchmark dataset loaders.

Provides unified loading for LoCoMo, LongMemEvalS, and REALTALK.
All datasets are downloaded from their official sources (GitHub / HuggingFace)
and cached under the project's data/ directory.

Example:
    from hmem.evaluation.datasets import LoCoMoLoader

    loader = LoCoMoLoader(data_dir="data")
    conversations = loader.load_conversations()  # list[Conversation]
    questions = loader.load_qa()                     # list[QAItem]
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hmem.types import MemoryFragment, SourceType


@dataclass
class Conversation:
    """A single conversation with multiple turns."""
    id: str
    turns: list[str]           # text of each turn
    timestamps: list[str]       # ISO 8601 timestamps
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAItem:
    """A question-answer pair with optional metadata."""
    conversation_id: str
    question: str
    answer: str
    question_type: str = ""      # e.g., single-hop, multi-hop, temporal
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDatasetLoader(ABC):
    """Abstract base for benchmark loaders."""

    name: str
    data_url: str
    data_dir: Path
    archive_type: str = "zip"    # zip, tar.gz, json, etc.

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.data_dir / self.name / "raw"
        self.processed_dir = self.data_dir / self.name / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──────────────────────────────────

    def _ensure_downloaded(self) -> Path:
        """Download dataset if not already cached."""
        local_path = self.raw_dir / f"{self.name}.{self.archive_type}"
        if local_path.exists():
            return local_path

        print(f"Downloading {self.name} from {self.data_url} ...")
        urllib.request.urlretrieve(self.data_url, local_path)
        print(f"Saved to {local_path}")
        return local_path

    def _extract(self, archive_path: Path, extract_to: Path) -> None:
        """Extract archive if applicable."""
        if self.archive_type == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_to)
        elif self.archive_type in ("tar", "tar.gz", "tgz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_to)
        # json files are kept as-is

    def download(self) -> None:
        """Download and extract dataset."""
        archive = self._ensure_downloaded()
        if self.archive_type != "json":
            self._extract(archive, self.raw_dir)

    # ── Processing ────────────────────────────────

    def _load_json(self, path: Path) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _is_ready(self, required_files: list[str]) -> bool:
        return all((self.processed_dir / f).exists() for f in required_files)

    # ── Abstract: implement per dataset ─────────────

    @abstractmethod
    def _process_raw(self) -> None:
        """Convert raw data to processed json files."""

    @abstractmethod
    def load_conversations(self) -> list[Conversation]:
        """Load all conversations."""

    @abstractmethod
    def load_qa(self) -> list[QAItem]:
        """Load all QA items."""

    @abstractmethod
    def to_memory_fragments(self, conversation: Conversation) -> list[MemoryFragment]:
        """Convert a conversation to memory fragments."""

    # ── Public convenience ────────────────────────

    def prepare(self) -> None:
        """Ensure dataset is downloaded + processed."""
        self.download()
        self._process_raw()


class LoCoMoLoader(BaseDatasetLoader):
    """LoCoMo: Long-term Conversational Memory benchmark.

    Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024)
    Source: https://github.com/snap-research/locomo
    """

    name = "locomo"
    data_url = "https://github.com/snap-research/locomo/releases/download/v1.0/locomo_v1.json"
    archive_type = "json"

    def _process_raw(self) -> None:
        raw_path = self.raw_dir / f"{self.name}.{self.archive_type}"
        data = self._load_json(raw_path)

        conversations: list[dict] = []
        qa_items: list[dict] = []

        for conv in data:
            conv_id = conv["conversation_id"]
            turns = []
            timestamps = []
            for turn in conv.get("turns", []):
                turns.append(turn["text"])
                timestamps.append(turn.get("timestamp", ""))
            conversations.append({
                "id": conv_id,
                "turns": turns,
                "timestamps": timestamps,
                "metadata": conv.get("metadata", {}),
            })
            for qa in conv.get("qa", []):
                qa_items.append({
                    "conversation_id": conv_id,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "question_type": qa.get("type", ""),
                    "metadata": qa.get("metadata", {}),
                })

        with open(self.processed_dir / "conversations.json", "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2)
        with open(self.processed_dir / "qa.json", "w", encoding="utf-8") as f:
            json.dump(qa_items, f, indent=2)

    def load_conversations(self) -> list[Conversation]:
        if not self._is_ready(["conversations.json"]):
            self.prepare()
        data = self._load_json(self.processed_dir / "conversations.json")
        return [Conversation(**item) for item in data]

    def load_qa(self) -> list[QAItem]:
        if not self._is_ready(["qa.json"]):
            self.prepare()
        data = self._load_json(self.processed_dir / "qa.json")
        return [QAItem(**item) for item in data]

    def to_memory_fragments(self, conversation: Conversation) -> list[MemoryFragment]:
        from datetime import datetime
        frags = []
        for i, (turn, ts_str) in enumerate(zip(conversation.turns, conversation.timestamps)):
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
            frags.append(MemoryFragment(
                text=turn,
                timestamp=ts,
                source_type=SourceType.CONVERSATION,
                metadata={"conversation_id": conversation.id, "turn_idx": i},
            ))
        return frags


class LongMemEvalSLoader(BaseDatasetLoader):
    """LongMemEvalS (LongMemEval short-context variant).

    Paper: "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
    Source: https://github.com/xiaowu0162/longmemeval
    HuggingFace: https://huggingface.co/datasets/xiaowu0162/longmemeval
    """

    name = "longmemevals"
    data_url = "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval.jsonl"
    archive_type = "jsonl"

    def _process_raw(self) -> None:
        raw_path = self.raw_dir / f"{self.name}.jsonl"
        conversations: list[dict] = []
        qa_items: list[dict] = []

        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                conv_id = item["conversation_id"]
                turns = [t["text"] for t in item.get("turns", [])]
                timestamps = [t.get("timestamp", "") for t in item.get("turns", [])]
                conversations.append({
                    "id": conv_id,
                    "turns": turns,
                    "timestamps": timestamps,
                    "metadata": item.get("metadata", {}),
                })
                for qa in item.get("qa", []):
                    qa_items.append({
                        "conversation_id": conv_id,
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "question_type": qa.get("type", ""),
                        "metadata": qa.get("metadata", {}),
                    })

        with open(self.processed_dir / "conversations.json", "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2)
        with open(self.processed_dir / "qa.json", "w", encoding="utf-8") as f:
            json.dump(qa_items, f, indent=2)

    def load_conversations(self) -> list[Conversation]:
        if not self._is_ready(["conversations.json"]):
            self.prepare()
        data = self._load_json(self.processed_dir / "conversations.json")
        return [Conversation(**item) for item in data]

    def load_qa(self) -> list[QAItem]:
        if not self._is_ready(["qa.json"]):
            self.prepare()
        data = self._load_json(self.processed_dir / "qa.json")
        return [QAItem(**item) for item in data]

    def to_memory_fragments(self, conversation: Conversation) -> list[MemoryFragment]:
        from datetime import datetime
        frags = []
        for i, (turn, ts_str) in enumerate(zip(conversation.turns, conversation.timestamps)):
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
            frags.append(MemoryFragment(
                text=turn,
                timestamp=ts,
                source_type=SourceType.CONVERSATION,
                metadata={"conversation_id": conversation.id, "turn_idx": i},
            ))
        return frags

    def _ensure_downloaded(self) -> Path:
        """Override: download jsonl line-by-line."""
        local_path = self.raw_dir / f"{self.name}.jsonl"
        if local_path.exists():
            return local_path
        print(f"Downloading {self.name} from {self.data_url} ...")
        urllib.request.urlretrieve(self.data_url, local_path)
        print(f"Saved to {local_path}")
        return local_path


class REALTALKLoader(BaseDatasetLoader):
    """REALTALK: 21-day real-world human conversation dataset.

    Paper: "REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation"
    Source: https://github.com/danny911kr/REALTALK
    """

    name = "realtalk"
    data_url = "https://github.com/danny911kr/REALTALK/archive/refs/heads/main.zip"
    archive_type = "zip"

    def _process_raw(self) -> None:
        # REALTALK stores conversations in several json files
        conv_path = self.raw_dir / "conversations.json"
        qa_path = self.raw_dir / "qa.json"
        conversations = self._load_json(conv_path) if conv_path.exists() else []
        qa_items = self._load_json(qa_path) if qa_path.exists() else []

        with open(self.processed_dir / "conversations.json", "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2)
        with open(self.processed_dir / "qa.json", "w", encoding="utf-8") as f:
            json.dump(qa_items, f, indent=2)

    def load_conversations(self) -> list[Conversation]:
        if not self._is_ready(["conversations.json"]):
            self.prepare()
        data = self._load_json(self.processed_dir / "conversations.json")
        return [Conversation(**item) for item in data]

    def load_qa(self) -> list[QAItem]:
        if not self._is_ready(["qa.json"]):
            self.prepare()
        data = self._load_json(self.processed_dir / "qa.json")
        return [QAItem(**item) for item in data]

    def to_memory_fragments(self, conversation: Conversation) -> list[MemoryFragment]:
        from datetime import datetime
        frags = []
        for i, (turn, ts_str) in enumerate(zip(conversation.turns, conversation.timestamps)):
            ts = datetime.fromisoformat(ts_str) if ts_str else datetime.utcnow()
            frags.append(MemoryFragment(
                text=turn,
                timestamp=ts,
                source_type=SourceType.CONVERSATION,
                metadata={"conversation_id": conversation.id, "turn_idx": i},
            ))
        return frags


# ── Loader registry ──────────────────────────

LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
    "locomo": LoCoMoLoader,
    "longmemevals": LongMemEvalSLoader,
    "realtalk": REALTALKLoader,
}


def get_loader(name: str, data_dir: str | Path = "data") -> BaseDatasetLoader:
    """Factory: get loader by dataset name."""
    name = name.lower()
    if name not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(LOADER_REGISTRY.keys())}")
    return LOADER_REGISTRY[name](data_dir)
