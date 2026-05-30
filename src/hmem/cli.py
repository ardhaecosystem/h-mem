"""CLI entry point for H-Mem.

Usage:
    hmem index --input conversations.jsonl
    hmem query "What did Alice say about the project?"
    hmem benchmark --dataset locom
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hmem.config import HMemConfig
from hmem.engine import HMemEngine
from hmem.types import MemoryFragment

app = typer.Typer(name="hmem", help="H-Mem: Hybrid agent memory system")
console = Console()


@app.command()
def index(
    input_file: str = typer.Option(..., "--input", "-i", help="Path to conversations JSONL"),
    output_dir: str = typer.Option("./hmem_index", "--output", "-o", help="Index output directory"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Config YAML/JSON"),
) -> None:
    """Index conversations into H-Mem hybrid structure."""
    config = _load_config(config_file)
    engine = HMemEngine(config)

    console.print(f"[bold blue]Indexing[/bold blue] {input_file} ...")

    fragments = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            frag = MemoryFragment(
                text=data.get("text", ""),
                timestamp=_parse_time(data.get("timestamp")),
                metadata=data.get("metadata", {}),
            )
            fragments.append(frag)

    engine.index_batch(fragments)
    engine.save(output_dir)

    stats = engine.stats
    table = Table(title="Indexing Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"[green]Saved index to {output_dir}[/green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer"),
    index_dir: str = typer.Option("./hmem_index", "--index", help="Path to indexed data"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Query the H-Mem hybrid memory."""
    config = _load_config(config_file)
    engine = HMemEngine(config)
    if Path(index_dir).exists():
        engine.load(index_dir)

    result = engine.query(question)
    console.print(f"[bold green]Q:[/bold green] {question}")
    console.print(f"[bold blue]A:[/bold blue] {result.final_answer}")

    if result.sub_queries:
        console.print("\n[dim]Sub-queries:[/dim]")
        for sq in result.sub_queries:
            console.print(f"  - {sq.text} ({sq.predicted_scope.value})")


@app.command()
def benchmark(
    dataset: str = typer.Option(..., "--dataset", help="Dataset: locom, longmemeval, realtalk"),
    index_dir: str = typer.Option("./hmem_index_benchmark", "--index", help="Benchmark index dir"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c"),
    model: Optional[str] = typer.Option(None, "--model", help="Override LLM model"),
) -> None:
    """Run benchmark evaluation (LoCoMo, LongMemEval, REALTALK)."""
    console.print(f"[yellow]Benchmark: {dataset} — not yet implemented.[/yellow]")
    console.print("See docs/BENCHMARK.md for status.")
    raise typer.Exit(1)


# ── Helpers ───────────────────────────────────

def _load_config(path: str | None) -> HMemConfig:
    if path is None:
        return HMemConfig()
    # TODO: load YAML/JSON config
    return HMemConfig()


def _parse_time(ts) -> Optional:
    from datetime import datetime
    if ts is None:
        return datetime.utcnow()
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return datetime.utcnow()
    return datetime.utcnow()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
