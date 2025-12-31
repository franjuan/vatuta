"""CLI client for Vatuta application.

Provides command-line interface for managing knowledge base and querying the RAG system.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.entities.manager import EntityManager
from src.models.config import ConfigLoader, VatutaConfig
from src.rag.engine import DSPyRAGModule, RAGState, build_graph, configure_dspy_lm
from src.rag.qdrant_manager import QdrantDocumentManager
from src.sources.confluence import ConfluenceSource
from src.sources.jira_source import JiraSource
from src.sources.slack import SlackSource

# Configure logging to stay quiet by default, will be adjusted by verbose flag
logging.basicConfig(
    level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)]
)

app = typer.Typer(help="Vatuta - Virtual Assistant for Task Understanding, Tracking & Automation")
console = Console()


class State:
    """Application state container."""

    def __init__(self) -> None:
        """Initialize application state."""
        self.config: VatutaConfig = VatutaConfig()
        self.data_dir: str = "data"
        self.verbose: bool = False


state = State()


class SourceType(str, Enum):
    """Enum for supported data source types."""

    slack = "slack"
    jira = "jira"
    confluence = "confluence"


def get_ids_help() -> str:
    """Generate help string listing available source IDs from config."""
    config_path = "config/vatuta.yaml"
    if Path(config_path).exists():
        try:
            cfg = ConfigLoader.load(config_path)
            ids = []
            if cfg.sources.slack:
                ids.extend([f"slack/{k}" for k in cfg.sources.slack.keys()])
            if cfg.sources.jira:
                ids.extend([f"jira/{k}" for k in cfg.sources.jira.keys()])
            if cfg.sources.confluence:
                ids.extend([f"confluence/{k}" for k in cfg.sources.confluence.keys()])

            if ids:
                return f"Filter by source ID. Configured: {', '.join(ids)}"
        except Exception:
            pass

    return "Filter by source ID (e.g. slack-main)"


@app.callback()  # type: ignore[misc]
def main(
    ctx: typer.Context,
    config: str = typer.Option("config/vatuta.yaml", "--config", help="Path to configuration file"),
    data: str = typer.Option("data", "--data", help="Path to data directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging (DEBUG level)"),
) -> None:
    """Vatuta - Virtual Assistant for Task Understanding, Tracking & Automation."""
    state.data_dir = data
    state.verbose = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("src").setLevel(logging.INFO)

    # Load config
    state.config = ConfigLoader.load(config)

    # Ensure data directory exists
    Path(data).mkdir(parents=True, exist_ok=True)


@app.command()  # type: ignore[misc]
def reset() -> None:
    """Reset the Knowledge Base (Clear all documents)."""
    if typer.confirm("Are you sure you want to clear the entire Knowledge Base?", default=False):
        dm = QdrantDocumentManager(state.config.qdrant)
        dm.clear_all_documents()
        console.print("[green]Knowledge Base cleared successfully.[/green]")
    else:
        console.print("[yellow]Operation cancelled.[/yellow]")


@app.command()  # type: ignore[misc]
def remove_documents(
    source: Optional[SourceType] = typer.Option(None, help="Filter by source type"),
    source_id: Optional[str] = typer.Option(None, help=get_ids_help()),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Remove documents from the Knowledge Base with optional filtering."""
    dm = QdrantDocumentManager(state.config.qdrant)

    # Resolving source string
    s_val = source.value if source else None

    # Determine filter description
    criteria = []
    if s_val:
        criteria.append(f"source='{s_val}'")
    if source_id:
        criteria.append(f"source_id='{source_id}'")

    criteria_str = " AND ".join(criteria) if criteria else "ALL documents"

    # Confirm
    if not force:
        if not typer.confirm(f"Are you sure you want to delete {criteria_str}?"):
            console.print("[yellow]Operation cancelled.[/yellow]")
            return

    # Delete
    deleted = dm.delete_documents(source=s_val, source_instance_id=source_id)
    if deleted > 0:
        console.print(f"[green]Successfully deleted {deleted} documents matching {criteria_str}.[/green]")
    else:
        console.print(f"[yellow]No documents found matching {criteria_str} (or error occurred).[/yellow]")


def _get_enabled_sources(
    source_filter: Optional[str] = None, source_id_filter: Optional[str] = None
) -> List[tuple[str, str, Any]]:
    """Return list of enabled sources matching filters."""
    sources: List[tuple[str, str, Any]] = []
    cfg = state.config.sources

    def process_sources(source_type: str, source_dict: Dict[str, Any]) -> None:
        for sid, scfg in source_dict.items():
            if hasattr(scfg, "enabled") and not scfg.enabled:
                continue

            if source_filter and source_filter != source_type:
                continue
            if source_id_filter and source_id_filter != sid:
                continue

            sources.append((source_type, sid, scfg))

    if cfg.slack:
        process_sources("slack", cfg.slack)
    if cfg.jira:
        process_sources("jira", cfg.jira)
    if cfg.confluence:
        process_sources("confluence", cfg.confluence)

    return sources


def _ingest_docs(dm: QdrantDocumentManager, docs: list, chunks: list, source_name: str) -> None:
    if not docs:
        console.print(f"[yellow]No documents found for {source_name}.[/yellow]")
        return

    console.print(f"    Found {len(docs)} documents and {len(chunks)} chunks.")

    # Map docs by ID
    docs_by_id = {d.document_id: d for d in docs}

    # Add to Document Manager
    success = dm.add_chunk_records(chunks, docs_by_id)
    if success:
        console.print(f"[green]   Successfully ingested {source_name} data.[/green]")
    else:
        console.print(f"[red]   Failed to ingest {source_name} data.[/red]")


@app.command()  # type: ignore[misc]
def load(
    source: Optional[SourceType] = typer.Option(None, help="Filter by source type"),
    source_id: Optional[str] = typer.Option(None, help=get_ids_help()),
) -> None:
    """Load data from local cache into the Knowledge Base."""
    dm = QdrantDocumentManager(state.config.qdrant)
    # Convert enum to string if present
    s_val = source.value if source else None
    sources = _get_enabled_sources(s_val, source_id)

    if not sources:
        console.print("[yellow]No enabled sources matches the filters.[/yellow]")
        return

    # Instantiate Entity Manager
    entities_path = state.config.entities_manager.storage_path or "data/entities.json"
    entity_manager = EntityManager(storage_path=entities_path)

    for stype, sid, scfg in sources:
        console.print(f"[bold blue]Loading {stype} ({sid}) from cache...[/bold blue]")

        try:
            src: Any = None
            if stype == "slack":
                src = SlackSource.create(config=scfg, data_dir=state.data_dir, entity_manager=entity_manager)
                docs, chunks = src.collect_cached_documents_and_chunks()
                _ingest_docs(dm, docs, chunks, stype)

            elif stype == "jira":
                src = JiraSource.create(config=scfg, data_dir=state.data_dir, entity_manager=entity_manager)
                docs, chunks = src.collect_cached_documents_and_chunks()
                _ingest_docs(dm, docs, chunks, stype)

            elif stype == "confluence":
                src = ConfluenceSource.create(config=scfg, data_dir=state.data_dir, entity_manager=entity_manager)
                docs, chunks = src.collect_cached_documents_and_chunks()
                _ingest_docs(dm, docs, chunks, stype)

        except Exception as e:
            console.print(f"[red]Error loading {stype}: {e}[/red]")
            if state.verbose:
                logging.exception("Load error")


@app.command()  # type: ignore[misc]
def update(
    source: Optional[SourceType] = typer.Option(None, help="Filter by source type"),
    source_id: Optional[str] = typer.Option(None, help=get_ids_help()),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force fresh fetch (ignore cache)"),
) -> None:
    """Update the Knowledge Base by fetching data from sources.

    Defaults to using cached data if available (incremental), use --no-cache to force fresh fetch.
    """
    dm = QdrantDocumentManager(state.config.qdrant)
    # Convert enum to string if present
    s_val = source.value if source else None
    sources = _get_enabled_sources(s_val, source_id)

    if not sources:
        console.print("[yellow]No enabled sources matches the filters.[/yellow]")
        return

    use_cached_data = not no_cache

    # Instantiate Entity Manager
    entities_path = state.config.entities_manager.storage_path or "data/entities.json"
    entity_manager = EntityManager(storage_path=entities_path)

    for stype, sid, scfg in sources:
        console.print(f"[bold blue]Updating {stype} ({sid})...[/bold blue]")

        try:
            src: Any = None
            if stype == "slack":
                src = SlackSource.create(config=scfg, data_dir=state.data_dir, entity_manager=entity_manager)
                checkpoint = src.load_checkpoint()

                with Progress(
                    SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
                ) as progress:
                    progress.add_task(description=f"Fetching {stype}...", total=None)
                    docs, chunks = src.collect_documents_and_chunks(checkpoint, use_cached_data=use_cached_data)

                _ingest_docs(dm, docs, chunks, stype)

            elif stype == "jira":
                src = JiraSource.create(config=scfg, data_dir=state.data_dir, entity_manager=entity_manager)
                checkpoint = src.load_checkpoint()

                with Progress(
                    SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
                ) as progress:
                    progress.add_task(description=f"Fetching {stype}...", total=None)
                    docs, chunks = src.collect_documents_and_chunks(checkpoint, use_cached_data=use_cached_data)

                _ingest_docs(dm, docs, chunks, stype)

            elif stype == "confluence":
                src = ConfluenceSource.create(config=scfg, data_dir=state.data_dir, entity_manager=entity_manager)
                checkpoint = src.load_checkpoint()

                with Progress(
                    SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
                ) as progress:
                    progress.add_task(description=f"Fetching {stype}...", total=None)
                    docs, chunks = src.collect_documents_and_chunks(checkpoint, use_cached_data=use_cached_data)

                _ingest_docs(dm, docs, chunks, stype)

        except Exception as e:
            console.print(f"[red]Error updating {stype}: {e}[/red]")
            if state.verbose:
                logging.exception("Update error")


@app.command()  # type: ignore[misc]
def ask(
    question: str = typer.Argument(..., help="The question to ask"),
    k: int = typer.Option(4, "--k", help="Number of documents to retrieve"),
    show_sources: bool = typer.Option(False, "--show-sources", help="Display retrieved sources using rich tables"),
    show_stats: bool = typer.Option(False, "--show-stats", help="Display KB stats before answering"),
) -> None:
    """Ask a question to the RAG system."""
    dm = QdrantDocumentManager(state.config.qdrant)

    if show_stats:
        stats = dm.get_document_stats()
        table = Table(title="Knowledge Base Stats")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total Chunks", str(stats.get("total_documents", 0)))

        src_table = Table(title="Chunks by Source", show_header=False)
        for s, count in stats.get("sources", {}).items():
            src_table.add_row(s, str(count))

        console.print(table)
        console.print(src_table)

    try:
        configure_dspy_lm(state.config.rag)
        rag_module = DSPyRAGModule()
        graph = build_graph(dm, k, rag_module)

        with Progress(SpinnerColumn(), TextColumn("[bold green]Thinking..."), transient=True) as progress:
            progress.add_task("think", total=None)
            final_state: RAGState = cast(RAGState, graph.invoke({"question": question, "k": k}))

        console.print(f"\n[bold]Question:[/bold] {question}")
        console.print(f"\n[bold]Answer:[/bold]\n{final_state.get('answer', '')}")

        if show_sources:
            docs = final_state.get("docs", [])
            table = Table(title="Retrieved Sources")
            table.add_column("#", style="dim")
            table.add_column("Title", style="bold")
            table.add_column("Source", style="blue")
            table.add_column("Preview")

            for i, d in enumerate(docs, start=1):
                title = d.metadata.get("title") or d.metadata.get("summary") or "Untitled"
                src = d.metadata.get("source", "unknown")
                preview = d.page_content[:100].replace("\n", " ") + "..."
                table.add_row(str(i), title, src, preview)

            console.print("\n")
            console.print(table)

    except Exception as e:
        console.print(f"[red]Error answering question: {e}[/red]")
        if state.verbose:
            logging.exception("RAG Error")


@app.command()  # type: ignore[misc]
def stats() -> None:
    """Show detailed knowledge base statistics."""
    dm = QdrantDocumentManager(state.config.qdrant)

    with Progress(SpinnerColumn(), TextColumn("[bold blue]Analyzing Knowledge Base..."), transient=True) as progress:
        progress.add_task("analyze", total=None)
        stats = dm.get_detailed_stats()

    # General Stats Table
    table = Table(title="Knowledge Base Overview")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Total Chunks (Vectors)", str(stats.get("total_documents", 0)))
    table.add_row("Storage URL", stats.get("storage_directory", "Unknown"))
    console.print(table)
    console.print("\n")

    # Detailed Source Stats Table
    src_table = Table(title="Statistics by Source", show_lines=True)
    src_table.add_column("Source", style="bold green")
    src_table.add_column("Documents", justify="right")
    src_table.add_column("Chunks", justify="right")
    src_table.add_column("Avg Chunks/Doc", justify="right")
    src_table.add_column("Avg Chunk Size", justify="right")
    src_table.add_column("Avg Doc Size", justify="right")
    src_table.add_column("Total Size", justify="right")

    sources_data = stats.get("sources", {})
    if not sources_data:
        console.print("[yellow]No data found in Knowledge Base.[/yellow]")
        return

    def format_size(size_bytes: float) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    for source, data in sources_data.items():
        src_table.add_row(
            source,
            str(data["documents_count"]),
            str(data["chunks_count"]),
            f"{data['mean_chunks_per_document']:.1f}",
            f"{data['mean_chunk_size']:.0f} chars",
            f"{data['mean_document_size']:.0f} chars",
            format_size(data["total_size"]),
        )

    console.print(src_table)


if __name__ == "__main__":
    app()
