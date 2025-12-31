"""Verification script for Slack chunks.

Randomly samples chunks from the Knowledge Base and displays them with their
preceding and succeeding chunks to verify context and splitting logic.
"""
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.config import ConfigLoader
from src.rag.qdrant_manager import QdrantDocumentManager, QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def main():
    config = ConfigLoader.load("config/vatuta.yaml")
    dm = QdrantDocumentManager(config.qdrant)
    
    # 1. Filter for Slack chunks
    slack_chunks = dm.list_documents(source="slack", limit=1000)
            
    count = len(slack_chunks)
    console.print(f"[bold blue]Found {count} Slack chunks in Knowledge Base.[/bold blue]")
    
    if count == 0:
        console.print("[yellow]No Slack chunks found. Run 'vatuta load --source slack' first.[/yellow]")
        return

    # 2. Group by Parent Document (Thread/Channel)
    chunks_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for meta in slack_chunks:
        # qdrant_manager.list_documents returns dict with doc_id, source, etc.
        # But list_documents doesn't return full metadata like 'document_id' (parent ID).
        # We might need to fetch better data.
        # Let's scroll properly.
        pass
    
    # Re-fetch using low-level client for full payload
    scroll_result, _ = dm.client.scroll(
        collection_name=dm.collection_name,
        scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value="slack"))]),
        limit=1000,
        with_payload=True,
    )
    
    for point in scroll_result:
        meta = point.payload
        chunk_id = str(point.id)
        parent_id = meta.get("document_id")
        if not parent_id:
            continue
            
        if parent_id not in chunks_by_parent:
            chunks_by_parent[parent_id] = []

        meta["chunk_id"] = chunk_id # Ensure chunk_id is in meta
        chunks_by_parent[parent_id].append(meta)

    # SPECIAL REQUEST: Filter for specific ID
    target_id = "SLACK_CHANNEL_ID_PLACEHOLDER"
    target_parents = {}
    
    debug_ids = set()

    for pid, chunks in chunks_by_parent.items():
        meta = chunks[0]
        s_doc_id = meta.get("source_doc_id", "")
        debug_ids.add(s_doc_id)
        if target_id in s_doc_id:
            target_parents[pid] = chunks
            
    if target_parents:
        console.print(f"[bold green]Found {len(target_parents)} documents matching ID '{target_id}'[/bold green]")
        for pid, chunks in target_parents.items():
            console.print(f"Parent: {pid} (Chunks: {len(chunks)})")
            
        selected_parents = list(target_parents.keys())
    else:
        console.print(f"[red]No documents found matching '{target_id}'[/red]")
        console.print("Sample available IDs:")
        for i, did in enumerate(sorted(list(debug_ids))[:10]):
            console.print(f" - {did}")
        return
    
    for pid in selected_parents:
        parent_chunks = chunks_by_parent[pid]
        total_chunks = len(parent_chunks)
        
        console.print(f"\n[bold magenta]Document: {parent_chunks[0].get('title')}[/bold magenta] ({pid})")
        console.print(f"Total Chunks: {total_chunks}")
        
    for pid in selected_parents:
        parent_chunks = chunks_by_parent[pid]
        total_chunks = len(parent_chunks)
        
        console.print(f"\n[bold magenta]Document: {parent_chunks[0].get('title')}[/bold magenta] ({pid})")
        console.print(f"Total Chunks: {total_chunks}")
        
        # Show ALL chunks for this document
        for i, target_meta in enumerate(parent_chunks):
            chunk_id = target_meta.get('chunk_id')
            if not chunk_id: 
                chunk_id = target_meta.get("chunk_id")

            # Try to retrieve content from payload or VectorStore
            content = target_meta.get("page_content") or target_meta.get("content") or target_meta.get("content_preview") or "[Content not found in payload]"
            
            console.print(Panel(
                Markdown(content),
                title=f"Chunk {i+1}/{total_chunks} - {chunk_id[-8:]}",
                subtitle=f"{len(content)} chars",
                border_style="cyan"
            ))

if __name__ == "__main__":
    main()
