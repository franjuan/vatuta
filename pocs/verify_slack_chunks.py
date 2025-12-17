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

from src.rag.document_manager import DocumentManager
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def main():
    dm = DocumentManager()
    
    # 1. Filter for Slack chunks
    slack_chunks = []
    for chunk_id, meta in dm.documents_metadata.items():
        if meta.get("source") == "slack":
            slack_chunks.append((chunk_id, meta))
            
    count = len(slack_chunks)
    console.print(f"[bold blue]Found {count} Slack chunks in Knowledge Base.[/bold blue]")
    
    if count == 0:
        console.print("[yellow]No Slack chunks found. Run 'vatuta load --source slack' first.[/yellow]")
        return

    # 2. Group by Parent Document (Thread/Channel)
    chunks_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for chunk_id, meta in slack_chunks:
        parent_id = meta.get("document_id")
        if not parent_id:
            continue
            
        if parent_id not in chunks_by_parent:
            chunks_by_parent[parent_id] = []
        
        # We need the full content to show context. 
        # Metadata only has preview. We need to fetch from VectorStore or assume we can't?
        # DocumentManager.search_documents returns content.
        # But we want specific chunks.
        # DocumentManager doesn't expose "get_chunk_by_id".
        # We can implement a temporary lookup or just rely on metadata preview if it was full?
        # Metadata preview is chopped.
        # We really want the full content.
        # Let's use `vectorstore.docstore.search(chunk_id)` if available?
        # FAISS docstore is usually DictDocStore.
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

            # Try to retrieve content from VectorStore
            content = "[Content not retrievable directly from FAISS without docstore access]"
            if dm.vectorstore and hasattr(dm.vectorstore, "docstore"):
                try:
                    doc = dm.vectorstore.docstore.search(chunk_id)
                    if doc:
                        content = doc.page_content
                except Exception:
                    pass
            
            console.print(Panel(
                Markdown(content),
                title=f"Chunk {i+1}/{total_chunks} - {chunk_id[-8:]}",
                subtitle=f"{len(content)} chars",
                border_style="cyan"
            ))

if __name__ == "__main__":
    main()
