
import os
import sys
from typing import cast

import dspy
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from google import genai

# Add project root to path
sys.path.append(os.getcwd())

from src.models.config import ConfigLoader
from src.rag.engine import DSPyRAGModule, RAGState, build_graph
from src.rag.qdrant_manager import QdrantDocumentManager

# Load environment variables
load_dotenv()

console = Console()

def main():

    console.print("[bold blue]Checking available models[/bold blue]")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for m in client.models.list():
        print(m.name)


    console.print("[bold blue]Starting Gemini 3.0 Flash PoC[/bold blue]")

    # 1. Verify API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error: GEMINI_API_KEY not found in environment.[/red]")
        return

    # 2. Configure Gemini via LangChain
    console.print("Configuring Gemini 3.0 Flash...")
    try:
        # Initialize LangChain model
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=api_key,
            temperature=1.0, # Ask recommended in docs: https://ai.google.dev/gemini-api/docs/gemini-3?hl=es-419#temperature
            convert_system_message_to_human=True # Helper for dspy sometimes
        )
        
        # Configure DSPy to use this LM
        # DSPy supports LangChain models via dspy.Google if using google-generativeai directly, 
        # or dspy.LangChain for generic LC models.
        # We'll use dspy.LangChain.
        dspy_lm = dspy.LM(
            "gemini/gemini-3-flash-preview",
            api_key=api_key,
            max_tokens=512,
            temperature=1.0,
        )
        dspy.configure(lm=dspy_lm)
        console.print("[green]Gemini configured successfully.[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to configure Gemini: {e}[/red]")
        raise e

    # 3. Load RAG Components
    console.print("Loading Knowledge Base...")
    try:
        config = ConfigLoader.load("config/vatuta.yaml")
        dm = QdrantDocumentManager(config.qdrant)
        
        # Check if we have documents
        stats = dm.get_document_stats()
        count = stats.get("total_documents", 0)
        console.print(f"Found {count} chunks in Qdrant.")
        
        if count == 0:
            console.print("[yellow]Warning: Knowledge base is empty. Results might be poor.[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Failed to initialize RAG components: {e}[/red]")
        raise e

    # 4. Run Query
    question = "Tell me about the RFC5424 support?"
    console.print(f"\n[bold]Testing Query:[/bold] {question}")
    
    try:
        rag_module = DSPyRAGModule()
        # Compile graph (build_graph expects the module)
        graph = build_graph(dm, k=4, rag_module=rag_module)
        
        # Invoke
        final_state = cast(RAGState, graph.invoke({"question": question, "k": 4}))
        
        answer = final_state.get("answer", "")
        docs = final_state.get("docs", [])
        
        console.print(f"\n[bold green]Answer:[/bold green]\n{answer}")
        console.print(f"\n[dim]Retrieved {len(docs)} documents.[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error during RAG execution: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
