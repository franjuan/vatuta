"""RAG Agent using LangGraph for dynamic routing."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, TypedDict

import dspy
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.models.config import VatutaConfig
from src.rag.engine import DSPyRAGModule, build_dspy_lm
from src.rag.qdrant_manager import QdrantDocumentManager
from src.rag.tools import DateFilterTool, JiraTicketTool
from src.rag.tools.source_filter import SourceFilterTool
from src.sources.source import Source

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """The state of the RAG agent."""

    question: str
    messages: List[BaseMessage]  # Conversation history including tool calls
    dynamic_query: Dict[str, Any]  # Accumulated Qdrant filters
    specific_docs: List[Document]  # Docs retrieved via exact lookup (Jira tool)
    vector_docs: List[Document]  # Docs retrieved via semantic search
    answer: str
    router_cot: Dict[str, Any]  # Chain of Thought trace from Router
    generator_cot: str  # Chain of Thought rationale from Generator
    routing_summary: str  # Summary of routing decisions


class RouteSignature(dspy.Signature):
    """Routing agent for a RAG system.

    You are a routing agent for a RAG system.
    - The current time is provided in the input. Use it to resolve relative dates.
    - If the user implies a time range (e.g., yesterday, last week), call date_filter with precise ISO start/end based on the current time.
    - If the user wants to search in specific sources (e.g., "in Jira", "in Confluence documents"), call source_filter with the source_types or source_ids from the available sources list. You can combine multiple sources.
    - If the user mentions Jira ticket keys (e.g., PROJ-123), call jira_ticket_lookup with all keys.
    - Stop calling tools when no further filters/docs are needed.
    """

    question: str = dspy.InputField()
    routing_summary: str = dspy.OutputField(desc="Short summary of tools used and filters applied.")


class RAGAgent:
    """Dynamic RAG Agent."""

    def __init__(
        self,
        config: VatutaConfig,
        sources: List[Source],
        doc_manager: QdrantDocumentManager,
        retrieval_k: int = 4,
    ):
        """Initialize the agent."""
        self.sources = sources
        self.doc_manager = doc_manager
        self.retrieval_k = retrieval_k

        rag_conf = config.rag

        # 1) Initialize LMs (Router vs Generator)
        self.router_lm = build_dspy_lm(rag_conf, rag_conf.router_backend)
        self.generator_lm = build_dspy_lm(rag_conf, rag_conf.generator_backend)

        # 2) DSPy module for Generation
        self.dspy_module = DSPyRAGModule()

        # 3) Router Signature (ReAct initialized in route node to inject tools)
        self.router_signature = RouteSignature

        # 4) Build Graph
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("route", self.route)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("route")
        workflow.add_edge("route", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def route(self, state: AgentState) -> Dict[str, Any]:
        """Decide what to do: call tools using DSPy ReAct or finish."""
        # Calculate current time
        current_time = datetime.now(timezone.utc).isoformat()

        # Build available sources text
        sources_info = [f"- {s.source_type} (ID: {s.source_id})" for s in self.sources]
        sources_text = "\n".join(sources_info)

        # Prepend current time and sources to prompt
        original_question = state["question"]
        question = (
            f"Current Time: {current_time}\n" f"Available Sources:\n{sources_text}\n\n" f"Question: {original_question}"
        )

        # --- Tools as functions capturing state ---

        def date_filter(start_date: str, end_date: str) -> str:
            """Apply date filter to search query (ISO format YYYY-MM-DD)."""
            f = DateFilterTool().invoke({"start_date": start_date, "end_date": end_date})
            if f:
                dynamic_query = state.get("dynamic_query", {})
                if not dynamic_query:
                    state["dynamic_query"] = f
                else:
                    dynamic_query.setdefault("must", [])
                    if "must" in f:
                        dynamic_query["must"].extend(f["must"])
                    state["dynamic_query"] = dynamic_query
                return f"Applied date filter: {start_date} to {end_date}"
            return "No date filter applied."

        def jira_ticket_lookup(ticket_ids: List[str]) -> str:
            """Lookup specific Jira tickets by ID (e.g. ['PROJ-123'])."""
            tool = JiraTicketTool(sources=self.sources, manager=self.doc_manager)
            docs = tool.invoke({"ticket_ids": ticket_ids})
            if docs:
                state.setdefault("specific_docs", [])
                state["specific_docs"].extend(docs)
                return f"Retrieved {len(docs)} Jira tickets."
            return "No Jira tickets found."

        def source_filter(source_types: List[str] | None = None, source_ids: List[str] | None = None) -> str:
            """Apply filter to restrict search to specific source types or IDs."""
            f = SourceFilterTool().invoke({"source_types": source_types, "source_ids": source_ids})
            if not f:
                return "No source filter applied."

            dynamic_query = state.get("dynamic_query", {})
            if not dynamic_query:
                state["dynamic_query"] = f
            else:
                # TODO: In complex logics, we should use a more sophisticated approach to merge filters.
                # For now, we assume that the filters are not conflicting.
                if "must" in f:
                    dynamic_query.setdefault("must", [])
                    dynamic_query["must"].extend(f["must"])
                if "should" in f:
                    dynamic_query.setdefault("should", [])
                    dynamic_query["should"].extend(f["should"])
                state["dynamic_query"] = dynamic_query

            applied = []
            if source_types:
                applied.append(f"types={source_types}")
            if source_ids:
                applied.append(f"ids={source_ids}")
            return f"Applied source filter: {', '.join(applied)}"

        tools = [date_filter, jira_ticket_lookup, source_filter]

        # Initialize ReAct agent with tools
        router_agent = dspy.ReAct(self.router_signature, tools=tools, max_iters=5)

        # Execute with Router LM
        summary = ""
        router_cot = {}
        with dspy.context(lm=self.router_lm):
            try:
                pred = router_agent(question=question)
                summary = pred.routing_summary

                # Capture CoT trace
                history = self.router_lm.history
                if history:
                    last_interaction = history[-1]

                    # sanitize response
                    raw_response = last_interaction.get("response", {})
                    try:
                        if hasattr(raw_response, "model_dump"):
                            response_dict = raw_response.model_dump()
                        elif hasattr(raw_response, "dict"):
                            response_dict = raw_response.dict()
                        elif isinstance(raw_response, (dict, list, str, int, float, bool, type(None))):
                            response_dict = raw_response
                        else:
                            response_dict = str(raw_response)
                    except Exception:
                        response_dict = str(raw_response)

                    # Structure the trace
                    messages_list: List[Any] = []
                    for m in last_interaction.get("messages", []):
                        try:
                            if isinstance(m, dict):
                                messages_list.append(m)
                            elif hasattr(m, "dict"):
                                messages_list.append(m.dict())
                            elif hasattr(m, "model_dump"):
                                messages_list.append(m.model_dump())
                            else:
                                messages_list.append(str(m))
                        except Exception:
                            messages_list.append(str(m))

                    router_cot = {
                        "prompt": last_interaction.get("prompt", ""),
                        "response": response_dict,
                        "messages": messages_list,
                    }

            except Exception as e:
                logger.error(f"Router failed: {e}")
                summary = f"Routing failed: {e}"
                router_cot = {"error": str(e)}

        return {
            "messages": state.get("messages", []) + [AIMessage(content=summary)],
            "router_cot": router_cot,
            "dynamic_query": state.get("dynamic_query", {}),
            "routing_summary": summary,
        }

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Perform retrieval."""
        question = state["question"]
        dynamic_query = state.get("dynamic_query") or None

        logger.info(f"Retrieving with filter: {dynamic_query}")

        vector_docs = self.doc_manager.search(query=question, k=self.retrieval_k, filter=dynamic_query)

        return {"vector_docs": vector_docs}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Generate answer."""
        question = state["question"]
        specific_docs = state.get("specific_docs", [])
        vector_docs = state.get("vector_docs", [])

        # Combine contexts
        combined_docs = (specific_docs or []) + (vector_docs or [])

        # Deduplicate
        unique_docs = []
        seen = set()
        for d in combined_docs:
            # Use source_doc_id or content hash
            k = d.metadata.get("source_doc_id", d.page_content[:20])
            if k not in seen:
                unique_docs.append(d)
                seen.add(k)

        context_str = "\n\n".join(
            [
                f"Source: {d.metadata.get('source', 'unknown')} | Title: {d.metadata.get('title', 'Untitled')}\nContent: {d.page_content}"
                for d in unique_docs
            ]
        )

        # DSPy Generation with Generator LM
        generator_cot = ""
        try:
            with dspy.context(lm=self.generator_lm):
                pred = self.dspy_module(
                    question=question,
                    context=context_str,
                    routing_summary=state.get("routing_summary", ""),
                )
                generator_cot = getattr(pred, "rationale", "")

            answer = getattr(pred, "answer", str(pred))
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            answer = f"I encountered an error generating the answer. Details: {e}"
            generator_cot = f"Error: {e}"

        return {"answer": answer, "generator_cot": generator_cot}

    def run(self, question: str) -> Dict[str, Any] | Any:
        """Run the agent."""
        initial_state = AgentState(
            question=question,
            messages=[HumanMessage(content=question)],
            dynamic_query={},
            specific_docs=[],
            vector_docs=[],
            answer="",
            router_cot={},
            generator_cot="",
            routing_summary="",
        )

        return self.graph.invoke(initial_state)
