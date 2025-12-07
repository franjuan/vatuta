"""
Simple RAG PoC using LangChain 1.0 and AWS Bedrock

This module loads the existing FAISS vector store persisted by `DocumentManager`
and exposes a minimal Retrieval-Augmented Generation (RAG) chain built with LCEL.

Usage (module):
    python -m pocs.rag_poc "What are the epic requirements?" --k 4 --show-sources

It expects AWS credentials (preferably via a named profile) and a Bedrock-supported
chat model id. Defaults can be set via environment variables:
    - AWS_PROFILE (default: IAAdmin)
    - AWS_REGION (default: us-east-1)

Dependencies:
    - langchain (>=1.0)
    - langchain-aws
    - langchain-community
    - langchain-core
    - sentence-transformers, faiss-cpu (already used by DocumentManager)
"""

import argparse
import os
from typing import List

import boto3
from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from pocs.tools_example import kb_stats
from src.rag.document_manager import DocumentManager


def build_bedrock_chat_model(
    profile_name: str | None = None,
    region_name: str | None = None,
    model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> ChatBedrock:
    """Create a Bedrock-backed chat model for LangChain.

    Falls back to environment variables when args are not provided.
    """
    resolved_profile = profile_name or os.getenv("AWS_PROFILE", "IAAdmin")
    resolved_region = region_name or os.getenv("AWS_REGION", "us-east-1")

    session = boto3.Session(profile_name=resolved_profile)
    bedrock_client = session.client("bedrock-runtime", region_name=resolved_region)

    return ChatBedrock(
        model_id=model_id,
        client=bedrock_client,
        model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
    )


def ensure_retriever(document_manager: DocumentManager, k: int):
    """Return a retriever from the existing FAISS vector store.

    Raises ValueError if the store is not available.
    """
    if document_manager.vectorstore is None:
        raise ValueError("No vector store found. Import documents first (see doc_cli commands).")
    return document_manager.vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(documents: List[Document]) -> str:
    """Format retrieved documents as context text."""
    return "\n\n".join(d.page_content for d in documents)


def build_rag_chain(retriever, llm: ChatBedrock, tools: List | None = None):
    """Construct a RAG chain.

    - Without tools: LCEL pipeline (retriever -> prompt -> llm -> text)
    - With tools: LCEL branch that optionally executes tool calls and re-invokes LLM
    """
    if tools:
        tool_by_name = {t.name: t for t in tools}

        def make_messages(inputs: dict):
            question = inputs["question"]
            docs = inputs["docs"]
            context_text = format_docs(docs)
            system_text = (
                "You are a helpful assistant answering with the provided context. "
                "If you need high-level knowledge base statistics, you may use the tools provided. "
                "Do not fabricate information that is not in the context."
            )
            return [
                SystemMessage(content=system_text),
                HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}"),
            ]

        llm_with_tools = llm.bind_tools(tools)

        def run_llm_with_tools(msgs):
            return llm_with_tools.invoke(msgs)

        def execute_tools_and_finalize(payload: dict) -> str:
            messages = payload["msgs"]
            ai_msg = payload["ai"]
            messages = messages + [ai_msg]
            for call in ai_msg.tool_calls:
                name = call.get("name")
                call_id = call.get("id")
                tool = tool_by_name.get(name)
                try:
                    if tool is not None:
                        # kb_stats has no args; extend here to pass call.get("args") if needed
                        result_text = tool.invoke({})
                    else:
                        result_text = f"(unknown tool: {name})"
                except Exception:
                    result_text = f"(tool {name} failed)"
                messages.append(ToolMessage(content=result_text, tool_call_id=call_id, name=name))
            final_msg = llm.invoke(messages)
            return final_msg.content

        def direct_content(payload: dict) -> str:
            return payload["ai"].content

        def has_tool_calls(payload: dict) -> bool:
            return bool(getattr(payload["ai"], "tool_calls", None))

        chain = (
            {
                "question": RunnablePassthrough(),
                "docs": retriever,
            }
            | RunnableLambda(make_messages)  # -> msgs (list)
            | {
                "msgs": RunnablePassthrough(),  # pass msgs list forward
                "ai": RunnableLambda(lambda msgs: run_llm_with_tools(msgs)),
            }
            | RunnableBranch(
                (has_tool_calls, RunnableLambda(execute_tools_and_finalize)),
                RunnableLambda(direct_content),
            )
        )
        return chain

    # Non-agentic LCEL path
    prompt = ChatPromptTemplate.from_template(
        (
            "You are a helpful assistant. Use the provided context to answer the question.\n"
            "Be concise and cite facts from the context only. If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    parser = argparse.ArgumentParser(description="Simple LangChain 1.0 RAG PoC (Bedrock)")
    parser.add_argument("question", help="User question to answer with RAG")
    parser.add_argument("--k", type=int, default=4, help="Top-K documents to retrieve")
    parser.add_argument(
        "--model-id",
        default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        help="Bedrock model id",
    )
    parser.add_argument("--profile", default=None, help="AWS profile for Bedrock")
    parser.add_argument("--region", default=None, help="AWS region for Bedrock")
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print brief info about retrieved source documents",
    )
    parser.add_argument(
        "--show-kb-stats",
        action="store_true",
        help="Print knowledge base stats (via kb_stats tool) before answering",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Enable agentic tool-calling; model may call kb_stats if needed",
    )
    args = parser.parse_args()

    # Load existing FAISS vector store via DocumentManager
    doc_manager = DocumentManager()

    if args.show_kb_stats:
        print("📊 KB Stats:")
        try:
            print(kb_stats.invoke({}))
        except Exception as _e:
            print("(failed to get stats)")

    try:
        retriever = ensure_retriever(doc_manager, k=args.k)
    except ValueError as e:
        print(f"❌ {e}")
        return

    # Build Bedrock model
    llm = build_bedrock_chat_model(profile_name=args.profile, region_name=args.region, model_id=args.model_id)

    # Build chain (agentic when --agent)
    tools = [kb_stats] if args.agent else None
    chain = build_rag_chain(retriever, llm, tools=tools)

    print("🤖 Question:", args.question)
    answer = chain.invoke(args.question)
    print("\n✅ Answer:\n")
    print(answer)

    if args.show_sources:
        print("\n📚 Top retrieved sources:\n")
        docs = retriever.invoke(args.question)
        for i, d in enumerate(docs, start=1):
            title = d.metadata.get("title") or d.metadata.get("summary") or "Untitled"
            source = d.metadata.get("source", "unknown")
            preview = d.page_content[:120].replace("\n", " ") + ("..." if len(d.page_content) > 120 else "")
            print(f"  {i}. {title} ({source})")
            print(f"     {preview}")


if __name__ == "__main__":
    main()
