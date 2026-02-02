# RAG Agent Architecture

Vatuta uses a Retrieval-Augmented Generation (RAG) agent built with **DSPy** and **LangGraph**. This architecture
separates the reasoning logic (Routing) from the answer generation, allowing for optimized model usage and clear
"Chain of Thought" (CoT) tracing.

## Core Components

### 1. Router (dspy.ReAct)

The Router is responsible for understanding the user's question and deciding which tools to use. It uses a ReAct
(Reasoning + Acting) loop.

- **Model**: Can be configured separately (e.g., a faster/cheaper model like `gpt-4o-mini`).
- **Function**:
  - Analyzes the input.
  - Selects appropriate tools (`jira_ticket_lookup`, `date_filter`, etc.).
  - Observes tool outputs.
  - Decides when enough information has been gathered to answer.
- **Output**: A "Chain of Thought" trace showing the step-by-step reasoning and tool usage.

### 2. Generator (dspy.ChainOfThought)

The Generator synthesizes the final answer using the information retrieved by the Router.

- **Model**: Can be configured separately (e.g., a high-quality model like `gpt-4o` or `claude-3-5-sonnet`).
- **Function**:
  - Receives the original question and the context retrieved by the Router.
  - Generates a comprehensive, natural language answer.
  - Provides its own rationale/reasoning trace.

## Tools

The agent has access to specific tools to retrieve precise information.

### Jira Ticket Lookup (`jira_ticket_lookup`)

Retrieves specific Jira tickets by their keys (e.g., `PROJ-123`).

- **Trigger**: Used when the user mentions specific ticket IDs.
- **Function**: Looks up the ticket in the vector store (exact match) and returns it as a document.
- **Capabilities**: Can aggregate results from multiple connected Jira sources.

### Date Filter (`date_filter`)

Applies time-based filtering to the retrieval process.

- **Trigger**: Used when the user's query implies a time range (e.g., "last week", "since December", "in 2024").
- **Function**:
  - Parses natural language time references into a structured date range (start/end timestamps).
  - Returns a Qdrant filter structure.
- **Application**: This filter is applied to the *semantic search* phase, ensuring that only documents modified/created
  within the specified timeframe are retrieved.

## CLI Features

### Chain of Thought Display (`--show-cot`)

You can view the internal reasoning process of the agent using the `--show-cot` flag in the CLI.

```bash
vatuta ask "What is the status of VAT-123?" --show-cot
```

This will display:

1. **Router Trace**: The prompt, tool calls, and observations made by the router.
2. **Generator Trace**: The reasoning used to formulate the final answer.
