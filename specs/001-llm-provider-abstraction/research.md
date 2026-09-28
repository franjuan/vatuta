# Research: Provider-Independent LLM Abstraction

## Overview

This document records the architectural research, technology decisions, and design rationale for implementing a provider-independent LLM abstraction in Vatuta using LiteLLM, DSPy, and Prometheus, initially focused on the Gemini model family.

---

## 1. LiteLLM Integration with DSPy (`dspy.LM`)

### Context

Vatuta uses DSPy for structured prompt declarations, ReAct routing, and Chain of Thought answer generation (`RouteSignature`, `DSPyRAGModule`). The assistant requires an extensible LLM abstraction that allows switching models without modifying prompt signatures or business logic.

### Decision

Use DSPy's native `dspy.LM` runtime, which uses LiteLLM internally for provider routing and request dispatch. Model identifiers follow LiteLLM's standard prefix convention (`gemini/<model_name>`, e.g., `gemini/gemini-2.5-flash`, `gemini/gemini-1.5-pro`).

### Rationale

- DSPy 2.5+ and 3.x natively delegate completion requests to LiteLLM.
- Calling `dspy.LM(model="gemini/...", ...)` leverages LiteLLM's battle-tested provider translation, request normalization, and credential management out of the box.
- Preserves all DSPy signatures, modules (`dspy.ChainOfThought`, `dspy.ReAct`), and optimization capabilities without introducing custom middleware or code forks.

### Alternatives Considered

- **Custom DSPy LM subclass wrapping raw LiteLLM**: Rejected because `dspy.LM` already wraps LiteLLM directly; writing a custom subclass adds redundant maintenance overhead and risks desynchronization with upstream DSPy updates.
- **Direct Google GenAI SDK (`google-genai` / `langchain-google-genai`)**: Rejected because it hardcodes provider-specific client libraries, violating Constitution Principle III (LLM Provider Independence & Isolation) and FR-002.

---

## 2. Centralized Global LLM Provider Architecture

### Context

Vatuta requires language model backends for pipeline nodes across multiple subsystems (routing, generation, entity extraction, and future autonomous agents). In Vatuta, LangGraph orchestrates workflow states, but all language model interactions are executed through structured DSPy modules (`dspy.ReAct`, `dspy.ChainOfThought`) running on `dspy.LM`.

### Decision

Provide a centralized factory in `src/llm/provider.py` (`LLMProviderManager`) that:

1. Manages and caches configured `dspy.LM` instances using LiteLLM as the underlying provider gateway.
2. Performs broad, non-intrusive startup validation of configured backends and credentials.
3. Exposes a clean, decoupled interface in `src/llm/` so any subsystem (RAG, entities, MCP tools) can instantiate models without depending on RAG logic.
4. Omits any custom LangChain `BaseChatModel` wrapper (`ChatLiteLLM`), avoiding unnecessary complexity and dead code since all assistant interactions use DSPy structured signatures.

### Rationale

- Strictly aligns with Constitution Principle III (Structured & Atomic Prompt Isolation via DSPy; no ad-hoc chat prompt strings).
- `dspy.LM` natively uses LiteLLM for multi-provider routing and request dispatch.
- Completely avoids maintaining a redundant, custom ~100-line `BaseChatModel` implementation when zero components in Vatuta consume it.

### Alternatives Considered

- **Custom `BaseChatModel` adapter (`ChatLiteLLM`)**: Rejected per clarification 2026-09-27. Vatuta uses DSPy signatures and modules exclusively for prompting; LangGraph only orchestrates Python state transitions without invoking LangChain chat models directly.
- **Keeping LLM provider inside `src/rag/`**: Rejected per user request to make the provider global and reusable across all assistant subsystems.

---

## 3. Configuration Schema & Model Tier Selection

### Context

FR-001, FR-003, and FR-008 require replacing the existing `rag.llm_backends` schema in-place with a LiteLLM-native schema in `config/vatuta.yaml`, allowing independent backend selection for routing versus generation.

### Decision

Update `RagConfig` and `LLMBackendConfig` in `src/models/config.py`:

```yaml
rag:
  llm_backends:
    gemini_flash:
      model: "gemini/gemini-2.5-flash"
      temperature: 0.2
      max_tokens: 800
    gemini_pro:
      model: "gemini/gemini-1.5-pro"
      temperature: 0.2
      max_tokens: 2000

  router_backend: "gemini_flash"
  generator_backend: "gemini_pro"
```

Fields per backend:

- `model`: str (e.g., `"gemini/gemini-2.5-flash"`; supports `model_id` alias for backwards compatibility)
- `temperature`: float (default: `0.2`)
- `max_tokens`: Optional[int] (default: `800`)
- `top_p`: Optional[float] (default: `None`)
- `api_base`: Optional[str] (default: `None`)
- `extra_kwargs`: Dict[str, Any] (default: `{}`)

### Rationale

- Decouples pipeline roles (`router_backend`, `generator_backend`) from concrete backend implementations.
- Allows operators to use a fast, low-cost model for ReAct tool routing (Gemini Flash) and a high-reasoning model for synthesis (Gemini Pro), fulfilling User Story 2.
- Cleanly replaces the old schema while maintaining Pydantic validation.

### Alternatives Considered

- **Global single model setting**: Rejected because query routing and detailed synthesis have divergent cost/speed/reasoning requirements (FR-003).
- **Provider-specific sub-tables (`rag.gemini`, `rag.claude`)**: Rejected because it violates provider abstraction; provider selection should be expressed via model strings.

---

## 4. Credential Management & Zero-Trust Hygiene

### Context

FR-005 mandates delegating credential discovery directly to LiteLLM from environment variables without maintaining custom per-provider key registries, hardcoded provider logic, or storing secrets in YAML files.

### Decision

Credential discovery and validation are delegated entirely to LiteLLM's dynamic introspection:

- Vatuta relies on `litellm.validate_environment(model=...)` to dynamically inspect whether the environment variables required for any configured model string (`provider/model_name`) are present.
- Vatuta contains **zero** hardcoded provider names, **zero** provider-specific conditionals (`if provider == "gemini"`), and **zero** hardcoded environment variable names in application source code (`src/`).
- LiteLLM maintains the definitive, battle-tested registry of required environment variables across 100+ providers (e.g. `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, AWS credentials, etc.).
- Google Gemini is utilized solely as the live validation target for this release; application code remains 100% generic and provider-agnostic.

### Rationale

- Complies with Constitution Principle III (LLM Provider Independence) and Principle VI (Zero-Trust, Credential Hygiene).
- Avoids custom secret-loading or provider-specific branching logic that could drift from upstream provider SDK updates.
- Allows operators to switch to any provider (Gemini, Claude, OpenAI, Bedrock, etc.) simply by setting configuration in `config/vatuta.yaml` and providing the standard environment variables, with zero application code modifications.

### Alternatives Considered

- **Configuring API keys in `config/vatuta.yaml`**: Strictly prohibited by Constitution Principle VI.
- **Custom dictionary of environment variable names per provider in Vatuta**: Rejected because it introduces vendor coupling in domain code and duplicates LiteLLM's native `validate_environment` functionality.

---

## 5. Startup Pre-Flight Checks & Runtime Error Handling

### Context

FR-006, FR-009, and User Story 3 require:

- Halting at startup with actionable diagnostics on missing credentials or unsupported models.
- Catching runtime provider failures (rate limits, network outages), logging full diagnostic context at ERROR level, presenting a user-friendly message, and cleanly exiting with exit code 1.

### Decision

1. **Startup Pre-Flight Validation**:
   - `LLMProviderManager.validate_backends(config: RagConfig)` executes a broad, general, and non-intrusive evaluation of configured backends.
   - It validates:
     a) Backend existence in `config.llm_backends`.
     b) Model name syntax conforming to generic `provider/model` pattern.
     c) Presence of required environment variables using LiteLLM's dynamic `validate_environment(model)` without vendor-specific branching.
   - Strictly avoids live network probes, dummy test completion prompts, or provider-specific deep pinging during startup.
   - Raises `LLMConfigurationError` if any check fails, reporting missing keys dynamically from LiteLLM's check, halting cleanly with an actionable diagnostic, and exiting with code 1.
2. **Runtime Error Handling**:
   - Any runtime LLM failure that cannot be remediated during execution is treated as a fatal, unrecoverable execution event.
   - Wrap LM execution points and catch `litellm.exceptions.LiteLLMError`:
     - `AuthenticationError`: Report invalid/expired credentials for the target provider.
     - `RateLimitError`: Report provider quota/rate limit exhaustion (HTTP 429).
     - `APIConnectionError`: Report upstream network unreachable / DNS failure.
     - `BadRequestError`: Report parameter or model constraint violations.
   - All runtime errors are logged using lazy formatting at `ERROR`/`CRITICAL` severity level (`logger.critical("Fatal LLM execution error: %s", exc)`), printed clearly to the console via Rich `Panel`, and immediately cancel/abort the entire execution process via `typer.Exit(code=1)`.

### Rationale

- Startup check is immediate, synchronous, hermetic, and offline-compatible (no network traffic or billing charges incurred just by starting the application).
- Eliminates unhandled tracebacks or frozen sessions when network or quota errors occur.
- Categorizing unrecoverable LLM failures as fatal with `ERROR`/`CRITICAL` logs and non-zero exit codes guarantees immediate visibility for ops, monitoring systems, and automated orchestration scripts.

### Alternatives Considered

- **Active network ping or dummy prompt during startup**: Rejected per user clarification (unnecessary network dependency, potential billing costs, and startup latency).
- **Automatic retries / multi-provider failover**: Excluded by clarification 2026-09-25 to maintain simplicity and prevent unexpected billing surges.
- **Ignoring startup check and failing on first query**: Rejected because it degrades user experience when running offline commands or waiting for document ingestion.

---

## 6. Prometheus Observability Instrumentation

### Context

FR-010 and SC-006 require Prometheus metrics for LLM calls:

- Call latency histograms.
- Token usage counters (prompt and completion tokens).
- Error rate counters.
- Labeled by provider and pipeline role.

### Decision

Implement `VatutaMetricsLogger` subclassing `litellm.integrations.custom_logger.CustomLogger` and register it with `litellm.callbacks`:

- `LLM_CALL_LATENCY` (`vatuta_llm_call_latency_seconds`): Histogram with labels `[provider, role, model, status]`. Buckets: `[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, +Inf]`.
- `LLM_TOKENS_TOTAL` (`vatuta_llm_tokens_total`): Counter with labels `[provider, role, model, token_type]`. The `token_type` label distinguishes input tokens (`prompt`) from output tokens (`completion`), enabling independent financial cost accounting and pricing tier analysis since LLM providers bill output tokens at significantly different rates than input tokens.
- `LLM_CALLS_TOTAL` (`vatuta_llm_calls_total`): Counter with labels `[provider, role, model, status]`.

The current role (`router` vs `generator`) is passed via execution context (`litellm` call metadata or thread-local context).

### Rationale

- `litellm.callbacks` intercepts 100% of LLM calls automatically, including internal ReAct retry loops, without manual timer code in domain logic.
- Consistent with existing Prometheus metrics in `src/metrics/metrics.py`.
- Enables empirical benchmarking across models and prompt designs (Constitution Observability Pillar).

### Alternatives Considered

- **Manual timing around `dspy.Predict` / `dspy.ReAct`**: Prone to omissions, misses internal multi-step tool calls within ReAct, and clutters business logic.
- **DSPy callbacks only**: Misses calls made via LangChain / LangGraph components.
