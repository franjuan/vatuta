# Data Model: Provider-Independent LLM Abstraction

## Overview

This document specifies the data models, entities, relationships, validation rules, and error hierarchies for the provider-independent LLM abstraction in Vatuta.

---

## 1. Entities & Data Models

### 1.1 `LLMBackendConfig` (Value Object / Pydantic Model)

Represents the configuration parameters for a specific language model service backend.

```python
class LLMBackendConfig(BaseModel):
    """Configuration definition for a single LLM backend."""

    model: str = Field(
        ...,
        description="LiteLLM model identifier in 'provider/model_name' format (e.g. 'gemini/gemini-2.5-flash')",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for text generation",
    )
    max_tokens: Optional[int] = Field(
        default=800,
        gt=0,
        description="Maximum token generation limit",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability threshold",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Custom API base URL endpoint if proxying or using enterprise endpoints",
    )
    extra_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific passthrough kwargs for LiteLLM completion calls",
    )
```

#### Validation Rules:
- `model`: Must be a non-empty string containing a slash separating provider and model (e.g., `gemini/gemini-2.5-flash`). For backwards compatibility with previous configurations, an alias `model_id` is supported during deserialization.
- `temperature`: Bounded in `[0.0, 2.0]`.
- `max_tokens`: Must be strictly positive integer if specified.
- `top_p`: Bounded in `[0.0, 1.0]` if specified.

---

### 1.2 `RagConfig` (Updated Aggregate Root)

The RAG configuration model holding backend registries and active role bindings.

```python
class RagConfig(BaseModel):
    """Configuration for RAG system and LLM role bindings."""

    llm_backends: Dict[str, LLMBackendConfig] = Field(
        ...,
        min_length=1,
        description="Registry of configured LLM backends",
    )
    router_backend: str = Field(
        ...,
        description="Backend ID from llm_backends bound to query routing",
    )
    generator_backend: str = Field(
        ...,
        description="Backend ID from llm_backends bound to answer synthesis",
    )
```

#### Cross-Field Validation:
- `llm_backends` must contain at least one configured backend.
- `router_backend` must exist as a key in `llm_backends`.
- `generator_backend` must exist as a key in `llm_backends`.
- If either referenced backend is missing, Pydantic raises a `ValidationError` detailing the missing backend key and available keys.

---

### 1.3 `LLMProviderManager` (Factory / Lifecycle Service)

Manages initialization, caching, and lifecycle for DSPy language model instances (`dspy.LM`). The implementation is strictly provider-agnostic: `validate_backends()` dynamically checks credentials for any configured model string via `litellm.validate_environment(model)` with zero vendor-specific logic.

```python
class LLMProviderManager:
    """Centralized manager for LLM client instances."""

    def __init__(self, config: RagConfig) -> None: ...
    def validate_backends(self) -> None: ...
    def get_dspy_lm(self, backend_name: str, role: str) -> dspy.LM: ...
```

#### State Transitions & Lifecycle:
```text
[Startup / Configuration Load]
            │
            ▼
    [validate_backends()]
       ├── Missing Credentials / Invalid Model ──► Raise LLMConfigurationError (Exit 1)
       └── Valid ──► Ready
                       │
                       ▼
             [get_dspy_lm(role)]
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       [Router LM]         [Generator LM]
     (gemini_flash)         (gemini_pro)
```

---

### 1.4 Observability Entities & Metrics

Telemetry recorded by `VatutaMetricsLogger` via `litellm.callbacks`:

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `vatuta_llm_call_latency_seconds` | Histogram | `provider`, `role`, `model`, `status` | Call latency distribution in seconds |
| `vatuta_llm_tokens_total` | Counter | `provider`, `role`, `model`, `token_type` | Token usage separated by `token_type` (`prompt` for input tokens, `completion` for output tokens) to enable distinct pricing and cost tracking |
| `vatuta_llm_calls_total` | Counter | `provider`, `role`, `model`, `status` | Total LLM invocations |

---

## 2. Exception Hierarchy

Domain exceptions defined under `src/llm/errors.py`:

```text
VatutaError
  └── LLMError
        ├── LLMConfigurationError    # Missing backend, invalid syntax, missing env credentials
        ├── LLMAuthenticationError   # Provider rejected API key / auth token
        ├── LLMRateLimitError        # Provider rate limit or quota exceeded (HTTP 429)
        ├── LLMConnectionError       # Upstream network failure, DNS error, timeout
        └── LLMBadRequestError       # Invalid request payload, unsupported parameters
```

### Mapping from LiteLLM Exceptions:
- `litellm.exceptions.AuthenticationError` → `LLMAuthenticationError`
- `litellm.exceptions.RateLimitError` → `LLMRateLimitError`
- `litellm.exceptions.APIConnectionError` → `LLMConnectionError`
- `litellm.exceptions.BadRequestError` → `LLMBadRequestError`
- Other `litellm.exceptions.LiteLLMError` → `LLMError`
