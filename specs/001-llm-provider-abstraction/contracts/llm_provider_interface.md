# Contract: LLM Provider Interface

## Overview

This contract specifies the programmatic interface for language model backends within Vatuta, encompassing DSPy integration (`dspy.LM`), startup validation, and exception handling.

---

## 1. Provider Manager Interface

The provider manager acts as the entry point for retrieving configured language models.

```python
from typing import Optional
import dspy
from src.models.config import RagConfig

# Module location: src/llm/provider.py (exported from src/llm)
class LLMProviderManager:
    """Manages lifecycle, validation, and instantiation of LLM backends."""

    def __init__(self, config: RagConfig) -> None:
        """Initialize manager with validated RAG configuration.

        Args:
            config: Validated RAG configuration containing llm_backends.
        """
        ...

    def validate_backends(self) -> None:
        """Execute broad, non-intrusive pre-flight validation on configured backends.

        Validates:
        1. Model format conforms to 'provider/model_name' without making network calls.
        2. Required provider environment variables are present via dynamic LiteLLM inspection
           (`litellm.validate_environment(model)`) without hardcoding provider or variable names.

        Does NOT make active network probes, ping calls, or test completion prompts.

        Raises:
            LLMConfigurationError: If any backend has invalid configuration or missing credentials.
        """
        ...

    def get_dspy_lm(self, backend_name: str, role: str = "general") -> dspy.LM:
        """Instantiate or return cached DSPy language model instance.

        Args:
            backend_name: Key of backend in config.llm_backends.
            role: Pipeline role label ('router', 'generator', etc.) for telemetry.

        Returns:
            Configured dspy.LM instance.

        Raises:
            LLMConfigurationError: If backend_name is not configured.
        """
        ...
```

---

## 2. DSPy LM Interaction Contract

In accordance with Constitution Principle III, all prompts and model calls MUST use structured DSPy Signatures and Modules:

```python
# Router Invocation
with dspy.context(lm=provider_manager.get_dspy_lm(config.router_backend, role="router")):
    prediction = router_agent(question=question)

# Generator Invocation
with dspy.context(lm=provider_manager.get_dspy_lm(config.generator_backend, role="generator")):
    prediction = generator_module(question=question, context=context)
```

### Guarantees:
- **Zero Prompt Leakage**: Domain logic never constructs raw JSON or prompt strings.
- **Role Telemetry**: Each call carries the pipeline `role` through the context to the metrics logger.
- **Safe Rationale Extraction**: Intermediate reasoning traces are captured from `prediction.rationale` without modifying prompt outputs.

---

## 3. Error Handling Contract

All entry points (`client.py`, `agent.py`) must adhere to this failure contract:

1. **Catch Boundary**: Wrap model invocations with:
   ```python
   try:
       result = agent.run(question)
   except LLMError as e:
       logger.critical("Fatal LLM execution error: %s", e)
       console.print(Panel(f"[red]{e.message}[/red]", title="Model Execution Error", border_style="red"))
       raise typer.Exit(code=1)
   ```
2. **Standard Diagnostic Messages**:
   - Authentication: `"Authentication failed for model '{model}'. Check that required credentials ({missing_keys}) are properly set in your environment."`
   - Rate Limit: `"Provider '{provider}' rate limit exceeded (HTTP 429). Please wait before retrying."`
   - Connection: `"Network error connecting to provider '{provider}'. Please verify your internet connection."`
   - Configuration: `"Invalid model configuration for backend '{backend}': {reason}."`
