# Quickstart & Validation Guide: Provider-Independent LLM Abstraction

## Overview

This guide provides step-by-step procedures to validate that the provider-independent LLM abstraction functions correctly end-to-end with the Gemini model family in Vatuta.

---

## 1. Prerequisites

1. **Python & Poetry Environment**:
   - Python 3.12 active in the virtual environment.
   - All commands run via `poetry run` (or active venv).

2. **API Credentials**:
   - Export your Gemini API key:
     ```bash
     export GEMINI_API_KEY="your-gemini-api-key"
     ```
   - (Alternatively, set `GEMINI_API_KEY` in your local `.env` file).

---

## 2. Configuration Setup

Verify that `config/vatuta.yaml` specifies the new LiteLLM-based RAG configuration:

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

Refer to [contracts/config_schema.json](contracts/config_schema.json) for the full JSON schema definition.

---

## 3. Scenario 1: Successful End-to-End Query (Happy Path)

### Command:
```bash
poetry run python -m src.client.client ask "What are the latest updates from Slack?" --show-cot
```

### Expected Outcome:
1. **Startup**: Vatuta validates `config/vatuta.yaml` and confirms `GEMINI_API_KEY` is present.
2. **Routing Phase**: DSPy ReAct agent uses `gemini_flash` to resolve any filters or tools. The terminal displays the Router Chain of Thought panel.
3. **Generation Phase**: DSPy RAG module synthesizes the response using `gemini_pro`. The terminal displays the Generator Chain of Thought panel and the final answer.
4. **Exit Code**: `0` (Success).

---

## 4. Scenario 2: Switching Models via Configuration Only

### Procedure:
1. Update `config/vatuta.yaml` to switch `generator_backend` to `gemini_flash`:
   ```yaml
   router_backend: "gemini_flash"
   generator_backend: "gemini_flash"
   ```
2. Re-run the query:
   ```bash
   poetry run python -m src.client.client ask "Summarize recent tickets"
   ```

### Expected Outcome:
- Both routing and generation execute using `gemini-2.5-flash` with zero code modifications.
- Exit code: `0`.

---

## 5. Scenario 3: Pre-Flight Failure on Missing Credentials

### Procedure:
Temporarily unset the Gemini API key and execute the client:
```bash
GEMINI_API_KEY="" poetry run python -m src.client.client ask "Hello"
```

### Expected Outcome:
- Execution halts immediately during initialization.
- Console displays a clean diagnostic message:
  ```text
  [Model Execution Error]
  Missing credentials for provider 'gemini'. Please set the GEMINI_API_KEY environment variable.
  ```
- No uncaught Python stack traces are dumped.
- Exit code: `1`.

---

## 6. Scenario 4: Pre-Flight Failure on Malformed Model ID

### Procedure:
Set an invalid model name in `config/vatuta.yaml`:
```yaml
rag:
  llm_backends:
    broken:
      model: "invalid-format-without-slash"
```

### Expected Outcome:
- Pydantic configuration validation fails immediately on startup.
- Actionable error highlights the invalid model syntax requirement (`provider/model_name`).
- Exit code: `1`.

---

## 7. Scenario 5: Prometheus Metrics Verification

### Procedure:
Run a query and check that Prometheus metrics are updated:
```bash
poetry run python -c "
from src.metrics.llm_metrics import LLM_CALL_LATENCY, LLM_TOKENS_TOTAL, LLM_CALLS_TOTAL
print('Metrics registered successfully')
"
```

### Expected Outcome:
- `vatuta_llm_call_latency_seconds` observes duration labeled with `[provider="gemini", role="router"|"generator", model="...", status="success"]`.
- `vatuta_llm_tokens_total` records input tokens (`token_type="prompt"`) and output tokens (`token_type="completion"`) separately to support independent pricing and cost analysis.
- `vatuta_llm_calls_total` increments total calls count.
