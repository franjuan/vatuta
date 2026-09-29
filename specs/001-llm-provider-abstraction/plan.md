# Implementation Plan: Provider-Independent LLM Abstraction

**Branch**: `001-llm-provider-abstraction` | **Date**: 2026-09-26 | **Spec**: [specs/001-llm-provider-abstraction/spec.md](specs/001-llm-provider-abstraction/spec.md)

**Input**: Feature specification from `specs/001-llm-provider-abstraction/spec.md`

## Summary

Implement a provider-independent LLM abstraction using LiteLLM as the unified backend for DSPy (`dspy.LM`). The core
implementation (`src/llm/`, `src/models/`, `src/rag/`, `src/client/`) is strictly provider-agnostic with zero
vendor-specific branching, logic, or hardcoded credential environment variable names. Credential discovery and
environment validation are delegated entirely to LiteLLM's dynamic introspection (`litellm.validate_environment(model)`),
seamlessly supporting any provider. As a practical operational consideration, Google Gemini is used solely as the
concrete live validation target and example in `config/vatuta.yaml.example` for this release, without coupling application
code to it. The feature replaces the existing `rag.llm_backends` configuration in-place in `config/vatuta.yaml`, allowing
independent model assignment for query routing and answer generation. The system performs a broad, general, and
non-intrusive pre-flight validation on startup using LiteLLM's native mechanisms (syntax check and dynamic credential
presence without live network probes or dummy test prompts), catches runtime provider failures as fatal unrecoverable
errors logged at ERROR/CRITICAL level to cleanly abort execution with actionable diagnostics and exit code 1, and exposes
Prometheus metrics (call latency, token usage distinguishing input/prompt vs output/completion tokens for pricing
analysis, error rates) via LiteLLM callbacks.

## Technical Context

**Language/Version**: Python 3.12 (`>=3.12,<3.14`), managed exclusively with Poetry

**Primary Dependencies**: `litellm` (v1.96.2+), `dspy-ai` (v3.3.0+ / v3.4.0, latest stable available), `pydantic` (v2.12.4), `prometheus-client` (v0.23.1)

**Storage**: File configuration (`config/vatuta.yaml`); no persistent database changes

**Testing**: `poetry run pytest tests/` with unit tests for configuration validation, provider factory, error mapping, and Prometheus metrics

**Target Platform**: Linux CLI & API services

**Project Type**: Personal AI Assistant (CLI / Service)

**Performance Goals**: Immediate synchronous startup pre-flight validation; runtime telemetry overhead < 5ms per call

**Constraints**: Strict mypy compliance; Xenon cyclomatic complexity <= 30 per function, <= 10 average; functions <= 30 lines; PEP 8 line length <= 120 characters; lazy logging formatting (`logger.info("...", arg)`); DCO `Signed-off-by` trailers; zero hardcoded credentials; zero provider-specific conditionals in application code

**Scale/Scope**: Centralized global LLM provider factory (`src/llm/provider.py`), error definitions (`src/llm/errors.py`), telemetry handler (`src/metrics/llm_metrics.py`), updated configuration models (`src/models/config.py`), updated RAG engine (`src/rag/engine.py`), updated RAG agent (`src/rag/agent.py`), and CLI client integration (`src/client/client.py`, invoking `provider_manager.validate_backends()` during startup)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle / Rule | Compliance Status | Analysis & Verification |
| --- | --- | --- |
| **I. Python 3.12 & Poetry Standard Environment** | PASS | All dependencies managed via Poetry, targeting Python 3.12. |
| **II. Mandatory Quality Gates & Static Verification** | PASS | Code complies with Black (120 chars), isort, Ruff, pydocstyle Google convention, strict Mypy, and Xenon CC <= 30. |
| **III. LLM Provider Independence & Isolation** | PASS | Core requirement. Codebase contains zero vendor-specific branching or hardcoded provider keys. Centralizes LLM access through LiteLLM and DSPy structured signatures; Gemini is used solely as an external validation target. |
| **IV. Determinism & Simplicity** | PASS | Strict Pydantic models for configuration; simple, focused provider factory without unnecessary abstraction layers. |
| **V. Provenance & Fact-Conclusion Separation** | PASS | Preserves RAG retrieval citations and Chain of Thought trace capture from Router and Generator. |
| **VI. Zero-Trust & Credential Hygiene** | PASS | Zero secrets stored in configuration; dynamic credential discovery for any provider delegated entirely to LiteLLM environment introspection without hardcoded variable names in code. |
| **VII. Safeguards for Side-Effecting Operations** | PASS | Read-only model interactions; graceful, clean termination on provider errors with exit code 1. |
| **VIII. Behavioral Preservation & Stability** | PASS | In-place configuration replacement documented in `RELEASE.md` with clear migration guide. |
| **IX. Comprehensive Test Coverage** | PASS | Unit tests covering configuration parsing, provider instantiation, error mapping, and metrics. |
| **X. Standard Protocols & Ecosystem Reuse** | PASS | Reuses LiteLLM standard gateway and DSPy native LM instead of proprietary in-house wrappers. |
| **XI. IDE & AI-Agent Agnosticism** | PASS | CLI-first workflow executable via standard shell and Poetry commands. |
| **Observability (Tracing, Logging, Metrics)** | PASS | Exposes Prometheus histograms and counters; lazy printf-style logging; Rich UI panels for CLI presentation. |

## Project Structure

### Documentation (this feature)

```text
specs/001-llm-provider-abstraction/
├── spec.md                          # Feature specification with clarifications
├── plan.md                          # Implementation plan (this file)
├── research.md                      # Phase 0 research findings and architectural decisions
├── data-model.md                    # Phase 1 data entities and validation rules
├── quickstart.md                    # Phase 1 quickstart validation scenarios
├── contracts/
│   ├── config_schema.json           # JSON Schema contract for config/vatuta.yaml RAG section
│   └── llm_provider_interface.md    # Programmatic interface contract for LLM provider
└── checklists/
    ├── requirements.md              # Requirements verification checklist
    └── integration.md               # Integration verification checklist
```

### Release & Project Documentation Deliverables (to update during implementation)

The following documentation and release files MUST be updated in lockstep during the implementation phase (not beforehand), adhering to Constitution Section 5 (Release Requirements):

- `RELEASE.md`: Add feature changes into the existing `## [0.5.0]` release section (documenting in-place
  configuration migration from legacy `rag.llm_backends` to LiteLLM-native backend schema, LiteLLM provider
  abstraction, and Prometheus metrics telemetry under `## [0.5.0]`).
- `docs/models.md`: Create dedicated documentation covering model configuration, provider-agnostic abstraction,
  LiteLLM integration, role assignment (`router_backend`, `generator_backend`), dynamic credential discovery, and
  pre-flight verification (leaving `docs/integrations.md` dedicated exclusively to vector data sources).
- `README.md`: Reflect provider-independent architecture and Gemini defaults.
- `pyproject.toml`: Bump project version to `0.5.0` (`[project] version = "0.5.0"`) to align with `RELEASE.md`.

### Source Code (repository root)

```text
src/
├── models/
│   └── config.py                    # LLMBackendConfig, updated RagConfig with validator
├── llm/
│   ├── __init__.py                  # Public exports (LLMProviderManager, exceptions)
│   ├── provider.py                  # LLMProviderManager (factory & lifecycle for dspy.LM instances)
│   └── errors.py                    # LLM exception hierarchy and LiteLLM error mapping
├── rag/
│   ├── engine.py                    # Updated build_dspy_lm delegating to global LLMProviderManager
│   └── agent.py                     # Agent integration with global provider manager and error boundary
├── metrics/
│   ├── metrics.py                   # Existing source metrics
│   └── llm_metrics.py               # Prometheus metrics (latency, tokens, calls) & VatutaMetricsLogger
└── client/
    └── client.py                    # Startup pre-flight checks (validate_backends()) and clean exit code 1 handling

tests/
├── models/
│   └── test_llm_config.py           # Configuration schema and validation tests
├── llm/
│   ├── test_provider.py             # LLMProviderManager instantiation and dspy.LM caching tests
│   └── test_errors.py               # Error hierarchy and exception mapping tests
├── metrics/
│   └── test_llm_metrics.py          # Prometheus metrics and logger callback tests
├── rag/
│   └── test_engine.py               # Engine LM initialization tests delegating to LLMProviderManager
└── client/
    └── test_client.py               # CLI client startup pre-flight and LLMError exit code 1 tests
```

**Structure Decision**: Global modular layout introducing `src/llm/` as a top-level domain module alongside `src/rag/`, `src/sources/`, `src/entities/`, and `src/models/`, making the LLM provider reusable across RAG, future agents, MCP tools, and autonomous tasks. Test suite follows the mirror package structure (`tests/llm/`, `tests/models/`, `tests/metrics/`, `tests/rag/`, `tests/client/`), strictly adhering to the repository testing guidelines.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| *None* | No constitutional violations identified. | Architecture follows standard ecosystem patterns and simple factories. |
