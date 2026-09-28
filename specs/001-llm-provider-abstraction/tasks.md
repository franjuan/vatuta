---
description: "Task list template for feature implementation"
---

# Tasks: Provider-Independent LLM Abstraction

**Input**: Design documents from `specs/001-llm-provider-abstraction/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are explicitly requested in the feature specification (unit tests for configuration parsing, provider instantiation, error mapping, and metrics).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Initialize domain module `src/llm/__init__.py` and matching test structure `tests/llm/`
- [ ] T002 [P] Add/update dependencies `litellm` (v1.96.2+), `dspy-ai` (v3.1.3+ / 3.3.0), `pydantic` (v2.12.4), `prometheus-client` (v0.23.1) in `pyproject.toml`
- [ ] T003 Update configuration schema file `config/vatuta.yaml.example` with the new LiteLLM backend structure

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 [P] Define `VatutaError`, `LLMError`, and specific error subclasses (`LLMConfigurationError`, `LLMAuthenticationError`, `LLMRateLimitError`, `LLMConnectionError`, `LLMBadRequestError`) in `src/llm/errors.py`
- [ ] T005 [P] Create Prometheus metrics definitions (histograms for latency, counters for tokens and errors) and `VatutaMetricsLogger` in `src/metrics/llm_metrics.py`
- [ ] T006 Write tests for error mapping logic in `tests/llm/test_errors.py`
- [ ] T007 Write tests for metrics callbacks in `tests/metrics/test_llm_metrics.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Configure and Switch Model Providers via Central Configuration (Priority: P1) 🎯 MVP

**Goal**: Select and configure active language model backends by updating settings in the configuration file (`config/vatuta.yaml`) using a strictly provider-agnostic abstraction (with Gemini used solely as the live validation target for this release).

**Independent Test**: Can be tested independently by running live end-to-end queries against the Gemini provider across different Gemini models or parameters, verifying that configuration changes alter model selection without application code changes.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T008 [P] [US1] Unit test for configuration validation (`LLMBackendConfig`, `RagConfig`) in `tests/models/test_llm_config.py`
- [ ] T009 [P] [US1] Unit test for `LLMProviderManager` instantiation and `get_dspy_lm` caching in `tests/llm/test_provider.py`
- [ ] T010 [P] [US1] Unit test for Engine LM initialization delegating to `LLMProviderManager` in `tests/rag/test_engine.py`

### Implementation for User Story 1

- [ ] T011 [P] [US1] Implement `LLMBackendConfig` and `RagConfig` (incorporating `llm_backends`, `router_backend`, `generator_backend`) in `src/models/config.py`. Enforce constraints: model must be 'provider/model_name', temperature [0.0, 2.0], max_tokens > 0.
- [ ] T012 [US1] Implement provider-agnostic `LLMProviderManager` with dynamic `litellm.validate_environment` and `get_dspy_lm` in `src/llm/provider.py` (zero vendor-specific branching or hardcoded keys)
- [ ] T013 [US1] Update RAG engine configuration loading and LLM instantiation to use `LLMProviderManager` in `src/rag/engine.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Independent Model Assignment for Routing and Generation (Priority: P2)

**Goal**: Configure different model backends or tiers for query routing versus content generation (e.g., fast Gemini Flash for routing, Gemini Pro for reasoning).

**Independent Test**: Can be tested independently by setting the routing backend to one model tier and the generator backend to another, verifying each stage executes against its designated backend.

### Tests for User Story 2

- [ ] T014 [US2] Extend integration/engine tests to verify correct model selection per role in `tests/rag/test_engine.py`

### Implementation for User Story 2

- [ ] T015 [US2] Update `RagConfig` validation logic in `src/models/config.py` to ensure `router_backend` and `generator_backend` strictly resolve against existing keys in `llm_backends`.
- [ ] T016 [US2] Update RAG agent workflow to invoke router with `role="router"` and generator with `role="generator"` using `dspy.context` in `src/rag/agent.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Clean Failure Termination and Error Reporting (Priority: P3)

**Goal**: Catch LiteLLM exceptions, log detailed diagnostics, report a clear and actionable error message, and abort the operation cleanly with an error exit status.

**Independent Test**: Test by attempting initialization without credentials and simulating runtime rate limits, verifying actionable output and non-zero exit code.

### Tests for User Story 3

- [ ] T017 [US3] Unit test for LiteLLM exception mapping to domain errors in `tests/llm/test_errors.py`
- [ ] T018 [US3] Unit test for CLI client catching `LLMError` and terminating with exit code 1 in `tests/client/test_client.py`

### Implementation for User Story 3

- [ ] T019 [US3] Implement generic translation mapping from `litellm.exceptions.*` to domain `LLMError` classes in `src/llm/provider.py` (applicable to any provider without hardcoding)
- [ ] T020 [US3] Update CLI entry points to wrap agent executions in `try/except LLMError`, log at CRITICAL, and `raise typer.Exit(code=1)` with user-friendly formatting via Rich in `src/client/client.py`

**Checkpoint**: All user stories should now be independently functional

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T021 [P] Ensure cyclomatic complexity constraints (Xenon <= 30) are maintained and code passes all linters (`just lint`).
- [ ] T022 [P] Create dedicated documentation in `docs/models.md` covering model configuration, provider abstraction, and role assignment (leaving `docs/integrations.md` dedicated to vector sources).
- [ ] T023 [P] Update `RELEASE.md` under `## [0.5.0]` with in-place configuration migration steps (`rag.llm_backends`), dynamic credential discovery via LiteLLM, Prometheus metrics, and clean error handling.
- [ ] T024 [P] Update `README.md` to reflect the new provider-independent architecture, Gemini defaults, and 0.5.0 release updates.
- [ ] T025 [P] Review and update all remaining project documentation in `docs/` and repository root to ensure consistency with the new LLM abstraction.
- [ ] T026 Verify lazy printf-style logging (`logger.info("...", arg)`) is used throughout `src/llm/provider.py` and `src/metrics/llm_metrics.py`.
- [ ] T027 Run `quickstart.md` validation locally.
- [ ] T028 [P] Bump project version to `0.5.0` in `pyproject.toml` (`[project] version = "0.5.0"`) to maintain lockstep release alignment with `RELEASE.md` and `README.md`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can proceed sequentially in priority order (US1 → US2 → US3) or parallel if independent.
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2).
- **User Story 2 (P2)**: Depends heavily on US1 for configuration models and manager.
- **User Story 3 (P3)**: Builds on US1 and US2 for exception boundaries in the CLI.

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Tests for a user story marked [P] can run in parallel
- Different models/files within a story marked [P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "T008 [P] [US1] Unit test for configuration validation in tests/models/test_llm_config.py"
Task: "T009 [P] [US1] Unit test for LLMProviderManager instantiation in tests/llm/test_provider.py"
Task: "T010 [P] [US1] Unit test for Engine LM initialization in tests/rag/test_engine.py"

# Implementation tasks:
Task: "T011 [P] [US1] Implement LLMBackendConfig and RagConfig in src/models/config.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories
