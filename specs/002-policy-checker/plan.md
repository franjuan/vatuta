# Implementation Plan: Policy Validation Engine

**Branch**: `002-policy-checker` | **Date**: 2026-11-11 | **Spec**: [spec.md](spec.md)
**Input**: Stateless `PolicyService` contract, step-by-step creation & execution flows, pragmatic TDD & CI strategy,
minimal `quickstart.md`, and isolated GitHub/Jira blog demonstration ([demo-plan.md](demo-plan.md))
**Status**: Phase 0 research and Phase 1 design complete; runtime implementation not yet performed

## Summary

Implement a **source-independent, stateless `PolicyService`** (`src/policy/service.py`) exposing two domain
operations—create complete `ControlDefinition` objects (`create`) and execute supplied `ControlDefinition` objects
inline (`execute`)—plus trace resource reading (`read_trace`). All request and result models use JSON-serializable
Pydantic v2 types with `Field(description=...)` metadata so they map 1:1 to future MCP tools (`policy_create`,
`policy_execute`); `vatuta policy serve` and `policy_parameters` remain excluded from this increment.

Creation (`PolicyService.create`) follows a step-by-step DSPy pipeline:

1. **Step 0 — Pre-flight & Tool Catalog**: Validate `policy_text`. Always build the catalog from local
   `policy.evidence_profiles` (`arguments_schema` + `contract_reference`). No MCP startup, connection, or tool discovery
   occurs before Phase A succeeds. With zero attempts, creation never contacts MCP.
2. **Step 1 — Policy Decomposition (`PolicyDecompositionSignature`)**: Decompose `policy_text` into atomic normative
   rules (splitting conditions whenever they can fail independently or represent distinct violations to the policy
   author), failing with `NO_NORMATIVE_RULES` (`stage: "decomposition"`) if zero normative obligations exist. Assign
   each control a deterministic kebab-case `id`: `f"{policy_id}-{rule_id}"` (or `rule_id` if already prefixed with
   `f"{policy_id}-"`, deduplicated within the creation output).
3. **Step 2, Phase A — All-Rules Parameter & Evidence Sweep (Step 2.1 across `r_1 ... r_N`)**: Run
   `RuleEvidenceAndParametersSignature` across all rules first with the incremental `SharedParameterCatalog`, declaring
   only the minimum necessary parameters (`parameters: {}` when none are required). Two parameters share the same name
   only when `value_schema`, `construction_value`, and `default` match; otherwise the constructor adapts to the catalog
   or uses a distinct parameter name. When `trial_attempts == 0`, `construction_value` is optional. When
   `trial_attempts > 0`, aggregate and return **all** `MISSING_CONSTRUCTION_CONTEXT` errors and affected `control_id`s
   across all rules at once before invoking any MCP server.
4. **Step 2, Phase B — Per-Rule Deep Construction (`r_1 ... r_N`) with Fail-Fast**:
   - **Step 2.2 (Discovery, when `trial_attempts > 0`)**: After Phase A succeeds, discover and validate the required
     MCP tool `inputSchema` before evidence calls. Fetch declared evidence once in memory (cached per rule by
     `(server, tool, resolved_arguments)`) and run up to `policy.discovery_max_queries` (default `5`) exploratory
     JSONata projections over the root context `{"evidence": ..., "parameters": ...}`, truncating each output to
     `policy.discovery_max_output_bytes` (default `4096` bytes + `[TRUNCATED]`). Query errors or `missing` results are
     returned as observations within the query budget; `done=True` closes the loop early.
   - **Step 2.3 (`RuleSelectionSignature`)**: Can declare additional parameters (checked against
     `SharedParameterCatalog`), builds `selection` and `selection_tests`, normalizes single-object JSONata output to
     `[obj]`, enforces non-whitespace `id`/`descriptive_id` and rejects duplicate subject `id`s
     (`DUPLICATE_SUBJECT_ID`), resolves test parameters via `1. test_case.parameters` -> `2. default` ->
     `3. construction_value`, and validates against `selection_tests` (mandatory) and live evidence (when
     `trial_attempts > 0`).
   - **Step 2.4 (`RuleCriterionSignature`)**: Can declare additional parameters (checked against
     `SharedParameterCatalog`), builds `criterion` and `criterion_tests`, and validates against `criterion_tests`
     (mandatory, using CEL for `deterministic` and `SemanticCriterionSignature` LLM calls with `{{parameters.<name>}}`
     template substitution in `condition` for `mode: llm`) and live subjects (when `trial_attempts > 0`).
   - **Step 2.5 (Full-Context Bounded Retry & Fail-Fast)**: When `trial_attempts > 1` and a rule fails in Steps
     2.2–2.4, re-executes from Step 2.1 carrying the full accumulated context and errors from all prior stages, reusing
     cached in-memory MCP evidence whenever `evidence` bindings are unchanged. If any rule fails definitively after its
     attempts, Phase B aborts immediately (**Fail-Fast**). Persist `test_cases` in `ControlDefinition.test_cases`.

Shared declarations already used by other rules are immutable during later construction and retries. Candidate changes
remain local until validated; adapt the current rule or use a distinct parameter name rather than changing other rules.

The Vatuta client (`src/client/policy.py`) invokes `PolicyService` directly, manages local YAML control files under
`policy.controls_directory` (default `data/controls`, overridable via `--controls-dir`), provides `ClientControlQuery`
for local filtering, attribute projection, and `include_merged_parameters: bool = True` aggregation, and exposes three
CLI subcommands: `vatuta policy create`, `vatuta policy execute`, and the guided 2-step `vatuta policy check "<prompt>"`
pipeline.

Quality and validation strictly separate **three concerns**:

1. **Permanent automated tests** (`tests/`) driven by pragmatic TDD (unit tests primary, lightweight contract tests,
   minimal local wiring tests) running deterministically in GitHub Actions with zero Docker, database, or SaaS secrets.
2. **Minimal local E2E validation** ([quickstart.md](quickstart.md)) covering two representative acceptance scenarios
   through a local guide; the underlying hermetic tests also run in CI.
3. **Temporary blog demonstration** ([demo-plan.md](demo-plan.md), isolated under `pocs/policy_demo/`) showcasing real
   policy checking over GitHub Free and Jira Cloud Free via existing MCP containers, with zero coupling to `src/` or CI.

### Confirmed Review Decisions (2026-11-11)

The client rejects existing target files with `CONTROL_FILE_EXISTS` (exit `1`), checking the complete output set before
publishing and never overwriting. Each `ControlResult` includes `source` and effective per-control `parameters` under
both filters, including errors/inapplicability. An internal context supplies a common ID for `check` and execution
without new public request fields. Trace-write failure preserves calculated results, adds global `TRACE_WRITE_FAILED`,
and produces exit `1`; diagnostic failure does not change subject verdicts.

The optional demo uses a complete standalone configuration template with explicit LLM bindings and all required
general settings. It does not copy personal configuration or rely on implicit merging. Credentials stay in `.env`.

## Technical Context

**Language/Version**: Python >=3.12,<3.14; validated on Python 3.12 and 3.13 using Poetry.

**Primary Dependencies**: Existing Pydantic 2, PyYAML, Typer/Rich, DSPy/LiteLLM, MCP Python SDK, Docker SDK, and
Prometheus. Add `jsonata-python==0.7.1` and `cel-python==0.5.0`, plus direct declarations for `jsonschema` and
`google-re2` when imported directly. Dependency installation, lock updates, and `pip-audit` occur during implementation.

**Storage**: Client-owned YAML files under `policy.controls_directory` (`data/controls`) and sanitized local invocation
traces under `policy.trace_directory` (`logs/policy`). No database, history store, or server-side rule repository.

**Testing**: Pragmatic TDD with `pytest` in GitHub Actions (Python 3.12 & 3.13): fast mocked unit tests as the primary
mechanism, lightweight self-contained contract tests for Pydantic/MCP/YAML/evidence boundaries, and minimal in-process
wiring tests using `tmp_path`—with zero Docker containers, external credentials, or live third-party services in CI.

**Target Platform**: Linux CLI environment (`vatuta policy`). Upstream MCP evidence tools execute via Vatuta's existing
hardened Docker-stdio `MCPServer` (`src/mcp/server.py`) when run against live sources locally.

**Project Type**: Python application/library with stateless service API (`PolicyService`) and CLI (`vatuta policy`).

**Performance Goals**: None invented. Record stage durations and model token usage; execute sequentially.

**Constraints**:

- **Source independence**: Core policy creation and execution (`src/policy/`) MUST remain 100% independent of GitHub,
  Jira, or any specific MCP server; provider specifics exist only in YAML `mcp_servers` / `policy.evidence_profiles`
  configuration and generated `ControlDefinition` data.
- **Stateless boundary & bounded LLM queries**: JSON-serializable Pydantic models; configurable creation JSONata
  discovery (`policy.discovery_max_queries=5`, `policy.discovery_max_output_bytes=4096`) and check Step 1 filtering
  (`policy.check_max_filter_queries=3`, `policy.check_max_output_bytes=4096`); mandatory `test_cases`; no hidden LLM
  inference inside `PolicyService.execute`.
- **CI simplicity & secret hygiene**: No Docker container provisioning, third-party SaaS calls, or new secrets in
  `.github/workflows/ci.yml`; local credentials reside exclusively in `.env`.

**Scale/Scope**: Single-invocation inline `ControlDefinition` list and selected subjects; temporary blog demo isolated
to `pocs/policy_demo/` (~5 Jira issues, 3 `main` commits, 4 PRs, 4 policy rules).

## Constitution Check

Evaluated against [constitution.md](../../.specify/memory/constitution.md) (`v1.16.0`):

| Gate | Pre-research | Post-design evidence |
| --- | --- | --- |
| I–VIII: Simplicity, scope control, anti-overengineering | PASS | Stateless 2-operation `PolicyService`; no custom demo adapters or extra test frameworks; `policy serve` deferred. |
| IX/X: Poetry, Python 3.12/3.13, static quality gates | PASS | All commands via `poetry run`; strict mypy, Ruff, Black, Xenon, Bandit, Semgrep, and `pocs/` exclusion preserved. |
| XI/XII: Atomic DSPy prompts & deterministic core | PASS | Atomic DSPy signatures per step; deterministic `ClientControlQuery`, `merged_parameters`, unit-test runner, and CEL. |
| XIII: Provenance & evidence-logic separation | PASS | Attributable `control_id`, `tags`, `test_cases`, `source`, stage traces, and strict separation of evidence vs verdicts. |
| XIV/XV: Zero-Trust MCP isolation & read-only tools | PASS | Reuses unchanged hardened `src/mcp/server.py` (`--read-only`, `--cap-drop=ALL`, digest pinning, tool allowlists). |
| XVI: Public CLI/config compatibility & `pocs/` isolation | PASS | Adds `vatuta policy` (`create`, `execute`, `check`); isolates temporary blog demo strictly under `pocs/policy_demo/`. |
| XVII: Pragmatic TDD, contract tests & hermetic CI | PASS | Unit-first TDD + lightweight contract tests in CI; minimal integration scope; zero Docker/SaaS dependencies in CI. |
| XVIII/XIX: Ecosystem reuse & CLI/IDE agnosticism | PASS | Reuses JSONata, CEL, JSON Schema, Pydantic v2, and existing GitHub/Jira MCP images via standard CLI workflows. |
| Security & Observability | PASS | `.env`-only credentials, bounded queries/truncation, sanitized `logs/policy` traces (`read_trace`), stderr progress. |

The observability gate records application decisions, stage transitions, and returned model outputs; it does not
require unavailable private model reasoning (FR-037). No constitution amendment is introduced by this review.
Principle XVII was previously amended (`1.15.0` → `1.16.0`) to codify the permanent,
project-wide Pragmatic TDD, lightweight contract testing, minimal integration testing, and hermetic CI rules while
keeping all demo specifics out of the constitution. The 2026-11-11 confirmed decisions refine FR-010, FR-017,
FR-038, FR-041, and FR-047; provider-independent evidence access remains unchanged.

## Project Structure

### Planning & Specification Artifacts (`specs/002-policy-checker/`)

```text
specs/002-policy-checker/
├── spec.md                # Preserved authoritative functional specification
├── plan.md                # Master implementation, testing, validation, and demo boundary plan
├── research.md            # Technical decisions (JSONata, CEL, DSPy, MCP profiles, security)
├── data-model.md          # Pydantic v2 models, embedded test_cases, config, and state transitions
├── design-inputs.md       # Resolved design register and implementation qualification checks
├── quickstart.md          # Minimal 2-scenario local E2E acceptance guide (outside CI)
├── demo-plan.md           # Temporary GitHub Free + Jira Cloud Free blog demonstration design
├── contracts/
│   ├── operations.md      # Stateless PolicyService, ClientControlQuery, CLI, and trace URI contracts
│   ├── control-format.md  # YAML ControlDefinition, expression contexts, and completeness contracts
│   └── reference-control.yaml
└── checklists/
    └── requirements.md
```

### Permanent Production Code & Automated Tests (`src/`, `tests/`)

```text
src/policy/
├── service.py             # Stateless PolicyService (create, execute, read_trace) — provider-neutral
├── creation.py            # Step-by-step creation, SharedParameterCatalog, JSONata discovery, unit tests
├── execution.py           # Independent control/subject processing and result filtering
├── files.py               # Client safe YAML I/O, controls_directory containment, filtering, projection
├── parameters.py          # Shared parameter merging (include_merged_parameters), defaults, conflicts
├── agent.py               # Guided 2-step check pipeline (Step 1 rule selection, Step 2 param extraction)
├── evidence.py            # Generic MCPServer caller, payload normalization, completeness-profile check
├── expressions.py         # JSONata/CEL boundary adapters, output truncation, deterministic guards
├── semantic.py            # Singleton DSPy semantic judgment and strict correlation-token verification
├── trace.py               # Sanitized stage artifacts, URI resolution, and metric integration
├── config.py              # PolicyConfig (controls_directory, query/byte limits, profiles, trace_directory)
├── signatures/            # One atomic DSPy Signature class per file
└── models/                # One JSON-serializable Pydantic v2 model class per file
src/client/policy.py       # Typer group (create, execute, check), JSON/Rich output, client YAML saving
src/client/client.py       # Register policy group; skip eager LLM preflight only for policy commands
src/models/config.py       # Add optional PolicyConfig to VatutaConfig
src/llm/provider.py        # Selected-backend validation with existing default preserved
src/metrics/policy_metrics.py

tests/policy/              # Permanent unit, lightweight contract, and minimal local wiring tests (CI)
├── test_expressions.py    # JSONata envelope, builtin guards, truncation, CEL strict bool/error unit tests
├── test_parameters.py     # Shared parameters, defaults vs construction_value, conflict/missing unit tests
├── test_files.py          # Safe YAML load/save, containment, ClientControlQuery filter/projection tests
├── test_evidence.py       # MCP payload conversion and EvidenceProfile completeness unit tests
├── test_creation.py       # Step-by-step decomposition, catalog reuse, discovery budget, test_cases, trials
├── test_execution.py      # Control/subject evaluation, PRs 101/102/103, filters, isolated control errors
├── test_semantic.py       # Semantic mode: llm correlation and error-isolation unit tests
├── test_agent.py          # Guided 2-step policy check selection and parameter extraction unit tests
├── test_trace.py          # Sanitization, redaction, and PolicyService.read_trace unit tests
└── test_contracts.py      # Self-contained Pydantic/MCP JSON Schema, reference YAML & boundary contracts
tests/client/test_policy.py # CLI create/execute/check wiring, exit codes (0/1/2), stdout/stderr separation
tests/fixtures/policy/     # Deterministic provider-neutral JSON/YAML test fixtures (no live SaaS data)
```

### Temporary Blog Demonstration Assets (`pocs/policy_demo/` — Isolated Non-Production Scope)

```text
pocs/policy_demo/          # Excluded from production package, static gates, and GitHub Actions CI
├── README.md              # Minimal setup checklist (.env vars, digest pinning, screenshot commands)
├── policy.md              # Example Change Management & Release Traceability Policy (4 rules)
├── vatuta.demo.yaml       # Demo-only MCP config (github-mcp-server, mcp-atlassian) & evidence_profiles
└── controls/              # Generated <id>.yaml controls produced locally during the demo run
```

**Structure Decision**: Keep `src/policy/` and `tests/policy/` 100% provider-independent. Place all temporary blog
demonstration files under `pocs/policy_demo/`, which is already Vatuta's constitutionally approved exclusion boundary
for non-production PoCs and demonstrations.

## Phase 0 & Phase 1 Design Summary

- **[research.md](research.md)**: Resolves the stateless `PolicyService` boundary (R1), JSONata existence envelope and
  configurable exploratory projections (R2), CEL strict booleans (R3), atomic DSPy creation/check signatures (R4),
  client YAML filtering/projection and `include_merged_parameters` (R5), generic `EvidenceProfile` completeness checks
  (R6), and trace sanitization/security bounds (R7).
- **[data-model.md](data-model.md)**: Defines `ControlDefinition` (`tags`, `test_cases`), `SelectionTestCase`,
  `CriterionTestCase`, `ParameterDeclaration`, `SharedParameterCatalog`, `ClientControlQuery`, `EvidenceProfile`,
  `PolicyConfig`, and state transitions.
- **[operations.md](contracts/operations.md) & [control-format.md](contracts/control-format.md)**: Fix stateless
  `PolicyService` schemas, CLI subcommands (`create`, `execute`, `check`), trace URIs, YAML structure, and
  [reference-control.yaml](contracts/reference-control.yaml).

### Processing and Resource Ownership

1. **`PolicyService.create`**: Decomposes `policy_text` into atomic rules (or fails with `NO_NORMATIVE_RULES`), assigns
   deterministic slug `id`s (`f"{policy_id}-{rule_id}"`), runs Phase A (Step 2.1 across all rules with
   `SharedParameterCatalog` and minimum necessary parameters to collect all `MISSING_CONSTRUCTION_CONTEXT` errors at
   once when `trial_attempts > 0`), and then runs Phase B per rule with Fail-Fast: bounded JSONata discovery over
   `{"evidence", "parameters"}` (with error feedback and early exit when `done=True`), mandatory `selection_tests`
   (with `[obj]` normalization and `DUPLICATE_SUBJECT_ID` check) and `criterion_tests` (CEL or LLM with
   `{{parameters.<name>}}` template rendering for `mode: llm`) using `test_case.parameters` -> `default` ->
   `construction_value` precedence, full-context retries from Step 2.1 reusing cached in-memory MCP evidence, and
   returns complete `ControlDefinition` objects or `[]` on error.
2. **`vatuta policy check "<prompt>"`**: Associates all stages with a single execution `invocation_id` trace
   (`vatuta://traces/policy/execution/{invocation_id}`), runs Step 1 iterative lightweight `ClientControlQuery`
   selection (`check_max_filter_queries`, `check_max_output_bytes`, `done=True`, failing with `NO_CONTROLS_MATCHED` if
   empty), deterministically loads full controls and `merged_parameters`, skips Step 2 when `merged_parameters == []` or
   extracts explicit prompt parameters without inventing missing values, and calls `PolicyService.execute`.
3. **`PolicyService.execute`**: Validates nonempty unique `control_id`s and shared declared parameters (`Stage E0`,
   allowing extra unused keys in `request.parameters` and passing each control only its own declared parameters) before
   any external call. Then executes per control (`Stages E1–E4`), keeping started `MCPServer` containers alive across
   controls and caching valid evidence in memory by exact `(server, tool, resolved_arguments)`, fast-failing a control
   on its first failed evidence item, evaluating unique-ID valid subjects while reporting colliding subject IDs as
   `DUPLICATE_SUBJECT_ID` without criterion evaluation, rendering `{{parameters.<name>}}` templates in `mode: llm`
   (1 subject per call), and retaining only `NON_COMPLIANT` in `ControlResult.subjects` when
   `result_filter="violations"` while preserving all errors in `ControlResult.errors`.

---

## Testing, Validation & Demonstration Strategy

### Strict Separation of Responsibilities

| Concern | Location | Purpose | External Dependencies | Runs in GitHub Actions CI? | Maintenance Lifecycle |
| --- | --- | --- | --- | --- | --- |
| **1. Permanent Automated Tests** | `tests/policy/`, `tests/client/test_policy.py` | Prevent regressions across core logic, expressions, contracts, and CLI wiring | None (all MCP and LLM calls mocked/faked) | **Yes** (Python 3.12 & 3.13 matrix) | Permanent; maintained with production code |
| **2. Minimal Quickstart Validation** | [quickstart.md](quickstart.md) | Local run of two focused CLI wiring cases | None; controlled authoring and SDK-shaped MCP fixtures | Guide runs locally; underlying hermetic cases also run in CI | Updated with public CLI/service changes |
| **3. Temporary Blog Demonstration** | [demo-plan.md](demo-plan.md), `pocs/policy_demo/` | Capture realistic GitHub + Jira compliance screenshots for a technical blog post | Local Docker, `.env` tokens, GitHub Free repo, Jira Cloud Free project | **Never** (strictly isolated from CI) | Temporary one-off showcase; zero long-term maintenance |

### 1. Permanent Automated Testing — Pragmatic TDD & CI Constraints

Follow a pragmatic **Red → Green → Refactor** cycle per module: write failing unit/contract tests for the approved
behavior first, implement the smallest passing code, and refactor to satisfy complexity and style gates.

- **Unit Tests (Primary Mechanism — `tests/policy/test_*.py`)**:
  - Fast, isolated, deterministic tests mocking `MCPServer` and DSPy/LiteLLM calls.
  - Cover JSONata existence envelope (`missing` vs `null` vs `[]`), disabled nondeterministic builtins, byte truncation
    with `[TRUNCATED]`, CEL strict booleans/errors, safe YAML containment, `ClientControlQuery` filtering/projection,
    `include_merged_parameters`, `SharedParameterCatalog` reuse, mandatory `test_cases` execution, trial budgets
    (`0`/`1`/`N`), independent per-control failure isolation, `result_filter` (`violations` vs `all`), semantic
    correlation tokens, and trace redaction. Add focused cases for no MCP discovery before Phase A, immutable shared
    declarations through retries, refusal to overwrite, self-contained result attribution, pre-execution check errors,
    and preserved results with global `TRACE_WRITE_FAILED`/exit `1`.
- **Lightweight Contract Tests (`tests/policy/test_contracts.py` + existing `tests/test_mcp_*.py`)**:
  - Verify `PolicyCreateRequest`/`Result`, `PolicyExecuteRequest`/`Result`, and `ControlDefinition` generate valid,
    self-contained Pydantic v2 JSON Schemas (`model_json_schema()`), enforce `extra="forbid"`, and contain zero
    filesystem path types (`Path` or `control_file`).
  - Parse [reference-control.yaml](contracts/reference-control.yaml), execute its embedded `selection_tests` and
    `criterion_tests` with real `jsonata-python` and `cel-python`, and verify YAML round-trip stability.
  - Validate MCP `CallToolResult` normalization (`isError`, `structuredContent`, single JSON text block) and DSPy
    signature field contracts without starting containers or calling remote APIs.
- **Minimal Integration Scope (`tests/client/test_policy.py`)**:
  - Limit integration tests strictly to lightweight, in-process CLI → `ClientControlQuery` (`tmp_path`) →
    `PolicyService` → real JSONata/CEL → `trace.py` (`tmp_path`) flows with MCP lifecycle, `list_tools`, `call_tool`,
    and
    LLM completion boundaries stubbed using SDK-shaped tool schemas and responses.
  - Do **not** build Docker-in-CI harnesses, mock HTTP servers, or database fixtures.
- **GitHub Actions CI Constraints (`.github/workflows/ci.yml`)**:
  - Preserve the existing workflow structure (`pre-commit`, Python `3.12`/`3.13` `pytest` matrix, SAST, complexity,
    duplication) with **no new CI services, no Docker container pulls/runs, and no new GitHub Actions secrets**.

### 2. Minimal Feature Validation — `quickstart.md`

[quickstart.md](quickstart.md) defines exactly **two** local post-implementation E2E acceptance scenarios:

1. **Scenario 1 (Happy Path)**: Author a control with `vatuta policy create --trial-attempts 0` (verifying YAML output
   and embedded `test_cases`), execute that saved control against `tests/fixtures/policy/` under both result filters,
   verify `source`/effective `parameters`, and read back its sanitized trace via `PolicyService.read_trace(uri)`.
2. **Scenario 2 (Missing-Input Guardrail)**: Invoke `vatuta policy execute` with `--parameters '{}'`, asserting exit
   code `2`, complete missing-parameter reporting (`repository`, `project`), no fallback to `construction_value`, and
   zero MCP/LLM invocations.

### 3. Temporary Blog Demonstration — `demo-plan.md`

[demo-plan.md](demo-plan.md) specifies the self-contained, one-time demonstration under `pocs/policy_demo/`:

- **Infrastructure**: GitHub Free (`<owner>/vatuta-payments-demo`) + Jira Cloud Free (`PAY`) queried read-only through
  existing `ghcr.io/github/github-mcp-server` and `ghcr.io/sooperset/mcp-atlassian` Docker images via Vatuta's unchanged
  `src/mcp/server.py`, with credentials in local `.env` and a complete standalone `pocs/policy_demo/vatuta.demo.yaml`.
  Include required general settings, explicit LLM bindings, policy paths, and evidence profiles; do not depend on a
  personal config or implicit merging. Reject contradictory known totals in completeness proofs.
- **Scenario & Dataset**: 5 Jira issues (`PAY-100`..`PAY-104`), 3 `main` commits (2 compliant, 1 untracked `hotfix:`),
  and 4 PRs (`PR #1` compliant `In Review` + concrete rollback plan; `PR #2` non-compliant `Backlog` + `"TBD"` rollback
  placeholder; `PR #3` non-compliant unlinked; `PR #4` draft excluded).
- **Example Policy & 4 Rules**:
  1. `pr-jira-traceability` (GitHub + Jira, `deterministic` CEL): `PR #1` `COMPLIANT`; `PR #2`, `PR #3` `NON_COMPLIANT`.
  2. `commit-ticket-prefix` (GitHub `list_commits`, `deterministic` CEL): `PAY-100`, `PAY-103` `COMPLIANT`; `hotfix`
     `NON_COMPLIANT`.
  3. `high-priority-rollback-plan` (GitHub + Jira, `llm` semantic): `PR #1` `COMPLIANT`; `PR #2` `NON_COMPLIANT`.
  4. `critical-incident-postmortem-link` (Jira `jira_search`, `deterministic` CEL): `NOT_APPLICABLE` (`0` `Highest`
     resolved bugs).
- **Screenshots**: 6 terminal/UI captures using real `vatuta policy create`, `check`, `execute`, and trace JSON output.

### Task Organization Rules for `/speckit-tasks`

When generating `tasks.md`, `/speckit-tasks` MUST maintain a strict boundary between production and demonstration work:

1. **Permanent Production Phases (Phases 1–6)**: Cover `Setup`, `Foundational`, `US1`–`US4`, and `Polish & Quickstart
   Validation` across `src/`, `tests/`, `config/vatuta.yaml.example`, and `docs/`. Follow pragmatic TDD (unit and
   contract test tasks precede implementation tasks in each phase) with **zero** references to `pocs/policy_demo/` or
   live GitHub/Jira accounts.
2. **Temporary Blog Demonstration Phase (Final Standalone Phase — Non-Production / Local Only)**: Group all tasks from
   [demo-plan.md](demo-plan.md) into a clearly labeled final optional phase targeting **only `pocs/policy_demo/`**
   (`vatuta.demo.yaml`, `policy.md`, `README.md` setup/seed instructions, and local screenshot execution), explicitly
   noting that core feature delivery and CI completion do not depend on this phase.

## Documentation Updates

During permanent implementation update `docs/policy-checker/design.md`, `glossary.md`,
`adr-019-file-based-policy-operations.md`, and `config/vatuta.yaml.example` (adding only generic `policy` settings, not
demo GitHub/Jira profiles), plus user-facing command examples in `README.md` and `RELEASE.md`. Keep temporary demo
instructions isolated in `pocs/policy_demo/README.md` and [demo-plan.md](demo-plan.md).

## Complexity Tracking

No new constitutional exception. Reusing Vatuta's existing generic `MCPServer` container runner and isolating
all GitHub/Jira showcase configuration under `pocs/policy_demo/` keeps core `src/policy/` completely source-independent
while avoiding custom MCP wrappers, Docker-in-CI complexity, or throwaway demo UI code.
