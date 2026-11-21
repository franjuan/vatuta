# Implementation Plan: Policy Validation Engine (Schematic)

**Branch**: `002-policy-checker` | **Date**: 2026-11-21 | **Spec**: [spec.md](spec.md)
**Input**: Stateless MCP-ready `PolicyService`, CLI commands (`query`, `create`, `execute`, `check`), Agent Skill
(`skills/vatuta-policy/` per `agentskills.io`), pragmatic TDD & hermetic CI, and isolated blog demo ([demo-plan.md](demo-plan.md))
**Status**: Phase 0 research and Phase 1 design complete. Ready for `/speckit-tasks`.

## Summary (Schematic)

- **1. Stateless Core Service (`src/policy/service.py` — `PolicyService`)**:
  - Exposes 2 domain methods (`create`, `execute`) + trace reader (`read_trace`).
  - Uses JSON-serializable Pydantic v2 request/result models with `Field(description=...)` (1:1 ready for future MCP
    tools `policy_create`, `policy_execute`, and `resources/read`).
  - Manages zero `.yaml` files, directories, or globs. MCP server transport (`vatuta policy serve`) is deferred.
- **2. Step-by-Step Creation Pipeline (`PolicyService.create`)**:
  - **Step 0 (Pre-flight & Local Tool Catalog)**: Validate `policy_text`. Build tool catalog from local
    `policy.evidence_profiles` only (no MCP connection in Step 0 or Phase A; `trial_attempts == 0` never contacts MCP).
  - **Step 1 (Decomposition — `PolicyDecompositionSignature`)**: Split policy into atomic rules whenever conditions can
    fail independently or represent distinct violations (`NO_NORMATIVE_RULES` if zero). Assign deterministic slug `id`:
    `f"{policy_id}-{rule_id}"` (or `rule_id` if already prefixed).
  - **Step 2, Phase A (All-Rules Step 2.1 Sweep)**: Declare evidence and minimum necessary parameters (`parameters: {}`
    allowed) across `r_1..r_N` with incremental `SharedParameterCatalog`. Two rules share a parameter name only when
    `value_schema`, `construction_value`, and `default` match; shared entries are immutable in later rules and retries.
    When `trial_attempts > 0`, collect and return **all** `MISSING_CONSTRUCTION_CONTEXT` errors at once before starting
    any MCP server.
  - **Step 2, Phase B (Per-Rule Steps 2.2–2.5 with Fail-Fast)**:
    - **Step 2.2 (`trial_attempts > 0`)**: Connect to MCP, validate live `inputSchema`, fetch evidence once in memory,
      and run up to `policy.discovery_max_queries` (`5`) JSONata queries over `{"evidence", "parameters"}` (truncated to
      `policy.discovery_max_output_bytes`, `4096` bytes + `[TRUNCATED]`, early exit on `done=True`).
    - **Step 2.3 (`RuleSelectionSignature`)**: Build `selection` and mandatory `selection_tests` (`[obj]` normalization,
      non-whitespace `id`/`descriptive_id`, `DUPLICATE_SUBJECT_ID` check).
    - **Step 2.4 (`RuleCriterionSignature`)**: Build `criterion` (`deterministic` CEL or justified `llm`) and mandatory
      `criterion_tests` (CEL or `SemanticCriterionSignature` with `{{parameters.<name>}}` substitution).
    - **Step 2.5 (Retry & Fail-Fast)**: When `trial_attempts > 1` and Steps 2.2–2.4 fail, retry from Step 2.1 with full
      stage context and cached MCP evidence; abort immediately (Fail-Fast) if any rule exhausts its attempts.
- **3. Stateless Execution Pipeline (`PolicyService.execute`)**:
  - **Stage E0 (Whole-Request Pre-Flight)**: Validate non-empty unique `controls`, shared parameter compatibility,
    required parameters present (`MISSING_PARAMETER`, Exit `2`), and `value_schema` conformance
    (`INVALID_PARAMETER_VALUE`, Exit `2`). Allow extra keys in `request.parameters` and pass each control only its own
    declared parameters.
  - **Stages E1–E3 (Per-Control Isolation)**: Reuse started `MCPServer` containers and exact `(server, tool,
    resolved_arguments)` evidence cache across controls. Isolate tool and evidence validation errors per control.
    If a required MCP connection fails, disconnects, or times out, stop pending work and pass completed results
    and global `MCP_SERVER_UNAVAILABLE` to E4. During creation, stop before repair and save no generated YAML.
    Evaluate unique-ID valid subjects via CEL or singleton LLM while reporting
    `INVALID_SUBJECT` and `DUPLICATE_SUBJECT_ID` without criterion evaluation.
  - **Stage E4 & Traces**: Filter `subjects` by `result_filter` (`"violations"` vs `"all"`), keep all `errors`, include
    `source` and effective `parameters` in every `ControlResult`, and attempt to save the sanitized trace, including
    for inline execution and MCP outages. On write failure append `TRACE_WRITE_FAILED`, preserve results and
    existing errors, and exit `1`.
- **4. Client Directory Tree, CLI & Agent Skill (`https://agentskills.io/`)**:
  - **Hierarchical Folder Tree**: One `<id>.yaml` per rule under `policy.controls_directory` (`data/controls`,
    overridable via `--controls-dir` or `--output-dir`), supporting nested subfolders per user use case. Pre-check all
    targets on save and reject existing files with `CONTROL_FILE_EXISTS` (Exit `1`). If a later write fails,
    keep complete files and report the unsaved paths.
  - **Universal Rule Selection (`ClientControlQuery`)**: Select rules across subfolders by file (`--control` / `files`),
    exact or wildcard ID (`--id` / `ids`, e.g., `change-mgmt-*`), relative path or filename glob (`--glob` / `glob`,
    e.g., `team-a/**/*.yaml` or `*github*.yaml`), or metadata filters. Combine fields with **AND** and list items with
    **OR**. Apply path selectors before opening YAML. Invalid candidates fail with `INVALID_CONTROL_FILE` (Exit `2`).
    Duplicate IDs in the matched selection fail with `DUPLICATE_CONTROL_ID` (Exit `2`). Compute `merged_parameters`
    when `include_merged_parameters=True` (default).
    A required declaration without a default forces an explicit invocation value for the whole selection.
  - **CLI (`vatuta policy ...`)**: `query`, `create`, `execute`, and guided 2-step `check "<prompt>"`.
  - **Invocation identity**: Allocate one unique `invocation_id` per CLI command or Agent Skill script invocation
    before validating operation inputs. Pass the same ID through shared handlers and the service, including
    when a script delegates to the CLI. Direct service calls establish a context when called independently.
    Guided `check` shares its ID across rule selection, internal queries, parameter extraction, and execution.
    Standalone `query` records its ID in stderr diagnostics on success and failure and keeps its JSON response
    schema. The deferred MCP transport will allocate one ID per incoming operation call.
  - **Guided Check**: Require confirmation within the query budget. Verify each selected `(file, id)` pair against
    the query results, load only those files, and check their current IDs before merging parameters. Selection
    errors return empty results and a pre-execution trace before any evidence call.
  - **Agent Skill (`skills/vatuta-policy/`)**: Tracked skill package with `SKILL.md` and 3 thin Python scripts in
    `skills/vatuta-policy/scripts/` (`query.py`, `create.py`, `execute.py`) containing **PEP 723 inline script
    metadata** and sharing the exact client handlers with `vatuta policy query|create|execute --format json`.

### LLM input boundaries

- Keep engine-authored DSPy instructions separate from user and external data fields in provider requests.
- Validate MCP responses as JSON and apply tool-specific validation and completeness checks. Strip scripts
  from external HTML in the copy projected for the model.
- Verify generated policy excerpts against their source. Generate and test `subject_paths` for each semantic
  control. Guided checks use bounded candidate metadata and confirmed parameter declarations.
- Use the same semantic projection for synthetic tests, live trials, and execution. Mark the subject `ERROR`
  before the LLM call when a declared path is absent. Record `MISSING_SUBJECT_PATH` at stage `criterion` with
  the subject ID and missing pointer in `ControlResult.errors`.
- Save the creation trace before releasing controls to the caller or YAML writer. On failure retain earlier
  errors and the invocation ID, return `controls: []`, and report `TRACE_WRITE_FAILED`.

## Technical Context (Schematic)

Query and execution use the same shared parameter resolver. Creation and execution use the same violation message
renderer. The CLI and Agent Skill share the YAML writer.

The Typer root callback skips eager provider validation for the `policy` group. Shared client handlers initialize
an LLM when an operation needs one. Guide and script runs use the Poetry environment with Vatuta installed.
Guided `check` passes a per-call execution context to the service so its pre-execution steps and the core share
one invocation ID.

- **Language/Version**: Python `>=3.12,<3.14` (tested on Python 3.12 and 3.13 via Poetry).
- **Primary Dependencies**:
  - Existing: Pydantic v2, PyYAML, Typer, Rich, DSPy, LiteLLM, MCP Python SDK, Docker SDK, Prometheus.
  - Added during implementation: `jsonata-python==0.7.1`, `cel-python==0.5.0`, plus direct declarations for
    `jsonschema` and `google-re2`.
- **Storage**:
  - Client-owned `<id>.yaml` files in nested subfolders under `policy.controls_directory` (`data/controls`).
  - Sanitized JSON stage traces under `policy.trace_directory` (`logs/policy`).
  - Tracked Agent Skill package under `skills/vatuta-policy/`.
- **Testing**: Pragmatic TDD with `pytest` in GitHub Actions (unit tests primary, lightweight contract tests, minimal
  in-process CLI/Skill wiring tests using `tmp_path`, zero Docker/database/SaaS secrets in CI).
- **Target Platform**: Linux CLI (`vatuta policy ...`) and any `agentskills.io`-compatible AI coding agent
  (`skills/vatuta-policy/`).
- **Project Type**: Python application with stateless service API (`PolicyService`), CLI, and Agent Skill.
- **Performance Goals**: No arbitrary latency targets invented; record stage durations and token usage.
- **Constraints**:
  - 100% source-independent core (`src/policy/`); GitHub/Jira MCP profiles live only in `pocs/policy_demo/`.
  - Bounded LLM exploration/filtering (`discovery_max_queries=5`, `check_max_filter_queries=3`, `4096` byte truncation).
  - Hermetic CI with zero Docker containers or live SaaS calls; credentials only in `.env`.
- **Scale/Scope**: Single-invocation inline control execution; isolated blog demo in `pocs/policy_demo/` (4 rules).

## Constitution Check

Evaluated against [constitution.md](../../.specify/memory/constitution.md) (`v2.0.0`).

The gate reports **PASS**. Redact credentials and tokens in diagnostics. Ordinary application logs contain
stage and status information.
The post-design check also passes. Path scoping and pair validation share the local loader and preserve the inline
execution contract. The validation plan covers confirmation and selection errors with local fakes.

| Gate | Pre-research | Post-design Evidence |
| --- | --- | --- |
| I–VIII: Simplicity & scope control | PASS | Stateless 2-operation `PolicyService`; 3 thin PEP 723 skill scripts share CLI handlers with zero duplication; `policy serve` deferred. |
| IX/X: Poetry, Python 3.12/3.13, static quality gates | PASS | All commands via `poetry run`; strict mypy, Ruff, Black, Xenon, Pydocstyle, Markdownlint (`<=120` chars), Bandit, Semgrep. |
| XI/XII: Atomic DSPy prompts & deterministic core | PASS | Atomic DSPy signatures per step; deterministic `ClientControlQuery`, `merged_parameters`, unit-test runner, and CEL. |
| XIII: Provenance & evidence-logic separation | PASS | Self-contained `ControlResult` (`control_id`, `source`, effective `parameters`), `test_cases`, and sanitized stage traces. |
| XIV/XV: Zero-Trust MCP isolation & read-only tools | PASS | Reuse `MCPServer`. Validate the configured container and read-only tool profile before launch. |
| XVI: Public CLI/config compatibility & `pocs/` isolation | PASS | Adds `vatuta policy` (`query`, `create`, `execute`, `check`) and `skills/vatuta-policy/`; isolates demo in `pocs/policy_demo/`. |
| XVII: Pragmatic TDD, contract tests & hermetic CI | PASS | Unit-first TDD + lightweight contract tests in CI; minimal integration scope; zero Docker/SaaS dependencies in CI. |
| XVIII/XIX: Ecosystem reuse & CLI/IDE agnosticism | PASS | Adopts `agentskills.io` + PEP 723 standards in tracked `skills/vatuta-policy/` while keeping `.agents/`/`.cursor/` untracked. |
| Trust boundaries | PASS | Validate external data, separate DSPy instructions, and project inputs per stage. |
| Observability | PASS | Correlate diagnostics by invocation and redact secrets from traces. |
| Graceful degradation | PASS | Return a global error on required MCP outages and preserve completed results. |
| XX: Simple English & complete contracts | PASS | The plan and contracts record the agreed defaults, retry, save, and message behavior. |

## Project Structure

### 1. Planning & Specification Artifacts (`specs/002-policy-checker/`)

```text
specs/002-policy-checker/
├── spec.md                # Functional specification (User Stories 1–5, FR-001..FR-051, SC-001..SC-011)
├── plan.md                # Schematic implementation, testing, validation, and demo boundary plan
├── research.md            # Technical decisions R1–R10 and constitution check
├── data-model.md          # Pydantic v2 models, embedded test_cases, ClientControlQuery, config, and transitions
├── design-inputs.md       # Resolved design register and implementation qualification checks
├── quickstart.md          # Local E2E acceptance guide for the CLI, Agent Skill, and guided check
├── demo-plan.md           # Temporary GitHub Free + Jira Cloud Free blog demonstration design
├── contracts/
│   ├── operations.md      # Stateless PolicyService, ClientControlQuery, CLI, Agent Skill, and trace URI contracts
│   ├── control-format.md  # YAML ControlDefinition, expression contexts, and completeness contracts
│   └── reference-control.yaml
└── checklists/
    └── requirements.md
```

### 2. Permanent Production Code, Agent Skill & Automated Tests (`src/`, `skills/`, `tests/`)

```text
src/policy/
├── service.py             # Stateless PolicyService (create, execute, read_trace) — provider-neutral
├── creation.py            # Step-by-step creation, SharedParameterCatalog, JSONata discovery, unit tests
├── execution.py           # Independent control/subject processing and result filtering
├── files.py               # Safe YAML I/O, nested directory scan, containment, wildcard ID/glob filtering, projection
├── parameters.py          # Shared parameter merging (include_merged_parameters), defaults, conflicts
├── agent.py               # Guided 2-step check pipeline (Step 1 rule selection, Step 2 param extraction)
├── evidence.py            # Generic MCPServer caller, payload normalization, completeness-profile check
├── expressions.py         # JSONata/CEL boundary adapters, output truncation, deterministic guards
├── semantic.py            # Singleton DSPy semantic judgment and strict correlation-token verification
├── trace.py               # Sanitized stage artifacts, URI resolution, and metric integration
├── config.py              # PolicyConfig (controls_directory, query/byte limits, profiles, trace_directory)
├── signatures/            # One atomic DSPy Signature class per file
└── models/                # One JSON-serializable Pydantic v2 model class per file
src/client/policy.py       # Shared handlers + Typer group (query, create, execute, check), JSON/Rich output
src/client/client.py       # Register policy group; skip eager LLM preflight only for policy commands
src/models/config.py       # Add optional PolicyConfig to VatutaConfig
src/llm/provider.py        # Selected-backend validation with existing default preserved
src/metrics/policy_metrics.py

skills/vatuta-policy/      # Tracked Agent Skill package (https://agentskills.io/)
├── SKILL.md               # YAML frontmatter (name: vatuta-policy, description) + schematic agent workflow guide
└── scripts/               # Thin PEP 723 Python entrypoints sharing src/client/policy.py handlers
    ├── query.py           # Search YAML rules in nested folders, project fields, and get merged_parameters
    ├── create.py          # Create rules from a policy and save <id>.yaml files in target directory
    └── execute.py         # Select YAML rules (by file, wildcard ID, glob, or filter) and execute with parameters

tests/policy/              # Permanent unit, lightweight contract, and skill/CLI wiring tests (CI)
├── test_expressions.py    # JSONata envelope, builtin guards, truncation, CEL strict bool/error unit tests
├── test_parameters.py     # Shared parameters, defaults vs construction_value, conflict/missing unit tests
├── test_files.py          # Safe YAML load/save, nested folders, wildcard IDs, globs, DUPLICATE_CONTROL_ID tests
├── test_evidence.py       # MCP payload conversion and EvidenceProfile completeness unit tests
├── test_creation.py       # Step-by-step decomposition, catalog reuse, discovery budget, test_cases, trials
├── test_execution.py      # Control/subject evaluation, PRs 101/102/103, filters, isolated control errors
├── test_semantic.py       # Semantic mode: llm correlation, parameter templates, error-isolation unit tests
├── test_agent.py          # Guided 2-step policy check selection and parameter extraction unit tests
├── test_trace.py          # Sanitization, redaction, and PolicyService.read_trace unit tests
├── test_contracts.py      # Self-contained Pydantic/MCP JSON Schema, reference YAML & boundary contracts
└── test_skill.py          # Agent Skill SKILL.md frontmatter, PEP 723 metadata, and 3-script parity tests
tests/client/test_policy.py # CLI query/create/execute/check wiring, exit codes (0/1/2), stdout/stderr separation
tests/fixtures/policy/     # Deterministic provider-neutral JSON/YAML test fixtures (no live SaaS data)
```

### 3. Temporary Blog Demonstration Assets (`pocs/policy_demo/` — Isolated Non-Production Scope)

```text
pocs/policy_demo/          # Excluded from production package, static gates, and GitHub Actions CI
├── README.md              # Minimal setup checklist (.env vars, digest pinning, screenshot commands)
├── policy.md              # Example Change Management & Release Traceability Policy (4 rules)
├── vatuta.demo.yaml       # Standalone demo MCP config (github-mcp-server, mcp-atlassian) & evidence_profiles
└── controls/              # Generated <id>.yaml controls produced locally during the demo run
```

---

## Testing, Validation & Demonstration Strategy (Schematic)

| Layer | Location | Purpose | External Dependencies | Runs in CI? |
| --- | --- | --- | --- | --- |
| **1. Permanent Automated Tests** | `tests/policy/`, `tests/client/test_policy.py` | Unit, contract, CLI, and Agent Skill verification | None (MCP & LLM mocked/faked) | **Yes** (Python 3.12 & 3.13) |
| **2. Quickstart Validation** | [quickstart.md](quickstart.md) | CLI wiring, missing inputs, and guided selection | Local fakes | Yes |
| **3. Temporary Blog Demo** | [demo-plan.md](demo-plan.md), `pocs/policy_demo/` | Real GitHub Free + Jira Cloud Free compliance showcase | Local Docker + `.env` tokens | **Never** |

### Key Unit & Contract Test Focus Areas

- **Shared parameters**: Check query and execution agree for mixed defaults, optional inputs, explicit `null`,
  explicit overrides, and conflicts. Verify that an absent required input produces zero external calls.
- **File persistence**: Inject a failure after one successful write. Keep that file, list unsaved paths, and return
  exit `1`. Check the same result through the CLI and Agent Skill.
- **Retries and messages**: Exercise the synthetic failure path at trial settings `0`, `1`, and `>1`. Verify
  message rendering with valid fields, escaped braces, explicit `null`, and subject-level error isolation.
- **LLM input tests**: Inspect provider requests for separate instructions and data. Check that scripts in
  external HTML stay out of prompts, plain text remains data, and unused subject fields are excluded. Verify
  that an invalid policy excerpt fails validation and an absent projected value stops the subject before the call.
  Assert both result filters retain the same `MISSING_SUBJECT_PATH` diagnostic with the subject ID and pointer.
- **Creation trace tests**: Force trace writes to fail. Verify the same invocation ID, empty controls, preserved
  errors, exit code `1`, and an untouched YAML output directory.
- **MCP failures**: Test connection failure, disconnect, and timeout. Verify partial results, no subsequent calls,
  and trace finalization. Verify that controls continue after a tool error from a responding server.
- **MCP startup**: Test digest pinning, tool allowlisting, non-root execution, default-deny network policy,
  and safe subprocess diagnostics.
- **Trace paths**: Test JSON Pointer escaping, explicit array indices, and absent targets. Verify that
  `trace_redact_paths` overrides `trace_safe_text_paths` on overlap and that credentials and tokens stay redacted.
- **Traces and logs**: Verify independent invocations receive distinct IDs, including successive standalone queries.
  Check that query diagnostics carry the ID on success and validation failure, and that stdout matches the query
  response schema. Verify that a script and its delegated handler share one ID and that guided calls reuse their ID
  for internal queries and service stages. Test trace URI containment and credential redaction. Verify that ordinary
  logs contain stage and status information, including for MCP subprocess errors, without copying raw payloads.
- **Lazy startup**: Check that policy query and deterministic execution run without LLM credentials, and existing
  CLI groups still use their normal provider preflight.

- **Expressions (`test_expressions.py`)**: JSONata existence envelope (`missing` vs `null` vs `[]`), disabled builtins,
  4096-byte truncation (`[TRUNCATED]`), `google-re2` guard, CEL strict booleans.
- **Files & Selection (`test_files.py`)**: Nested folder scan under `controls_directory`, root containment, `--control`
  (`files`), exact + wildcard `--id` (`ids`), relative path + bare filename `--glob` (`glob`), `INVALID_CONTROL_FILE`,
  `DUPLICATE_CONTROL_ID` on matched selection, `CONTROL_FILE_EXISTS` pre-check on save, and `merged_parameters`.
  Verify file and glob scopes exclude unrelated invalid YAML before reading, including when combined. With only an
  ID or metadata filter, malformed YAML remains an error across the candidate tree.
- **Guided Check (`test_agent.py`)**: Cover confirmation on the last allowed query, exhaustion with provisional
  choices, a confirmed empty result, a pair not shown to the model, and a changed ID on reload. Verify that one of
  two same-ID files can run by path and that selecting both fails. Assert zero evidence calls on failure.
- **Creation & Execution (`test_creation.py`, `test_execution.py`, `test_semantic.py`)**: Local catalog in Step 0 &
  Phase A, immutable `SharedParameterCatalog`, `MISSING_CONSTRUCTION_CONTEXT` aggregation, mandatory `selection_tests`
  & `criterion_tests`, `Stage E0` pre-flight vs `Stages E1–E3` per-control isolation, MCP container reuse, exact
  evidence cache, `DUPLICATE_SUBJECT_ID`, singleton `llm` with `{{parameters.<name>}}`, self-contained `ControlResult`,
  and `TRACE_WRITE_FAILED` result preservation.
- **CLI & Agent Skill (`tests/client/test_policy.py`, `tests/policy/test_skill.py`)**:
  - Validate `skills/vatuta-policy/SKILL.md` frontmatter (`name: vatuta-policy`, `description`) and markdownlint compliance.
  - Validate PEP 723 `# /// script ... # ///` blocks on `skills/vatuta-policy/scripts/{query,create,execute}.py`.
  - Verify 100% output, exit code (`0`, `1`, `2`), and `stdout`/`stderr` parity between `vatuta policy
    query|create|execute --format json` and `skills/vatuta-policy/scripts/{query,create,execute}.py`.

### Task Organization Rules for `/speckit-tasks`

1. **Permanent Production Phases (Phases 1–8)**:
   - Cover `Setup`, `Foundational`, `US1` (Create), `US2` (Search, Merged Parameters & Check), `US3` (Execute), `US4`
     (Traces), `US5` (Agent Skill `skills/vatuta-policy/` & PEP 723 Scripts), and `Polish & Quickstart Validation`
     across `src/`, `skills/vatuta-policy/`, `tests/`, `config/vatuta.yaml.example`, and `docs/`.
   - Follow Pragmatic TDD (unit/contract test tasks before implementation tasks in every phase).
   - Include **zero** references to `pocs/policy_demo/` or live GitHub/Jira accounts.
2. **Temporary Blog Demonstration (Phase 9, Optional and Local Only)**:
   - Group all tasks from [demo-plan.md](demo-plan.md) into a final optional phase targeting **only
     `pocs/policy_demo/`**.

## Documentation Updates

During implementation, update:

- `docs/policy-checker/design.md`, `docs/policy-checker/glossary.md`, and
  `docs/policy-checker/adr-019-file-based-policy-operations.md` (documenting `vatuta policy query`, the Agent Skill in
  `skills/vatuta-policy/`, wildcard `--id`, and nested folder selection in simple, schematic English).
- `config/vatuta.yaml.example` (generic `policy` section).
- `README.md` and `RELEASE.md` (CLI and Agent Skill usage examples).
