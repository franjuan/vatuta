# Policy Validation Engine: Design Inputs Register

**Updated**: 2026-11-11
**Specification**: [spec.md](spec.md)
**Plan**: [plan.md](plan.md) | **Demo Plan**: [demo-plan.md](demo-plan.md)
**Research**: [research.md](research.md)
**Scope decision**: [ADR-019](../../docs/policy-checker/adr-019-file-based-policy-operations.md)

## Role and Status

The 2026-11-04 through 2026-11-11 scope, UX, creation, and execution refinements are authoritative. Phase 0 research and
Phase 1 design are complete; this register records the decisions and separates implementation qualification from
stakeholder ambiguities. It is not a second specification. No runtime feature, dependency installation, or live-source
certification has been completed.

## Resolved Inputs

| Earlier investigation | Resolution | Reference |
| --- | --- | --- |
| Stateless service and MCP boundary | `PolicyService` exposes `create`, `execute`, and `read_trace` over JSON-serializable Pydantic models (`policy_parameters` and `policy serve` excluded). | Research R1 |
| Deterministic startup | Client filtering and deterministic execution bypass unrelated eager LLM preflight; selected backends initialize only when needed. | Research R1, R4 |
| Step-by-step creation LLM flow | Local catalog through Phase A for every trial mode; live MCP discovery only after Phase A succeeds with enabled trials. Shared declarations remain immutable through Phase B and retries. | Research R4, R8 |
| Embedded unit tests (`test_cases`) | Mandatory `selection_tests` (`[obj]` normalization, `DUPLICATE_SUBJECT_ID` check) and `criterion_tests` (CEL or LLM with `{{parameters.<name>}}` template rendering) with `test_case.parameters -> default -> construction_value` priority. | Research R4; control-format contract |
| Control, tags, and parameter models | Pydantic `ControlDefinition` with `tags: list[str]` and `test_cases: ControlTestCases`, parameter minimization (`parameters: {}` allowed), extra keys in `request.parameters` allowed, per-control parameter scoping. | Research R5; data model |
| Client filtering & projection tool | Client-side tool over `policy.controls_directory`: **AND** across fields, **OR** within list fields, fail immediately on malformed YAML (`INVALID_CONTROL_FILE`), `projection`, and `include_merged_parameters=True` (default). | Operation contracts |
| Guided 2-step check LLM flow | Iterative Step 1 (`done=True`, `NO_CONTROLS_MATCHED` on empty), deterministic `merged_parameters` load, Step 2 skipped when `merged_parameters == []`, and unified `invocation_id` trace across Steps 1, 2, and `execute`. | Research R4; operation contracts |
| JSONata missing/null & subject IDs | Existence envelope; non-whitespace `id`/`descriptive_id`; unique `id`s evaluate normally while colliding `id`s emit `DUPLICATE_SUBJECT_ID` without criterion evaluation and set control `outcome="ERROR"`. | Research R2; control-format contract |
| Deterministic language execution | JSONata function guards and CEL strict booleans over explicit inputs. | Research R2, R3 |
| Semantic evaluation | Singleton (1 subject per call) DSPy signature with correlation token; no separate `parameters` dict sent to LLM, but `condition` supports `{{parameters.<name>}}` template substitution. | Research R4 |
| Evidence acquisition & caching | Reuse started `MCPServer` containers across controls in one `execute` call, cache valid responses in memory by exact `(server, tool, resolved_arguments)`, and fast-fail a control on its first evidence failure. | Research R6 |
| Pre-flight vs per-control boundary | `Stage E0` validates only unique `controls` and declared parameter schemas/presence; `evidence_profiles`, allowlists, `arguments_schema`, `inputSchema`, `complete_when`, and expressions are isolated per control in `Stages E1–E3`. | Operation contracts |
| Reference source candidates | Core engine stays 100% source-independent; GitHub/Jira MCP profiles live only in isolated demo YAML (`pocs/policy_demo/`). | Control-format contract; [demo-plan.md](demo-plan.md) |
| Testing, validation & demo split | Pragmatic TDD (unit + lightweight contract tests in CI, minimal integration, zero Docker/SaaS in CI), minimal 2-scenario local `quickstart.md`, and isolated blog demo under `pocs/policy_demo/`. | [plan.md](plan.md), [quickstart.md](quickstart.md), [demo-plan.md](demo-plan.md) |
| Runtime errors and outputs | Per-control errors by `control_id`; `result_filter="violations"` retains only `NON_COMPLIANT` in `subjects` while `errors` retains all control/subject errors under both filters. | Operation contracts |
| Traces and privacy | Private sanitized stage artifacts, `PolicyService.read_trace(uri)`, existing logging/metrics, no expiry or lifecycle history. | Research R7 |
| Creation trials | Existing bounded attempts and complete-or-error behavior; empty error-free live trial passes with visible unexecuted live criterion while `criterion_tests` still verify the criterion. | Spec FR-026, FR-046 |

### Confirmed Review Resolutions (2026-11-11)

- Reject existing target control files with `CONTROL_FILE_EXISTS`, preserving existing contents and checking the
  complete target set before publishing. No overwrite option or automatic renaming.
- Include `source` and effective per-control `parameters` directly in every `ControlResult`, as well as the trace.
- Preserve calculated execution results on trace-write failure, adding global `TRACE_WRITE_FAILED` and exit `1`.
- Allocate the check/execution ID in an internal invocation context before check Step 1. Preserve its error envelope
  and attempt trace persistence even when the check fails before execution; public request schemas stay unchanged.
- Provide a complete standalone demo configuration, including required general sections and explicit LLM bindings,
  without relying on personal settings or configuration merging. Credentials stay in `.env`.
- Correct the Jira completeness proof so a short response never overrides a contradictory known total. Align the two
  quickstart cases with SDK-shaped discovery/evidence fakes, both result filters, and explicit zero-call assertions.
- Retire active batching and obsolete service-interface descriptions; keep explicitly superseded ADR history intact.

These decisions are reflected in Research R8, the operation contract, the data model, and ADR-019.

## Implementation Qualification

These checks are required before claiming working implementations, not unresolved functional choices:

- Install/lock the selected libraries through Poetry; satisfy release-age, license, dependency audit, and Python
  3.12/3.13 checks. Verify JSONata envelope behavior, disabled builtins including aliases, regex/time guards,
  and CEL boolean/null/error conversion against the actual pinned releases.
- Verify the generated Pydantic JSON schemas (`model_json_schema()`), policy decomposition, incremental
  `SharedParameterCatalog`, bounded JSONata exploration (`policy.discovery_max_queries`,
  `policy.discovery_max_output_bytes`), mandatory `selection_tests` and `criterion_tests`, `tags` inference and
  merging, typed semantic outputs, absent-versus-null defaults, whole-argument binding, client directory containment,
  and stdout/stderr separation.
- Test complete, incomplete, and unknown evidence (including empty responses) in fast, self-contained unit and contract
  tests under `tests/policy/` using provider-neutral fixtures—requiring zero Docker containers, databases, or external
  API credentials in GitHub Actions CI.
- Validate read-only credentials, explicit source allowlists, digest-pinned images, sanitization, private trace files,
  bounded calls, and failure isolation using existing infrastructure. Never use logs as a substitute for policy history.
- Run the minimal 2-scenario local [quickstart](quickstart.md) acceptance check after core implementation. Keep the
  temporary GitHub Free + Jira Cloud Free blog showcase strictly isolated under `pocs/policy_demo/` per
  [demo-plan.md](demo-plan.md) and out of CI.

## Exclusions and Next Phase

No policy registry, server-side control repository, revisions, hashes for comparison, managed review/approval,
publication, maintenance, identity reconciliation, historical retrieval/restoration, or trace-retention scheduler is
planned. `vatuta policy serve` and the `policy_parameters` MCP tool are excluded from this increment. No dependency
graph, automatic prerequisites, evidence pagination, provider-specific executor adapters, Docker-in-CI test harnesses,
custom demo UI, runtime model planning, new retry queue, cross-invocation cache, parallel executor, or violation
explanation
is added.

The next command is `/speckit-tasks`. It must derive permanent production tasks (`src/`, `tests/`, `docs/`) separately
from the optional final temporary demonstration phase (`pocs/policy_demo/`) per [plan.md](plan.md) and
[demo-plan.md](demo-plan.md).
