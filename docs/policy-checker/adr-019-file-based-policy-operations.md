# ADR-019: Stateless Policy Service, Step-by-Step Creation & Check LLM Flows

**Date**: 2026-10-25
**Updated**: 2026-11-11
**Status**: Accepted — stakeholder scope, UX, and creation/execution flows refinement
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Context

The earlier specification included review, publication, stable identity reconciliation, immutable revisions,
comparison, restoration, and configurable diagnostic expiry. Subsequent UX and LLM-flow reviews on 2026-11-04
decoupled the core policy service (the future MCP interface) from any file system or control repository, moved local
YAML file storage, filtering, projection, and parameter merging to the Vatuta client, and established structured,
scalable LLM flows for control creation and prompt-based policy checking.

## Decision

Expose a stateless `PolicyService` class using JSON-serializable Pydantic v2 models that map 1:1 to two MCP domain
tools (`policy_create` and `policy_execute`) plus trace resource reads (`read_trace`), while the Vatuta client manages
local YAML control files, a client-side filtering/projection tool, direct CLI subcommands, and a guided 2-step
prompt-driven `vatuta policy check "<prompt>"` pipeline.

- **Stateless `PolicyService` (MCP-ready boundary, no `policy serve` in this increment)**:
  - The Vatuta client invokes `PolicyService` directly; wrapping it with `FastMCP` (`vatuta policy serve`) is deferred
    to the next increment.
  - `PolicyService` manages no control directory, `--root`, file selectors, or rule repository.
  - `PolicyService.create` (`policy_create`) accepts required `policy_text`, optional provenance identifier
    `policy_id`, optional descriptive `construction_context`, optional `tags: list[str]`, and optional nonnegative
    `trial_attempts`. The CLI can read a UTF-8 plain-text/Markdown file and supply its contents through the same
    request.
  - `ControlDefinition` includes `tags: list[str]` and `test_cases: ControlTestCases` (`selection_tests` and
    `criterion_tests`).
  - `PolicyService.create` returns `{invocation_id, controls: list[ControlDefinition], errors}` with structured
    `ControlDefinition` objects directly. The client may serialize each control to `<id>.yaml` in
    `policy.controls_directory` (or `--output-dir`); managing that directory is not the service's responsibility.
    Saving rejects existing target files with `CONTROL_FILE_EXISTS`/exit `1`, preserving existing contents. Check all
    destinations before publishing; no overwrite option or automatic renaming is introduced.
  - `PolicyService.execute` (`policy_execute`) accepts `{controls: list[ControlDefinition], parameters, result_filter}`
    with inline structured `ControlDefinition` objects (requiring unique `id` values within the invocation) and returns
    `{invocation_id, results: list[ControlResult], errors}`. `ControlResult` and `ErrorDetail` identify controls by
    `control_id` without `control_file`. Every `ControlResult` also includes `source` and its effective resolved
    `parameters`, including error/inapplicable results under both filters. Trace-only attribution is insufficient.
  - Remove `policy_parameters` as a service/MCP operation.
- **Step-by-Step Creation LLM Flow & Embedded Unit Tests**:
  - **Step 0 (Pre-flight & Tool Catalog)**: Always use local `policy.evidence_profiles` (`arguments_schema` and
    `contract_reference`). No MCP startup, connection, or tool discovery occurs before Phase A succeeds. Zero attempts
    never contacts MCP; enabled trials discover and validate live `inputSchema` in Phase B before evidence calls.
  - **Step 1 (Policy Decomposition)**: Decompose `policy_text` into atomic normative rules (failing with
    `NO_NORMATIVE_RULES` when zero normative rules are extracted). Split conditions into separate rules whenever they
    can fail independently and a user wants to know which condition failed, or whenever the policy author would consider
    them distinct violations. Assign each control a deterministic kebab-case `id` (`f"{policy_id}-{rule_id}"`, or
    `rule_id` if already prefixed with `f"{policy_id}-"`).
  - **Step 2, Phase A (All-Rules Step 2.1 Sweep & `SharedParameterCatalog`)**: Run Step 2.1 across all rules first,
    passing the cumulative `SharedParameterCatalog`. Two parameters share a name only when `value_schema`,
    `construction_value`, and `default` match; otherwise the constructor adapts to the catalog or uses a distinct name.
    Declarations already referenced by other rules remain immutable through later stages and retries. Keep candidate
    additions local until validated and discard failed-attempt changes; never invalidate another rule's declaration.
    When `trial_attempts == 0`, `construction_value` is optional. When `trial_attempts > 0`, aggregate and return all
    `MISSING_CONSTRUCTION_CONTEXT` errors and affected `control_id`s across all rules at once before calling any MCP
    server.
  - **Step 2, Phase B (Per-Rule Steps 2.2–2.5 with Fail-Fast)**:
    1. **Scalable Source Exploration (Step 2.2, when `trial_attempts > 0`)**: Fetch declared evidence once in memory
       (cached per rule by `(server, tool, resolved_arguments)`) and allow up to `policy.discovery_max_queries`
       (default `5`) exploratory JSONata queries over `{"evidence": ..., "parameters": ...}`, truncating outputs to
       `policy.discovery_max_output_bytes` (default `4096` bytes + `[TRUNCATED]`), returning query errors or `missing`
       outcomes as observations, and closing the loop early when `done=True`.
    2. **Selection & `selection_tests` (Step 2.3)**: Can declare additional parameters; generates `selection` and
       `selection_tests` (normalizing single-object JSONata output to `[obj]`, requiring non-whitespace
       `id`/`descriptive_id`, rejecting duplicate subject `id`s with `DUPLICATE_SUBJECT_ID`, and resolving test
       parameters with `1. test_case.parameters` -> `2. default` -> `3. construction_value` priority); validates against
       `selection_tests` (mandatory) and live evidence (when `trial_attempts > 0`).
    3. **Criterion & `criterion_tests` (Step 2.4)**: Can declare additional parameters; generates `criterion` and
       `criterion_tests`; validates against `criterion_tests` (mandatory, using CEL for `deterministic` and singleton
       `SemanticCriterionSignature` LLM calls with `{{parameters.<name>}}` template substitution in `condition` for
       `mode: llm`) and live trial subjects (when `trial_attempts > 0`).
    4. **Full-Context Retry & Fail-Fast (Step 2.5)**: When `trial_attempts > 1` and a rule fails in Steps 2.2–2.4,
       re-executes from Step 2.1 carrying the full accumulated context and errors from all prior stages, reusing cached
       in-memory MCP evidence whenever `evidence` bindings are unchanged. If any rule fails definitively after its
       attempts, Phase B aborts immediately (**Fail-Fast**). Persist `test_cases` inside `ControlDefinition`.
- **Client-Side Local Controls, Filtering/Projection Tool, Guided 2-Step Check Flow, and Execution Semantics**:
  - The client stores and reads YAML control files in configurable `policy.controls_directory` (default
    `data/controls`, overridable via `--controls-dir`).
  - The client provides a local filtering and projection function/tool (`ClientControlQuery`) supporting criteria
    (`ids`, `policy_ids`, `rule_ids`, `tags`, `evaluation_mode`, `evidence_server`, `evidence_tool`, `parameter_names`,
    `text_query`, `files`, `glob`, combining distinct filter fields with logical **AND** and multi-value list fields
    with logical **OR**), failing immediately with `INVALID_CONTROL_FILE` if any encountered YAML control file is
    malformed, projecting requested attributes (`projection`, e.g., `["id", "description", "tags", "source"]` or
    `["full"]`), and providing a configurable `include_merged_parameters: bool = True` parameter (enabled by default).
  - In `vatuta policy check "<prompt>"`, the client executes a **guided 2-step LLM pipeline** associated with the same
    execution `invocation_id` trace (`vatuta://traces/policy/execution/{invocation_id}`):
    An internal context allocates this ID before Step 1 and is shared with execution without new public request fields.
    A failure before execution returns `{invocation_id, results: [], errors}` and attempts to retain the same trace.
    1. **Iterative Rule Selection**: The LLM queries `ClientControlQuery` with lightweight projection (`["id",
       "description", "tags", "source"]`, `include_merged_parameters=False`, bounded by configurable
       `policy.check_max_filter_queries`, default `3`, and `policy.check_max_output_bytes`, default `4096` bytes,
       exiting early when `done=True`) to choose applicable `control_id`s (failing with `NO_CONTROLS_MATCHED` if empty);
       the client then deterministically loads those controls with `projection=["full"]` and
       `include_merged_parameters=True`.
    2. **Parameter Deduction**: Skipped when `merged_parameters` is empty (`parameters={}`); otherwise a dedicated DSPy
       signature receives the user's prompt and `merged_parameters` and extracts explicit parameter values from the
       prompt before invoking `PolicyService.execute` under the same `invocation_id`.
  - During `PolicyService.execute`:
    - `Stage E0` (pre-flight) validates nonempty unique `controls` and declared shared parameters (`value_schema` and
      required parameter presence), allowing extra unused keys in `request.parameters` and passing each control only its
      own declared parameters.
    - `Stages E1–E4` (per control) keep started `MCPServer` containers alive across controls, cache valid evidence in
      memory by exact `(server, tool, resolved_arguments)`, fast-fail a control on its first failed evidence item,
      evaluate unique-ID valid subjects while reporting duplicate `id`s as `DUPLICATE_SUBJECT_ID` without criterion
      evaluation (`outcome="ERROR"`), evaluate `mode: llm` 1 subject per call with `{{parameters.<name>}}` template
      substitution in `condition`, and retain only `NON_COMPLIANT` subjects in `ControlResult.subjects` when
      `result_filter="violations"` while preserving all errors in `ControlResult.errors`.
- **Diagnostics and Trace Resource URIs**:
  - Preserve diagnostic traces in the log directory, including model interactions and required redaction. Remove
    retention configuration and automatic age-based cleanup. Traces remain until explicitly deleted and do not create
    a policy history service.
  - Clients derive trace resource URIs from the invoked operation and returned `invocation_id`, using the templates in
    [the operation contract](../../specs/002-policy-checker/contracts/operations.md#diagnostic-trace-resources).
  - On execution trace-write failure, preserve calculated results, append global `TRACE_WRITE_FAILED` at stage `trace`,
    and return CLI exit `1`. This diagnostic error does not change subject states or control outcomes.
- **Standalone Optional Demo**:
  - Supply a complete configuration template with required general sections, explicit LLM bindings, MCP profiles,
    and control/trace paths. Reproduction does not depend on a personal configuration or implicit YAML merging.
    Document placeholders and keep credentials in `.env`; demo execution remains outside production and CI.

## Retained Contracts

The independent-evidence, response-conversion, completeness, subject-selection, boolean-result, filtering, and
singleton semantic-evaluation contracts remain. Technical failures are errors, never automatic violations. Results
remain per
subject and per control, without an overall policy compliance percentage.

Configured source permissions, read-only external access, privacy, and provenance remain required. Execution records
which control definitions and inputs were used without requiring a policy revision or cross-generation identity.

## Supersession

This decision and the current specification take precedence over incompatible earlier text.

| Record | Current applicability |
| ------ | --------------------- |
| ADR-001, ADR-002, ADR-009 | Keep trials and complete-or-error behavior; add mandatory synthetic unit tests; remove publication/approval. |
| ADR-007 | Superseded: retain traces, with no retention configuration or automatic expiry. |
| ADR-008 | Keep UTF-8 text/Markdown support; inline policy text is also an accepted input. |
| ADR-010 | Versioning, document comparison hashes, and history are outside current scope. |
| ADR-011 | Keep creation/execution separation; remove maintenance and model-free-only execution. |
| ADR-012 | Keep YAML, MCP evidence, JSONata, and deterministic CEL; allow declared LLM criteria under ADR-018. |
| ADR-013, ADR-017 | Keep evidence and error contracts; remove the unavailable-source invocation-abort exception. |
| ADR-014 | Subject-selection and result contracts remain applicable. |
| ADR-015 | Keep policy authority and complete creation; remove managed regeneration, review, and comparison. |
| ADR-016 | Superseded: no immutable revision, historical retrieval, comparison, or restoration workflow. |
| ADR-018 | Keep declared semantic evaluation; singleton calls replace batching; lifecycle/source-abort text is historical. |

Earlier interview answers are decision history, not a second current specification. Retired requirements FR-028
through FR-031 and FR-033 must not become implementation tasks. FR-016 and FR-020 were already absent. Existing
identifiers for retained responsibilities remain; new parameter, tag, client, and LLM-flow contracts use FR-038
through FR-047.

## Consequences

- `PolicyService` is completely stateless and file-system-free for controls, accepting and returning Pydantic
  `ControlDefinition` objects that map directly to future MCP tool schemas.
- Creation scales to large evidence responses via bounded JSONata exploration and produces controls with embedded,
  verified unit tests (`selection_tests` and `criterion_tests`).
- Shared parameters across rules of the same policy are consistently named via the incremental
  `SharedParameterCatalog`.
- `vatuta policy check "<prompt>"` uses a predictable 2-step LLM pipeline backed by the client's local
  filtering/projection and `include_merged_parameters` tool.

## Related Artifacts

See [design](design.md), [glossary](glossary.md), [interview](interview.md), and
[design inputs](../../specs/002-policy-checker/design-inputs.md).
