# ADR-019: Stateless Policy Service, Step-by-Step Creation & Check LLM Flows

**Date**: 2026-10-25
**Updated**: 2026-11-04
**Status**: Accepted — stakeholder scope, UX, and LLM flows refinement
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
  - `PolicyService.execute` (`policy_execute`) accepts `{controls: list[ControlDefinition], parameters, result_filter}`
    with inline structured `ControlDefinition` objects (requiring unique `id` values within the invocation) and returns
    `{invocation_id, results: list[ControlResult], errors}`. `ControlResult` and `ErrorDetail` identify controls by
    `control_id` without `control_file`.
  - Remove `policy_parameters` as a service/MCP operation.
- **Step-by-Step Creation LLM Flow & Embedded Unit Tests**:
  - **Step 1 (Policy Decomposition)**: Decompose `policy_text` into atomic normative rules. Split conditions into
    separate rules whenever they can fail independently and a user wants to know which condition failed, or whenever
    the policy author would consider them distinct violations.
  - **Step 2 (Sequential Rule Creation with Incremental `SharedParameterCatalog`)**: Construct rules one by one,
    passing the cumulative `SharedParameterCatalog` (`name`, `description`, `value_schema`, `construction_value`,
    `default`) from previously created rules of the same policy so conceptually identical parameters reuse the exact
    same name and compatible schema.
  - **Scalable Source Exploration (Bounded JSONata Projections)**: When `trial_attempts > 0`, fetch the declared
    evidence once in memory using `construction_value` inputs and allow the constructor to inspect data structure and
    semantics via bounded exploratory JSONata queries (configurable via `policy.discovery_max_queries`, default `5`,
    with each output truncated to `policy.discovery_max_output_bytes`, default `4096` bytes, and marked `[TRUNCATED]`
    if exceeded) rather than dumping raw evidence JSON into the prompt.
  - **Step-by-Step Selection & Criterion with Mandatory Synthetic Unit Tests (`ControlDefinition.test_cases`)**:
    1. Build `test_cases.selection_tests` (`mock_evidence` -> `expected_subjects` or `NOT_APPLICABLE`) and `selection`,
       then validate `selection` against `selection_tests` (always mandatory, even when `trial_attempts == 0`) and
       against the in-memory live evidence (when `trial_attempts > 0`) to verify resulting `Subject` objects.
    2. Build `test_cases.criterion_tests` (`subject` -> `COMPLIANT` and `NON_COMPLIANT`) and `criterion` (preferring
       deterministic CEL; declaring LLM mode only with explicit insufficiency reason), then validate `criterion`
       against `criterion_tests` (always mandatory) and against the live trial subjects (when `trial_attempts > 0`).
    3. Persist `test_cases` inside `ControlDefinition` so saved `<id>.yaml` files include reproducible unit tests.
- **Client-Side Local Controls, Filtering/Projection Tool, and Guided 2-Step Check Flow**:
  - The client stores and reads YAML control files in configurable `policy.controls_directory` (default
    `data/controls`, overridable via `--controls-dir`).
  - The client provides a local filtering and projection function/tool (`ClientControlQuery`) supporting criteria
    (`ids`, `policy_ids`, `rule_ids`, `tags`, `evaluation_mode`, `evidence_server`, `evidence_tool`, `parameter_names`,
    `text_query`, `files`, `glob`), attribute `projection` (e.g., `["id", "description", "tags", "source"]` or
    `["full"]`), and a configurable `include_merged_parameters: bool = True` parameter (enabled by default).
  - In `vatuta policy check "<prompt>"`, the client executes a **guided 2-step LLM pipeline**:
    1. **Rule Selection**: The LLM queries `ClientControlQuery` with lightweight projection (`["id", "description",
       "tags", "source"]`, `include_merged_parameters=False`, bounded by configurable `policy.check_max_filter_queries`,
       default `3`, and `policy.check_max_output_bytes`, default `4096` bytes) to choose applicable `control_id`s; the
       client then deterministically loads those controls with `projection=["full"]` and
       `include_merged_parameters=True`.
    2. **Parameter Deduction**: A dedicated DSPy signature receives the user's prompt and `merged_parameters` and
       extracts explicit parameter values from the prompt before invoking `PolicyService.execute`.
- **Diagnostics and Trace Resource URIs**:
  - Preserve diagnostic traces in the log directory, including model interactions and required redaction. Remove
    retention configuration and automatic age-based cleanup. Traces remain until explicitly deleted and do not create
    a policy history service.
  - Clients derive trace resource URIs from the invoked operation and returned `invocation_id`, using the templates in
    [the operation contract](../../specs/002-policy-checker/contracts/operations.md#diagnostic-trace-resources).

## Retained Contracts

The independent-evidence, response-conversion, completeness, subject-selection, boolean-result, filtering, and
semantic-batch contracts remain. Technical failures are errors, never automatic violations. Results remain per
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
| ADR-018 | Keep declared semantic evaluation and batching; remove approval, revisions, and source-abort references. |

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
