# Policy Validation Engine: Data Model

**Date**: 2026-11-11
**Authority**: [spec.md](spec.md); formats are defined in [contracts/control-format.md](contracts/control-format.md)

Use Pydantic v2 with `extra='forbid'`, explicit JSON-compatible types (`str`, `int`, `float`, `bool`, `list`, `dict`,
`Literal`, `Optional`), `Field(description=...)` annotations for direct MCP schema generation, and strict boundary
validation. Each new model class has its own file under `src/policy/models/`. There is no database, server-side
control repository, policy registry, revision entity, or migration.

## ControlDefinition

| Field | Meaning and validation |
| --- | --- |
| `id` | Deterministic kebab-case slug `f"{policy_id}-{rule_id}"` (or `rule_id` if already prefixed); unique per call. |
| `description` | Human-readable obligation. |
| `tags` | List of nonempty strings (deduplicated). Combines constructor-inferred tags and caller-supplied request `tags`. |
| `source` | Policy reference, rule/section reference, and supporting excerpt; no required version or content hash. |
| `parameters` | Name-to-ParameterDeclaration map. Same-name entries across selected controls share one invocation value. |
| `evidence` | Nonempty list of independent EvidenceDeclaration objects with unique evidence names in this control. |
| `selection` | JSONata expression over `{evidence, parameters}`. |
| `criterion` | Discriminated deterministic or LLM criterion; exactly one mode. |
| `test_cases` | `ControlTestCases` containing mandatory synthetic `selection_tests` and `criterion_tests`. |
| `creation_check` | Trial status and whether the criterion was exercised; generated metadata, not an execution permission. |

`ControlDefinition` is the canonical JSON-serializable representation returned by `PolicyService.create` and accepted
inline by `PolicyService.execute`. At execution, `id` identifies each control within the request (duplicate `id`s in a
single `PolicyExecuteRequest` are rejected as an input error). On the client side, the client serializes each
`ControlDefinition` as `<id>.yaml` in `policy.controls_directory` (or `--output-dir`), rejecting duplicate YAML keys
and unsafe tags when loading.

Before saving, the client checks all target paths for containment and existing files. A collision produces
`CONTROL_FILE_EXISTS` with exit `1`, preserves existing files, and prevents publishing the proposed set. Publication
must not replace a file created after the check. No overwrite mode or automatic renaming is supported.

## ControlTestCases (`selection_tests` and `criterion_tests`)

Every `ControlDefinition` includes `test_cases: ControlTestCases`, generated and verified step by step during creation
and persisted in the control model and YAML file:

- **`SelectionTestCase`**:
  - `description: str`: Human-readable description of the selection scenario.
  - `mock_evidence: dict[str, Any]`: Synthetic JSON map keyed by declared evidence `name`.
  - `parameters: dict[str, Any] = {}`: Optional parameter values used by `selection`.
  - `expected_outcome: Optional[Literal["NOT_APPLICABLE"]] = None`: Set when the case tests an empty/no-result
    selection.
  - `expected_subjects: Optional[list[dict[str, Any]]] = None`: Expected list of selected `Subject` objects (each with
    non-empty, non-whitespace string `id` and `descriptive_id`, `len(s.strip()) > 0`, and unique `id`s within the list).
    Exactly one of `expected_outcome` or `expected_subjects` must be supplied. When comparing, a single JSONata-selected
    object `{...}` is normalized to a 1-element list `[{...}]` before checking equality against `expected_subjects`, and
    duplicate `id`s fail validation with `DUPLICATE_SUBJECT_ID`.
- **`CriterionTestCase`**:
  - `description: str`: Human-readable description of the criterion scenario.
  - `subject: dict[str, Any]`: Synthetic `Subject` JSON object (must have non-empty, non-whitespace string `id` and
    `descriptive_id`).
  - `parameters: dict[str, Any] = {}`: Optional parameter values used by `criterion` (in CEL activation for
    `mode: deterministic` or rendered into `condition` via `{{parameters.<name>}}` template substitution for
    `mode: llm`).
  - `expected_state: Literal["COMPLIANT", "NON_COMPLIANT"]`: Expected evaluation verdict for this subject.

When running any `SelectionTestCase` or `CriterionTestCase`, parameters are resolved with strict priority:
`1. test_case.parameters` (explicit override) -> `2. parameter.default` -> `3. parameter.construction_value`.
`ControlTestCases` requires nonempty `selection_tests` (at least one populated subject case and one `NOT_APPLICABLE`
case) and nonempty `criterion_tests` (at least one `COMPLIANT` case and one `NON_COMPLIANT` case). `criterion_tests`
run via CEL for `mode: deterministic` and via `SemanticCriterionSignature` (LLM) for `mode: llm`. All `selection_tests`
and `criterion_tests` must pass during creation, regardless of whether `trial_attempts` is `0` or `>= 1`.

## ParameterDeclaration, SharedParameterCatalog, MergedParameter, and ResolvedParameters

| Field | Meaning and validation |
| --- | --- |
| `description` | Meaning used by the calling agent; merged parameter summaries retain descriptions per `control_id`. |
| `value_schema` | Self-contained JSON Schema 2020-12 describing a JSON value. No network-resolved references. |
| `construction_value` | Optional trial value derived during construction (optional when `trial_attempts == 0`). |
| `default` | Optional explicitly declared invocation default; presence is tracked separately from null. |

Parameters may be declared in any creation stage (Step 2.1 evidence/parameters, Step 2.3 selection, Step 2.4 criterion,
or Step 2.5 repair), following a **parameter minimization principle**: declare only the minimum necessary inputs
(`parameters: {}` is valid when no external input is needed). During sequential per-rule creation, an incremental
`SharedParameterCatalog` (`dict[str, ParameterDeclaration]`) is passed from rule `i` to rule `i + 1` within the same
`PolicyService.create` call. Two parameters share the same parameter name in `SharedParameterCatalog` **only** when
their `value_schema`, `construction_value`, and `default` are identical/compatible. If a rule requires a different
`value_schema`, `construction_value`, or `default` despite similar semantics, it is effectively a distinct parameter and
must either adapt to be compatible with the existing catalog entry or use a different parameter name.

Declarations referenced by other rules are immutable throughout Phase B and retries. Keep candidate additions local
until validated, and discard failed-attempt changes. Later stages adapt only the current rule or allocate a distinct
name; they cannot silently change another rule's schema, default, or construction value.

Construction/default values must satisfy the schema when present. Missing `default` means required at execution;
`default: null` is permitted only when the schema admits null. A supplied null is a supplied value, then
schema-validated. Schema validation performs no string-to-number/boolean coercion. Reject non-finite numbers and
non-JSON YAML values.

For each shared name across selected controls, compare normalized schemas and default presence/value; object key
ordering is irrelevant and scalar types remain distinct (`true` is not `1`). Differences are declaration conflicts.
Construction values need not match because they are historical trial inputs, not invocation defaults. Do not use a
model to compare descriptions. Merged parameter views list each referring `control_id` and its `description`.

Explicit invocation values override defaults. Aggregate missing inputs before any source access, returning the
parameter name, semantics, expected value schema, and affected `control_id`s. Extra parameter keys in
`request.parameters` that are not declared by any supplied control are permitted (ignored at request level so callers
can reuse shared parameter files across filtered control subsets), while each individual control receives only the
subset of resolved parameters declared in its own `control.parameters`. Invalid values for declared parameters are input
errors. Evidence argument bindings reference values structurally; semantic `mode: llm` criteria support deterministic
`{{parameters.<name>}}` template substitution in `condition`.

## Client Control Filter, Projection, and Parameter Aggregation

The Vatuta client provides a local filtering and projection model/tool (`ClientControlQuery`) over
`policy.controls_directory` (`--controls-dir`). Distinct filter fields combine with logical **AND**; multiple values
within any list filter field (`ids`, `policy_ids`, `rule_ids`, `tags`, `parameter_names`, `files`) combine with logical
**OR** (`any`). If any encountered YAML control file is malformed or fails schema validation, `ClientControlQuery` fails
the call immediately with `INVALID_CONTROL_FILE`:

| Field | Meaning and validation |
| --- | --- |
| `ids` | Optional list of exact `control_id` strings to match (logical OR within list). |
| `policy_ids` | Optional list of `source.policy` identifiers to match (logical OR within list). |
| `rule_ids` | Optional list of `source.rule` identifiers to match (logical OR within list). |
| `tags` | Optional list of tag strings; matches controls containing **any** specified tag (logical OR). |
| `evaluation_mode` | Optional `"deterministic"` or `"llm"` filter against `criterion.mode`. |
| `evidence_server` | Optional MCP server name filter against `evidence[].server`. |
| `evidence_tool` | Optional MCP tool name filter against `evidence[].tool`. |
| `parameter_names` | Optional list of parameter names declared in `parameters` (logical OR within list). |
| `text_query` | Optional case-insensitive substring search across `description` and `source.excerpt`. |
| `files` / `glob` | Optional file list (logical OR) or glob filter resolved safely within `controls_directory`. |
| `projection` | Attributes to project per matched control (e.g., `["id", "description", "tags", "source"]` or `["full"]`). |
| `include_merged_parameters` | Boolean, default `True`. When `True`, computes `merged_parameters` and checks declaration conflicts. |

`ClientControlQueryResult` contains `controls` (projected dicts or full `ControlDefinition` objects), optional
`merged_parameters` (present when `include_merged_parameters=True`), and `errors` (`list[ErrorDetail]`).

## EvidenceDeclaration and EvidenceProfile

An EvidenceDeclaration has `name`, `server`, `tool`, and `arguments`. Each argument is either `{literal: <JSON>}` or
`{parameter: <name>}`; never both. Bind only whole argument values, including structured JSON, to avoid a template
language. Source relationships are computed in Selection, never by feeding an evidence result into another call.

EvidenceProfile is deployment configuration keyed by server/tool, not generated policy content. It contains:

- `contract_reference`: documentation identifying the exact tool response and scope guarantee.
- `arguments_schema`: trusted self-contained JSON Schema restricting allowed arguments and scope. Generated controls
  cannot relax it; also validate the tool's discovered input schema before its call.
- `complete_when`: a JSONata boolean expression over `data`, resolved `arguments`, and SDK `response` metadata.
- `trace_redact_paths`: JSON Pointer paths to sensitive values in retained source data.
- `trace_safe_text_paths`: explicit non-sensitive text projections allowed in persisted diagnostics;
  other free text is masked.

A profile must correspond to an explicitly whitelisted read-only tool on a configured, digest-pinned source server.
The existing server configuration supplies connection credentials, mounts, and network settings. Profiles do not grant
new access. Unknown/missing profiles fail that control with unknown-completeness ERROR; absence is not proof of safety.
Only strict boolean true establishes completeness. False means incomplete, and absent/non-boolean/failed proof means
unknown. Both prevent Selection and Criterion. Record the applied contract and proof outcome in diagnostics.

## Criterion

- **DeterministicCriterion**: `mode: deterministic`, `expression`: CEL boolean expression. Activation contains only
  `subject` and `parameters` (the subset of resolved parameters declared by this control); external access and model
  calls are unavailable.
- **SemanticCriterion**: `mode: llm`, `condition`: semantic obligation (supporting optional `{{parameters.<name>}}`
  template placeholders substituted deterministically with the control's resolved parameters before calling the LLM),
  `deterministic_insufficiency`: why CEL cannot decide it. Model/backend settings are deployment-owned and are not
  selectable through an untrusted control definition.

Both modes consume the same valid selected subject objects. All extra JSON fields are preserved, including null.
Subjects require non-empty, non-whitespace string `id` and `descriptive_id` (`isinstance(v, str) and len(v.strip()) >
0`); scalar roots/array members and missing/blank identifiers are contract errors (`INVALID_SUBJECT`, reported by
`selection_index` without generated business identifiers). Within a control's selection output, subjects whose `id` is
unique are evaluated by `criterion`, whereas subjects sharing a duplicate `id` are not evaluated by `criterion`, emit
`ErrorDetail` with `code="DUPLICATE_SUBJECT_ID"`, `stage="selection"`, `subject_id`, and `selection_index`, and set the
control's `outcome` to `"ERROR"` (both in `execute` and in `create`).

Semantic evaluation executes one valid unique-ID subject per `SemanticCriterionSignature` call (no separate `parameters`
dict is sent to the LLM; only the rendered `condition`, private correlation token `subject_ref`, and single `subject`
object). Response `{subject_ref, verdict}` requires an exact matching token and strict boolean verdict. Unexpected,
missing, or invalid responses are errors. Internal correlation tokens do not replace subject identifiers and cannot make
malformed subjects valid.

## Service Requests and Results (`PolicyService` / Future MCP Contract)

[Operation contracts](contracts/operations.md) define the public envelopes. `PolicyService` models use pure
JSON-serializable Pydantic types with no filesystem paths (`control_file` is absent from service models).

- **`PolicyCreateRequest`**: required nonempty `policy_text`, optional nonempty provenance identifier `policy_id`,
  optional descriptive `construction_context` defaulting to `""`, optional `tags: list[str]` defaulting to `[]`, and
  optional nonnegative integer `trial_attempts`. It accepts neither file paths nor caller-defined parameter maps.
- **`PolicyCreateResult`**: `{invocation_id: str, controls: list[ControlDefinition], errors: list[ErrorDetail]}`.
- **`PolicyExecuteRequest`**: `{controls: list[ControlDefinition], parameters: dict[str, Any] = {}, result_filter:
  Literal["violations", "all"] = "violations"}`. `controls` must be nonempty and have unique `id` values.
- **`PolicyExecuteResult`**: `{invocation_id: str, results: list[ControlResult], errors: list[ErrorDetail]}`.

| Record | Required content |
| --- | --- |
| ControlDefinition | `id`, `description`, `tags`, `source`, `parameters`, `evidence`, `selection`, `criterion`, `test_cases`, `creation_check`. |
| ControlTestCases | `selection_tests` (`list[SelectionTestCase]`) and `criterion_tests` (`list[CriterionTestCase]`). |
| PolicyCreateResult | `invocation_id`, `controls` (`list[ControlDefinition]`), and `errors` (`list[ErrorDetail]`). |
| ErrorDetail | `code`, `stage`, safe `message`; optional `control_id`, `parameter`, `subject_id`, `descriptive_id`, `selection_index`. |
| SubjectResult | Original `id`, `descriptive_id`, and COMPLIANT/NON_COMPLIANT/ERROR. |
| ControlResult | `control_id`, `source`, `parameters`, `evaluation_mode`, `subjects`, `errors`, optional `outcome`. |
| PolicyExecuteResult | `invocation_id`, `results`, and request-level or invocation-level diagnostic `errors`. |
| InvocationTrace | Operation inputs, executed definitions, stage outputs/errors, measurements, model-call records. |

`ControlResult.source` contains the executed `ControlDefinition.source` (`policy`, `rule`, `excerpt`). Its `parameters`
map contains only that control's effective invocation values after explicit values override defaults. Include both
fields under both filters and for `ERROR`/`NOT_APPLICABLE`; omit construction-only values and unrelated request keys.
Use the existing privacy boundary when exposing identifying data. Preserve the same attribution in the trace.

An internal invocation context owns one execution ID and stage records. `check` allocates it before rule selection and
passes it internally to execution, without adding fields to public request models. Direct execution allocates its own
context. A check failure before execution returns the execution envelope with `results: []` and that same ID.

A trace-write failure adds a top-level `ErrorDetail` with `code="TRACE_WRITE_FAILED"` and `stage="trace"`, retaining
calculated results and producing CLI exit `1`. It does not change compliance states or control outcomes by itself.
Request errors still return `results: []`; control-specific runtime errors remain inside their control result.

The client derives a trace resource URI from the operation it invoked and the returned `invocation_id`, following
[the operation contract](contracts/operations.md#diagnostic-trace-resources). `PolicyService.read_trace(uri)` (and
future MCP `resources/read`) returns sanitized InvocationTrace JSON (`application/json`) without evidence or model
calls.

## Processing Transitions

### Creation Transitions (`PolicyService.create`)

1. **Step 0 — Pre-flight & Tool Catalog**: Validate nonempty `policy_text`. Build the tool catalog solely from local
   `policy.evidence_profiles` (`arguments_schema` and `contract_reference`) for every trial setting. No container
   startup, MCP connection, or `list_tools` occurs before Phase A succeeds; zero attempts never contacts MCP.
2. **Step 1 — Policy Decomposition**: Split `policy_text` into atomic normative rules (`PolicyDecompositionSignature`)
   whenever conditions can fail independently or represent distinct violations to the policy author. Assign each
   control its deterministic slug `id` (`f"{policy_id}-{rule_id}"`, or `rule_id` if already prefixed with
   `f"{policy_id}-"`). If zero normative rules are extracted, fail immediately with `NO_NORMATIVE_RULES` (`stage:
   "decomposition"`).
3. **Step 2, Phase A — All-Rules Parameter & Evidence Sweep (Step 2.1 across `r_1 ... r_N`)**:
   - Run `RuleEvidenceAndParametersSignature` sequentially across all decomposed rules `r_1 ... r_N`, passing and
     updating `SharedParameterCatalog` (reusing a parameter name only when `value_schema`, `construction_value`, and
     `default` match; otherwise adapting to the catalog or declaring a distinct parameter name).
   - When `trial_attempts > 0`, check all rules for required parameters lacking both `default` and
     `construction_value`. If any are missing, aggregate and return **all** `MISSING_CONSTRUCTION_CONTEXT` errors and
     affected `control_id`s across all rules in a single response before calling any MCP server.
4. **Step 2, Phase B — Per-Rule Deep Construction (`r_1 ... r_N`) with Fail-Fast**:
   - **Step 2.2 — In-Memory Source Exploration (when `trial_attempts > 0`)**: After Phase A succeeds, connect to the
     required MCP servers and validate discovered `inputSchema` before calling evidence tools. Fetch evidence in memory
     (caching by `(server, tool, resolved_arguments)` across retries of the same rule) and run up to configurable
     `policy.discovery_max_queries` (default 5) exploratory JSONata queries against the root context `{"evidence": ...,
     "parameters": ...}`, truncating each output to `policy.discovery_max_output_bytes` (default 4096 bytes +
     `[TRUNCATED]`). Query errors or `missing` outcomes are returned as observations within the query budget; the loop
     exits early as soon as `done=True`.
   - **Step 2.3 — Selection & `selection_tests`**: Generate `selection`, `selection_tests`, and any additional
     selection parameters (validated against `SharedParameterCatalog`). Normalize single-object JSONata output to
     `[obj]`, resolve test parameters via `test_case.parameters` -> `default` -> `construction_value`, and validate
     `selection` against all `selection_tests` (mandatory) and against in-memory live evidence (when
     `trial_attempts > 0`).
   - **Step 2.4 — Criterion & `criterion_tests`**: Generate `criterion`, `criterion_tests`, and any additional
     criterion parameters (validated against `SharedParameterCatalog`). Execute `criterion` against all
     `criterion_tests` (mandatory, using CEL for `mode: deterministic` and `SemanticCriterionSignature` LLM calls for
     `mode: llm`) and against live trial subjects (when `trial_attempts > 0`).
   - **Step 2.5 — Full-Context Bounded Retry & Fail-Fast**: When `trial_attempts > 1` and a rule fails in Steps
     2.2–2.4 (including a newly added parameter in 2.3/2.4 lacking `default`/`construction_value`), re-execute from
     Step 2.1 carrying the full accumulated context and errors from all stages, reusing cached in-memory MCP evidence
     whenever `evidence` bindings are unchanged. If a rule still fails after exhausting its configured attempts, abort
     creation immediately (**Fail-Fast**) without executing Phase B for remaining rules.

### Client Check Transitions (`vatuta policy check "<prompt>"`)

All stages of `vatuta policy check "<prompt>"` share a single execution `invocation_id` and record their artifacts in
`vatuta://traces/policy/execution/{invocation_id}`:

1. **Step 1 — Iterative Rule Selection (`check_rule_selection`)**: Run `PolicyCheckRuleSelectionSignature` in a bounded
   loop (up to configurable `policy.check_max_filter_queries`, default 3 queries, with each query output truncated to
   configurable `policy.check_max_output_bytes`, default 4096 bytes + `[TRUNCATED]`), calling `ClientControlQuery` with
   `projection=["id", "description", "tags", "source"]` and `include_merged_parameters=False` until `done=True` (or the
   query budget is reached) to select applicable `control_id`s. If `selected_control_ids` is empty (`[]`), fail
   immediately with `NO_CONTROLS_MATCHED` (exit code `2`) without calling `PolicyService.execute`.
2. **Deterministic Load**: Call `ClientControlQuery(ids=selected_ids, projection=["full"],
   include_merged_parameters=True)` to obtain full `ControlDefinition` objects and conflict-checked `merged_parameters`.
3. **Step 2 — Parameter Deduction (`check_parameter_extraction`)**: If `merged_parameters` is empty (`[]`), skip the LLM
   call and set `extracted_parameters = {}`; otherwise extract explicit parameter values from the prompt against
   `merged_parameters` via `PolicyCheckParameterExtractionSignature` without inventing missing values.
4. **Step 3 — Execution**: Invoke `PolicyService.execute(controls=selected_controls, parameters=extracted_parameters,
   result_filter=...)` under the same `invocation_id`.

### Execution Transitions (`PolicyService.execute`)

1. **Stage E0 — Pre-flight request validation (Whole-Request Gate)**: Validate that `controls` is nonempty and
   `control.id` values are unique within the request, merge shared parameter declarations across supplied controls,
   resolve values (`request.parameters[name]` -> `default`; never `construction_value`), ignore extra keys in
   `request.parameters` not declared by any supplied control, and validate all declared resolved parameters against
   their `value_schema`. Any missing required parameter or declaration/schema error aborts the entire request before
   evidence acquisition.
2. **Stage E1 — Per-control evidence acquisition & completeness (Isolated per control with Fast-Fail)**: Keep started
   `MCPServer` containers alive across controls for the duration of the `execute` invocation and cache valid evidence
   responses in memory by exact `(server, tool, resolved_arguments)`. For each control, pass only the resolved
   parameters declared in its own `control.parameters`, validate each `EvidenceDeclaration` against local server/tool
   allowlists, `evidence_profile` existence, `arguments_schema`, and discovered `inputSchema`, call the tool (or reuse
   the invocation cache), convert the payload, and verify `complete_when == True`. If any evidence item fails,
   **fast-fail that control immediately** (`outcome="ERROR"`, skipping its remaining evidence items, selection, and
   criterion) while remaining controls continue.
3. **Stage E2 — Selection (Isolated per control)**: Evaluate `selection` in the existence envelope. No result/empty
   collection yields `NOT_APPLICABLE`. Root null or scalar is `ERROR`. Single object `{...}` normalizes to `[{...}]`.
   Validate each array item for non-empty, non-whitespace string `id` and `descriptive_id` (`len(s.strip()) > 0`).
   Subjects with a unique valid `id` proceed to Criterion; malformed items (`INVALID_SUBJECT`) and items sharing a
   duplicate `id` (`DUPLICATE_SUBJECT_ID`, with `subject_id` and `selection_index`) are not evaluated by Criterion, are
   recorded in `ControlResult.errors`, and produce global control `outcome="ERROR"`.
4. **Stage E3 — Evaluation (Per valid unique-ID subject)**: Evaluate each valid unique-ID subject using its declared
   mode (`deterministic` CEL with `{subject, parameters}`, or singleton `llm` `SemanticCriterionSignature` with
   `{{parameters.<name>}}` rendered into `condition`), retaining `COMPLIANT`/`NON_COMPLIANT`/`ERROR` outcomes.
   Individual failures do not suppress other valid subjects or independent controls.
5. **Stage E4 — Presentation & Filtering**: When `result_filter="violations"` (default), `ControlResult.subjects`
   retains only `NON_COMPLIANT` subjects; when `result_filter="all"`, `ControlResult.subjects` retains all evaluated
   subjects (`COMPLIANT`, `NON_COMPLIANT`, and `ERROR`). `ControlResult.errors` retains all control and subject errors
   under either filter.
6. **Trace Finalization**: Persist the invocation trace. On failure, preserve `results`, append global
   `TRACE_WRITE_FAILED`, and return exit `1`. Do not misclassify the diagnostic failure as a compliance failure.

## Configuration and Storage

Add an optional `policy` section (`PolicyConfig`) to `VatutaConfig` with:

- `controls_directory` (default `"data/controls"`),
- `creation_backend` and `evaluation_backend` (defaulting to the existing generator binding),
- `trial_attempts` (default `1`),
- `discovery_max_queries` (default `5`, `>= 0`) and `discovery_max_output_bytes` (default `4096`, `> 0`) for creation
  Step 2.2 in-memory JSONata evidence exploration,
- `check_max_filter_queries` (default `3`, `>= 1`) and `check_max_output_bytes` (default `4096`, `> 0`) for `policy
  check` Step 1 rule-selection filter queries,
- `trace_directory` (default `"logs/policy"`), and
- `evidence_profiles`.

No separate provider registry is introduced. Client-side filtering and parameter merging require neither credentials nor
a running source; model validation occurs only when needed (`create`, `check`, or `llm` criteria). Source-backed
execution validates only selected profiles.
