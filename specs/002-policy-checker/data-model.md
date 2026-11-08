# Policy Validation Engine: Data Model

**Date**: 2026-11-05
**Authority**: [spec.md](spec.md); formats are defined in [contracts/control-format.md](contracts/control-format.md)

Use Pydantic v2 with `extra='forbid'`, explicit JSON-compatible types (`str`, `int`, `float`, `bool`, `list`, `dict`,
`Literal`, `Optional`), `Field(description=...)` annotations for direct MCP schema generation, and strict boundary
validation. Each new model class has its own file under `src/policy/models/`. There is no database, server-side
control repository, policy registry, revision entity, or migration.

## ControlDefinition

| Field | Meaning and validation |
| --- | --- |
| `id` | Nonempty control identifier; creation assigns an opaque UUID, unique within its output. Unique per `execute` call. |
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
    nonempty string `id` and `descriptive_id`). Exactly one of `expected_outcome` or `expected_subjects` must be
    supplied.
- **`CriterionTestCase`**:
  - `description: str`: Human-readable description of the criterion scenario.
  - `subject: dict[str, Any]`: Synthetic `Subject` JSON object (must have nonempty string `id` and `descriptive_id`).
  - `parameters: dict[str, Any] = {}`: Optional parameter values used by `criterion`.
  - `expected_state: Literal["COMPLIANT", "NON_COMPLIANT"]`: Expected evaluation verdict for this subject.

`ControlTestCases` requires nonempty `selection_tests` (at least one populated subject case and one `NOT_APPLICABLE`
case) and nonempty `criterion_tests` (at least one `COMPLIANT` case and one `NON_COMPLIANT` case). All `selection_tests`
and `criterion_tests` must pass during creation, regardless of whether `trial_attempts` is `0` or `>= 1`.

## ParameterDeclaration, SharedParameterCatalog, MergedParameter, and ResolvedParameters

| Field | Meaning and validation |
| --- | --- |
| `description` | Meaning used by the calling agent; merged parameter summaries retain descriptions per `control_id`. |
| `value_schema` | Self-contained JSON Schema 2020-12 describing a JSON value. No network-resolved references. |
| `construction_value` | Optional value derived during construction for trials; does not affect runtime requiredness. |
| `default` | Optional explicitly declared invocation default; presence is tracked separately from null. |

During sequential per-rule creation, an incremental `SharedParameterCatalog` (`dict[str, ParameterDeclaration]`) is
passed from rule `i` to rule `i + 1` within the same `PolicyService.create` call. When rule `i + 1` needs a conceptual
parameter already present in `SharedParameterCatalog`, the constructor reuses that exact parameter name and compatible
`value_schema`.

Construction/default values must satisfy the schema when present. Missing `default` means required at execution;
`default: null` is permitted only when the schema admits null. A supplied null is a supplied value, then
schema-validated. Schema validation performs no string-to-number/boolean coercion. Reject non-finite numbers and
non-JSON YAML values.

For each shared name across selected controls, compare normalized schemas and default presence/value; object key
ordering is irrelevant and scalar types remain distinct (`true` is not `1`). Differences are declaration conflicts.
Construction values need not match because they are historical trial inputs, not invocation defaults. Do not use a
model to compare descriptions. Merged parameter views list each referring `control_id` and its `description`.

Explicit invocation values override defaults. Aggregate missing inputs before any source access, returning the
parameter name, semantics, expected value schema, and affected `control_id`s. Unknown names and invalid values are
input errors. No expression-source interpolation occurs. Evidence argument bindings reference values structurally.

## Client Control Filter, Projection, and Parameter Aggregation

The Vatuta client provides a local filtering and projection model/tool (`ClientControlQuery`) over
`policy.controls_directory` (`--controls-dir`):

| Field | Meaning and validation |
| --- | --- |
| `ids` | Optional list of exact `control_id` strings to match. |
| `policy_ids` | Optional list of `source.policy` identifiers to match. |
| `rule_ids` | Optional list of `source.rule` identifiers to match. |
| `tags` | Optional list of tag strings; matches controls containing any/all specified tags. |
| `evaluation_mode` | Optional `"deterministic"` or `"llm"` filter against `criterion.mode`. |
| `evidence_server` | Optional MCP server name filter against `evidence[].server`. |
| `evidence_tool` | Optional MCP tool name filter against `evidence[].tool`. |
| `parameter_names` | Optional list of parameter names declared in `parameters`. |
| `text_query` | Optional case-insensitive substring search across `description` and `source.excerpt`. |
| `files` / `glob` | Optional file or glob filter resolved safely within `controls_directory`. |
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
  `subject` and `parameters`; external access and model calls are unavailable.
- **SemanticCriterion**: `mode: llm`, `condition`: semantic obligation, `deterministic_insufficiency`: why CEL cannot
  decide it. Model/backend settings are deployment-owned and are not selectable through an untrusted control definition.

Both modes consume the same valid selected subject objects. All extra JSON fields are preserved, including null.
Subjects require nonempty string `id` and `descriptive_id`; scalar roots/array members and missing identifiers are
contract errors. Malformed objects are reported separately by selection index without generated business identifiers.

Semantic requests carry a private correlation token plus the original subject. Response `{subject_ref, verdict}`
requires an exact matching token and strict boolean verdict. First implementation sends one subject per request; no
batching setting is exposed. Unexpected, missing, or invalid responses are errors. Internal correlation tokens do not
replace subject identifiers and cannot make malformed subjects valid.

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
| ControlResult | `control_id`, `evaluation_mode`, `subjects`, `errors`, optional `outcome`. |
| PolicyExecuteResult | `invocation_id`, `results` (`list[ControlResult]`), and request-level `errors`. |
| InvocationTrace | Operation inputs, executed definitions, stage outputs/errors, measurements, model-call records. |

The client derives a trace resource URI from the operation it invoked and the returned `invocation_id`, following
[the operation contract](contracts/operations.md#diagnostic-trace-resources). `PolicyService.read_trace(uri)` (and
future MCP `resources/read`) returns sanitized InvocationTrace JSON (`application/json`) without evidence or model
calls.

## Processing Transitions

### Creation Transitions (`PolicyService.create`)

1. **Policy Decomposition**: Split `policy_text` into atomic normative rules (`PolicyDecompositionSignature`) whenever
   conditions can fail independently or represent distinct violations to the policy author.
2. **Sequential Per-Rule Construction (with `SharedParameterCatalog`)**:
   - **Parameters & Evidence**: Deduce parameters (reusing names/schemas from `SharedParameterCatalog` when matching),
     `construction_value`s, optional `default`s, and `evidence` tool bindings.
   - **In-Memory Source Exploration (when `trial_attempts > 0`)**: Fetch declared evidence once in memory (failing with
     `MISSING_CONSTRUCTION_CONTEXT` if required trial values are missing) and run up to configurable
     `policy.discovery_max_queries` (default 5) exploratory JSONata queries with outputs truncated to configurable
     `policy.discovery_max_output_bytes` (default 4096 bytes, appending `[TRUNCATED]` when exceeded).
   - **Selection & `selection_tests`**: Generate `selection` and `selection_tests`; execute `selection` against all
     `selection_tests` (mandatory) and against in-memory live evidence (when `trial_attempts > 0`) to validate the
     resulting `Subject` structure.
   - **Criterion & `criterion_tests`**: Generate `criterion` and `criterion_tests`; execute `criterion` against all
     `criterion_tests` (mandatory) and against live trial subjects (when `trial_attempts > 0`).
   - **Bounded Correction & Catalog Update**: Apply bounded candidate repairs within attempt limits; on rule success,
     persist `test_cases` on `ControlDefinition` and merge the rule's parameters into `SharedParameterCatalog`.

### Client Check Transitions (`vatuta policy check "<prompt>"`)

1. **Step 1 — Rule Selection**: Query `ClientControlQuery` with `projection=["id", "description", "tags", "source"]`
   and `include_merged_parameters=False` (up to configurable `policy.check_max_filter_queries`, default 3 queries, with
   each query output truncated to configurable `policy.check_max_output_bytes`, default 4096 bytes) via
   `PolicyCheckRuleSelectionSignature` to select applicable `control_id`s.
2. **Deterministic Load**: Call `ClientControlQuery(ids=selected_ids, projection=["full"],
   include_merged_parameters=True)` to obtain full `ControlDefinition` objects and conflict-checked `merged_parameters`.
3. **Step 2 — Parameter Deduction**: Extract explicit parameter values from the prompt against `merged_parameters` via
   `PolicyCheckParameterExtractionSignature` without inventing missing values.
4. **Step 3 — Execution**: Invoke `PolicyService.execute(controls=selected_controls, parameters=extracted_parameters,
   result_filter=...)`.

### Execution Transitions (`PolicyService.execute`)

1. **Request validation**: Validate unique `control_id`s in `controls`, merge parameter declarations, apply explicit
   values/defaults, and validate configured permissions and arguments. Any request error ends before evidence
   acquisition.
2. **Per-control acquisition**: Resolve that control's permitted source connections lazily and obtain evidence.
   Unavailable source/tool, tool error, invalid payload, or failed completeness stops this control with ERROR.
3. **Selection**: No result/empty collection yields NOT_APPLICABLE. Root null or scalar is ERROR. Mixed invalid members
   preserve valid subjects, collect object errors, and ultimately produce global control ERROR.
4. **Evaluation**: Evaluate each valid subject using its declared mode, retaining true/false/error outcomes.
   Individual failures do not suppress other valid subjects or independent controls.
5. **Presentation**: Apply filtering to findings after all selected subjects have been considered. Preserve errors.

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
