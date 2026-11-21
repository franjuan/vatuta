# Policy Validation Engine: Data Model (Schematic)

**Date**: 2026-11-20
**Authority**: [spec.md](spec.md) | **YAML Format**: [contracts/control-format.md](contracts/control-format.md)

## General Modeling Rules

- **Framework**: Pydantic v2 with `extra='forbid'` and `Field(description=...)` for 1:1 MCP schema generation.
- **Types**: Pure JSON-serializable types (`str`, `int`, `float`, `bool`, `list`, `dict`, `Literal`, `Optional`). No
  `Path` objects inside `PolicyService` request or result models.
- **Code Layout**: One model class per file under `src/policy/models/`.
- **Persistence**: Client saves each rule as one `<id>.yaml` file under `policy.controls_directory` (supporting nested
  subfolders). No database, server-side rule store, or policy revision entity.

## 1. ControlDefinition (`<id>.yaml`)

| Field | Type | Validation & Meaning |
| --- | --- | --- |
| `id` | `str` | Deterministic kebab-case slug `f"{policy_id}-{rule_id}"` (or `rule_id` if already prefixed). Unique per call. |
| `description` | `str` | Human-readable rule obligation. |
| `tags` | `list[str]` | Deduplicated non-empty tags (combines LLM-inferred tags + caller request `tags`). |
| `source` | `ControlSource` | `{policy: str, rule: str, excerpt: str}` provenance back to the policy text. |
| `parameters` | `dict[str, ParameterDeclaration]` | Map of parameter name -> declaration (`{}` allowed when no inputs needed). |
| `evidence` | `list[EvidenceDeclaration]` | Non-empty list of independent MCP tool calls with unique `name`s in the control. |
| `selection` | `str` | JSONata expression evaluated over `{evidence, parameters}`. |
| `criterion` | `DeterministicCriterion \| SemanticCriterion` | Discriminated by `mode` (`"deterministic"` or `"llm"`). |
| `test_cases` | `ControlTestCases` | Mandatory embedded unit tests (`selection_tests` and `criterion_tests`). |
| `creation_check` | `CreationCheck` | `{trial: "skipped" \| "passed", criterion_executed: bool, outcome?: "NOT_APPLICABLE"}`. |

- **Client YAML Persistence Rules**:
  - Saved as `<id>.yaml` in `policy.controls_directory` or `--output-dir` (creating missing subfolders).
  - Safe YAML loader rejects duplicate keys, multiple documents, unsafe tags, and non-JSON types.
  - Pre-check all target `<id>.yaml` paths before writing any file: if any target exists, abort with
    `CONTROL_FILE_EXISTS` (Exit `1`) and write nothing. No overwrite mode or auto-renaming.

## 2. ControlTestCases (`selection_tests` & `criterion_tests`)

- **`SelectionTestCase`**:
  - `description: str`
  - `mock_evidence: dict[str, Any]` (keyed by declared evidence `name`)
  - `parameters: dict[str, Any] = {}`
  - `expected_outcome: Optional[Literal["NOT_APPLICABLE"]] = None`
  - `expected_subjects: Optional[list[dict[str, Any]]] = None`
  - **Rules**:
    - Exactly one of `expected_outcome` or `expected_subjects` must be set.
    - Single selected JSON object `{...}` normalizes to `[{...}]` before comparison.
    - Every subject in `expected_subjects` must have non-whitespace string `id` and `descriptive_id` (`len(s.strip()) >
      0`) and unique `id`s (duplicate `id`s fail with `DUPLICATE_SUBJECT_ID`).
- **`CriterionTestCase`**:
  - `description: str`
  - `subject: dict[str, Any]` (non-whitespace `id` and `descriptive_id`)
  - `parameters: dict[str, Any] = {}`
  - `expected_state: Literal["COMPLIANT", "NON_COMPLIANT"]`
  - **Rules**:
    - Evaluated with CEL for `mode: deterministic` and via `SemanticCriterionSignature` (LLM, after substituting
      `{{parameters.<name>}}` in `condition`) for `mode: llm`.
- **Execution & Parameter Priority**:
  - `ControlTestCases` requires at least 1 populated and 1 `NOT_APPLICABLE` `selection_tests` case, and at least 1
    `COMPLIANT` and 1 `NON_COMPLIANT` `criterion_tests` case.
  - Parameter resolution priority in unit tests:
    `1. test_case.parameters` -> `2. parameter.default` -> `3. parameter.construction_value`.
  - Mandatory during creation even when `trial_attempts == 0`.

## 3. ParameterDeclaration, SharedParameterCatalog & MergedParameter

| Field | Type | Validation & Meaning |
| --- | --- | --- |
| `description` | `str` | Parameter meaning for users and agents. |
| `required` | `bool` | Defaults to `true`. Set to `false` to allow omission when there is no default. |
| `value_schema` | `dict[str, Any]` | Self-contained JSON Schema 2020-12 (no network `$ref`). |
| `construction_value` | `Optional[Any]` | Trial value for creation only (optional when `trial_attempts == 0`). Never used as execution default. |
| `default` | `Optional[Any]` | Explicit execution default (presence tracked separately from `null`). |

- **Creation Rules (`SharedParameterCatalog`)**:
  - Declare minimum necessary parameters (`parameters: {}` is valid).
  - Pass `SharedParameterCatalog` (`dict[str, ParameterDeclaration]`) sequentially across rules `r_1..r_N`.
  - Two rules share a parameter name **only** when `value_schema`, `construction_value`, and `default` match; otherwise
    the rule adapts to the catalog or declares a distinct name.
  - Parameters already shared with earlier rules are **immutable** during Phase B and retries.
- **Execution & Merging Rules (`MergedParameter`)**:
  - Same-name parameters across selected controls share one invocation value.
  - Normalized `value_schema` values and any explicitly declared defaults must match across selected controls;
    incompatible values fail `PARAMETER_CONFLICT` (Exit `2`), even if an explicit invocation value is supplied.
    A missing default is not a conflict, and `construction_value` is ignored during merging and execution.
  - If any control requires the parameter without a default, `MergedParameter.required` is `true`, its `default` is
    omitted, and an explicit invocation value is required. Otherwise `required` is `false`, and `default` is included
    only when an eligible default exists. A declared `null` is a present value, not an absent default.
  - Execution resolution priority: `1. request.parameters[name]` -> `2. eligible merged default` (missing required ->
    `MISSING_PARAMETER`, schema mismatch -> `INVALID_PARAMETER_VALUE`, Exit `2`).
  - Leave an omitted optional parameter with no eligible default absent from the resolved parameter map.
  - Keep an explicit JSON null when the schema allows it. Serialize optional output fields by presence so an
    absent `default` is omitted and a declared `default: null` is written.
  - Extra keys in `request.parameters` not declared by any selected control are allowed at request level; each control
    receives only the parameters declared in its own `control.parameters`.

## 4. ClientControlQuery & ClientControlQueryResult

Operates over the hierarchical folder tree under `policy.controls_directory` (`--controls-dir`):

| Field | Type | Validation & Meaning |
| --- | --- | --- |
| `ids` | `Optional[list[str]]` | Exact `control.id` or glob/wildcard patterns on `control.id` (e.g., `change-mgmt-*`) (OR). |
| `policy_ids` | `Optional[list[str]]` | Match `source.policy` (OR). |
| `rule_ids` | `Optional[list[str]]` | Match `source.rule` (OR). |
| `tags` | `Optional[list[str]]` | Match controls containing any listed tag (OR). |
| `evaluation_mode` | `Optional[Literal["deterministic", "llm"]]` | Match `criterion.mode`. |
| `evidence_server` | `Optional[str]` | Match `evidence[].server`. |
| `evidence_tool` | `Optional[str]` | Match `evidence[].tool`. |
| `parameter_names` | `Optional[list[str]]` | Match controls declaring any listed parameter name (OR). |
| `text_query` | `Optional[str]` | Case-insensitive substring match in `description` or `source.excerpt`. |
| `files` | `Optional[list[str]]` | Specific `.yaml` file paths relative to or contained in `controls_directory` (OR). |
| `glob` | `Optional[str]` | Glob pattern (including `**`) matching relative paths (`team-a/**/*.yaml`) or bare filenames (`*github*.yaml`). |
| `projection` | `list[str]` | Default `["id", "description", "tags", "source", "parameters"]`, or `["full"]`. Always includes relative `file` in projected summary dicts. |
| `include_merged_parameters` | `bool` | Default `True`. Computes conflict-checked `merged_parameters` across matched controls. |

- **Query Rules**:
  - Distinct filter fields combine with **AND**; list values combine with **OR**.
  - Select candidate paths using `files` and `glob` before reading YAML. Intersect the path matches if both are set.
    Without either selector, all `.yaml` files under the root are candidates.
  - Validate containment and YAML for each candidate, then apply ID and metadata filters. Invalid candidate YAML
    returns `INVALID_CONTROL_FILE` (Exit `2`).
  - Different subfolders on disk may hold files with the same `control.id`, but if the active filter/selection matches
    two files with the same `control.id`, fail immediately with `DUPLICATE_CONTROL_ID` (Exit `2`).
  - `ClientControlQueryResult`: `{controls: list[dict[str, Any] | ControlDefinition], merged_parameters?:
    list[MergedParameter], errors: list[ErrorDetail]}`.

## 5. EvidenceDeclaration & EvidenceProfile

- **`EvidenceDeclaration` (inside `ControlDefinition`)**:
  - `{name: str, server: str, tool: str, arguments: dict[str, ArgumentBinding]}`.
  - Each `ArgumentBinding` is either `{literal: <JSON>}` or `{parameter: <name>}` (whole-value binding, never both).
- **`EvidenceProfile` (trusted deployment config in `vatuta.yaml` under `policy.evidence_profiles`)**:
  - Keyed by `<server>.<tool>` in configuration, as in the [demo YAML](demo-plan.md):
    - `contract_reference: str`
    - `arguments_schema: dict[str, Any]` (JSON Schema restricting allowed arguments)
    - `complete_when: str` (JSONata boolean expression over `{data, arguments, response}`)
    - `trace_redact_paths: list[str]` (JSON Pointers to redact)
    - `trace_safe_text_paths: list[str]` (non-sensitive text paths allowed in traces)
  - Only strict boolean `True` from `complete_when` proves completeness (`False` -> incomplete `ERROR`; other/missing ->
    unknown completeness `ERROR`).

## 6. Criterion (`DeterministicCriterion` vs `SemanticCriterion`)

- **`DeterministicCriterion`**:
  - `{mode: "deterministic", expression: str, violation_message: str}` (CEL over `{subject, parameters}`).
- **`SemanticCriterion`**:
  - `{mode: "llm", condition: str, deterministic_insufficiency: str, violation_message: str}`.
  - Supports `{{parameters.<name>}}` placeholders in `condition`, substituted deterministically before the LLM call.
  - Evaluates 1 unique-ID valid subject per `SemanticCriterionSignature` call (`{condition, subject_ref, subject}` ->
    `{subject_ref, verdict: bool}`).

## 7. Service Requests, Results & Traces (`PolicyService`)

- **`PolicyCreateRequest`**: `{policy_text: str, policy_id?: str, construction_context: str = "", tags: list[str] = [],
  trial_attempts?: int}`.
- **`PolicyCreateResult`**: `{invocation_id: str, controls: list[ControlDefinition], errors: list[ErrorDetail]}`.
- **`PolicyExecuteRequest`**: `{controls: list[ControlDefinition], parameters: dict[str, Any] = {}, result_filter:
  Literal["violations", "all"] = "violations"}`.
- **`PolicyExecuteResult`**: `{invocation_id: str, results: list[ControlResult], errors: list[ErrorDetail]}`.
- **`ControlResult`**:
  - `{control_id: str, source: ControlSource, parameters: dict[str, Any], evaluation_mode: Literal["deterministic",
    "llm"], subjects: list[SubjectResult], errors: list[ErrorDetail], outcome?: Literal["ERROR", "NOT_APPLICABLE"]}`.
  - Always includes `source` and effective per-control `parameters` (even on `ERROR` or `NOT_APPLICABLE`).
- **`SubjectResult`**: `{id: str, descriptive_id: str, state, violation_message?: str}`. The states are `COMPLIANT`,
  `NON_COMPLIANT`, and `ERROR`. Include the rendered `violation_message` only for `NON_COMPLIANT`. A render failure
  sets the subject and control to `ERROR` and adds `VIOLATION_MESSAGE_RENDER_ERROR` to `ControlResult.errors`.
- **`ErrorDetail`**: `{code: str, stage: str, message: str, control_id?: str, parameter?: str, subject_id?: str,
  descriptive_id?: str, selection_index?: int}`.
  A client `CONTROL_FILE_WRITE_FAILED` error also has `unsaved_paths: list[str]` relative to the output directory.
- **`InvocationTrace`**: Sanitized JSON trace read via `PolicyService.read_trace(uri)`.
- **Internal execution context**: Carries the engine-generated `invocation_id` and stage records for one call.
  Guided `check` initializes it before rule selection and shares it with the core execution path. Direct execution
  initializes its own context. The context stays outside the public request schema.

## 8. Processing State Transitions (Schematic)

### A. Creation Flow (`PolicyService.create`)

1. **Step 0 (Pre-flight & Local Tool Catalog)**: Validate `policy_text`; build tool catalog from local
   `policy.evidence_profiles` only (no MCP connection).
2. **Step 1 (Decomposition)**: Split policy into atomic rules (`NO_NORMATIVE_RULES` if zero); assign slug `id`s.
3. **Step 2, Phase A (All-Rules Step 2.1 Sweep)**: Declare evidence & minimum parameters across `r_1..r_N` with
   `SharedParameterCatalog`. If `trial_attempts > 0`, collect and return all `MISSING_CONSTRUCTION_CONTEXT` errors at
   once before starting MCP.
4. **Step 2, Phase B (Per-Rule Steps 2.2–2.5, Fail-Fast)**:
   - **Step 2.2 (`trial_attempts > 0`)**: Connect to MCP, validate live `inputSchema`, fetch evidence in memory, run up
     to `discovery_max_queries` (5) JSONata queries (truncated to `discovery_max_output_bytes`, 4096 bytes, early exit
     on `done=True`).
   - **Step 2.3**: Build `selection` + `selection_tests`; normalize `{...}` to `[{...}]`; reject `DUPLICATE_SUBJECT_ID`;
     run `selection_tests` (mandatory) + live selection (`trial_attempts > 0`).
   - **Step 2.4**: Build `criterion` + `criterion_tests`; run `criterion_tests` (mandatory) + live criterion
     (`trial_attempts > 0`).
   - **Step 2.5**: Synthetic and live failures share the same attempt budget. When `trial_attempts > 1`, retry
     from Step 2.1 with full context and cached MCP evidence for unchanged bindings.
     Abort creation with `controls: []` when any rule exhausts its attempts.

### B. Guided Check Flow (`vatuta policy check "<prompt>"`)

1. **Allocate Context**: Create shared execution `invocation_id` before Step 1.
2. **Step 1 (`check_rule_selection`)**: The model queries `ClientControlQuery` (default 3 queries, 4096 bytes per
   response). For this invocation, retain the complete `{file: str, id: str}` pairs shown in query responses.
   - `file` is the path relative to `controls_directory` returned by the query. `id` is the corresponding control ID.
   - The model confirms `selected_controls: list[{file: str, id: str}]` with `done=True`. No confirmation returns
     `CONTROL_SELECTION_INCOMPLETE` (Exit `1`). A confirmed empty list returns `NO_CONTROLS_MATCHED` (Exit `2`).
3. **Deterministic Load**: Reuse the local loader on exactly the confirmed paths. Keep the path associated with its
   parsed control until verifying the pair against query history and the current file ID. An unknown pair or ID mismatch
   returns `INVALID_CONTROL_SELECTION` (Exit `1`). Check the verified selection for duplicate IDs and compute
   `merged_parameters` only after verification. Send the full controls inline for execution.
4. **Step 2 (`check_parameter_extraction`)**: Skip if `merged_parameters == []`; else extract explicit prompt values.
5. **Step 3**: Call `PolicyService.execute` under the same `invocation_id`.

A selection failure returns empty results and the pre-execution trace ID. It skips Step 2 and Step 3.

### C. Execution Flow (`PolicyService.execute`)

1. **Stage E0 (Whole-Request Pre-Flight)**: Check non-empty unique `controls`, shared parameter compatibility, required
   parameters present (`MISSING_PARAMETER`), and `value_schema` validation (`INVALID_PARAMETER_VALUE`).
2. **Stage E1 (Per-Control Evidence)**: Reuse started MCP containers and exact `(server, tool, resolved_arguments)`
   cache across controls. Validate profile, allowlist, schemas, response JSON, and `complete_when == True`. Fast-fail
   only the affected control on error.
3. **Stage E2 (Per-Control Selection)**: Run JSONata in existence envelope. Empty -> `NOT_APPLICABLE`; root `null` ->
   `ERROR`. Validate non-whitespace `id`/`descriptive_id`. Unique valid `id`s proceed to Stage E3; `INVALID_SUBJECT` and
   `DUPLICATE_SUBJECT_ID` skip Stage E3, record errors, and set control `outcome="ERROR"`.
4. **Stage E3 (Per-Subject Criterion)**: Run CEL (`deterministic`) or singleton LLM (`llm` with `{{parameters.<name>}}`
   substitution).
5. **Stage E4 (Filtering & Trace)**: Keep only `NON_COMPLIANT` in `subjects` when `result_filter="violations"` (or all
   when `"all"`); always keep all `errors`. Save trace; on write failure append global `TRACE_WRITE_FAILED` and exit
   `1` while preserving `results`.

## 9. PolicyConfig (`config/vatuta.yaml`)

- `controls_directory: str = "data/controls"`
- `creation_backend: Optional[str] = None` (defaults to generator binding)
- `evaluation_backend: Optional[str] = None` (defaults to generator binding)
- `trial_attempts: int = 1`
- `discovery_max_queries: int = 5`
- `discovery_max_output_bytes: int = 4096`
- `check_max_filter_queries: int = 3`
- `check_max_output_bytes: int = 4096`
- `trace_directory: str = "logs/policy"`
- `evidence_profiles: dict[str, EvidenceProfile] = {}`
