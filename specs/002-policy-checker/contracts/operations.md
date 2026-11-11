# Policy Operations: Stateless Service, Future MCP, and Client Contract

**Date**: 2026-11-11
**Status**: Planned interface; commands and service classes are not implemented yet

`PolicyService` (`src/policy/service.py`) defines the public stateless operation contracts using JSON-serializable
Pydantic v2 request and result models with `Field(description=...)` metadata. These models map 1:1 to the two MCP
domain tools (`policy_create` and `policy_execute`) plus trace resource reads (`read_trace` / `resources/read`). In
this increment, the Vatuta client invokes `PolicyService` directly; a stdio MCP transport command (`vatuta policy
serve`) is deferred to a follow-up increment.

`PolicyService` manages no control directory, `--root`, file selectors, or rule repository. Local YAML control storage
(`policy.controls_directory`), control filtering, attribute projection, shared parameter merging
(`include_merged_parameters=True` by default), and the guided 2-step prompt-driven check flow (`vatuta policy check
"<prompt>"`) belong to the Vatuta client.

## Create — `PolicyService.create` (`policy_create`)

Decompose a policy into atomic normative rules and construct the complete set of executable `ControlDefinition` objects
one by one with an incremental `SharedParameterCatalog`, bounded in-memory JSONata evidence exploration, mandatory
synthetic unit tests (`selection_tests` and `criterion_tests`), and configured live trials. Creation returns reusable
controls only when the complete set succeeds.

### Input (`PolicyCreateRequest`)

The request has only the following JSON-serializable fields; unknown fields are rejected (`extra='forbid'`):

| Field | Type | Required / default | Meaning |
| --- | --- | --- | --- |
| `policy_text` | string | Required | Complete nonempty policy in plain text or Markdown. |
| `policy_id` | string | Optional; `invocation_id` | Nonempty identifier used for policy provenance. |
| `construction_context` | string | Optional; `""` | Descriptive information for generation, tag inference, and trials. |
| `tags` | array of strings | Optional; `[]` | Explicit nonempty tags merged into every generated control. |
| `trial_attempts` | integer | Optional; deployment setting | Maximum live trial attempts for this creation; `>= 0`. |

Example request payload:

```json
{
  "policy_text": "Every pull request must reference an approved issue.",
  "policy_id": "change-management",
  "construction_context": "Build controls for acme/api. Issues are tracked in Jira project PAY.",
  "tags": ["compliance", "q4"],
  "trial_attempts": 1
}
```

### Output (`PolicyCreateResult`)

Return `{invocation_id, controls, errors}`:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `invocation_id` | string | Required | Engine-assigned identifier of this creation invocation. |
| `controls` | array of ControlDefinition | Required | Complete generated controls on success; `[]` on failure. |
| `errors` | array of ErrorDetail | Required | Creation errors; `[]` on success. See Error Mapping. |

Each element of `controls` is a full structured `ControlDefinition` object (`id`, `description`, `tags`, `source`,
`parameters`, `evidence`, `selection`, `criterion`, `test_cases`, `creation_check`) as defined in
[control-format.md](control-format.md) and [data-model.md](../data-model.md).

### Step-by-Step Creation Behavior

1. **Step 0 — Pre-flight & Tool Catalog**:
   - Validate nonempty `policy_text`. Resolve `policy_id` (defaults to `invocation_id`) and `trial_attempts` (defaults
     to `policy.trial_attempts`, which defaults to `1`).
   - For every trial mode, build the tool catalog exclusively from local `policy.evidence_profiles`
     (`arguments_schema` and `contract_reference`). Do not start containers, connect to MCP, or call `list_tools`
     before Phase A succeeds. With `trial_attempts == 0`, no MCP connection occurs anywhere in creation.
2. **Step 1 — Policy Decomposition into Atomic Rules (`PolicyDecompositionSignature`)**:
   - Split `policy_text` into atomic normative obligations, ignoring non-normative preamble/scope prose. Two obligations
     MUST be split into separate rules whenever they can fail independently and a user wants to know which condition
     failed, or whenever the policy author would consider them two distinct violations.
   - If zero normative obligations are extracted from `policy_text`, fail immediately with `NO_NORMATIVE_RULES`
     (`stage: "decomposition"`).
   - Assign each control its deterministic kebab-case `id`: `f"{policy_id}-{rule_id}"` (or `rule_id` directly when
     `rule_id` already starts with `f"{policy_id}-"`, deduplicated within the creation output).
3. **Step 2, Phase A — All-Rules Parameter & Evidence Sweep (Step 2.1 across `r_1 ... r_N`)**:
   - Execute `RuleEvidenceAndParametersSignature` sequentially across all decomposed rules `r_1 ... r_N`, declaring only
     the minimum necessary parameters (`parameters: {}` when none are needed) and passing and updating the cumulative
     `SharedParameterCatalog`.
   - **`SharedParameterCatalog` Compatibility Rule**: Two parameters across rules of the same policy share the same
     parameter name **only** when their `value_schema`, `construction_value`, and `default` are identical/compatible. If
     any of those three differ despite similar semantics, it is effectively a distinct parameter: the constructor must
     either adapt its declaration to be compatible with the catalog or use a distinct parameter name.
   - Declarations already referenced by other rules are immutable throughout Phase B and all retries. Keep changes
     local to the current candidate until validated; discard failed-attempt additions. Adapt the current rule or
     choose a distinct name instead of invalidating another rule's declaration.
   - When `trial_attempts == 0`, `construction_value` may be omitted; explicit null still requires schema validation.
   - When `trial_attempts > 0`, inspect all rules after Phase A: if any required parameter (`default` absent) lacks a
     `construction_value`, aggregate and return **all** `MISSING_CONSTRUCTION_CONTEXT` errors and affected `control_id`s
     across all rules in a single response before calling any MCP server.
4. **Step 2, Phase B — Per-Rule Deep Construction (`r_1 ... r_N`) with Fail-Fast**:
   - Process each rule `r_i` through Steps 2.2–2.5; if any rule fails definitively after exhausting its configured
     attempts, abort creation immediately (**Fail-Fast**) without executing Phase B for subsequent rules.
   - **Step 2.2 — Scalable Source Exploration (when `trial_attempts > 0`)**: After Phase A succeeds, connect to the
     required MCP servers, discover tool `inputSchema`, and validate arguments before evidence calls.
     Fetch declared evidence once in memory
     (cached per rule by `(server, tool, resolved_arguments)`) and execute up to configurable
     `policy.discovery_max_queries` (default **5**) exploratory JSONata queries evaluated over the root context
     `{"evidence": ..., "parameters": ...}`, truncating each output to `policy.discovery_max_output_bytes` (default
     **4096** bytes, appending `[TRUNCATED]` if exceeded). Query syntax/runtime errors or `missing` outcomes are
     returned as observations within the query budget so the model can refine the next query; the loop closes early as
     soon as `done=True`.
   - **Step 2.3 — Selection & `selection_tests` (`RuleSelectionSignature`)**: May declare additional parameters
     (validated against `SharedParameterCatalog`) and generates `test_cases.selection_tests` (at least one populated
     `expected_subjects` case and one `NOT_APPLICABLE` case) plus the JSONata `selection` expression. Normalize single
     selected JSON objects `{...}` to `[{...}]`, require non-whitespace `id` and `descriptive_id` (`len(s.strip()) >
     0`), reject duplicate subject `id`s with `DUPLICATE_SUBJECT_ID`, resolve unit-test parameters with priority `1.
     test_case.parameters` -> `2. default` -> `3. construction_value`, and execute `selection` against all
     `selection_tests` (always mandatory, even when `trial_attempts == 0`) and against in-memory live evidence (when
     `trial_attempts > 0`).
   - **Step 2.4 — Criterion & `criterion_tests` (`RuleCriterionSignature`)**: May declare additional parameters
     (validated against `SharedParameterCatalog`) and generates `test_cases.criterion_tests` (at least one `COMPLIANT`
     and one `NON_COMPLIANT` synthetic `subject`) plus `criterion` (`deterministic` CEL or justified `llm`). Execute
     `criterion` against all `criterion_tests` (always mandatory, using CEL for `mode: deterministic` and calling
     `SemanticCriterionSignature` on the LLM with `{{parameters.<name>}}` template substitution in `condition` for
     `mode: llm`) and against live trial subjects (when `trial_attempts > 0`).
   - **Step 2.5 — Full-Context Bounded Repair & Persistence**: When `trial_attempts > 1` and a validation, unit test, or
     live trial fails in Steps 2.2–2.4 (including a newly added parameter in 2.3/2.4 lacking `default` and
     `construction_value`), re-execute from Step 2.1 carrying the full accumulated context and errors from all prior
     stages, reusing cached in-memory MCP evidence whenever `(server, tool, resolved_arguments)` are unchanged. On
     success, persist `test_cases` in the returned `ControlDefinition`.

- Success: complete `controls` array of `ControlDefinition` objects; `errors` is empty.
- Failure: `controls` is empty and `errors` explains failed rules/controls/inputs. Diagnostic candidates may exist only
  in the sanitized trace. Never present passing controls from a failed creation as a reusable partial result.

### CLI Mapping (`vatuta policy create`)

```bash
poetry run vatuta policy create --policy-file policy.md --policy-id change-management \
  --construction-context 'Build controls for acme/api; issues are tracked in Jira project PAY.' \
  --tag compliance --tag q4 --trial-attempts 1 --output-dir data/controls --format json
poetry run vatuta policy create --policy-text 'Every change must reference an approved issue.' \
  --construction-context 'Build controls for acme/api; issues are tracked in Jira project PAY.' --format json
```

When saving controls (via `--output-dir` or configured `policy.controls_directory`), the client serializes each
returned `ControlDefinition` (including `tags` and `test_cases`) as `<id>.yaml` in a staging directory and writes them
without replacing existing files. Validate target containment and check the complete output set for collisions before
publishing any file. If a target already exists, return `CONTROL_FILE_EXISTS` at stage `client_persistence` with CLI
exit `1`; preserve existing files and publish none of the proposed set. Publication must also reject a destination
created after the initial check. No overwrite option or automatic renaming is provided. File persistence failures
belong to the client response, not to the stateless service.

## Execute — `PolicyService.execute` (`policy_execute`)

Execute the supplied `ControlDefinition` objects inline with explicit invocation values and declared defaults,
preserving independent results and errors per `control_id` and subject.

### Input (`PolicyExecuteRequest`)

Request: `{controls: list[ControlDefinition], parameters: dict[str, Any], result_filter: "violations" | "all"}`.

| Field | Type | Required / default | Meaning |
| --- | --- | --- | --- |
| `controls` | array of ControlDefinition | Required | Nonempty list of inline `ControlDefinition` objects with unique `id`s. |
| `parameters` | object | Optional; `{}` | Map from declared parameter names to explicit JSON invocation values. |
| `result_filter` | string | Optional; `"violations"` | `"violations"` or `"all"`; always retain errors. |

`PolicyService.execute` accepts structured `ControlDefinition` objects directly (not file paths or globs). Explicit
values override declared defaults; construction values do not fill missing inputs. No execution prompt or
natural-language parameter inference is accepted by `PolicyService.execute`.

### Output (`PolicyExecuteResult`)

Return `{invocation_id, results, errors}`:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `invocation_id` | string | Required | Engine-assigned identifier of this execution invocation. |
| `results` | array of ControlResult | Required | Per-control results; `[]` when request validation fails. |
| `errors` | array of ErrorDetail | Required | Request or invocation-level diagnostic errors. |

Each `ControlResult` in `results` has these fields:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `control_id` | string | Required | Identifier (`id`) declared by the executed `ControlDefinition`. |
| `source` | object | Required | Executed definition's provenance: `{policy, rule, excerpt}`. |
| `parameters` | object | Required | Effective resolved invocation values declared by this control. |
| `evaluation_mode` | string | Required | Declared criterion mode: `deterministic` or `llm`. |
| `subjects` | array of SubjectResult | Required | `NON_COMPLIANT` subjects when `result_filter="violations"`; all evaluated subjects (`COMPLIANT`, `NON_COMPLIANT`, `ERROR`) when `result_filter="all"`. |
| `errors` | array of ErrorDetail | Required | All control/subject errors, retained under either result filter. |
| `outcome` | string | Optional | `ERROR` or `NOT_APPLICABLE`; absent for error-free nonempty evaluations. |

Every `ControlResult`, including `ERROR` and `NOT_APPLICABLE`, includes `source` and `parameters` under both filters.
Explicit invocation values override defaults. Exclude construction-only values, unused request keys, and parameters
belonging only to other controls. Preserve this attribution in the trace too; trace lookup is not required to obtain it.
Apply existing credential/personal-data redaction at presentation boundaries. Request validation failures return no
`ControlResult`, and their errors identify affected controls where known. Control runtime errors remain in their
`ControlResult.errors`; top-level `errors` also permits invocation-level diagnostic failures.

Each `SubjectResult` has these fields:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `id` | string | Required | Original non-empty, non-whitespace technical identifier of the selected subject. |
| `descriptive_id` | string | Required | Original non-empty, non-whitespace human-readable identifier of the subject. |
| `state` | string | Required | `COMPLIANT`, `NON_COMPLIANT`, or `ERROR` for this subject's evaluation. |

### Behavior

1. **Pre-Flight Request Validation (`Stage E0`)**: Validate that `controls` is nonempty, `id` values are unique within
   the request, shared parameter declarations across supplied controls are compatible (`value_schema` and `default`),
   and explicit values/defaults satisfy declared `value_schema`s. Missing required values across all controls produce
   one request-error result (`MISSING_PARAMETER`) with all names, semantics, expected schemas, and affected
   `control_id`s; zero evidence calls and zero subject evaluations occur. Extra keys in `request.parameters` not
   declared by any supplied control are allowed and ignored at request level; each control receives only the subset of
   resolved parameters declared in its own `control.parameters`.
2. **Per-Control Evidence Acquisition & Completeness (`Stage E1`)**: Keep started `MCPServer` containers alive across
   controls for the duration of the `execute` invocation and cache valid responses in memory by exact `(server, tool,
   resolved_arguments)`. For each control, validate `evidence_profile` existence, tool allowlists, `arguments_schema`,
   and discovered `inputSchema` per control. If any evidence declaration in a control fails validation, connection,
   conversion, or `complete_when == True`, **fast-fail that control immediately** (`outcome="ERROR"`, skipping its
   remaining evidence declarations, `selection`, and `criterion`) while independent controls continue.
3. **Selection & Criterion Evaluation (`Stages E2 & E3`)**: Normalize a single selected JSON object `{...}` to
   `[{...}]` and require non-empty, non-whitespace `id` and `descriptive_id` (`len(s.strip()) > 0`). Evaluate subjects
   whose `id` is unique within the control's selection under the declared criterion (`deterministic` CEL with `{subject,
   parameters}`, or singleton `llm` `SemanticCriterionSignature` with `{{parameters.<name>}}` template substitution in
   `condition`). Invalid subject objects (`INVALID_SUBJECT`) and subjects sharing a duplicate `id`
   (`DUPLICATE_SUBJECT_ID`, with `subject_id` and `selection_index`) are not evaluated by `criterion`, are reported in
   `ControlResult.errors`, and set the control's `outcome` to `"ERROR"`.
4. **Result Filtering (`Stage E4`)**: Under `result_filter="violations"` (default), `ControlResult.subjects` contains
   only `NON_COMPLIANT` findings; under `result_filter="all"`, `ControlResult.subjects` contains all evaluated subjects
   (`COMPLIANT`, `NON_COMPLIANT`, and `ERROR`). `ControlResult.errors` retains all control and subject errors under
   both filters. No automatic retries, evidence pagination, policy reanalysis, or runtime mode switching.
5. **Trace Finalization**: If writing the execution trace fails, preserve all calculated `results` and append
   `ErrorDetail(code="TRACE_WRITE_FAILED", stage="trace", message=...)` to top-level `errors`; return CLI exit `1`.
   This diagnostic failure alone changes neither subject states nor a control's `outcome`.

## Client-Side Filtering, Projection, Parameter Merging, and Guided 2-Step Check Flow

Because `PolicyService` is stateless, the Vatuta client manages local YAML control files under
`policy.controls_directory` (default `"data/controls"`, overridable via `--controls-dir`) and provides a local
filtering and projection tool used by both the `vatuta policy check "<prompt>"` pipeline and the direct CLI. Distinct
filter fields combine with logical **AND**; multiple values within a list field (`ids`, `policy_ids`, `rule_ids`,
`tags`, `parameter_names`, `files`) combine with logical **OR**. If any encountered YAML control file is malformed or
fails schema validation, `ClientControlQuery` fails the call immediately with `INVALID_CONTROL_FILE`.

### Input (`ClientControlQuery`)

| Field | Type | Required / default | Meaning |
| --- | --- | --- | --- |
| `ids` | array of strings | Optional; `None` | Match controls whose `id` is in this list (OR). |
| `policy_ids` | array of strings | Optional; `None` | Match controls whose `source.policy` is in this list (OR). |
| `rule_ids` | array of strings | Optional; `None` | Match controls whose `source.rule` is in this list (OR). |
| `tags` | array of strings | Optional; `None` | Match controls whose `tags` contain any of these tags (OR). |
| `evaluation_mode` | string | Optional; `None` | `"deterministic"` or `"llm"`. |
| `evidence_server` | string | Optional; `None` | Match controls using this MCP server in `evidence`. |
| `evidence_tool` | string | Optional; `None` | Match controls using this MCP tool in `evidence`. |
| `parameter_names` | array of strings | Optional; `None` | Match controls declaring any of these parameter names (OR). |
| `text_query` | string | Optional; `None` | Case-insensitive substring search in `description` and `source.excerpt`. |
| `files` | array of strings | Optional; `None` | Specific YAML file paths within `controls_directory` (OR). |
| `glob` | string | Optional; `None` | Glob pattern expanded safely within `controls_directory`. |
| `projection` | array of strings | Optional; `["id", "description", "tags", "source", "parameters"]` | Attributes to return per control, or `["full"]` for full `ControlDefinition`. |
| `include_merged_parameters` | boolean | Optional; `True` | When `True` (default), compute and validate shared `merged_parameters`. |

### Output (`ClientControlQueryResult`)

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `controls` | array of objects | Required | Matched controls projected to `projection` (or full `ControlDefinition`s). |
| `merged_parameters` | array of MergedParameter | When `include_merged_parameters=True` | Shared parameter descriptions across matched controls. |
| `errors` | array of ErrorDetail | Required | File, schema, empty-match, or parameter-conflict errors; `[]` on success. |

Each `MergedParameter` in `merged_parameters` has these fields:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `name` | string | Required | Shared parameter name to use as a key in execution's `parameters` object. |
| `value_schema` | JSON Schema | Required | Self-contained JSON Schema 2020-12 for accepted invocation values. |
| `required` | boolean | Required | True when there is no declared default, regardless of any construction value. |
| `default` | JSON value | Optional | Declared invocation default; an explicit null must satisfy the schema. |
| `controls` | array of objects | Required | `[{control_id, description}]` for each referencing control. |

### Client CLI Subcommands (`vatuta policy ...`)

1. **Direct Execution (`vatuta policy execute`)**:

   ```bash
   poetry run vatuta policy execute --controls-dir data/controls --policy-id change-management \
     --tag compliance --parameters @values.json --result-filter all --format json
   poetry run vatuta policy execute --control data/controls/a.yaml --parameters '{"repository": "acme/api"}'
   ```

   Uses the client filtering utility (`--controls-dir`, `--id`, `--policy-id`, `--rule-id`, `--tag`, `--mode`,
   `--control`, `--glob`) to load matching `ControlDefinition` objects from disk and passes them inline to
   `PolicyService.execute`.

2. **Guided 2-Step Check Flow (`vatuta policy check "<prompt>"`)**:

   ```bash
   poetry run vatuta policy check "Verify change-management compliance for repository acme/api and Jira project PAY"
   ```

   Executes a deterministic 2-step LLM flow on the client (separate from `vatuta ask`). Before Step 1, an internal
   invocation context allocates the execution `invocation_id` and collects stage artifacts. Client orchestration and
   `execute` share this context without adding public request fields or mutable global state. Direct execution
   allocates its own context. All stages use `vatuta://traces/policy/execution/{invocation_id}`:
   - **Step 1 (Iterative Rule Selection — `check_rule_selection`)**: `PolicyCheckRuleSelectionSignature` inspects the
     prompt and iteratively calls `ClientControlQuery` with `projection=["id", "description", "tags", "source"]` and
     `include_merged_parameters=False`, bounded by configurable `policy.check_max_filter_queries` (default **3**
     queries) and output truncation `policy.check_max_output_bytes` (default **4096** bytes, appending `[TRUNCATED]` if
     exceeded, exiting early when `done=True`), to select the applicable `control_id`s. If `selected_control_ids` is
     empty (`[]`), fail immediately with `NO_CONTROLS_MATCHED` (exit code `2`) without calling `PolicyService.execute`.
   - **Deterministic Load**: The client calls `ClientControlQuery(ids=selected_ids, projection=["full"],
     include_merged_parameters=True)` to load the full `ControlDefinition` objects and compute `merged_parameters`.
   - **Step 2 (Parameter Deduction — `check_parameter_extraction`)**: When `merged_parameters` is empty (`[]`), skip the
     Step 2 LLM call completely (`extracted_parameters = {}`). Otherwise, `PolicyCheckParameterExtractionSignature`
     receives the user's prompt and `merged_parameters` and extracts explicit JSON parameter values present in the
     prompt (without inventing missing values).
   - **Step 3 (Stateless Execution)**: The client invokes `PolicyService.execute(controls=selected_controls,
     parameters=extracted_parameters, result_filter=...)` under the same `invocation_id` and renders the results and
     errors.

If a check stage fails before `execute`, return `{invocation_id, results: [], errors}` and attempt to finalize the
same context's trace without invoking execution. Input/configuration errors use exit `2`; technical model or trace
failures use exit `1`. A trace failure is appended to existing errors and takes exit-code precedence, preserving any
already computed results. Creation retains its complete-or-error contract if a required creation trace cannot be saved.

## Diagnostic Trace Resources

The client derives the trace URI from the operation it invoked and the returned `invocation_id`; no additional input
or output field is needed. Use these resource URI templates:

| Operation | Trace resource URI |
| --- | --- |
| `PolicyService.create` (`policy_create`) | `vatuta://traces/policy/construction/{invocation_id}` |
| `PolicyService.execute` (`policy_execute`) | `vatuta://traces/policy/execution/{invocation_id}` |

`PolicyService.read_trace(uri: str)` resolves the URI and returns the sanitized `InvocationTrace` as JSON
(`application/json`), including stage records, executed definitions, results/errors, measurements, and model
interactions.

## Error Mapping

All `errors` arrays contain `ErrorDetail` objects with these fields:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `code` | string | Required | Stable machine-readable identifier of the failure. |
| `stage` | string | Required | Processing stage where the failure was detected. |
| `message` | string | Required | Actionable human-readable explanation after privacy filtering. |
| `control_id` | string | When known | Identifier (`id`) of the affected control. |
| `parameter` | string | When applicable | Parameter with missing, invalid, or conflicting data or declarations. |
| `subject_id` | string | When known | Original `id` of the subject whose processing failed. |
| `descriptive_id` | string | When known | Original human-readable identifier of the affected subject. |
| `selection_index` | integer | When applicable | Zero-based array index of a malformed selected item. |

CLI exit codes: `0` for an operation without technical errors (including violations and NOT_APPLICABLE), `1` for
creation/runtime/client persistence errors or `TRACE_WRITE_FAILED`, `2` for invalid requests, empty filter selections,
or configuration errors. A global trace error has `stage="trace"` and does not require `control_id`.
JSON mode writes exactly one result object to stdout; progress and logging go to stderr.
