# Policy Operations: Stateless Service, Future MCP, and Client Contract

**Date**: 2026-11-05
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

1. **Policy Decomposition into Atomic Rules**:
   - Split `policy_text` into atomic normative obligations. Two obligations MUST be split into separate rules whenever
     they can fail independently and a user wants to know which condition failed, or whenever the policy author would
     consider them two distinct violations.
2. **Sequential Rule Construction with `SharedParameterCatalog`**:
   - Construct each rule one by one, passing the cumulative `SharedParameterCatalog` from earlier rules of the same
     policy so conceptually identical inputs reuse the exact same parameter name and compatible `value_schema`.
   - **Parameters & Evidence**: Deduce parameters, `construction_value`s, optional explicit `default`s, and `evidence`
     declarations (`server`, `tool`, `arguments`).
   - **Scalable Source Exploration (when `trial_attempts > 0`)**: Require all needed `construction_value` inputs (or
     return `MISSING_CONSTRUCTION_CONTEXT`), fetch the declared evidence once in memory, and allow the constructor to
     execute up to configurable `policy.discovery_max_queries` (default **5**) exploratory JSONata queries against the
     in-memory evidence, truncating each query output to configurable `policy.discovery_max_output_bytes` (default
     **4096** bytes, appending `[TRUNCATED]` if exceeded).
   - **Selection & `selection_tests`**: Generate `test_cases.selection_tests` (`mock_evidence` -> `expected_subjects`
     or `NOT_APPLICABLE`, with at least one populated subject case and one `NOT_APPLICABLE` case) and the JSONata
     `selection` expression. Execute `selection` against all `selection_tests` (always mandatory, even when
     `trial_attempts == 0`) and against the in-memory live evidence (when `trial_attempts > 0`) to verify the resulting
     `Subject` objects.
   - **Criterion & `criterion_tests`**: Generate `test_cases.criterion_tests` (synthetic `subject` objects with at
     least one `COMPLIANT` and one `NON_COMPLIANT` expected state) and the `criterion` (`deterministic` CEL or
     justified `llm`). Execute `criterion` against all `criterion_tests` (always mandatory) and against the live trial
     subjects (when `trial_attempts > 0`).
   - **Persistence**: Include `test_cases` (`selection_tests` and `criterion_tests`) in the returned
     `ControlDefinition`.

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
atomically.

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
| `errors` | array of ErrorDetail | Required | Request errors; runtime errors stay with their control results. |

Each `ControlResult` in `results` has these fields:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `control_id` | string | Required | Identifier (`id`) declared by the executed `ControlDefinition`. |
| `evaluation_mode` | string | Required | Declared criterion mode: `deterministic` or `llm`. |
| `subjects` | array of SubjectResult | Required | Subject results retained by `result_filter`; may be empty. |
| `errors` | array of ErrorDetail | Required | All control/subject errors, retained under either result filter. |
| `outcome` | string | Optional | `ERROR` or `NOT_APPLICABLE`; absent for error-free nonempty evaluations. |

Each `SubjectResult` has these fields:

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `id` | string | Required | Original nonempty technical identifier of the selected subject. |
| `descriptive_id` | string | Required | Original nonempty human-readable identifier of the selected subject. |
| `state` | string | Required | `COMPLIANT`, `NON_COMPLIANT`, or `ERROR` for this subject's evaluation. |

### Behavior

1. Validate that `controls` is nonempty, `id` values are unique within the request, control structures and shared
   parameter declarations are compatible, and explicit values/defaults satisfy schemas and locally configured scope and
   permissions. Missing values across all controls produce one request-error result with all names, semantics,
   expected schemas, and affected `control_id`s. Zero evidence calls and zero subject evaluations occur.
2. Resolve source connections and discovered tool schemas per control only after the complete request passes local
   validation. A connection failure, unavailable tool, or invalid response affects only that control.
3. Evaluate all valid subjects under each declared criterion. Preserve subject errors and valid results. Filter
   findings only after evaluation. No automatic retries, evidence pagination, policy reanalysis, or runtime mode
   switching.

## Client-Side Filtering, Projection, Parameter Merging, and Guided 2-Step Check Flow

Because `PolicyService` is stateless, the Vatuta client manages local YAML control files under
`policy.controls_directory` (default `"data/controls"`, overridable via `--controls-dir`) and provides a local
filtering and projection tool used by both the `vatuta policy check "<prompt>"` pipeline and the direct CLI.

### Input (`ClientControlQuery`)

| Field | Type | Required / default | Meaning |
| --- | --- | --- | --- |
| `ids` | array of strings | Optional; `None` | Match controls whose `id` is in this list. |
| `policy_ids` | array of strings | Optional; `None` | Match controls whose `source.policy` is in this list. |
| `rule_ids` | array of strings | Optional; `None` | Match controls whose `source.rule` is in this list. |
| `tags` | array of strings | Optional; `None` | Match controls whose `tags` intersect/contain these tags. |
| `evaluation_mode` | string | Optional; `None` | `"deterministic"` or `"llm"`. |
| `evidence_server` | string | Optional; `None` | Match controls using this MCP server in `evidence`. |
| `evidence_tool` | string | Optional; `None` | Match controls using this MCP tool in `evidence`. |
| `parameter_names` | array of strings | Optional; `None` | Match controls declaring these parameter names. |
| `text_query` | string | Optional; `None` | Case-insensitive substring search in `description` and `source.excerpt`. |
| `files` | array of strings | Optional; `None` | Specific YAML file paths within `controls_directory`. |
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

   Executes a deterministic 2-step LLM flow on the client (separate from `vatuta ask`):
   - **Step 1 (Rule Selection)**: `PolicyCheckRuleSelectionSignature` inspects the prompt and calls
     `ClientControlQuery` with `projection=["id", "description", "tags", "source"]` and
     `include_merged_parameters=False`, bounded by configurable `policy.check_max_filter_queries` (default **3**
     queries) and output truncation `policy.check_max_output_bytes` (default **4096** bytes, appending `[TRUNCATED]` if
     exceeded), to select the applicable `control_id`s.
   - **Deterministic Load**: The client calls `ClientControlQuery(ids=selected_ids, projection=["full"],
     include_merged_parameters=True)` to load the full `ControlDefinition` objects and compute `merged_parameters`.
   - **Step 2 (Parameter Deduction)**: `PolicyCheckParameterExtractionSignature` receives the user's prompt and
     `merged_parameters` and extracts explicit JSON parameter values present in the prompt (without inventing missing
     values).
   - **Step 3 (Stateless Execution)**: The client invokes `PolicyService.execute(controls=selected_controls,
     parameters=extracted_parameters, result_filter=...)` and renders the results and errors.

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
creation/runtime/client persistence errors, `2` for invalid requests, empty filter selections, or configuration errors.
JSON mode writes exactly one result object to stdout; progress and logging go to stderr.
