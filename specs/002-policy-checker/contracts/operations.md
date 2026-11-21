# Policy Operations: Service, CLI, Agent Skill & Future MCP Contract (Schematic)

**Date**: 2026-11-20
**Status**: Planned interface (stateless service, CLI commands, and Agent Skill)

## Architecture Summary

- **Stateless `PolicyService` (`src/policy/service.py`)**:
  - Uses JSON-serializable Pydantic v2 request/result models with `Field(description=...)`.
  - Ready 1:1 for future MCP tools (`policy_create`, `policy_execute`) and `resources/read` (`read_trace`).
  - Manages zero `.yaml` files, directories, or globs.
  - Note: MCP server transport (`vatuta policy serve`) is deferred.
- **Client Layer (`src/client/policy.py`, `src/policy/files.py`, `src/policy/parameters.py`)**:
  - Manages `<id>.yaml` control files (1 file per rule) in a nested directory tree under `policy.controls_directory`
    (default `"data/controls"`, overridable via `--controls-dir` or `--output-dir`).
  - Exposes 4 CLI subcommands (`vatuta policy query`, `create`, `execute`, `check`) and 3 Agent Skill scripts
    (`skills/vatuta-policy/scripts/{query,create,execute}.py` with PEP 723 metadata) sharing the exact client handlers.

---

## LLM Input Boundaries

Keep engine-authored DSPy instructions separate from structured input data in the provider request. Validate MCP
structured responses and parsed JSON text as JSON data with string keys and finite numbers. Apply configured
tool validation and completeness checks. Parse rule YAML with a safe loader and validate the data against
the strict `ControlDefinition` model.

| Stage | Data passed to the LLM |
| --- | --- |
| Creation and refinement | Policy text, rule context, validated tool metadata, and bounded evidence projections. |
| Guided selection | User prompt and bounded rule metadata. Track exact candidate pairs delivered to the model. |
| Guided parameter extraction | User prompt and merged declarations from confirmed and reloaded controls. |
| Semantic evaluation | Rendered condition, correlation token, and values selected by `subject_paths`. |

Strip scripts from external HTML in a copy of the projected input. Keep plain text as data and verify generated
policy excerpts against the submitted policy text. Validate model outputs before authorizing any tool call.

## 1. Create — `PolicyService.create` (`policy_create`)

Save the creation trace before returning generated controls. On write failure, return the same `invocation_id`,
`controls: []`, and global `TRACE_WRITE_FAILED`, retaining earlier errors. The CLI and Skill exit with code `1`
before writing any generated YAML.

Required MCP server unavailability during discovery or live trials stops creation before repair. Return a global
`MCP_SERVER_UNAVAILABLE` error and `controls: []`. The CLI and Skill exit with code `1` and save no generated YAML.

Split a policy into atomic rules and build the complete list of `ControlDefinition` objects step by step with
`SharedParameterCatalog`, bounded JSONata exploration, mandatory `selection_tests` and `criterion_tests`, and live
trials. Returns controls only when all rules succeed (`controls: []` on any failure).

### Input (`PolicyCreateRequest`, `extra='forbid'`)

| Field | Type | Required / Default | Meaning |
| --- | --- | --- | --- |
| `policy_text` | `string` | Required | Non-empty policy in plain text or Markdown. |
| `policy_id` | `string` | Optional; `invocation_id` | Non-empty identifier for `source.policy` and control slug prefix. |
| `construction_context` | `string` | Optional; `""` | Context for tag inference, parameters, and trial values. |
| `tags` | `array[string]` | Optional; `[]` | Explicit non-empty tags merged into every generated control. |
| `trial_attempts` | `integer` | Optional; `policy.trial_attempts` (`1`) | Max live trial attempts (`>= 0`). |

### Output (`PolicyCreateResult`)

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `invocation_id` | `string` | Required | ID of the invocation that started this creation run. |
| `controls` | `array[ControlDefinition]` | Required | Complete generated controls on success; `[]` on failure. |
| `errors` | `array[ErrorDetail]` | Required | Creation errors; `[]` on success. |

### Step-by-Step Creation Pipeline

1. **Step 0 (Pre-flight & Local Tool Catalog)**:
   - Validate non-empty `policy_text`; resolve `policy_id` and `trial_attempts`.
   - Build tool catalog from local `policy.evidence_profiles` only (no MCP connection in Step 0 or Phase A).
2. **Step 1 (Policy Decomposition — `PolicyDecompositionSignature`)**:
   - Split `policy_text` into atomic normative rules whenever obligations can fail independently or represent separate
     violations.
   - If zero rules are found, fail with `NO_NORMATIVE_RULES` (`stage: "decomposition"`).
   - Assign deterministic slug `id`: `f"{policy_id}-{rule_id}"` (or `rule_id` if already prefixed, deduplicated).
3. **Step 2, Phase A (All-Rules Parameter & Evidence Sweep — Step 2.1 across `r_1..r_N`)**:
   - Run `RuleEvidenceAndParametersSignature` sequentially across all rules with `SharedParameterCatalog`.
   - Declare minimum necessary parameters (`parameters: {}` allowed).
   - Two rules share a parameter name only when `value_schema`, `construction_value`, and `default` match. Shared
     declarations are immutable during Phase B and retries.
   - When `trial_attempts > 0`, if a required parameter lacks both `default` and `construction_value`, collect and
     return **all** `MISSING_CONSTRUCTION_CONTEXT` errors across all rules at once before starting any MCP server.
4. **Step 2, Phase B (Per-Rule Deep Construction `r_1..r_N` with Fail-Fast)**:
   - **Step 2.2 (JSONata Exploration, when `trial_attempts > 0`)**: Connect to MCP, validate live `inputSchema`, fetch
     evidence once in memory (cached by `(server, tool, resolved_arguments)`), and run up to
     `policy.discovery_max_queries` (`5`) JSONata queries over `{"evidence", "parameters"}`, truncating each output to
     `policy.discovery_max_output_bytes` (`4096` bytes + `[TRUNCATED]`). Return query errors/missing within budget; stop
     early on `done=True`.
   - **Step 2.3 (Selection & `selection_tests`)**: Build `selection` and `test_cases.selection_tests` (at least 1
     populated and 1 `NOT_APPLICABLE` case). Normalize `{...}` to `[{...}]`, require non-whitespace
     `id`/`descriptive_id`, reject `DUPLICATE_SUBJECT_ID`, resolve test parameters (`test_case.parameters` -> `default`
     -> `construction_value`), and run `selection_tests` (mandatory) + live selection (`trial_attempts > 0`).
   - **Step 2.4 (Criterion & `criterion_tests`)**: Build `criterion` (`deterministic` CEL or justified `llm`) and
     `test_cases.criterion_tests` (at least 1 `COMPLIANT` and 1 `NON_COMPLIANT` case). Run `criterion_tests`
     (mandatory, via CEL or LLM with `{{parameters.<name>}}` substitution) + live criterion (`trial_attempts > 0`).
   - **Step 2.5 (Full-Context Retry & Fail-Fast)**: If `trial_attempts > 1` and Steps 2.2–2.4 fail, retry from Step 2.1
     carrying full stage context and errors while reusing cached in-memory MCP evidence. Abort immediately (Fail-Fast)
     when any rule exhausts its attempts.

   - Synthetic failures share the per-rule attempt budget with live failures. With `trial_attempts=0`, run the
     mandatory synthetic tests once and skip live MCP. With `trial_attempts=1`, run one attempt without a retry.
     If the final attempt fails, service creation returns `controls: []`.

### Client YAML Saving Behavior (`create`)

- Saves each returned `ControlDefinition` as `<id>.yaml` in `--output-dir` (or `policy.controls_directory`), creating
  missing target subdirectories.
- Pre-checks all target paths for root containment and file collisions before writing any file.
- If any target `<id>.yaml` already exists, aborts with `CONTROL_FILE_EXISTS` (`stage: "client_persistence"`, Exit `1`)
  and writes zero files. No overwrite flag or auto-renaming.
- If a write fails after saving starts, stop the batch and keep all complete files already written. Return
  `CONTROL_FILE_WRITE_FAILED` at `client_persistence` (exit `1`) with `unsaved_paths` for the failed and
  unattempted targets, relative to the output directory. A failed write must leave no partial target YAML.

---

## 2. Execute — `PolicyService.execute` (`policy_execute`)

Execute inline `ControlDefinition` objects with explicit parameters and declared defaults.

### Input (`PolicyExecuteRequest`, `extra='forbid'`)

| Field | Type | Required / Default | Meaning |
| --- | --- | --- | --- |
| `controls` | `array[ControlDefinition]` | Required | Non-empty list of inline controls with unique `id`s. |
| `parameters` | `object` | Optional; `{}` | Map of parameter name -> explicit JSON invocation value. |
| `result_filter` | `string` | Optional; `"violations"` | `"violations"` or `"all"`; errors are always retained. |

### Output (`PolicyExecuteResult`)

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `invocation_id` | `string` | Required | ID of the invocation that started this execution run. |
| `results` | `array[ControlResult]` | Required | Per-control results (`[]` when Stage E0 pre-flight fails). |
| `errors` | `array[ErrorDetail]` | Required | Request-level or invocation-level errors (`[]` on clean run). |

#### `ControlResult` Fields

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `control_id` | `string` | Required | `id` of the executed `ControlDefinition`. |
| `source` | `object` | Required | `{policy, rule, excerpt}` provenance (always included). |
| `parameters` | `object` | Required | Effective resolved parameters for this control only (always included). |
| `evaluation_mode` | `string` | Required | `"deterministic"` or `"llm"`. |
| `subjects` | `array[SubjectResult]` | Required | `NON_COMPLIANT` only when `"violations"`; all evaluated when `"all"`. |
| `errors` | `array[ErrorDetail]` | Required | All control and subject errors (retained under both filters). |
| `outcome` | `string` | Optional | `"ERROR"` or `"NOT_APPLICABLE"`; omitted on error-free non-empty runs. |

#### `SubjectResult` Fields

| Field | Type | Presence | Meaning |
| --- | --- | --- | --- |
| `id` | `string` | Required | Non-empty, non-whitespace technical subject ID. |
| `descriptive_id` | `string` | Required | Non-empty, non-whitespace human-readable subject ID. |
| `state` | `string` | Required | `"COMPLIANT"`, `"NON_COMPLIANT"`, or `"ERROR"`. |
| `violation_message` | `string` | For `NON_COMPLIANT` | Explanation and remediation from the criterion template. |

### Execution Stages

1. **Stage E0 (Whole-Request Pre-Flight)**:
   - Validate non-empty unique `controls`, compatible shared parameters, all required parameters present
     (`MISSING_PARAMETER`, Exit `2`), and values matching `value_schema` (`INVALID_PARAMETER_VALUE`, Exit `2`).
   - Use the same effective required/default rules as `ClientControlQuery`; require an explicit value if any selected
     control requires the parameter without a default. Reject conflicting schemas or declared defaults even when an
     explicit value is supplied.
   - Extra keys in `request.parameters` not declared by any control are ignored at request level; each control receives
     only its own declared parameters.
   - Zero MCP or LLM calls occur if Stage E0 fails.
2. **Stage E1 (Per-Control Evidence & Completeness)**:
   - Reuse started `MCPServer` containers and exact `(server, tool, resolved_arguments)` evidence cache across controls.
   - Validate `evidence_profile`, tool allowlist, `arguments_schema`, discovered live `inputSchema`, JSON payload, and
     `complete_when == True` per control.
   - Fail only the affected control on tool response, schema, parsing, or completeness errors.
   - A required MCP connection failure, disconnect, or response timeout adds a top-level
     `MCP_SERVER_UNAVAILABLE` at stage `evidence`, identifies the server in the message, and stops further work.
     Proceed to E4 with completed results or `results: []` if no control has completed. Exit with code `1`.
3. **Stage E2 & E3 (Selection & Criterion)**:
   - Normalize single selected object `{...}` to `[{...}]`; require non-whitespace `id` and `descriptive_id`.
   - Evaluate valid subjects with CEL or a singleton `SemanticCriterionSignature`. For the semantic call, render
     the parameter template and resolve `subject_paths`. Send only the map of projected values, condition, and
     correlation token. If a projection target is absent, mark only that subject `ERROR` and skip its LLM call.
     Add an `ErrorDetail` to `ControlResult.errors` with `code: "MISSING_SUBJECT_PATH"`, `stage: "criterion"`,
     and the affected `subject_id`. Include the missing JSON Pointer in the message, for example
     `Required subject path '/title' is missing`. Keep the same diagnostic with both `violations` and `all`.
   - Render the violation message when a subject is `NON_COMPLIANT`. If rendering fails, mark that subject and
     the control `ERROR`, record `VIOLATION_MESSAGE_RENDER_ERROR`, and continue with other subjects.
   - Skip `criterion` for invalid objects (`INVALID_SUBJECT`) and duplicate IDs (`DUPLICATE_SUBJECT_ID` with
     `subject_id` and `selection_index`), add them to `ControlResult.errors`, and set control `outcome: "ERROR"`.
4. **Stage E4 & Trace Finalization**:
   - Apply `result_filter` (`"violations"` vs `"all"`) to `ControlResult.subjects`, keeping all `errors`.
   - Attempt to save the trace for every execution, including inline execution and MCP outages. On write failure,
     preserve results and existing errors, append top-level `TRACE_WRITE_FAILED`, and return exit code `1`.

---

## 3. Client Query & Parameter Merging — `ClientControlQuery`

Scans the nested folder tree under `policy.controls_directory` (`--controls-dir`), filters YAML controls, projects
attributes, and merges shared parameters without LLM or MCP calls.

### Input (`ClientControlQuery`)

| Field | CLI Flag | Type / Default | Meaning |
| --- | --- | --- | --- |
| `files` | `--control` | `list[str] \| None` | Specific `.yaml` file paths within `controls_directory` (OR). |
| `ids` | `--id` | `list[str] \| None` | Exact `control.id` or glob/wildcard patterns on `control.id` (`change-mgmt-*`) (OR). |
| `glob` | `--glob` | `str \| None` | Glob pattern matching relative paths (`team-a/**/*.yaml`) or bare filenames (`*github*.yaml`). |
| `policy_ids` | `--policy-id` | `list[str] \| None` | Match `source.policy` (OR). |
| `rule_ids` | `--rule-id` | `list[str] \| None` | Match `source.rule` (OR). |
| `tags` | `--tag` | `list[str] \| None` | Match controls containing any listed tag (OR). |
| `evaluation_mode` | `--mode` | `str \| None` | `"deterministic"` or `"llm"`. |
| `evidence_server` | `--server` | `str \| None` | Match `evidence[].server`. |
| `evidence_tool` | `--tool` | `str \| None` | Match `evidence[].tool`. |
| `parameter_names` | `--parameter` | `list[str] \| None` | Match controls declaring any listed parameter name (OR). |
| `text_query` | `--text` | `str \| None` | Case-insensitive substring search in `description` and `source.excerpt`. |
| `projection` | `--projection` | `list[str]` | Default `["id", "description", "tags", "source", "parameters"]`, or `["full"]`. Projected dicts include relative `file` path. |
| `include_merged_parameters` | `--include-merged-parameters` / `--no-merged-parameters` | `bool = True` | Compute conflict-checked `merged_parameters` across matched controls. |

### Query Validation Rules & Output (`ClientControlQueryResult`)

- Combine distinct filter fields with logical **AND** and list values with logical **OR**.
- Apply `files` and `glob` before opening YAML. If both are supplied, only files matching both are candidates.
  If neither is supplied, all `.yaml` files under `controls_directory` are candidates.
- Check containment before opening each candidate. Return `INVALID_CONTROL_FILE` (Exit `2`) if a candidate
  is malformed or fails schema validation. Apply ID and metadata filters after candidate validation.
- Allow same `control.id` in different folders on disk, but fail immediately with `DUPLICATE_CONTROL_ID` (Exit `2`) if
  the active filter/selection matches two files with the same `control.id`.
- Fail with `NO_CONTROLS_MATCHED` (Exit `2`) when zero controls match.
- Fail with `PARAMETER_CONFLICT` (Exit `2`) when `include_merged_parameters=True` and two matched controls declare
  incompatible normalized `value_schema` or different explicitly declared defaults for the same name. A default absent
  from some declarations is not a conflict; `construction_value` is ignored.
- Output: `{controls: [...], merged_parameters?: list[MergedParameter], errors: list[ErrorDetail]}`.
- Each `MergedParameter`: `{name, value_schema, required, default?, controls: [{control_id, description}]}`.
- Return one entry per parameter name. Set `required: true` and omit `default` if any matched control requires that
  parameter without declaring a default. Otherwise set `required: false` and include `default` only if an eligible
  default exists. Preserve an explicitly declared `null` as `default: null`.

For example, if control A declares `repository` with default `"app"` and control B requires it without a default,
the merged entry is:

```json
{
  "name": "repository",
  "value_schema": {"type": "string"},
  "required": true,
  "controls": [
    {"control_id": "A", "description": "Control A"},
    {"control_id": "B", "description": "Control B"}
  ]
}
```

Execution without an explicit `repository` fails with `MISSING_PARAMETER`. If B declares `required: false`, the
entry has `required: false` and `"default": "app"`. Execution uses that default unless a caller supplies a value.

---

## 4. Client CLI Commands (`vatuta policy ...`)

The policy group uses the root configuration and sends diagnostics to a stderr console through centralized logging.
It bypasses eager provider validation. `query` and deterministic `execute` work without LLM credentials.
`query` and `execute` share the selector flags listed in the [query input](#input-clientcontrolquery).
`execute` accepts inline JSON or `@file.json` for `--parameters`.

1. **Search Rules & Get Merged Parameters (`vatuta policy query`)**:

   ```bash
   poetry run vatuta policy query --controls-dir data/controls \
     --glob "payments/**/*.yaml" --tag compliance --format json
   poetry run vatuta policy query --id "change-management-*" --no-merged-parameters --format json
   ```

2. **Create Rules from a Policy (`vatuta policy create`)**:

   ```bash
   poetry run vatuta policy create --policy-file policy.md --policy-id change-management \
     --construction-context "Repo acme/api; Jira project PAY." --tag compliance \
     --trial-attempts 1 --output-dir data/controls/payments --format json
   ```

3. **Execute Selected Rules (`vatuta policy execute`)**:

   ```bash
   poetry run vatuta policy execute --controls-dir data/controls --glob "payments/**/*.yaml" \
     --parameters '{"repository": "acme/api", "project": "PAY"}' --result-filter all --format json
   ```

4. **Guided 2-Step Natural-Language Check (`vatuta policy check "<prompt>"`)**:

   ```bash
   poetry run vatuta policy check "Check change-management rules for repo acme/api and Jira project PAY"
   ```

   - Shares one `invocation_id` trace (`vatuta://traces/policy/execution/{invocation_id}`) across all stages:
     - **Step 1 (`check_rule_selection`)**: The model calls `ClientControlQuery` using the configured query limit
       (default `3`) and byte limit per response (default `4096`). The client records the complete `(file, id)` pairs
       shown to the model during this call. An intermediate no-match response can be used to refine the filters.
     - **Confirmation**: The model returns `done=True` and `selected_controls`, a list of `{file, id}` objects.
       It may confirm after the last permitted query result. Without confirmation at the limit, return
       `CONTROL_SELECTION_INCOMPLETE` (Exit `1`). A confirmed empty list returns `NO_CONTROLS_MATCHED` (Exit `2`).
     - **Deterministic Load**: Confirm that each pair was shown together. Reload only the selected relative paths
       within the root. Check each current ID against the selection before merging parameters. An unseen pair or
       changed ID returns `INVALID_CONTROL_SELECTION` (Exit `1`). Two selected files with the same verified ID
       return `DUPLICATE_CONTROL_ID` (Exit `2`). Pass full controls and `merged_parameters` to the next stage.
     - **Step 2 (`check_parameter_extraction`)**: Skip if `merged_parameters == []`; else extract explicit values from
       prompt without guessing missing inputs.
   - **Step 3**: Run the service execution path with the same internal context allocated before Step 1.
     The context stays outside `PolicyExecuteRequest`. A selection failure skips parameter extraction and all
     evidence calls. On a Step 1 or Step 2 failure, return the same `invocation_id`, empty `results`, and the error.
     Write the trace for the steps completed so far.

---

## 5. Agent Skill Contract (`skills/vatuta-policy/` — `https://agentskills.io/`)

Run the scripts with `poetry run python <script_path>` from the Vatuta project with dependencies installed.
`query.py` and `execute.py` accept all shared selectors listed in the [query input](#input-clientcontrolquery).
`execute.py` accepts inline JSON or `@file.json` for `--parameters`.

Implemented in this feature as a tracked skill directory at `skills/vatuta-policy/`:

- **`skills/vatuta-policy/SKILL.md`**:
  - YAML frontmatter: `name: vatuta-policy`, `description: ...`.
  - Simple, schematic workflow instructions showing an AI agent how to search rules/parameters, create rules into nested
    folders, and execute rules.
- **`skills/vatuta-policy/scripts/` (3 Python Scripts with PEP 723 Metadata)**:
  - Each script starts with `#!/usr/bin/env python3` and a PEP 723 `# /// script ... # ///` block, and calls the exact
    same client handler in `src/client/policy.py` as `vatuta policy <query|create|execute> --format json`:
    1. `skills/vatuta-policy/scripts/query.py`:
       Accepts `--controls-dir`, `--control`, `--id`, `--glob`, metadata filter flags, `--projection`, and
       `--include-merged-parameters` / `--no-merged-parameters`.
    2. `skills/vatuta-policy/scripts/create.py`:
       Accepts `--policy-file` or `--policy-text`, `--policy-id`, `--construction-context`, `--tag`,
       `--trial-attempts`, and `--output-dir`.
    3. `skills/vatuta-policy/scripts/execute.py`:
       Accepts `--controls-dir`, `--control`, `--id`, `--glob`, metadata filter flags, `--parameters`, and
       `--result-filter` (`violations` | `all`).
  - All 3 scripts always emit a single JSON object to `stdout`, write logs/progress to `stderr`, and exit with `0`, `1`,
    or `2`.

---

## 6. Diagnostic Trace Resources & Error Codes

### Invocation Identity

Allocate one unique `invocation_id` when an external operation starts, before validating operation inputs.

| Entry point | Invocation boundary |
| --- | --- |
| CLI | Each invocation of a `vatuta policy` command (`query`, `create`, `execute`, or `check`). |
| Agent Skill | Each invocation of `query.py`, `create.py`, or `execute.py`. |
| Future MCP transport | Each incoming operation call, including calls within the same MCP session. |

A script that delegates to the CLI shares its context with the CLI. The entry point passes the context to shared
handlers, internal stages, and service calls. A direct service call creates a context when invoked independently.
Guided `check` uses one ID for selection, internal queries, parameter extraction, and execution.

Standalone `query` records its ID in stderr diagnostics on success and failure. Its stdout JSON follows
the `ClientControlQueryResult` schema. The following resource URI templates apply to creation and execution,
including guided `check`.

### Trace URI Templates (`PolicyService.read_trace(uri)`)

| Operation | Trace Resource URI |
| --- | --- |
| `PolicyService.create` (`policy_create`) | `vatuta://traces/policy/construction/{invocation_id}` |
| `PolicyService.execute` (`policy_execute`) | `vatuta://traces/policy/execution/{invocation_id}` |

- Return a resource error for a missing trace, an invalid URI, or a path outside the configured trace directory.
- Redact credentials and tokens in stored trace payloads, results, and error messages.

### Error Detail Schema & Exit Codes

- **`ErrorDetail`**: `{code, stage, message, control_id?, parameter?, subject_id?, descriptive_id?, selection_index?}`.
  Client write failures also include `unsaved_paths: list[str]`.
- **Key Error Codes**:
  - Guided selection (Exit `1`, stage `check_rule_selection`): `CONTROL_SELECTION_INCOMPLETE` for unconfirmed
    selection at the limit, and `INVALID_CONTROL_SELECTION` for a pair not shown or a changed file ID.
  - Input / Query (Exit `2`): `NO_CONTROLS_MATCHED`, `INVALID_CONTROL_FILE`, `DUPLICATE_CONTROL_ID`,
    `PARAMETER_CONFLICT`, `MISSING_PARAMETER`, `INVALID_PARAMETER_VALUE`.
  - Creation / Runtime / Persistence / Trace (Exit `1`): `NO_NORMATIVE_RULES`, `MISSING_CONSTRUCTION_CONTEXT`,
    `CONTROL_FILE_EXISTS`, `CONTROL_FILE_WRITE_FAILED`, `INVALID_SUBJECT`, `DUPLICATE_SUBJECT_ID`,
    `MISSING_SUBJECT_PATH`, `VIOLATION_MESSAGE_RENDER_ERROR`, `MCP_SERVER_UNAVAILABLE`, `TRACE_WRITE_FAILED`.
- **Exit Codes**:
  - `0`: Operation completed without technical errors (includes `COMPLIANT`, `NON_COMPLIANT`, and `NOT_APPLICABLE`).
  - `1`: Guided selection, creation, control or subject runtime, client persistence, or trace write failure.
  - `2`: Invalid request, filter/query failure (`NO_CONTROLS_MATCHED`, `INVALID_CONTROL_FILE`, `DUPLICATE_CONTROL_ID`,
    `PARAMETER_CONFLICT`, `MISSING_PARAMETER`, `INVALID_PARAMETER_VALUE`), or configuration error.
