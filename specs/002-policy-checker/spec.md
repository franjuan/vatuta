# Feature Specification: Policy Validation Engine

**Feature Branch**: `002-policy-checker`
**Created**: 2026-10-06
**Updated**: 2026-11-21
**Status**: Ready for planning (includes CLI UX, Agent Skill UX, and MCP-ready service contract)
**Input**: Policy validation engine with stateless MCP-ready service, CLI commands, and Agent Skill (`agentskills.io`).

## System Overview (Schematic)

- **Core Service (`PolicyService`)**:
  - 100% stateless regarding rule files. No rule repository or file selectors inside the service.
  - Uses JSON-serializable Pydantic models (1:1 ready for future MCP tools).
  - Exposes 3 methods:
    1. `create` (`policy_create`): Policy text -> list of `ControlDefinition` objects (with `tags` and `test_cases`).
    2. `execute` (`policy_execute`): Inline `ControlDefinition` objects + `parameters` -> findings and errors.
    3. `read_trace` (`resources/read`): Trace URI -> sanitized JSON trace.
  - Note: Running an MCP transport server (`vatuta policy serve`) is prepared by design but deferred.
- **Local Rule Storage (Client Side)**:
  - Each rule (control) is saved as one YAML file: `<id>.yaml`.
  - Files live under `policy.controls_directory` (default `data/controls`, overridable via `--controls-dir`).
  - Supports nested folders so users can group rules by use case (for example: team, system, environment, or policy).
- **Consistent Rule Selection (All Scripts & CLI)**:
  - Rules can be selected across the folder tree by:
    - File path (`files` / `--control`, one or more `.yaml` files).
    - Rule name / ID (`ids` / `--id`, exact `ControlDefinition.id` or glob/wildcard pattern like `change-mgmt-*`).
    - Glob pattern (`glob` / `--glob`, matching relative paths like `team-a/**/*.yaml` or bare filenames like
      `*github*.yaml`).
    - Metadata filters (`policy_ids`, `rule_ids`, `tags`, `evaluation_mode`, `evidence_server`, `evidence_tool`,
      `parameter_names`, `text_query`).
- **User Experience (UX) Surfaces Implemented in This Feature**:
  1. **CLI Commands (`vatuta policy ...`)**:
     - `vatuta policy query`: Search YAML rule files, project attributes, and get merged parameters.
     - `vatuta policy create`: Create rules from a policy and save `<id>.yaml` files in a target folder.
     - `vatuta policy execute`: Select YAML rules and execute them with explicit parameters.
     - `vatuta policy check "<prompt>"`: Guided 2-step LLM flow (Step 1: select rules; Step 2: extract parameters from
       prompt; Step 3: execute).
  2. **Agent Skill (`https://agentskills.io/`)**:
     - Implemented in this feature increment.
     - Standard skill layout: `SKILL.md` (frontmatter + simple agent workflow guide) and `scripts/` directory.
     - Provides **3 scripts** (combining rule search and parameter retrieval into the first script):
       1. **Search Rules & Get Parameters Script**: Filter rules in the folder tree, project rule attributes, and
          return `merged_parameters` for the selected rules.
       2. **Create Rules Script**: Create rules from a policy text/file and save `<id>.yaml` files into a target folder.
       3. **Execute Rules Script**: Select `.yaml` rule files (by file, ID, glob, or filter) and execute them with
          explicit parameters.

Related docs: [design](../../docs/policy-checker/design.md), [glossary](../../docs/policy-checker/glossary.md), and
[design inputs](design-inputs.md).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create Controls Step by Step from a Policy (Priority: P1)

As a user or agent, I want to give a policy text, optional context, and optional tags so the engine splits the policy
into atomic rules, builds each control step by step with unit tests, and saves one `<id>.yaml` file per rule in my
target directory.

**Why this priority**: Generated YAML controls and their embedded unit tests are the foundation for all compliance
checks.

**Independent Test**: Provide a policy with multiple obligations, context, and tags. Verify rule splitting, shared
parameter names, bounded JSONata exploration, embedded `test_cases`, live trials, and saved `<id>.yaml` files.

**Acceptance Scenarios**:

1. **Given** a policy with obligations that can fail separately, **When** creation splits the policy, **Then** it
   creates a separate atomic rule and `ControlDefinition` for each independent violation.
2. **Given** multiple rules from one policy, **When** parameters are defined rule by rule, **Then** an incremental
   `SharedParameterCatalog` is passed forward so identical inputs reuse the same parameter name and `value_schema`.
3. **Given** `trial_attempts > 0` and resolved `construction_value` inputs, **When** creation explores live evidence,
   **Then** it fetches evidence once in memory and runs at most `policy.discovery_max_queries` (default 5) JSONata
   queries (truncated to `policy.discovery_max_output_bytes`, default 4096 bytes) instead of sending full raw JSON to
   the LLM.
4. **Given** a rule's evidence and parameters, **When** creation builds `selection` and `criterion`, **Then** it also
   defines a clear, actionable `violation_message` format template in `criterion` (explaining the violation and giving
   remediation hints/instructions to resolve it, completed from the analyzed `subject` and `parameters` context) and
   creates, runs, and saves unit tests in `ControlDefinition.test_cases`: `selection_tests` (valid subjects +
   `NOT_APPLICABLE` empty case) and `criterion_tests` (at least one `COMPLIANT` and one `NON_COMPLIANT` subject,
   verifying that `violation_message` renders a non-empty explanation and remediation guide on `NON_COMPLIANT`
   subjects).
5. **Given** `trial_attempts=0`, **When** creation runs, **Then** live MCP trials are skipped (marked `skipped`), but
   schema checks, permission checks, and synthetic unit tests (`selection_tests` and `criterion_tests`) still run and
   must pass.
6. **Given** deterministic obligations and a semantic obligation, **When** creation chooses evaluation modes, **Then**
   it uses deterministic CEL by default and uses `llm` mode only when deterministic logic cannot solve it (explaining
   why).
7. **Given** `trial_attempts > 0`, **When** a rule fails its unit tests, **Then** creation retries within the configured
   attempt limit; if the rule still fails, or a required trial value is missing, creation fails with errors and returns
   zero controls (`controls: []`).
8. **Given** an obligation that cannot be checked with available tools, **When** creation runs, **Then** it fails
   explicitly and never silently drops the rule.
9. **Given** an invalid UTF-8 file or unsupported document format, **When** creation starts, **Then** it rejects the
   input before calling the LLM.
10. **Given** `trial_attempts > 0` and passing unit tests, **When** a live trial finishes without error but selects zero
    subjects (`NOT_APPLICABLE`) or finds `NON_COMPLIANT` subjects, **Then** the live trial passes.
11. **Given** construction context with sample project values, **When** creation defines parameters, **Then** it puts
    trial-only values in `construction_value` and never turns them into execution `default` values unless the policy or
    context defines a true default.
12. **Given** `trial_attempts > 0` and insufficient context, **When** a required parameter has no `default` and no
    `construction_value`, **Then** creation returns `MISSING_CONSTRUCTION_CONTEXT` with parameter details and affected
    `control_id`s without inventing values.
13. **Given** inline policy text or a policy file in the CLI/Skill script, **When** creation is called, **Then** both
    send `policy_text`, optional `policy_id` (defaulting to `invocation_id`), `construction_context`, `tags`, and
    `trial_attempts` to `PolicyService.create`.
14. **Given** construction context that contradicts the policy or allowed tool profiles, **When** controls are created,
    **Then** the context cannot weaken policy rules or expand tool permissions.
15. **Given** any `trial_attempts` value, **When** Step 0 and Phase A run, **Then** creation uses only local
    `policy.evidence_profiles` without starting containers or connecting to MCP. Live MCP discovery happens only in
    Phase B when `trial_attempts > 0`.
16. **Given** a parameter already in `SharedParameterCatalog` from an earlier rule, **When** a later rule or retry needs
    a different schema, default, or construction value, **Then** the shared entry stays unchanged and the current rule
    either aligns with it or uses a new parameter name.
17. **Given** a target folder (including nested subfolders) where `<id>.yaml` already exists, **When** the client saves
    generated controls, **Then** it checks all target paths before writing, rejects the collision with
    `CONTROL_FILE_EXISTS` (exit `1`), and overwrites nothing.

---

### User Story 2 - Search Controls, Get Merged Parameters, and Run Guided Check (Priority: P1)

As a user or agent, I want to search local YAML controls across my folder structure, inspect their merged parameters,
or run a 2-step natural-language `vatuta policy check "<prompt>"` command that selects controls, extracts parameters
from my prompt, and executes them.

**Why this priority**: Fast local filtering and parameter merging let users, CLI commands, and the Agent Skill work with
large rule folders without putting file management into `PolicyService`.

**Independent Test**: Create a nested folder of `.yaml` controls with tags and shared parameters. Test filtering and
parameter merging via `ClientControlQuery`, `vatuta policy query`, Skill script 1, and `vatuta policy check`.

**Acceptance Scenarios**:

1. **Given** `.yaml` control files in `policy.controls_directory` (including nested subfolders), **When**
   `ClientControlQuery` runs with filters (`ids`, `policy_ids`, `rule_ids`, `tags`, `evaluation_mode`,
   `evidence_server`, `evidence_tool`, `parameter_names`, `text_query`, `files`, or `glob`), **Then** it combines
   different filter fields with **AND** and list values with **OR**, fails immediately on any malformed YAML file
   in the search scope (`INVALID_CONTROL_FILE`), and returns matching controls projected to `projection` without
   external calls.
2. **Given** `include_merged_parameters=True` (default), **When** `ClientControlQuery` runs on selected controls (by
   file, ID, glob, or search filter), **Then** it returns `merged_parameters` (`name`, `value_schema`, `required`,
   `default`, and referencing `controls`). With `include_merged_parameters=False`, it omits `merged_parameters`.
3. **Given** `vatuta policy check "<prompt>"`, **When** Step 1 (Rule Selection) runs, **Then** the LLM queries
   `ClientControlQuery` (up to `policy.check_max_filter_queries`, default 3, truncated to
   `policy.check_max_output_bytes`, default 4096 bytes, stopping early when `done=True`) to select controls by
   relative file path and ID. A confirmed empty selection returns `NO_CONTROLS_MATCHED`. The client loads the
   selected files, checks their IDs, and returns full controls and `merged_parameters`.
4. **Given** selected controls and `merged_parameters`, **When** Step 2 (Parameter Extraction) of `vatuta policy check`
   runs, **Then** if `merged_parameters` is empty it skips the LLM call (`parameters={}`), otherwise the LLM extracts
   only explicit values from the prompt (never guessing missing values) and calls `PolicyService.execute` under the
   same `invocation_id`.
5. **Given** multiple selected controls sharing a parameter name, **When** parameters are merged or passed to
   execution, **Then** one value satisfies all controls using that name. Extra keys in `request.parameters` not used by
   any control are allowed, and each control receives only its own declared parameters.
6. **Given** a required parameter that only has a `construction_value`, **When** execution omits that parameter,
   **Then** `construction_value` is ignored and execution fails with `MISSING_PARAMETER`. A declared `default` can
   fill an omitted value when no selected control requires an explicit value; an invocation value overrides it.
7. **Given** missing required parameters for one or more controls, **When** execution validates the request in `Stage
   E0`, **Then** it rejects the whole call before fetching any evidence and lists all missing parameters and affected
   `control_id`s.
8. **Given** an empty control list, duplicate `control_id`s, unreadable/malformed YAML files, empty filter match
   (`NO_CONTROLS_MATCHED`), or incompatible same-name parameters, **When** validation runs, **Then** it fails
   immediately with an input error before any evidence call.
9. **Given** a shared parameter that one control requires without a default and another declares with a default,
   **When** the query merges parameters, **Then** it returns one entry with `required: true`, no `default` field, and
   both controls; execution without an explicit value fails with `MISSING_PARAMETER`.
10. **Given** a shared parameter with a compatible declared default and no control requiring it without a default,
    **When** the query merges parameters, **Then** it returns `required: false` and that default, including `null` if
    explicitly declared. Different declared defaults or incompatible schemas fail with `PARAMETER_CONFLICT`.
11. **Given** `policy check` exhausts its filter queries without the model confirming a final control selection,
    **When** Step 1 ends, **Then** it returns `CONTROL_SELECTION_INCOMPLETE` (Exit `1`) with empty results and a
    pre-execution trace. It skips parameter extraction and makes no evidence calls.
12. **Given** valid YAML files in one folder and a malformed YAML file in another, **When** the client selects
    only the valid files by path or path pattern, **Then** it reads and validates only those candidates. If the
    path selection is widened to include the malformed file, it returns `INVALID_CONTROL_FILE` (Exit `2`).
13. **Given** two folders contain controls with the same ID, **When** the model chooses one relative path and ID,
    **Then** the client loads that file only and checks its ID. If the file now declares a different ID, selection
    fails before evidence calls. Selecting both files still fails with `DUPLICATE_CONTROL_ID`.

---

### User Story 3 - Execute Supplied Controls and Inspect Results (Priority: P1)

As a user or agent, I want to execute selected controls with explicit parameters and get per-control and per-subject
results and errors by `control_id`.

**Why this priority**: Execution must isolate control errors and stop when a required MCP server is unavailable.

**Independent Test**: Execute deterministic and LLM controls via `PolicyService.execute`, `vatuta policy execute`, and
Skill script 3 against fixed evidence. Verify subject verdicts, filters, and error isolation.

**Acceptance Scenarios**:

1. **Given** valid `ControlDefinition` objects and complete parameters, **When** execution runs, **Then** it fetches
   each control's evidence, selects subjects, and evaluates `criterion` per subject without reading policy text or
   rule YAML files. Trace finalization follows FR-010.
2. **Given** a deterministic control evaluating pull requests `101` (approved issue), `102` (pending issue), and `103`
   (no issue), **When** execution finishes, **Then** results are `COMPLIANT` (no `violation_message`), `NON_COMPLIANT`,
   and `NON_COMPLIANT` with zero LLM calls, and each `NON_COMPLIANT` `SubjectResult` includes an actionable
   `violation_message` (explaining the violation and how an agent or operator can resolve it) formatted
   deterministically from the rule's template using that subject's data context.
3. **Given** a control with `mode: llm`, **When** execution runs, **Then** it evaluates the semantic criterion per
   subject (`true` -> `COMPLIANT`, `false` -> `NON_COMPLIANT` with rendered `violation_message` containing the
   violation explanation and remediation hints from the rule's template over the subject context, failure -> `ERROR`).
   Deterministic failures never fall back to LLM evaluation.
4. **Given** complete evidence where JSONata selection returns nothing or `[]`, **When** selection finishes, **Then**
   control outcome is `NOT_APPLICABLE` and criterion is skipped. If selection returns root `null`, outcome is `ERROR`.
5. **Given** selected subjects containing valid unique IDs, objects missing non-whitespace `id`/`descriptive_id`, and
   duplicate `id`s, **When** selection finishes, **Then** valid unique-ID subjects run `criterion`, while invalid
   objects (`INVALID_SUBJECT`) and duplicate-ID objects (`DUPLICATE_SUBJECT_ID` with `subject_id` and
   `selection_index`) skip `criterion`, add errors, and set control `outcome: "ERROR"`.
6. **Given** multiple controls and one control fails profile validation, receives a tool error from a responding
   MCP server, or fails completeness or selection, **When** execution runs, **Then** that control returns
   `outcome: "ERROR"` and the other controls continue. When a required MCP server cannot connect, disconnects, or
   times out after earlier controls complete, execution stops, returns those completed results and a top-level
   `MCP_SERVER_UNAVAILABLE` error, and exits with code `1`. Pending controls produce no verdicts.
7. **Given** incomplete evidence (`complete_when != True`), **When** evidence is checked, **Then** the affected control
   fast-fails with `ERROR` before running remaining evidence calls, selection, or criterion.
8. **Given** an MCP tool response, **When** evidence is parsed, **Then** structured JSON content is used first;
   otherwise a single text block containing valid JSON is parsed. Prose, malformed JSON, or tool errors become `ERROR`.
9. **Given** `result_filter="all"`, **When** results are returned, **Then** `ControlResult.subjects` includes
   `COMPLIANT`, `NON_COMPLIANT`, and `ERROR` subjects. With `result_filter="violations"` (default), `subjects` includes
   only `NON_COMPLIANT` subjects. `ControlResult.errors` keeps all errors under both filters.
10. **Given** an error-free execution with evaluated subjects, **When** results are returned, **Then** `outcome` is
    omitted (no aggregate `COMPLIANT`/`NON_COMPLIANT` percentage or status). Any subject error sets control `outcome:
    "ERROR"`.
11. **Given** multiple subjects on an `llm` control, **When** singleton LLM calls run, **Then** each call evaluates one
    subject with a correlation token (`subject_ref`). Failed or mismatched responses mark only that subject as `ERROR`.
12. **Given** equivalent requests via `PolicyService.execute`, CLI `vatuta policy execute`, or Skill script 3, **When**
    they finish, **Then** they produce identical validation, results, and error structures.
13. **Given** any executed control (`COMPLIANT`, `NON_COMPLIANT`, `NOT_APPLICABLE`, or `ERROR`), **When** its
    `ControlResult` is returned, **Then** it includes the control's `source` and its own effective resolved
    `parameters` (excluding construction-only values and other controls' parameters).
14. **Given** a `NON_COMPLIANT` subject and a `criterion.violation_message` format template (for example referencing
    `{descriptive_id}` / `{subject.descriptive_id}`, `{ticket_status}` / `{subject.ticket_status}`, or
    `{parameters.repository}`), **When** the template is rendered, **Then** `SubjectResult.violation_message` contains
    both the human-readable violation explanation and concrete remediation hints/instructions so a downstream agent or
    user can resolve the finding; if a placeholder references a missing field or has invalid format syntax, only that
    subject is marked `ERROR` (`VIOLATION_MESSAGE_RENDER_ERROR`).

---

### User Story 4 - Inspect Diagnostic Traces (Priority: P1)

As a user or developer, I want detailed creation and execution traces on disk so I can audit evidence, LLM prompts, and
errors without building a policy versioning system.

**Why this priority**: Traces explain why a control was created or how a compliance verdict was reached.

**Independent Test**: Run successful and failed creations and executions. Read traces from disk and via
`PolicyService.read_trace(uri)` and verify stage files and secret redaction.

**Acceptance Scenarios**:

1. **Given** any creation or execution call, **When** stages run, **Then** a trace folder under the configured log path
   saves stage inputs, outputs, evidence payloads, results, and errors as sanitized JSON (redacting secrets).
2. **Given** LLM calls during creation, `policy check`, or `mode: llm` execution, **When** the model is called,
   **Then** traces record prompts, responses, model metadata, and subject mappings.
3. **Given** saved traces on disk, **When** time passes, **Then** the system never auto-deletes them by age and provides
   no trace retention scheduler or policy revision history.
4. **Given** CLI or Skill script execution, **When** stages advance, **Then** progress and logs go to `stderr` while
   structured JSON output goes cleanly to `stdout`.
5. **Given** an `invocation_id` from creation or execution, **When** a client needs the trace, **Then** it builds the
   URI (`vatuta://traces/policy/construction/{invocation_id}` or `vatuta://traces/policy/execution/{invocation_id}`)
   and reads it via `PolicyService.read_trace` without extra trace fields in the response.
6. **Given** a missing, invalid, or out-of-bounds trace URI, **When** `read_trace` is called, **Then** it returns a
   resource error without exposing arbitrary host files.
7. **Given** completed execution results, **When** saving the trace to disk fails, **Then** the response keeps all
   calculated `results`, adds a top-level `TRACE_WRITE_FAILED` error, and exits with code `1`.
8. **Given** `vatuta policy check` failing in Step 1 or Step 2 before execution starts, **When** the error is returned,
   **Then** it returns the shared `invocation_id`, `results: []`, and `errors`, and writes the pre-execution trace.
9. **Given** validated controls ready to save, **When** writing the creation trace fails, **Then** the response
   returns the same `invocation_id`, `controls: []`, and a top-level `TRACE_WRITE_FAILED` error. The CLI and Skill
   exit with code `1` and write no generated YAML. If a creation error already exists, keep it in `errors`.

---

### User Story 5 - Use Policy Checker via an Agent Skill (Priority: P1)

As an AI coding agent supporting the Agent Skills standard (`https://agentskills.io/`), I want a self-contained Vatuta
Policy Skill with a schematic `SKILL.md` guide and 3 scripts so I can search rules and get their parameters, create new
YAML rules from policies into a folder structure, and execute selected YAML rules.

**Why this priority**: External AI agents already have their own reasoning LLM; giving them a standard Agent Skill with
deterministic scripts lets any `agentskills.io`-compatible agent author, discover, parameterize, and run rules directly.

**Independent Test**: Load the skill folder (`SKILL.md` + `scripts/`), invoke each of the 3 scripts using file paths,
rule IDs, and glob patterns over a nested controls directory, and verify JSON outputs and exit codes (`0`, `1`, `2`).

**Acceptance Scenarios**:

1. **Given** the repository, **When** an `agentskills.io`-compatible agent inspects the Policy Skill directory,
   **Then** it finds a valid `SKILL.md` (YAML frontmatter with `name` and `description`, plus concise schematic usage
   instructions) and 3 executable scripts under `scripts/`.
2. **Given** a hierarchical controls directory (with subfolders per user use case, one `<id>.yaml` file per rule),
   **When** any skill script selects rules, **Then** it accepts selection file by file (`--control`), by rule name/ID
   (`--id`), by glob pattern (`--glob` across subfolders and filenames), or by metadata filters (`--policy-id`,
   `--rule-id`, `--tag`, `--mode`, `--server`, `--tool`, `--parameter`, `--text`).
3. **Given** Skill Script 1 (Search Rules & Get Parameters), **When** the agent calls it to search rules or inspect
   parameters for a selected rule set, **Then** it returns matching rules (with relative YAML file path and requested
   `projection`) and, when enabled (default), the deduplicated `merged_parameters` JSON schema block.
4. **Given** Skill Script 2 (Create Rules from Policy), **When** the agent passes a policy file or text, optional
   context/tags/trials, and a target directory (`--output-dir`, including nested subfolders), **Then** it creates the
   controls via `PolicyService.create`, saves each control as `<id>.yaml` in that directory, and outputs JSON results.
5. **Given** Skill Script 3 (Execute Rules), **When** the agent selects rules (by file, ID, glob, or filter) and passes
   explicit parameter values (`--parameters`), **Then** it loads the `.yaml` definitions, runs `PolicyService.execute`,
   and outputs the structured JSON `PolicyExecuteResult` to `stdout` with standard exit codes (`0`, `1`, `2`).

---

### Edge Cases (Schematic)

- **Invalid inputs before execution (Exit `2`)**:
  - Empty control list or filter/glob matching zero files (`NO_CONTROLS_MATCHED`).
  - Malformed or schema-invalid `.yaml` file in the path scope of the scan (`INVALID_CONTROL_FILE`).
  - File path or glob escaping `controls_directory` (`..` or external symlink).
  - Duplicate `control_id` in an execution request.
  - Same-name parameters across selected controls with incompatible `value_schema` or `default`.
  - Missing required execution parameter (`MISSING_PARAMETER`) or value failing schema (`INVALID_PARAMETER_VALUE`).
  - Note: None of these may look like a passing compliance check; zero MCP calls are made.
- **Creation boundaries**:
  - `trial_attempts=0`: Skips all MCP calls and live trials, but synthetic `selection_tests` and `criterion_tests` are
    still mandatory.
  - `trial_attempts > 0`: Phase A reports required parameters that lack both `construction_value` and `default`
    (`MISSING_CONSTRUCTION_CONTEXT`) before starting any MCP server.
  - Empty live selection (`NOT_APPLICABLE`) or live `NON_COMPLIANT` subjects pass the live trial if no technical error
    occurs (for live `NON_COMPLIANT` subjects, `criterion.violation_message` must also render without error).
  - If `criterion.violation_message` has invalid placeholder syntax, references a field missing from a `NON_COMPLIANT`
    test or live subject, or renders to an empty/whitespace string, Step 2.4 fails validation.
  - Failed service creation returns `controls: []` (never a partial subset of rules).
  - Existing target `<id>.yaml` file aborts saving all files with `CONTROL_FILE_EXISTS` (Exit `1`).
  - A client save failure after writing begins keeps already written files and reports the files not saved.
- **Context window protection**:
  - Creation discovery queries: max `policy.discovery_max_queries` (default 5), each output truncated to
    `policy.discovery_max_output_bytes` (default 4096 bytes, marked `[TRUNCATED]`), early exit on `done=True`.
  - Guided `policy check` filter queries: max `policy.check_max_filter_queries` (default 3), each output truncated to
    `policy.check_max_output_bytes` (default 4096 bytes, marked `[TRUNCATED]`). If the model never sets `done=True`
    within that budget, report `CONTROL_SELECTION_INCOMPLETE` (Exit `1`) before execution.
- **Execution runtime errors (Exit `1`)**:
  - A required MCP server connection failure, disconnect, or response timeout stops the operation. Keep results
    from completed controls, apply the requested result filter, and report a top-level `MCP_SERVER_UNAVAILABLE`.
  - Tool errors from a responding server, evidence validation failures, incomplete evidence, and root `null`
    selection fail the affected control (`outcome: "ERROR"`); remaining controls continue.
  - Invalid selected subject (`INVALID_SUBJECT`) or duplicate subject ID (`DUPLICATE_SUBJECT_ID`) skips `criterion` for
    those items, records errors, sets control `outcome: "ERROR"`, and still evaluates remaining valid unique-ID
    subjects.
  - If rendering `criterion.violation_message` fails on a `NON_COMPLIANT` subject (missing field, undeclared parameter,
    invalid placeholder syntax, or empty rendered message), only that subject becomes `ERROR`
    (`VIOLATION_MESSAGE_RENDER_ERROR` in `ControlResult.errors`, setting control `outcome: "ERROR"`).
  - Technical errors or ambiguous LLM outputs always produce `ERROR`, never `NON_COMPLIANT` or `NOT_APPLICABLE`.
  - Independent rules are reported separately per `control_id`; no global compliance percentage is calculated.

## Requirements *(mandatory)*

### Functional Requirements (Schematic)

*Note: Retired IDs `FR-011`, `FR-016`, `FR-020`, `FR-028`..`FR-031`, and `FR-033` remain omitted for traceability.*

#### 1. Core Service & Architecture (`PolicyService`)

- **FR-001 (Create Input)**: `PolicyService.create` (`policy_create`) MUST accept:
  - `policy_text: str` (required, non-empty plain text or Markdown; invalid UTF-8/formats rejected before LLM).
  - `policy_id: Optional[str]` (defaults to creation `invocation_id`).
  - `construction_context: str` (optional, default `""`).
  - `tags: list[str]` (optional, default `[]`).
  - `trial_attempts: Optional[int]` (`>= 0`, overrides `policy.trial_attempts`, default `1`).
  - No pre-declared parameter map or file path in the service request.
- **FR-002 (Create Output)**: Return complete `list[ControlDefinition]` (`id`, `description`, `tags`, `source`,
  `parameters`, `evidence`, `selection`, `criterion`, `test_cases`, `creation_check`). If any rule cannot be built or
  validated, fail the whole call (`controls: []`) with diagnostic `errors`.
- **FR-003 (Evaluation Mode Choice)**: Choose `deterministic` (CEL) or `llm` mode at creation time. Prefer
  `deterministic`; use `llm` only when deterministic logic cannot evaluate the rule. Execution MUST NOT switch or
  repair modes at runtime.
- **FR-004 (Execute Pipeline)**: `PolicyService.execute` (`policy_execute`) MUST accept inline `ControlDefinition`
  objects, fetch declared evidence, run JSONata `selection` to extract subjects, and evaluate `criterion` per subject
  with separate stage inputs/outputs.
- **FR-005 (Independent Evidence)**: Evidence calls MUST NOT depend on other evidence calls or other controls. No
  dependency graphs, dynamic planning, or automatic pagination.
- **FR-006 (Reproducibility)**: Deterministic CEL criteria MUST return identical results for identical inputs without
  LLM calls. Semantic `llm` criteria MUST map fixed responses deterministically to subject states.
- **FR-007 (States & Outcomes)**:
  - Subject states: `COMPLIANT`, `NON_COMPLIANT`, `ERROR`.
  - Empty valid selection -> control `outcome: "NOT_APPLICABLE"` (criterion not run).
  - Any control, selection, or subject error -> control `outcome: "ERROR"`.
  - Error-free non-empty evaluation -> `outcome` omitted (no aggregate pass/fail state).
- **FR-008 (Read-Only Source Agnosticism)**: Evidence access MUST use configured read-only MCP tool profiles. The core
  engine MUST NOT hardcode GitHub or Jira logic.
- **FR-009 (Stateless Service, CLI & Skill Surfaces)**:
  - `PolicyService` is stateless, uses JSON-serializable Pydantic models (1:1 MCP schema ready), and never infers
    parameters using an LLM during `execute`.
  - `vatuta policy serve` (MCP server transport) is deferred.
  - Client provides CLI commands (`query`, `create`, `execute`, `check`) and the Agent Skill (`SKILL.md` + 3 scripts).

#### 2. Creation Pipeline Details

- **FR-026 (Creation Trials & Unit Tests)**:
  - `trial_attempts == 0`: Skip live MCP calls (mark `skipped`), still run schema/permission checks and synthetic
    `selection_tests` + `criterion_tests`.
  - `trial_attempts == 1`: Run 1 live trial without repair retries.
  - `trial_attempts > 1`: Run bounded full-context retries from Step 2.1 on failure, reusing cached in-memory MCP
    evidence if `(server, tool, resolved_arguments)` did not change.
  - Synthetic test failures use the same configured attempt limit. If no attempt passes, fail the entire creation and
    return `controls: []`; with `trial_attempts == 0`, a failed synthetic test also fails creation.
  - Live trials pass if error-free, even for results `NOT_APPLICABLE` or `NON_COMPLIANT`.
  - Required MCP server unavailability during discovery or live trials stops creation immediately with
    `MCP_SERVER_UNAVAILABLE` (Exit `1`). Return `controls: []` and write no generated YAML files.
- **FR-032 (Policy Authority)**: Policy text is the single source of truth. Users may inspect saved `.yaml` files
  manually; no approval workflow, versioning, or diff engine is included.
- **FR-034 (Criterion & Actionable Violation Message Template Declaration)**: Every control declares `deterministic`
  (`expression`) or `llm` (`condition` and `deterministic_insufficiency` justification). Both criterion modes MUST also
  declare a non-empty `violation_message: str` format template that:
  1. Explains clearly why the analyzed `subject` violates the rule, and
  2. Provides concrete remediation hints or instructions on how an agent or operator analyzing the policy check output
     can resolve the violation for that `subject` (grounded in the rule's obligation and completed from the `subject`
     and `parameters` data context).
- **FR-038 (Two-Phase Creation & Shared Parameters)**:
  - Derive minimum necessary parameters (`parameters: {}` allowed).
  - Pass incremental `SharedParameterCatalog` across rules `r_1..r_N`.
  - Two rules share a parameter name ONLY if `value_schema`, `construction_value`, and `default` match. Existing shared
    parameters are immutable during later rules and retries.
  - **Step 0 & Phase A (Step 2.1 across all rules)**: Use local `policy.evidence_profiles` only (no MCP connection). If
    `trial_attempts > 0` and any required parameter lacks both `construction_value` and `default`, return all
    `MISSING_CONSTRUCTION_CONTEXT` errors across all rules at once before starting MCP.
  - **Phase B (Steps 2.2–2.5 per rule)**: Connect to MCP (only when `trial_attempts > 0`), validate live `inputSchema`,
    and fail fast if any rule exhausts its attempts.
- **FR-041 (Control IDs & Safe YAML Saving)**:
  - The feature does not acquire shared locks when reading or writing rule YAML files.
  - Deterministic kebab-case slug: `f"{policy_id}-{rule_id}"` (or `rule_id` if already prefixed, deduplicated).
  - Client saves each control as `<id>.yaml` in `policy.controls_directory` or `--output-dir` (creating missing target
    subfolders).
  - Pre-check all target files before writing: if any `<id>.yaml` already exists, fail with `CONTROL_FILE_EXISTS` (Exit
    `1`) and write no files. No overwrite flag or auto-renaming.
  - If a later write fails, keep already written files, report the files not saved, and return a persistence error
    (Exit `1`).
- **FR-043 (Control Tags)**: `ControlDefinition` includes `tags: list[str]`, merging LLM-inferred tags from
  policy/context with caller-supplied `tags`.
- **FR-044 (Atomic Decomposition)**: Step 1 splits `policy_text` into atomic normative rules whenever conditions can
  fail independently or represent distinct violations. If zero rules are found, fail with `NO_NORMATIVE_RULES`
  (`stage: "decomposition"`).
- **FR-045 (Bounded JSONata Exploration)**: In Step 2.2 (`trial_attempts > 0`), explore in-memory evidence over
  `{"evidence": ..., "parameters": ...}` with at most `policy.discovery_max_queries` (default 5) JSONata queries,
  truncating each output to `policy.discovery_max_output_bytes` (default 4096 bytes, appending `[TRUNCATED]`), returning
  query errors/missing as observations within budget, and stopping early on `done=True`.
- **FR-046 (Mandatory Synthetic Unit Tests)**: Steps 2.3 and 2.4 generate, run, and persist `ControlDefinition.test_cases`:
  - `selection_tests`: Normalize single object `{...}` to `[{...}]`, check `expected_subjects` and `NOT_APPLICABLE`,
    reject `DUPLICATE_SUBJECT_ID`.
  - `criterion_tests`: Test at least one `COMPLIANT` and one `NON_COMPLIANT` mock subject (via CEL for `deterministic`
    or LLM with `{{parameters.<name>}}` substitution for `llm`), and verify that `criterion.violation_message` renders a
    non-empty, non-whitespace string without placeholder errors for every `NON_COMPLIANT` test case.
  - Unit-test parameter priority: `1. test_case.parameters` -> `2. default` -> `3. construction_value`.

#### 3. Client Rule Directory, Selection, Search & Guided Check

- **FR-012 (Inline Service Execution vs Client Directory)**:
  - `PolicyService.execute` receives inline `ControlDefinition` list with unique `id`s.
  - Client stores `<id>.yaml` files in `policy.controls_directory` (`--controls-dir`), supporting nested folders.
- **FR-039 (`ClientControlQuery` — Search, Projection & Merged Parameters)**:
  - Deterministically scans and filters `.yaml` files under `policy.controls_directory` (`--controls-dir`).
  - Combines distinct filter fields with **AND** and multi-value list fields (`ids`, `policy_ids`, `rule_ids`, `tags`,
    `parameter_names`, `files`) with **OR**.
  - Apply `files` and `glob` to paths before opening any YAML file. Read and validate only files that match the
    path selectors. With no path selector, all `.yaml` files under the control directory are candidates.
    Apply ID and metadata filters after candidate validation.
  - Fails immediately with `INVALID_CONTROL_FILE` (Exit `2`) if any candidate YAML file is malformed, or with
    `DUPLICATE_CONTROL_ID` (Exit `2`) if the matched selection contains two files with the same `control.id`
    (different folders on disk may hold same-ID files when not matched together).
  - Projects requested attributes (`projection`, plus relative file path) and, when `include_merged_parameters=True`
    (default), returns `merged_parameters` (`name`, `value_schema`, `required`, optional `default`, and referencing
    `controls`) without LLM or MCP calls. Merge fields follow the execution rules in FR-040.
- **FR-040 (Shared Parameter Resolution)**:
  - Same-name parameters across selected controls share one invocation value.
  - A declaration may set `required: false` to allow omission without a default. If the flag is omitted, it is `true`.
  - Resolution priority at execution: `1. explicit request value` -> `2. eligible declared default` (never
    `construction_value`).
  - If any selected control requires a shared parameter without declaring a default, an explicit request value is
    required for that parameter. Otherwise, an available compatible default supplies the shared value.
  - A default present on only some declarations is not itself a `PARAMETER_CONFLICT`.
  - The merged entry has `required: true` exactly when an explicit request value is required. It includes `default` only
    when an eligible default exists; an explicitly declared `null` remains a present `default: null`.
  - Incompatible normalized `value_schema` declarations or different explicitly declared defaults are a
    `PARAMETER_CONFLICT`, even when a request supplies an explicit value. `construction_value` is ignored in merging.
  - Extra keys in `request.parameters` not declared by any selected control are ignored at request level; each control
    receives only its own declared parameters.
- **FR-042 (Pre-Flight Input Errors)**: Empty control selection (`NO_CONTROLS_MATCHED`), duplicate `control_id`s in the
  matched selection or execution request (`DUPLICATE_CONTROL_ID`), malformed YAML (`INVALID_CONTROL_FILE`), and
  incompatible same-name parameter declarations (`PARAMETER_CONFLICT`) MUST fail before evidence acquisition (Exit `2`).
- **FR-047 (Guided 2-Step `vatuta policy check "<prompt>"`)**:
  - Shares one `invocation_id` trace (`vatuta://traces/policy/execution/{invocation_id}`) across pre-execution stages
    and `execute`.
  - **Step 1 (`check_rule_selection`)**: LLM calls `ClientControlQuery` (`projection=["id", "description", "tags",
    "source"]`, `include_merged_parameters=False`) up to `policy.check_max_filter_queries` (default 3, truncated to
    `policy.check_max_output_bytes`, default 4096 bytes) until `done=True`.
    - If the model has not set `done=True` when the budget runs out, return `CONTROL_SELECTION_INCOMPLETE` (Exit `1`).
      Do not extract parameters or fetch evidence.
    - A confirmed empty selection fails with `NO_CONTROLS_MATCHED` (Exit `2`).
    - A confirmed selection identifies each control by a relative file path and ID returned by the query. Load only
      those paths with `projection=["full"]` and `include_merged_parameters=True`. Verify the ID in each reloaded file.
      If a selected path and ID were not returned together, or the file no longer has that ID, return
      `INVALID_CONTROL_SELECTION` (Exit `1`) before parameter extraction or evidence calls. Keep
      `DUPLICATE_CONTROL_ID` (Exit `2`) if two selected paths have the same ID.
  - **Step 2 (`check_parameter_extraction`)**: Skipped if `merged_parameters` is empty (`parameters={}`). Otherwise LLM
    extracts explicit parameter values from the prompt without guessing missing values, then calls
    `PolicyService.execute`.
  - Pre-execution failures return `{invocation_id, results: [], errors}` and save the pre-execution trace.

#### 4. Execution Pipeline Details

- **FR-013 (Evidence Acquisition & In-Memory Cache)**:
  - Each evidence item specifies `name`, `server`, `tool`, and `arguments`. Its matching tool profile defines
    `complete_when`.
  - Keep started MCP containers alive across controls during a single `execute` call and cache valid responses in memory
    by exact `(server, tool, resolved_arguments)`.
  - Accept structured JSON response content first; otherwise accept a single text block containing valid JSON.
  - Fast-fail the control on evidence validation, tool response, parsing, or completeness errors.
  - Fail the operation on required MCP server unavailability as specified in FR-018.
  - MCP services manage their own concurrency and blocking. Vatuta adds no MCP rate limit, call budget,
    or circuit breaker for this feature.
- **FR-014 (Stage E0 Pre-Flight vs Stages E1–E3 Per-Control Validation)**:
  - **Stage E0 (Whole Request)**: Validate non-empty unique `controls`, compatible shared parameters, all required
    parameters present (`MISSING_PARAMETER`), and values matching `value_schema` (`INVALID_PARAMETER_VALUE`). Reject
    entire call on failure before any MCP call.
  - **Stages E1–E3 (Runtime)**: Isolate profile, allowlist, schema, tool response, completeness, selection,
    and criterion errors to the affected control or subject. Apply FR-018 on required MCP server unavailability.
- **FR-015 (Subject Selection & Duplicate ID Guard)**:
  - Selection output (object normalized to 1-item list, or list of objects) requires non-empty, non-whitespace string
    `id` and `descriptive_id` (`len(s.strip()) > 0`).
  - Unique valid `id`s run `criterion`.
  - Invalid objects (`INVALID_SUBJECT`) and duplicate `id`s (`DUPLICATE_SUBJECT_ID` with `subject_id` and
    `selection_index`) skip `criterion`, record errors in `ControlResult.errors`, and set control `outcome: "ERROR"`.
- **FR-017 (Self-Contained `ControlResult` Provenance & Parameters)**:
  - Every `ControlResult` (including `ERROR` and `NOT_APPLICABLE`, under any `result_filter`) MUST include
    `control_id`, `source` (`{policy, rule, excerpt}`), and `parameters` (only that control's effective resolved values,
    excluding `construction_value` and other controls' parameters).
- **FR-018 (Runtime Failure Isolation & MCP Unavailability)**:
  - A required MCP server is unavailable when its connection cannot be established, the transport disconnects,
    or a response times out. Stop the operation on that failure and report a top-level `MCP_SERVER_UNAVAILABLE`
    error identifying the server. Skip pending controls and further evidence or criterion calls.
  - For `execute` and the execution stage of `check`, keep results from completed controls under the requested
    `result_filter`, save the failure in the execution trace, and return exit code `1` from the CLI or Skill.
    When no control has completed, return `results: []`.
  - Control and subject failures from other causes stay isolated. Incomplete evidence (`complete_when != True`)
    yields a control-level `ERROR` with zero subject verdicts.
- **FR-019 (Criterion Evaluation & Violation Message Rendering)**: Evaluate one criterion independently per valid
  unique-ID subject (`true` -> `COMPLIANT`, `false` -> `NON_COMPLIANT`, non-boolean/error -> `ERROR`). When a subject is
  `NON_COMPLIANT`, deterministically render `criterion.violation_message` with that subject's data context (and the
  control's resolved `parameters`) into `SubjectResult.violation_message` (`COMPLIANT` and `ERROR` subjects omit
  `violation_message`).
- **FR-021 (Result Filter)**: `result_filter` accepts `"violations"` (default: `ControlResult.subjects` has only
  `NON_COMPLIANT`) or `"all"` (`subjects` has `COMPLIANT`, `NON_COMPLIANT`, and `ERROR`). `ControlResult.errors` retains
  all errors under both filters.
- **FR-035 (Singleton Semantic Evaluation & Parameter Templates)**: Each `mode: llm` criterion declares
  `subject_paths` as RFC 6901 JSON Pointers to values needed by its condition. Creation stores and verifies
  the projection in criterion tests. Each call sends the rendered condition with deterministic
  `{{parameters.<name>}}` substitution, `subject_ref`, and the projected values.
  A missing target yields subject-level `ERROR` before the call. Add `MISSING_SUBJECT_PATH` at stage `criterion`
  to `ControlResult.errors` with `subject_id` and the missing pointer in the message. Both filters retain the error.
  Keep explicit JSON `null` values.
- **FR-036 (Semantic Error Handling)**: Missing, mismatched `subject_ref`, ambiguous, or non-boolean LLM outputs mark
  only the affected subject as `ERROR`.
- **FR-051 (Safe Format-String Template for Violation & Remediation Messages)**:
  - `criterion.violation_message` adapts a single, self-contained template string to state both **what failed** and
    **how to resolve it** (remediation hints/instructions for a downstream agent or user), using safe Python `.format` /
    f-string style field placeholders (`{<path>}`) resolved deterministically against the evaluated `subject` data
    context (supporting direct subject keys like `{descriptive_id}` or `{ticket_status}`, nested dot paths, explicit
    `{subject.<path>}`, and `{parameters.<name>}`; literal braces escaped as `{{` and `}}`; JSON `null` values rendered
    as `"null"`).
  - Remediation guidance inside `violation_message` MUST be derived from the policy rule's normative requirement,
    selected `subject` identifiers/fields, and declared `parameters` (indicating which subject resource and state/value
    must change to become compliant).
  - Arbitrary expression evaluation (`eval`), method calls, and dunder/attribute traversal are forbidden.
  - Any placeholder syntax error, missing field/parameter reference, or empty/whitespace rendered result fails Step 2.4
    validation during creation and marks the affected `NON_COMPLIANT` subject as `ERROR`
    (`VIOLATION_MESSAGE_RENDER_ERROR`, setting control `outcome: "ERROR"`) during execution.

- **FR-052 (LLM Input Boundaries & Payload Sanitization)**: Keep policy text, user prompts, rule metadata,
  evidence, and rendered conditions in data fields separate from engine instructions. Check that generated source
  excerpts occur in the submitted policy. Validate MCP responses as JSON data and strip scripts from external HTML
  before sending it to the LLM. Send only the data each stage needs.

#### 5. Observability, Tracing & Privacy

- **FR-010 (Trace Directories & Resource URIs)**:
  - Save stage traces under the configured log directory.
  - Clients derive URIs without extra response fields:
    - Creation: `vatuta://traces/policy/construction/{invocation_id}`
    - Execution: `vatuta://traces/policy/execution/{invocation_id}`
  - Read via `PolicyService.read_trace(uri)` returning sanitized `InvocationTrace` JSON.
  - If writing an execution trace fails, keep calculated `results`, append global `TRACE_WRITE_FAILED`, and exit `1`.
  - Creation must save its trace before returning controls or writing YAML. On trace write failure, return the
    same `invocation_id` and `controls: []`, keep any existing errors, and append global `TRACE_WRITE_FAILED`.
    The CLI and Skill exit with code `1` and write no generated YAML.
  - Redact credentials and tokens in saved traces, results, and error messages.
- **FR-022 (Stream Separation)**: Send human progress and logs to `stderr`; send machine-readable JSON output to
  `stdout`.
- **FR-023 (Logging Standards)**: Identify component and stage in diagnostic logs using lazy formatting and central
  logging setup.
- **FR-024 (LLM Prompt Samples)**: Save creation, check, and semantic evaluation LLM calls as structured samples in
  traces for prompt auditing.
- **FR-025 (Per-Control Metrics)**: Record durations, token/model usage, and error counts per control without computing
  an overall compliance percentage.
- **FR-027 (Trace Persistence)**: Keep traces on disk until manually deleted; no automatic age cleanup or retention
  config.
- **FR-037 (Semantic Trace Attribution)**: Record rendered `condition`, model metadata, `subject_ref`, input, and
  output for every `mode: llm` evaluation.

#### 6. Agent Skill (`https://agentskills.io/`) & Hierarchical Rule Selection

- **FR-048 (Agent Skill Package)**:
  - Provide a tracked Agent Skill package at `skills/vatuta-policy/` following the `https://agentskills.io/`
    specification (ready to copy or symlink into any agent's skill directory):
    - `skills/vatuta-policy/SKILL.md` with required YAML frontmatter (`name: vatuta-policy`, `description`) and a
      simple, schematic agent workflow guide.
    - `skills/vatuta-policy/scripts/` folder containing 3 executable Python scripts (`query.py`, `create.py`,
      `execute.py`) with PEP 723 inline script metadata (`# /// script ... # ///`), invoked via `poetry run`.
  - The skill MUST be usable by any `agentskills.io`-compatible agent without requiring a running MCP server for
    `PolicyService`.
- **FR-049 (Three Skill Scripts & CLI Parity)**:
  - Combine rule search and parameter retrieval into Script 1 so the Skill exposes 3 thin Python scripts (`query.py`,
    `create.py`, `execute.py`, with PEP 723 metadata) sharing the exact client handler with CLI `vatuta policy
    query|create|execute --format json`:
    1. **Search Rules & Get Parameters (`query`)**: Filter YAML rules in `--controls-dir`, project fields (`projection`,
       including relative file path), and return `merged_parameters` (`--include-merged-parameters` /
       `--no-merged-parameters`, default `True`).
    2. **Create Rules (`create`)**: Read `--policy-file` or `--policy-text`, optional `--policy-id`,
       `--construction-context`, `--tag`, `--trial-attempts`, and `--output-dir`, call `PolicyService.create`, and save
       each rule as `<id>.yaml`.
    3. **Execute Rules (`execute`)**: Select YAML rules from `--controls-dir`, accept `--parameters` (JSON string or
       `@file.json`) and `--result-filter` (`violations` | `all`), call `PolicyService.execute`, and print JSON output.
  - All 3 scripts MUST output one JSON object to `stdout`, write logs/progress to `stderr`, and use exit codes `0`
    (success), `1` (creation/runtime/persistence/trace error), and `2` (input/filter/schema/config error).
- **FR-050 (Hierarchical Directory Structure & Universal Selection)**:
  - Users can organize `<id>.yaml` files (1 file per rule) in nested subdirectories under `policy.controls_directory`
    (`--controls-dir`) to reflect their use cases, and target any directory/subdirectory via `--output-dir` on creation.
  - Every rule-reading script and command (`query` and `execute`) MUST support selecting rules:
    - File by file (`--control` / `files`: relative or contained file paths).
    - By rule name or ID (`--id` / `ids`: matching exact `ControlDefinition.id` or glob/wildcard patterns on `id`,
      e.g., `change-mgmt-*`, anywhere in the directory tree).
    - By glob pattern (`--glob` / `glob`: supporting recursive `**` patterns matching relative file paths like
      `team-a/**/*.yaml` or bare `<id>.yaml` filenames like `*github*.yaml` within `controls_directory`).
    - By metadata filters (`--policy-id`, `--rule-id`, `--tag`, `--mode`, `--server`, `--tool`, `--parameter`,
      `--text`).
  - Directory scanning MUST enforce root containment under `controls_directory` (rejecting path traversal or external
    symlinks).

### Key Entities (Schematic)

- **Policy**: Input text or Markdown document containing one or more compliance rules.
- **Normative Rule**: Single atomic obligation extracted from a policy in Creation Step 1.
- **Construction Context**: Free-text description of target repos/projects used to infer tags, parameters, and
  `construction_value` trial inputs.
- **Shared Parameter Catalog**: Incremental map of parameter declarations (`name`, `description`, `value_schema`,
  `construction_value`, `default`) shared across rules created from the same policy.
- **Control (`ControlDefinition`)**: Complete rule definition saved as `<id>.yaml`: `id`, `description`, `tags`,
  `source`, `parameters`, `evidence`, `selection`, `criterion`, `test_cases`, `creation_check`.
- **Control Directory Tree**: Hierarchical folder structure under `policy.controls_directory` storing one `<id>.yaml`
  file per control, organized by user use case.
- **Control Test Cases (`ControlTestCases`)**: Embedded unit tests inside `<id>.yaml` (`selection_tests` and
  `criterion_tests`).
- **Parameter Declaration**: Named input with JSON Schema (`value_schema`), optional `construction_value` for trials,
  optional `default` for execution, and a `required` flag that defaults to `true`.
- **Client Control Query (`ClientControlQuery`)**: Local search, projection, and parameter-merging request over the
  Control Directory Tree (`files`, `ids`, `glob`, metadata filters, `projection`, `include_merged_parameters`).
- **Merged Parameter (`MergedParameter`)**: Deduplicated parameter requirement across selected controls (`name`,
  `value_schema`, `required`, optional effective `default`, and referencing `controls`).
- **Policy Agent Skill**: `agentskills.io`-compliant skill folder (`SKILL.md` + 3 scripts in `scripts/`) for searching
  rules/parameters, creating YAML rules from a policy, and executing selected YAML rules.
- **Evidence**: Read-only JSON payload fetched from a configured MCP tool.
- **Subject Result (`SubjectResult`)**: Evaluated item (`id`, `descriptive_id`, `state`: `COMPLIANT`, `NON_COMPLIANT`,
  or `ERROR`, plus rendered `violation_message` with violation explanation and remediation guidance when `state ==
  "NON_COMPLIANT"`).
- **Control Result (`ControlResult`)**: Per-control execution output (`control_id`, `source`, effective `parameters`,
  `evaluation_mode`, `subjects`, `errors`, optional `outcome`).
- **Invocation Trace & URI**: Sanitized on-disk JSON diagnostic record addressed by
  `vatuta://traces/policy/{construction|execution}/{invocation_id}`.

## Success Criteria *(mandatory)*

### Measurable Outcomes (Schematic)

- **SC-001 (Policy to YAML Rules)**: A reference policy with multiple obligations is split into atomic rules and
  generates complete `ControlDefinition` YAML files (`<id>.yaml`) with `tags`, shared parameters, and embedded
  `test_cases` in the target folder. Unrepresentable rules fail the entire creation with zero files written.
- **SC-002 (Creation Pipeline Verification)**: Automated tests verify policy splitting, `SharedParameterCatalog`
  immutability, bounded JSONata exploration (max 5 queries, 4096 bytes), mandatory `selection_tests` and
  `criterion_tests` (including `criterion.violation_message` rendering on `NON_COMPLIANT` test cases), `trial_attempts`
  (`0`, `1`, `>1`), `MISSING_CONSTRUCTION_CONTEXT` aggregation, and `NOT_APPLICABLE` / `NON_COMPLIANT` live trials.
- **SC-003 (Search & Merged Parameters)**: Across a nested folder of `.yaml` controls, `ClientControlQuery`, CLI
  `vatuta policy query`, and Skill Script 1 return identical filtered controls, relative file paths, projections, and
  `merged_parameters` when selecting by file, ID, glob, or metadata filters—with zero LLM or MCP calls.
  Path-scoped queries succeed when malformed files exist only outside the selected paths.
- **SC-004 (Pre-Flight Parameter Guard)**: Execution requests with missing required parameters or schema errors fail in
  `Stage E0` (Exit `2`), report all missing parameters and affected `control_id`s, and perform zero MCP calls.
- **SC-005 (Deterministic Execution & Actionable Violation Messages)**: Reference pull requests `101`, `102`, and `103`
  produce `COMPLIANT` (no violation message), `NON_COMPLIANT`, and `NON_COMPLIANT` deterministically with zero LLM
  calls, clear subject-contextualized `violation_message` strings (including violation explanation and remediation
  instructions) on `102` and `103`, and proper `violations` vs `all` filtering.
- **SC-006 (Runtime Failure Isolation & MCP Unavailability)**: In multi-control runs, incomplete evidence,
  tool errors from a responding server, and invalid expressions fail only the affected control or subject.
  Simulate a required server outage after one control completes. Verify the completed result is returned with
  a top-level `MCP_SERVER_UNAVAILABLE` error and exit code `1`, and pending controls make zero external calls.
- **SC-007 (Subject Edge Cases)**: Empty selection (`NOT_APPLICABLE`), root `null` (`ERROR`), invalid subjects
  (`INVALID_SUBJECT`), duplicate subject IDs (`DUPLICATE_SUBJECT_ID`), and all-compliant runs are distinguished without
  invented IDs or verdicts.
- **SC-008 (Evaluation Modes)**: Deterministic CEL and singleton `llm` controls run in their declared modes (with
  `{{parameters.<name>}}` template substitution in `llm` mode and deterministic `criterion.violation_message` formatting
  from the `subject` context on `NON_COMPLIANT` verdicts in both modes) and never switch modes at runtime.
- **SC-009 (Traceability & Self-Contained Results)**: Each creation or execution run attempts to save a sanitized
  stage trace, including inline execution and failed runs. A successful write yields a readable trace at the derived
  URI. On execution trace write failure, preserve calculated results and report `TRACE_WRITE_FAILED` (Exit `1`).
  On creation trace write failure, verify `controls: []`, `TRACE_WRITE_FAILED` (Exit `1`), and no generated YAML.
  Every returned `ControlResult` includes its `source` and effective `parameters`.
- **SC-010 (CLI & Service Parity)**: For equivalent creation and execution requests, CLI and service calls
  produce matching validation and results. CLI entry points map the same errors to the same exit codes.
  Guided `check` runs only confirmed controls whose file paths and IDs match, with zero evidence calls on
  incomplete or invalid selection.
- **SC-011 (Agent Skill Workflow)**: An `agentskills.io`-compatible agent can read `SKILL.md` and run the 3 skill
  scripts to: (1) create `<id>.yaml` rules in nested folders from a policy, (2) search rules and retrieve
  `merged_parameters` using file, ID, or glob selectors, and (3) execute selected rules with explicit parameters.

## Assumptions

- **Scope & Architecture**:
  - `PolicyService` is stateless and MCP-ready (`policy_create`, `policy_execute`, `read_trace`).
  - `vatuta policy serve` (MCP server process) is deferred; the CLI and the Agent Skill are implemented now.
  - Local `.yaml` files, folder hierarchy, filtering, `merged_parameters`, and `vatuta policy check` belong to the
    client layer.
- **No Speculative Lifecycle or NFRs**:
  - No rule update/merge engine, policy revision history, approval workflow, or automatic trace cleanup.
  - Re-creating from a policy is a fresh run; existing `<id>.yaml` collisions are rejected (`CONTROL_FILE_EXISTS`).
  - No arbitrary latency, throughput, or LLM extraction accuracy percentages are invented.

## Clarifications (Schematic Summary)

### Session 2026-11-21

- Q: What should happen when writing the creation trace fails?
  → A: Fail creation as a whole, return `controls: []` with `TRACE_WRITE_FAILED`, and write no generated YAML.

- Q: What should happen to completed results when a required MCP server becomes unavailable?
  → A: Stop the operation, keep completed results, and report a global failure with exit code 1.

- Q: What should happen if saving a set of generated control files fails after some files have been written? → A: Keep
  the files already written and report which files are missing.
- Q: When two selected controls share a parameter name but only one declares a default, how should execution resolve
  that parameter if the caller omits it? → A: Require an explicit value if any control requires the parameter without
  a default; otherwise use the declared default.
- Q: If a generated control fails a synthetic test and multiple trial attempts are allowed, should creation try to
  repair it? → A: Retry within the configured attempt limit; fail the whole creation if the control still fails.
- Q: What should `vatuta policy check` do if it runs out of queries before the model confirms the control selection?
  → A: Stop with `CONTROL_SELECTION_INCOMPLETE` before parameter extraction or evidence calls.
- Q: Should malformed YAML outside the selected paths block an operation? → A: Validate only files in the path scope.
- Q: How should `policy check` reload a control when the same ID exists in another folder?
  → A: Load the selected relative path and verify its ID. Selecting two paths with the same ID fails.

### 2026-10-25 to 2026-10-30 — Core Simplification, Inputs & Traces

- **Two service operations**: `create` and `execute` (+ `read_trace`), no approval or revision lifecycle.
- **Creation inputs**: `policy_text`, optional `policy_id` (fallback to `invocation_id`), `construction_context`,
  `trial_attempts`. Trial values (`construction_value`) do not become execution `default`s automatically.
- **Creation trials**: Error-free `NOT_APPLICABLE` or `NON_COMPLIANT` live trials pass. Missing trial context with
  `trial_attempts > 0` fails with `MISSING_CONSTRUCTION_CONTEXT`.
- **Trace URIs**: Derived from operation + `invocation_id` (`vatuta://traces/policy/{construction|execution}/{id}`).

### 2026-11-04 to 2026-11-11 — Stateless Service, Step-by-Step Flows & Review Decisions

- **Stateless `PolicyService`**: Accepts inline `ControlDefinition`s on `execute` and returns `ControlDefinition`s on
  `create`. No file paths or globs inside `PolicyService`.
- **Creation pipeline**:
  - Step 0 & Phase A (Step 2.1 across all rules): Local tool catalog only; incremental `SharedParameterCatalog`
    (existing shared parameters are immutable); collect all `MISSING_CONSTRUCTION_CONTEXT` errors before starting MCP.
  - Phase B (Steps 2.2–2.5 per rule, Fail-Fast): Bounded JSONata discovery (max 5 queries, 4096 bytes, `done=True`),
    mandatory synthetic `selection_tests` and `criterion_tests` saved in `test_cases`, and full-context retry reusing
    in-memory MCP evidence.
  - Client saving: Check all target `<id>.yaml` paths first; reject collisions `CONTROL_FILE_EXISTS` (Exit `1`).
- **Execution & Check pipeline**:
  - `ClientControlQuery`: Apply path selectors before reading YAML. Fail on malformed candidates
    (`INVALID_CONTROL_FILE`), combine filters (**AND** across fields, **OR** within lists), project fields, and compute
    `merged_parameters` (`include_merged_parameters=True`).
  - `vatuta policy check "<prompt>"`: Step 1 iterative rule selection (max 3 queries, 4096 bytes) -> reload selected
    paths and verify IDs -> Step 2 extract explicit prompt parameters (skipped if `merged_parameters` empty) ->
    Step 3 `execute` under one shared `invocation_id` trace. An unconfirmed selection at the query limit stops at Step 1.
  - `Stage E0` (request pre-flight: unique controls, parameter schemas, `MISSING_PARAMETER`, extra parameter keys
    allowed and scoped per control) vs `Stages E1–E3` (per-control failure isolation, MCP container reuse, exact
    evidence cache, `DUPLICATE_SUBJECT_ID` guard, singleton `llm` calls with `{{parameters.<name>}}` substitution).
  - Self-contained `ControlResult`: Always includes `source` and effective per-control `parameters`.
  - `MCP_SERVER_UNAVAILABLE`: Stops execution, keeps completed results, and exits `1`.
  - `TRACE_WRITE_FAILED`: Appends a global error and exits `1`. Execution retains calculated results. Creation
    returns `controls: []` and writes no generated YAML.

### 2026-11-19 — Agent Skill UX (`agentskills.io`), 3 Scripts & Schematic Format

- **Agent Skill Implemented**: Extend UX with an Agent Skill following `https://agentskills.io/` (`SKILL.md` +
  `scripts/`). While MCP transport remains prepared/deferred, the Agent Skill is implemented in this feature.
- **3 Skill Scripts (combining rule search and parameter retrieval)**:
  1. Search rules & get merged parameters (`query`).
  2. Create rules from a policy and save `<id>.yaml` files in a target folder (`create`).
  3. Execute selected `.yaml` rules with explicit parameters (`execute`).
- **Hierarchical Rule Folders & Consistent Selection**:
  - One `.yaml` file per rule (`<id>.yaml`), stored in a nested directory structure under `policy.controls_directory`
    that represents the user's use cases.
  - All scripts support selecting rules file by file (`--control`), by rule name/ID (`--id`), by glob pattern
    (`--glob` across subfolders), and by metadata filters.
- **Schematic Specification Style**: Use simple, accessible English and a schematic outline structure across all
  specification artifacts without losing technical completeness.

### Session 2026-11-20

- Q: Should the Agent Skill bundle **3 scripts** (combining rule search and parameter retrieval into one `query`
  script) or **4 distinct scripts** (`search`, `parameters`, `create`, `execute`)? (FR-049) → A: **3 scripts**
  (`query`, `create`, `execute`): `query` searches rules and optionally returns `merged_parameters` (enabled by
  default).
- Q: How should rule selection by file, rule name/ID, and glob pattern behave across the folder tree in all scripts?
  (FR-050) → A: **Wildcard IDs + relative path/filename glob**: `--control` (`files`) selects specific `.yaml` paths;
  `--id` (`ids`) matches exact `control.id` or glob patterns on `control.id` (e.g., `change-mgmt-*`); `--glob`
  (`glob`) matches relative paths (`team-a/**/*.yaml`) or bare filenames (`*github*.yaml`).
- Q: When rules are organized in nested subdirectories, how should the client handle two `.yaml` files in different
  folders that have the same `control.id`? (FR-012) → A: **Fail on duplicate ID in matched selection**: Allow same IDs
  in different folders on disk, but fail with `DUPLICATE_CONTROL_ID` (Exit `2`) if the active filter/selection matches
  two files with the same `control.id`.
- Q: Where should the implemented Agent Skill directory (`SKILL.md` + `scripts/`) live in the repository? (FR-048) → A:
  **Tracked `skills/vatuta-policy/` directory**: Store `SKILL.md` and the 3 scripts under `skills/vatuta-policy/` in
  version control so it can be copied or symlinked into any agent's skills folder.
- Q: How should the 3 scripts inside `skills/vatuta-policy/scripts/` be implemented in relation to the `vatuta policy`
  CLI? (FR-049) → A: **Thin Python entrypoints with PEP 723 metadata**: `query.py`, `create.py`, and `execute.py` in
  `skills/vatuta-policy/scripts/` include PEP 723 inline script metadata and share the exact client handler with CLI
  `vatuta policy query|create|execute --format json`.

### 2026-11-21 — Explanatory & Actionable Violation Messages via Subject Format Templates

- **Adapted Single Template with Violation Explanation + Remediation Guidance (`FR-034`, `FR-051`)**:
  - Every rule's `criterion` (`deterministic` and `llm`) declares a mandatory non-empty `violation_message` template
    string using safe Python `.format` / f-string style field placeholders (`{<field>}`, `{subject.<path>}`,
    `{parameters.<name>}`) populated from the analyzed `subject` (and control `parameters`) data context.
  - Rather than adding a second template field, `criterion.violation_message` is adapted to include both: (1) a clear
    explanation of why the `subject` violates the rule, and (2) concrete remediation hints or instructions so an AI
    agent or operator analyzing the policy check response knows how to resolve the violation on that `subject`.
  - Using a single safe dot-path template over the `subject` data context avoids schema bloat, runtime LLM calls, or
    unsafe `eval()`, keeps both deterministic and semantic rules 100% consistent, and grounds every violation and
    remediation hint in actual subject evidence fields and parameters.
- **Validation & Output (`FR-019`, `FR-046`)**:
  - Creation Step 2.4 validates `violation_message` rendering against all `NON_COMPLIANT` `criterion_tests` (and any
    live `NON_COMPLIANT` trial subjects when `trial_attempts > 0`).
  - Execution populates `SubjectResult.violation_message` whenever `state == "NON_COMPLIANT"` (omitting it for
    `COMPLIANT` and `ERROR`), and isolates any runtime rendering error to that subject (`VIOLATION_MESSAGE_RENDER_ERROR`,
    `state: "ERROR"`).
