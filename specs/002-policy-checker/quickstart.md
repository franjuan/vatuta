# Policy Validation Engine: Minimal E2E Acceptance Quickstart (Schematic)

**Date**: 2026-11-21
**Status**: Acceptance guide for the planned hermetic tests. Run the commands after implementation.

## 1. Purpose & Boundaries

- **Goal**: Verify end-to-end (E2E) wiring across CLI (`create`, `query`, `execute`, `check`), Agent Skill scripts
  (`skills/vatuta-policy/scripts/{create,query,execute}.py`), `PolicyService`, and trace persistence (`read_trace`).
- **Scope**:
  - Uses local hermetic fixtures (`tests/fixtures/policy/`) with zero Docker containers, network calls, or SaaS tokens.
  - Exhaustive unit/contract cases live in `tests/policy/`.
  - Live GitHub Free + Jira Cloud Free demo lives exclusively in [demo-plan.md](demo-plan.md) (`pocs/policy_demo/`).
- **Authoritative Contracts**: [operations.md](contracts/operations.md) and
  [control-format.md](contracts/control-format.md).

## 2. Prerequisites & Fixture Setup

- **Environment**: Python 3.12 or 3.13 via Poetry (`poetry run pytest -q`).
- **Standalone Fixture Config (`tests/fixtures/policy/vatuta.yaml`)**:
  - Includes required `rag`, `qdrant`, empty `sources`, and local reference MCP profiles (`reference_prs`,
    `reference_issues`).
  - Uses temporary `controls_directory` (with nested subfolder support) and `trace_directory`.
  - Deterministic `query` and `execute` must NOT initialize RAG, embeddings, Qdrant, or LLM backends.
- **Canonical Evidence (`tests/fixtures/policy/reference_evidence.json`)**:
  - PR `101`: linked `Approved` issue -> `COMPLIANT`.
  - PR `102`: linked `Pending` issue -> `NON_COMPLIANT`.
  - PR `103`: no linked issue (`ticket_status: null`) -> `NON_COMPLIANT`.
- **Boundary Fakes (Only MCP Transport & Authoring LLM Stubbed)**:
  - `MCPServer.start` / `stop`: async no-op fakes (zero Docker calls).
  - `MCPServer.list_tools`: returns SDK `ListToolsResult(tools=[Tool(...)])` with `inputSchema` requiring string
    `repository` and `project`.
  - `MCPServer.call_tool`: returns SDK `CallToolResult(content=[], structuredContent=payload, isError=False)`.
  - LLM call counters: allow controlled DSPy authoring outputs only during `create`; assert zero LLM calls during
    `query` and deterministic `execute`.

---

## Scenario 1 — Complete Happy-Path E2E (Create -> Query -> Execute -> Read Trace)

- **Objective**: Verify that a policy is split and saved as `<id>.yaml` in a target subfolder (`create`), discovered and
  inspected for `merged_parameters` via wildcard ID/glob (`query`), executed under both result filters (`execute`), and
  audited via `PolicyService.read_trace(uri)` across both CLI and Agent Skill entrypoints.
- **Command**:

  ```bash
  poetry run pytest -q tests/client/test_policy.py::test_create_query_execute_and_read_trace
  ```

- **Steps**:
  1. **Create (`create` / `skills/vatuta-policy/scripts/create.py`)**:
     - Policy: `"Every open pull request must reference an approved issue."`
     - Args: `--policy-id acceptance-change-mgmt`, `--construction-context "Repository example/project; issues in
       project EXAMPLE."`, `--tag acceptance`, `--trial-attempts 0`, `--output-dir <tmp_controls>/team-a`,
       `--format json`.
     - Runs real creation pipeline and synthetic `selection_tests` + `criterion_tests` with zero MCP calls.
  2. **Search & Inspect Parameters (`query` / `skills/vatuta-policy/scripts/query.py`)**:
     - Args: `--controls-dir <tmp_controls>`, `--id "acceptance-change-mgmt-*"`, `--glob "team-a/*.yaml"`,
       `--include-merged-parameters`.
     - Returns the saved control (with relative `file: "team-a/<id>.yaml"`) and `merged_parameters` (`repository`,
       `project` required) with zero LLM or MCP calls.
  3. **Execute (`execute` / `skills/vatuta-policy/scripts/execute.py`)**:
     - Run twice on the saved control with `--parameters '{"repository":"example/project","project":"EXAMPLE"}'`:
       - Default `--result-filter violations` -> returns only `102` and `103` (`NON_COMPLIANT`).
       - `--result-filter all` -> returns `101` (`COMPLIANT`), `102` (`NON_COMPLIANT`), and `103` (`NON_COMPLIANT`).
       - Findings for `102` and `103` explain the linked issue status and how to resolve it. The message for
         `103` renders the status as `null`. The result for `101` omits `violation_message`.
  4. **Read Trace (`PolicyService.read_trace`)**:
     - Resolve `vatuta://traces/policy/execution/{invocation_id}` and verify sanitized JSON trace and secret redaction.
     - Use a synthetic credential in a trace fixture to check redaction on disk and via `read_trace`. Verify the
       command's stdout contains one parseable JSON response even when the MCP fake emits a diagnostic.
- **Pass Criteria**:
  - Exit code `0` on `create`, `query`, and both `execute` runs; pure JSON on `stdout`, progress on `stderr`.
  - Saved `<id>.yaml` in `team-a/` contains verified `test_cases` and tag `"acceptance"`.
  - `ControlResult` includes `source` and effective `parameters` under both filters.
  - Zero MCP calls during `create` (`trial_attempts=0`) and `query`; zero LLM calls during `query` and `execute`.

---

## Scenario 2 — Pre-Flight Guardrail (Reject Missing Required Parameters in Stage E0)

- **Objective**: Confirm that `construction_value` never acts as an execution default and missing required parameters
  abort in `Stage E0` (Exit `2`) before any MCP or LLM call.
- **Command**:

  ```bash
  poetry run pytest -q tests/client/test_policy.py::test_missing_parameters_rejects_before_external_calls
  ```

- **Steps**:
  - Invoke `execute` (and `skills/vatuta-policy/scripts/execute.py`) on [reference-control.yaml](contracts/reference-control.yaml)
    with `--parameters '{}'` and `--format json`.
- **Pass Criteria**:
  - Exit code `2`.
  - `stdout` JSON has `results == []` and two `MISSING_PARAMETER` errors (`repository` and `project`) referencing
    `control_id: "reference-approved-ticket"`.
  - Zero calls to `MCPServer.start`, `list_tools`, `call_tool`, or any LLM backend.

### Variant — Mixed defaults

Copy [reference-control.yaml](contracts/reference-control.yaml) twice into a temporary controls directory.
Use distinct IDs and matching filenames (`reference-a.yaml` and `reference-b.yaml`). Keep the parameter schemas.
Give both controls a `project` default of `EXAMPLE` and give only the first a `repository` default of
`example/project`. Invoke the commands in the in-process test harness with the fixtures and fakes described above.
Use these argument lists, with `fixture_dir` set to the temporary directory.

```bash
poetry run vatuta policy query --controls-dir "$fixture_dir" --id "reference-*" --format json
poetry run vatuta policy execute --controls-dir "$fixture_dir" --id "reference-*" --parameters '{}' --format json
```

1. Expect one `repository` entry with both control IDs, `required: true`, and no `default`. Execution returns
   exit `2` with `MISSING_PARAMETER` for `repository` and zero MCP or LLM calls.
2. Set `required: false` on the second control's `repository` declaration and repeat. The query now reports
   `required: false` and `default: "example/project"`. Execution uses that value in both controls.

---

## Scenario 3 — Scoped search and confirmed control selection

Run the planned CLI acceptance test using local YAML and the fake model and MCP transport.

```bash
poetry run pytest -q tests/client/test_policy.py::test_check_selection_is_scoped_and_confirmed
```

Copy [reference-control.yaml](contracts/reference-control.yaml) into `team-a/` and `team-b/` under the temporary
controls root. Keep the same ID in both files and use distinct descriptions to identify the variant. Place broken YAML
in `scratch/broken.yaml`. Configure the fake model to search `team-a/*.yaml` and select the returned file and ID.
Supply explicit `repository` and `project` values through the fake parameter extraction step.

- A query scoped to the valid file or glob succeeds. A query that includes `scratch/broken.yaml` returns
  `INVALID_CONTROL_FILE` (Exit `2`).
- Confirmation after the last allowed filter query succeeds. Only the `team-a` variant is passed to execution.
  Its results and the execution trace use the same `invocation_id` as rule selection.
- Repeat with a model that never confirms, including one that offers provisional choices. Expect
  `CONTROL_SELECTION_INCOMPLETE` (Exit `1`).
- Repeat with an ID changed between search and reload, or with a pair that never appeared in a query response.
  Expect `INVALID_CONTROL_SELECTION` (Exit `1`).
- Confirm an empty list to get `NO_CONTROLS_MATCHED` (Exit `2`). Discover the two variants via separate path-scoped
  queries and confirm both to get `DUPLICATE_CONTROL_ID` (Exit `2`).

On any selection failure, check that `results` is empty, the pre-execution trace can be read using the returned
`invocation_id`, and neither the parameter extractor nor the MCP fake was called.
