# Policy Validation Engine: Minimal E2E Acceptance Quickstart

**Date**: 2026-11-11
**Status**: Local post-implementation acceptance runbook (not executed in GitHub Actions CI).

## Purpose and Scope Boundaries

This document defines the **minimal end-to-end (E2E) acceptance scenarios** used to verify locally that the complete
Policy Validation Engine works cohesively once implemented.

- **Not a unit/contract test duplicate**: Exhaustive rule, expression, schema, and error-branch coverage lives in the
  permanent `pytest` suite (`tests/policy/`, `tests/client/test_policy.py`) executed automatically in GitHub Actions.
- **Not the blog demonstration**: Live GitHub Free and Jira Cloud Free showcasing belongs exclusively to the temporary
  blog demonstration in [demo-plan.md](demo-plan.md) (`pocs/policy_demo/`).
- **Local execution**: Run these two scenarios locally during feature acceptance; they require no CI secrets, Docker
  orchestration in GitHub Actions, or third-party SaaS accounts.

Contracts and schemas are authoritative in [operations.md](contracts/operations.md) and
[control-format.md](contracts/control-format.md).

## Prerequisites

- Python 3.12 or 3.13 with Poetry and the development dependencies already installed.
- Permanent automated tests already green locally (`poetry run pytest -q`).
- Both scenarios use local fixtures and controlled authoring responses; neither needs Docker, network, or credentials.
- Implement the two focused cases below in the already planned `tests/client/test_policy.py` CLI wiring tests.
  The commands are post-implementation validation entry points, not tests available or executed during planning.
  Reuse those permanent hermetic tests rather than adding a separate acceptance suite.
- The planned `tests/fixtures/policy/vatuta.yaml` must be a complete standalone configuration with `rag` backend
  declarations and bindings, `qdrant`, empty `sources`, and local reference MCP profiles. Use temporary controls and
  trace directories. Keep the generic configuration loader unchanged; policy execution must not initialize RAG,
  embeddings, Qdrant, or LLM providers for deterministic controls.

### Fixture Contract for Both Scenarios

Use [reference-control.yaml](contracts/reference-control.yaml) and the planned
`tests/fixtures/policy/reference_evidence.json` as the canonical deterministic example: PR `101` links an `Approved`
issue, PR `102` links a `Pending` issue, and PR `103` has no linked issue. Keep their identifiers stable.

The hand-authored reference declares `id: reference-approved-ticket`. Execution uses that declared identifier.
For creation, assert the generated `<policy_id>-<rule_id>` convention against the supplied policy ID and returned rule.

Replace only MCP transport and authoring model boundaries. Keep CLI parsing, creation orchestration, synthetic tests,
JSONata, CEL, file saving/loading, execution, and trace persistence real. The planned fixtures must provide:

- `MCPServer.start` and `stop` async fakes, so no container or Docker client is contacted.
- An async `MCPServer.list_tools` fake returning SDK `ListToolsResult(tools=[Tool(...)])`, with `name` and `inputSchema`
  for both `reference_prs` and `reference_issues`. Require string `repository` and `project`, respectively, matching the
  local profile schemas. Execution discovers these schemas before calling tools; an empty list or raw dict is invalid.
- Async `call_tool` responses as SDK `CallToolResult(content=[], structuredContent=payload, isError=False)` objects.
  Each payload contains the canonical records and explicit completeness proof required by its local evidence profile.
  Load serialized SDK envelopes with `CallToolResult.model_validate(record)`; do not return bare evidence dictionaries.
- Call counters and failure-on-use guards on real LLM/provider entry points and network/transport boundaries. Controlled
  authoring responses are allowed only during creation; execution must make zero LLM calls.

---

## Scenario 1 — Complete Happy-Path E2E (Create, Execute, and Trace Read)

**Objective**: Confirm that Vatuta authors a self-tested YAML control from policy text, executes deterministic
compliance evaluation end-to-end through the CLI using existing reference fixtures, separates `stdout` JSON from
`stderr` progress, and persists a readable sanitized trace.

### 1A. Author a Control with Embedded Synthetic Unit Tests (`vatuta policy create`)

The focused case invokes the real CLI via `CliRunner` with the policy text "Every open pull request must reference an
approved issue.", `--policy-id acceptance-change-mgmt`, construction context "Repository example/project; issues in
project EXAMPLE.", `--tag acceptance`, `--trial-attempts 0`, `--format json`, and a temporary `--output-dir`.

Supply controlled DSPy signature responses derived from the reference obligation, including its deterministic
selection, criterion, and synthetic cases. The real creation pipeline must run every synthetic test and save its
returned control. Do not substitute the reference YAML for the saved result when executing the roundtrip.

Assert zero MCP starts, `list_tools`, and `call_tool` calls throughout this creation. Step 0 catalogs are local for
every trial setting; live discovery can start only after the all-rules Phase A sweep succeeds. With trials disabled,
creation never contacts MCP. Shared parameter declarations already used by other rules remain immutable; later rules
must match
them or use distinct names. Saving rejects filename/ID collisions without overwriting existing controls.

### 1B. Execute Compliance Check & Inspect Diagnostic Trace (`vatuta policy execute` + `read_trace`)

In the same case, load and execute the actual control saved in 1A with explicit
`{"repository":"example/project","project":"EXAMPLE"}` parameters and the SDK-shaped MCP fakes above. Invoke the
CLI twice: once without `--result-filter` to test default `violations`, then with `--result-filter all`. Read each
execution trace using `PolicyService.read_trace` and its returned invocation ID. Run the complete case with:

```bash
poetry run pytest -q tests/client/test_policy.py::test_create_execute_and_read_trace
```

### Expected Behavior & Pass/Fail Criteria

- **Pass**:
  1. `policy create` exits `0`, writes one `<id>.yaml` in the temporary output directory, `errors` is `[]`, `tags`
     includes
     `"acceptance"`, and `test_cases.selection_tests` / `test_cases.criterion_tests` are populated and verified.
  2. `policy execute` exits `0`, writes valid JSON exclusively to `stdout`, reports `102` and `103` as `NON_COMPLIANT`
     (omitting compliant `101` under default `violations`). With `all`, assert exactly `101: COMPLIANT`,
     `102: NON_COMPLIANT`, and `103: NON_COMPLIANT`. Both runs have no request/control errors or aggregate percentage.
  3. Each `ControlResult` preserves the saved control's `source` and the effective execution `parameters` under both
     filters. Its `control_id` matches the generated YAML, proving the create-to-execute roundtrip.
  4. `PolicyService.read_trace(trace_uri)` returns sanitized `InvocationTrace` JSON with matching `invocation_id`.
     Check redacted fixture secrets are absent and progress is confined to `stderr`; parse all `stdout` as JSON.
  5. MCP schema discovery and evidence fakes are called during execution, while all LLM call counters remain zero.
- **Fail**: Non-zero exit code, polluted `stdout` JSON, missing `102`/`103` violations, `101` reported as a violation,
  or unreadable/unsanitized trace output.

Trace write failure belongs to focused permanent error tests, not an additional E2E scenario: retain computed results,
append top-level `TRACE_WRITE_FAILED`, and exit `1`. Do not replace successful evaluations with empty results.

---

## Scenario 2 — Guardrail Edge Case (Reject Missing Required Inputs Before Evidence Acquisition)

**Objective**: Confirm that a control's `construction_value` never silently acts as a runtime default and that missing
mandatory parameters abort the entire request before any MCP or LLM call is attempted.

*(Justified as the single high-risk negative E2E check: silent fallback to authoring values would evaluate the wrong
repository/project and produce false compliance signals.)*

### Execution Instructions

The focused case invokes `policy execute` through `CliRunner` using the standalone fixture config, the canonical
reference control, `--parameters '{}'`, and `--format json`. Use failure-on-use mocks and counters for MCP start,
`list_tools`, `call_tool`, and every LLM/provider entry point. Keep request validation real.

```bash
poetry run pytest -q tests/client/test_policy.py::test_missing_parameters_rejects_before_external_calls
```

### Expected Behavior & Pass/Fail Criteria

- **Pass**:
  1. Assert `result.exit_code == 2` (invalid request / missing inputs); printing the exit code is insufficient.
  2. JSON output has `results == []` and two `ErrorDetail` entries naming both `repository` and `project` along with
     `control_id: "reference-approved-ticket"`.
  3. Assert every external-call mock is uncalled, including MCP start, schema discovery, evidence calls, and LLM calls.
     This must pass with credentials absent and Docker unavailable; both errors must be reported together.
- **Fail**: Exit code `0` or `1`, fallback to `example/project` or `EXAMPLE` `construction_value`s, partial missing
  parameter reporting, or any attempted Docker/MCP connection.
