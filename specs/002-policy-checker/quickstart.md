# Policy Validation Engine: Minimal E2E Acceptance Quickstart

**Date**: 2026-11-08
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

- Python 3.12 or 3.13 with Poetry installed (`poetry install --with dev`).
- Permanent automated tests already green locally (`poetry run pytest -q`).
- For Scenario 1A (live LLM control creation with `--trial-attempts 0`), a local `.env` with one configured LLM key
  (for example `GEMINI_API_KEY`); Scenario 1B and Scenario 2 require zero network or API credentials.

---

## Scenario 1 — Complete Happy-Path E2E (Create, Execute/Check, and Trace Read)

**Objective**: Confirm that Vatuta authors a self-tested YAML control from policy text, executes deterministic
compliance evaluation end-to-end through the CLI using existing reference fixtures, separates `stdout` JSON from
`stderr` progress, and persists a readable sanitized trace.

### 1A. Author Controls with Embedded Synthetic Unit Tests (`vatuta policy create`)

Run creation into a temporary directory with `--trial-attempts 0` (exercises DSPy decomposition, sequential rule
creation, and mandatory local execution of `selection_tests` and `criterion_tests` via real JSONata and CEL without
requiring live upstream MCP servers):

```bash
TMP_CONTROLS="$(mktemp -d)"
poetry run vatuta --config tests/fixtures/policy/vatuta.yaml policy create \
  --policy-text "Every open pull request must reference an approved issue." \
  --policy-id acceptance-change-mgmt \
  --construction-context "Repository example/project; issues in project EXAMPLE." \
  --tag acceptance \
  --trial-attempts 0 \
  --output-dir "$TMP_CONTROLS" \
  --format json > "$TMP_CONTROLS/create-result.json"
```

### 1B. Execute Compliance Check & Inspect Diagnostic Trace (`vatuta policy execute` + `read_trace`)

Execute the canonical [reference-control.yaml](contracts/reference-control.yaml) end-to-end against the deterministic
reference fixture responses (`PR 101` approved, `PR 102` pending, `PR 103` unlinked) and verify both `violations`
(default) and `all` filters plus trace retrieval:

```bash
poetry run python - <<'PY'
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from typer.testing import CliRunner

from src.client.client import app
from src.policy.service import PolicyService

fixtures = json.loads(Path("tests/fixtures/policy/reference_evidence.json").read_text(encoding="utf-8"))

async def fake_call_tool(name: str, arguments: dict | None = None):
    return fixtures[name]

runner = CliRunner()
with patch("src.mcp.server.MCPServer.start", new=AsyncMock()), \
     patch("src.mcp.server.MCPServer.stop", new=AsyncMock()), \
     patch("src.mcp.server.MCPServer.call_tool", side_effect=fake_call_tool):
    res = runner.invoke(
        app,
        [
            "--config", "tests/fixtures/policy/vatuta.yaml",
            "policy", "execute",
            "--control", "specs/002-policy-checker/contracts/reference-control.yaml",
            "--parameters", '{"repository":"example/project","project":"EXAMPLE"}',
            "--format", "json",
        ],
    )

assert res.exit_code == 0, res.stderr
payload = json.loads(res.stdout)
assert payload["errors"] == []
subjects = {s["id"]: s["state"] for s in payload["results"][0]["subjects"]}
assert subjects == {"102": "NON_COMPLIANT", "103": "NON_COMPLIANT"}, subjects

trace_uri = f"vatuta://traces/policy/execution/{payload['invocation_id']}"
trace = PolicyService.from_config_path("tests/fixtures/policy/vatuta.yaml").read_trace(trace_uri)
assert trace["invocation_id"] == payload["invocation_id"]
print("SCENARIO 1 PASSED:", payload["invocation_id"])
PY
```

### Expected Behavior & Pass/Fail Criteria

- **Pass**:
  1. `policy create` exits `0`, writes one `<id>.yaml` in `$TMP_CONTROLS`, `errors` is `[]`, `tags` includes
     `"acceptance"`, and `test_cases.selection_tests` / `test_cases.criterion_tests` are populated and verified.
  2. `policy execute` exits `0`, writes valid JSON exclusively to `stdout`, reports `102` and `103` as `NON_COMPLIANT`
     (omitting compliant `101` under the default `violations` filter), and emits no aggregate compliance percentage.
  3. `PolicyService.read_trace(trace_uri)` returns sanitized `InvocationTrace` JSON with matching `invocation_id`.
- **Fail**: Non-zero exit code, polluted `stdout` JSON, missing `102`/`103` violations, `101` reported as a violation,
  or unreadable/unsanitized trace output.

---

## Scenario 2 — Guardrail Edge Case (Reject Missing Required Inputs Before Evidence Acquisition)

**Objective**: Confirm that a control's `construction_value` never silently acts as a runtime default and that missing
mandatory parameters abort the entire request before any MCP or LLM call is attempted.

*(Justified as the single high-risk negative E2E check: silent fallback to authoring values would evaluate the wrong
repository/project and produce false compliance signals.)*

### Execution Instructions

```bash
poetry run vatuta --config tests/fixtures/policy/vatuta.yaml policy execute \
  --control specs/002-policy-checker/contracts/reference-control.yaml \
  --parameters '{}' \
  --format json > /tmp/vatuta-missing-params.json
echo "EXIT_CODE=$?"
poetry run python -c '
import json
data = json.loads(open("/tmp/vatuta-missing-params.json", encoding="utf-8").read())
missing = {e["parameter"] for e in data["errors"]}
assert data["results"] == [], data
assert missing == {"repository", "project"}, missing
assert all(e["control_id"] == "reference-approved-ticket" for e in data["errors"])
print("SCENARIO 2 PASSED:", sorted(missing))
'
```

### Expected Behavior & Pass/Fail Criteria

- **Pass**:
  1. Command exits with code `2` (invalid request / missing inputs) without requiring LLM keys or running Docker.
  2. JSON output has `results == []` and two `ErrorDetail` entries naming both `repository` and `project` along with
     `control_id: "reference-approved-ticket"`.
  3. Zero MCP container startups, evidence calls, or LLM calls occur.
- **Fail**: Exit code `0` or `1`, fallback to `example/project` or `EXAMPLE` `construction_value`s, partial missing
  parameter reporting, or any attempted Docker/MCP connection.
