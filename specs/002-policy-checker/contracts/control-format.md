# YAML Control and Evaluation Contract (Schematic)

**Date**: 2026-11-20
**Models**: [data-model.md](../data-model.md) | **Example**: [reference-control.yaml](reference-control.yaml)

## 1. File & Model Basics

- **1 Control = 1 YAML File (`<id>.yaml`)**:
  - Backed by the `ControlDefinition` Pydantic v2 model (`extra='forbid'`).
  - Saved by the client under `policy.controls_directory` (or `--output-dir`), supporting nested subfolders organized by
    user use case.
- **Safe YAML Loading Rules**:
  - Reject duplicate keys, multiple documents, unsafe tags, unknown fields, recursive aliases, and non-JSON values.
  - Pre-check all target paths before saving: if any `<id>.yaml` already exists, abort with `CONTROL_FILE_EXISTS` (Exit
    `1`) and overwrite nothing.

## 2. ControlDefinition Sections

| Section | Required Content |
| --- | --- |
| Identity | `id` (`f"{policy_id}-{rule_id}"`), `description`, `tags` (`list[str]`), `source: {policy, rule, excerpt}`. |
| Parameters | Map of `{description, value_schema, required?, construction_value?, default?}`. May be empty. |
| Evidence | Non-empty list of `{name, server, tool, arguments}` with unique `name`s in the control. |
| Selection | JSONata expression string over `{evidence, parameters}`. |
| Criterion | Both modes require `violation_message`. See the criterion context below for mode-specific fields. |
| Test Cases | `test_cases: {selection_tests: [...], criterion_tests: [...]}` (mandatory, verified during creation). |
| Creation Check | `{trial: "skipped" \| "passed", criterion_executed: bool, outcome?: "NOT_APPLICABLE"}`. |

## 3. Parameter, Evidence & Expression Bindings

- **Parameter Schemas & Sharing**:
  - `value_schema` uses self-contained JSON Schema 2020-12 (no network `$ref`, no string-to-number/bool coercion).
  - A parameter with no `default` requires an explicit value unless `required: false` is set.
  - `default: null` is valid only if `value_schema` permits `null`.
  - `construction_value` is for creation trials only (optional when `trial_attempts == 0`) and never acts as an
    execution default.
  - Across rules created from the same policy, `SharedParameterCatalog` shares a parameter name only when
    `value_schema`, `construction_value`, and `default` match; shared entries are immutable in later rules and retries.
- **Evidence Argument Bindings**:
  - Each argument in `evidence[].arguments` is either `{literal: <JSON>}` or `{parameter: <declared_name>}` (whole-value
    binding, no string template inside evidence arguments).
  - If a bound optional parameter is absent after resolution, omit that argument. Keep explicit JSON null values.
    The evidence profile and live tool schema still validate the final argument object before the call.
  - Step 0 and Phase A validate against local `policy.evidence_profiles` only; live MCP `inputSchema` validation runs
    before calling tools in Phase B (`trial_attempts > 0`) and `execute` (`Stage E1`).
- **Selection Context (JSONata)**:
  - Root context: `{evidence: {<name>: <normalized_json>}, parameters: {<name>: <resolved_value>}}` (scoped to this
    control's declared parameters).
  - Non-deterministic builtins (`now`, `millis`, `random`, `shuffle`, `eval`) are disabled.
- **Criterion Context (CEL vs LLM)**:
  - **`mode: "deterministic"` (CEL)**: Evaluated with `{subject, parameters}` activation.
  - **`mode: "llm"` (DSPy `SemanticCriterionSignature`)**: Evaluates 1 unique-ID valid subject per LLM call, passing
    rendered `condition`, `subject_ref`, and `subject` (no raw `parameters` dict). Supports deterministic
    `{{parameters.<name>}}` template substitution in `condition` before calling the LLM.

### Violation message template

Both criterion modes require a non-empty `violation_message` template. The template explains what failed and
how to correct it. Fill direct subject keys, nested mapping paths, `{subject.<path>}`, and
`{parameters.<name>}` by looking up JSON values. Use `{{` and `}}` for literal braces. Render explicit
JSON null as `null`. Reject missing fields, undeclared parameters, invalid placeholders, and empty output.
Reject expression evaluation, method calls, and dunder or Python attribute access. Validate rendering on every
`NON_COMPLIANT` synthetic test and live trial result.

## 4. Embedded Synthetic Unit Tests (`test_cases`)

Mandatory for every control (executed and required to pass even when `trial_attempts == 0`):

- **`selection_tests`**:
  - Fields: `description`, `mock_evidence`, optional `parameters`, and either `expected_outcome: "NOT_APPLICABLE"` or
    `expected_subjects` (list of valid `Subject` dicts with non-whitespace `id`/`descriptive_id` and unique `id`s).
  - Must include at least 1 populated case and 1 `NOT_APPLICABLE` case.
  - Single selected JSON object `{...}` normalizes to `[{...}]` before comparing; duplicate `id`s fail with
    `DUPLICATE_SUBJECT_ID`.
- **`criterion_tests`**:
  - Fields: `description`, `subject` (valid `Subject` dict), optional `parameters`, `expected_state` (`"COMPLIANT"` or
    `"NON_COMPLIANT"`).
  - Must include at least 1 `"COMPLIANT"` and 1 `"NON_COMPLIANT"` case.
  - Runs CEL for `deterministic` and `SemanticCriterionSignature` (with `{{parameters.<name>}}` substitution) for
    `llm`.
- **Unit-Test Parameter Priority**:
  `1. test_case.parameters` -> `2. parameter.default` -> `3. parameter.construction_value`.

## 5. Selection & Criterion Output Rules

- **Selection Outcomes**:
  - `exists=false` (no result) or `[]` -> control `outcome: "NOT_APPLICABLE"`, criterion skipped.
  - Explicit root `null` (`exists=true, value=null`) or root scalar -> control `outcome: "ERROR"`, criterion skipped.
  - Single object `{...}` -> normalized to `[{...}]`.
- **Subject Validation & Duplicate ID Guard**:
  - Every subject requires non-empty, non-whitespace string `id` and `descriptive_id` (`len(s.strip()) > 0`).
  - Unique valid `id`s run `criterion`.
  - Invalid objects (`INVALID_SUBJECT`) and duplicate `id`s (`DUPLICATE_SUBJECT_ID` with `subject_id` and
    `selection_index`) skip `criterion`, record errors in `ControlResult.errors`, and set control `outcome: "ERROR"`.
- **Criterion Verdicts**:
  - Strict boolean `true` -> `COMPLIANT`, `false` -> `NON_COMPLIANT`.
  - Non-boolean, exception, or mismatched `subject_ref` -> `ERROR` (never coerced via truthiness).

## 6. Reference Evidence Fixtures & Live Profile Examples

- **Hermetic Test Fixtures (`tests/fixtures/policy/`)**:
  - Use provider-neutral fixture tools `reference_prs` and `reference_issues` with `complete_when: "data.complete =
    true"`.
  - Canonical PRs: `101` (`Approved` issue -> `COMPLIANT`), `102` (`Pending` issue -> `NON_COMPLIANT`), `103` (no issue,
    `ticket_status: null` -> `NON_COMPLIANT`).
- **Live Tool Completeness Profiles (`policy.evidence_profiles` — Configuration Only)**:
  - Core `src/policy/` has zero GitHub/Jira code. Live profiles in `pocs/policy_demo/vatuta.demo.yaml` use JSONata
    `complete_when` proofs:
    - GitHub `search_pull_requests`: Page 1, `items` array, `total_count >= 0`, `total_count = $count(items)`,
      `incomplete_results = false`.
    - GitHub `list_commits`: Page 1, explicit `perPage > 0`, `data` array, `$count(data) < arguments.perPage`.
    - Jira `jira_search`: Page 1, no `next_page_token`, and either exact nonnegative `total = $count(issues)` or
      documented cursor sentinel `total = -1` with `$count(issues) < arguments.limit`. A short array never overrides a
      contradictory nonnegative `total`.
