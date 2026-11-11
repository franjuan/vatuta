# YAML Control and Evaluation Contract

**Date**: 2026-11-11
**Models**: [data-model.md](../data-model.md)
**Example**: [reference-control.yaml](reference-control.yaml)

One `ControlDefinition` Pydantic model represents one executable control across `PolicyService.create` and
`PolicyService.execute`, and serializes to one UTF-8 YAML document (`<id>.yaml`) when saved by the Vatuta client. Safe
loading must reject duplicate keys, multiple documents, unsafe tags, unknown fields, recursive aliases, and values
outside JSON. Models generate the implementation's JSON schemas; there is no separate hand-maintained schema framework
or policy schema-version lifecycle.

## Fields and Bindings

| Section | Required content |
| --- | --- |
| Identity | `id`, `description`, `tags` (array of nonempty strings), `source: {policy, rule, excerpt}`. |
| Parameters | Map of `{description, value_schema, construction_value?, default?}`. |
| Evidence | Array of `{name, server, tool, arguments}`. |
| Selection | JSONata expression string. |
| Criterion | `{mode: deterministic, expression}` or `{mode: llm, condition, deterministic_insufficiency}`. |
| Test cases | `test_cases: {selection_tests: [...], criterion_tests: [...]}` generated and verified during creation. |
| Creation check | `trial` is `skipped` or `passed`; `criterion_executed` is boolean; optional `outcome` is NOT_APPLICABLE. |

`value_schema` uses self-contained JSON Schema 2020-12. Any `$ref` must remain within that schema document; no network
resolution. Construction values and defaults validate strictly. Omitted default means required; explicit null is a
valid default only if permitted by its schema. Construction metadata does not bypass later mandatory validation.

Creation decomposes the policy into atomic normative rules (failing with `NO_NORMATIVE_RULES` if zero normative rules
are found), assigns each control a deterministic kebab-case `id` (`f"{policy_id}-{rule_id}"`, or `rule_id` if already
prefixed with `f"{policy_id}-"`), and derives only the minimum necessary parameter declarations (`parameters: {}` when
no parameter is needed) across all creation stages (`2.1`, `2.3`, `2.4`, `2.5`) using the incremental
`SharedParameterCatalog`. Two parameters share a name in `SharedParameterCatalog` only when their `value_schema`,
`construction_value`, and `default` match; otherwise the constructor adapts to the catalog or declares a distinct
parameter name. Declarations already shared with other rules remain immutable through later stages and retries.
Relevant `tags` are inferred and merged with any explicit request `tags`. With `trial_attempts == 0`,
`construction_value` is optional. Step 0 and Phase A always use only local `policy.evidence_profiles`; live tool schema
discovery is deferred until Phase A succeeds with enabled trials. In generated controls,
`source.policy` is the supplied `policy_id`, or the creation invocation ID when omitted; `source.rule` and
`source.excerpt` identify the obligation in the supplied policy text.

Each evidence argument is exactly `{literal: <JSON value>}` or `{parameter: <declared name>}`. A parameter reference
binds the entire value without text substitution. Required tool arguments must conform to its discovered input schema.
The discovered-schema check occurs before live evidence calls, never during the local-only creation preflight.
Evidence names are unique within a control; parameters and literals are the only evidence argument sources.

Selection's root context is `{evidence: {<name>: <normalized JSON>}, parameters: {<name>: <resolved value>}}`
(containing
only the parameters declared in that control's `control.parameters`). It can relate independently fetched collections.
It cannot call external tools. Evaluation guards disable ambient clock/random/dynamic-evaluation functions; policy time
must be an explicit parameter.

A deterministic criterion uses CEL activation `{subject: <selected object>, parameters: <resolved values>}`.
A semantic criterion (`mode: llm`) evaluates one subject per LLM call (`SemanticCriterionSignature`) and receives its
rendered `condition`, correlation reference (`subject_ref`), and that single `subject` JSON object only (no separate
`parameters` dictionary is sent to the LLM). However, `criterion.condition` may reference declared parameters via
`{{parameters.<name>}}` template placeholders, which the engine deterministically substitutes with the control's
resolved parameter values prior to calling the LLM. No whole evidence corpus, other subjects, or previous verdicts are
appended to semantic inputs. The declared mode cannot change at runtime.

## Embedded Synthetic Unit Tests (`test_cases`)

Every `ControlDefinition` contains `test_cases` with two nonempty arrays:

- **`selection_tests`**: Each item has `description`, `mock_evidence` (map keyed by declared evidence names), optional
  `parameters`, and either `expected_outcome: "NOT_APPLICABLE"` or `expected_subjects` (array of expected `Subject`
  objects with non-whitespace string `id` and `descriptive_id` and unique `id`s). Must include at least one populated
  subject case and one `NOT_APPLICABLE` case. When comparing against `expected_subjects`, a single selected JSON object
  `{...}` is normalized to a 1-element list `[{...}]`, and duplicate `id`s fail with `DUPLICATE_SUBJECT_ID`.
- **`criterion_tests`**: Each item has `description`, `subject` (a valid `Subject` object with non-whitespace `id` and
  `descriptive_id`), optional `parameters`, and `expected_state` (`"COMPLIANT"` or `"NON_COMPLIANT"`). Must include at
  least one `"COMPLIANT"` case and one `"NON_COMPLIANT"` case. Evaluated via CEL for `mode: deterministic` and via
  `SemanticCriterionSignature` (LLM, after rendering any `{{parameters.<name>}}` placeholders in `condition`) for
  `mode: llm`.

Unit test parameter resolution follows strict precedence: `1. test_case.parameters` -> `2. parameter.default` ->
`3. parameter.construction_value`. Both `selection_tests` and `criterion_tests` are executed and required to pass during
creation even when `trial_attempts == 0`.

## Selection and Criterion Results

- Evaluate the selection once using the internal existence envelope from [research](../research.md).
- No result or `[]`: NOT_APPLICABLE, no criterion call. Explicit root null: ERROR, no criterion call.
- One valid object: normalized to a 1-element list `[{...}]`. Array: validate each member independently.
- `id` and `descriptive_id` must be non-empty, non-whitespace strings (`isinstance(s, str) and len(s.strip()) > 0`);
  source numeric identifiers can be explicitly converted in Selection. The engine never invents an identifier for a
  missing field. Other fields may contain any JSON value, including null.
- Within a control's selection output, subjects with a unique valid `id` are evaluated by `criterion`. Invalid
  objects/scalars (`INVALID_SUBJECT`) and subjects sharing a duplicate `id` (`DUPLICATE_SUBJECT_ID`, reported with
  `subject_id` and `selection_index`) are **not** evaluated by `criterion`, are recorded in `ControlResult.errors`, and
  produce global control `outcome="ERROR"`.
- Criterion results must be strict booleans; true/false map to COMPLIANT/NON_COMPLIANT. Errors and ambiguous semantic
  outputs map to ERROR, never truthiness or automatic non-compliance. Under `result_filter="violations"` (default),
  `ControlResult.subjects` retains only `NON_COMPLIANT` subjects while `ControlResult.errors` retains all control and
  subject errors.

## Reference Evidence Fixtures

The example uses fixture-only tool names `reference_prs` and `reference_issues`. They are not claimed to be tool names
or response schemas of a live GitHub/Jira server. Each returns `{items: [...], complete: true|false}` for a documented
bounded fixture scope. This metadata is a test contract, not a universal completeness flag.

A trusted test profile binds each fixture tool to the assertion `data.complete = true`, its fixture contract reference,
and the exact input scope. Unknown/missing `complete` fails the proof. A real deployment must replace the profile with
an assertion justified by the exact deployed tool contract, not relabel a production response as this fixture.

Reference data contains PRs `101`, `102`, `103` linked to an approved issue, a pending issue, and no issue. Selection
must retain all PRs, including the one with no related issue. Its `ticket_status: null` is evaluated as non-compliance.
Expected findings under `violations` are `102` and `103`; `all` includes `101` as compliant.

## Live Tool Profile Examples (Configuration Only — Not Core Dependencies)

Core policy evaluation (`src/policy/`) is strictly source-independent. Live upstream MCP tools (such as GitHub and Jira
in the temporary blog demonstration under `pocs/policy_demo/`, see [demo-plan.md](../demo-plan.md)) are bound purely
through YAML `policy.evidence_profiles` configuration with explicit pagination arguments:

| Example tool | Positive single-page completeness assertion (`complete_when`) |
| --- | --- |
| GitHub `search_pull_requests` | Page 1; `items` array; integer `total_count >= 0` equal to `$count(items)`; boolean `incomplete_results = false`. |
| GitHub `list_commits` | Page 1 with explicit `perPage > 0`; `data` array; `$count(data) < arguments.perPage` (under-limit single page). |
| Jira `jira_search` | First page, no next token; exact known total, or a documented unknown-total cursor proof. |
| GitHub `list_pull_requests` | Unsupported when array count equals page size or lacks explicit page-bound arguments; unknown completeness yields ERROR. |

The proof is a conjunction, not a fallback: missing/wrong-type metadata, truncation flags, subsequent-page requests,
or hitting `perPage`/`limit` without an exact total must fail (`false` → incomplete-evidence ERROR). Never add
GitHub-specific or Jira-specific branches to `src/policy/` or require live SaaS credentials in GitHub Actions CI.

For Jira, a nonnegative integer `total` requires exact equality with the issue count; an under-limit response cannot
override a contradictory known total. Only a documented unknown-total cursor response (such as `total == -1` in the
pinned tool contract), with first-page arguments, no continuation token, and an explicit positive limit, may use
`$count(issues) < arguments.limit`. Missing or wrongly typed metadata without a documented proof remains ERROR.
