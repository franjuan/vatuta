# Policy Validation Engine: Research

**Date**: 2026-11-11
**Scope**: [spec.md](spec.md), including the stateless service, client tool, LLM flows, and source-independent testing
**Outcome**: Technical choices resolved; library conformance and local E2E checks belong to the implementation phase.

## R1 — Stateless Service Boundary and MCP-Ready Design

**Decision**: Implement a stateless `PolicyService` class (`src/policy/service.py`) exposing two domain operations
(`create` and `execute`) plus `read_trace`, using JSON-serializable Pydantic v2 request and result models with
`Field(description=...)` metadata. The Vatuta client (`src/client/policy.py`) invokes `PolicyService` directly in this
increment; `vatuta policy serve` and the stdio `FastMCP` server wrapper are deferred to a follow-up increment. Keep
upstream source access through `src/mcp/server.py:MCPServer` (which is an MCP client despite its class name). Do not
route policy execution through the conversational RAG `vatuta ask` command or its stringifying MCP wrapper.

**Evidence**: `src/client/client.py` has no implemented policy command. Its root callback loads configuration and
validates every LLM backend before any command. `State.provider_manager` already permits `None`.
`MCPServer.call_tool` returns the SDK result unchanged, preserving `structuredContent`, `content`, and `isError`.
A local probe of installed MCP 1.30.0 confirmed `FastMCP` inspects Pydantic v2 models (`BaseModel`, `Literal`,
`Optional`, `Field(description=...)`) to generate `inputSchema`, `outputSchema`, and `structuredContent` directly.

**Rationale**: Making `PolicyService` stateless and free of file-system paths (`Path`), sets, or host-root state ensures
zero-friction 1:1 wrapping as MCP tools (`policy_create`, `policy_execute`) and `resources/read` later. Removing
`policy_parameters` from the service boundary eliminates the need for a server-side control repository. Skip eager LLM
preflight only for the new `policy` CLI group; preserve existing commands' behavior. Creation, `vatuta policy check`,
and semantic execution obtain the selected backend lazily through `LLMProviderManager`.

**Alternatives considered**: Keeping file/glob selectors (`--root`) or a control repository inside the service/MCP
boundary couples the engine to host disk paths. Reusing the general RAG `vatuta ask` loop would mix document retrieval
with policy verification. Implementing `policy serve` now was explicitly deferred by the stakeholder.

## R2 — JSONata Selection, Exploratory Projections, and Completeness Expressions

**Decision**: Select `jsonata-python==0.7.1` and its `jsonata.Jsonata` API. Use the standard language to evaluate a
selection once inside an internal envelope: `($v := (<expression>); {"exists": $exists($v), "value": $v})`.
Classify absent output from `exists=false`, explicit root null from `exists=true,value=null`, and an empty array
separately. Reuse the same guarded JSONata evaluator for bounded exploratory projections over in-memory evidence during
rule creation (governed by configurable `policy.discovery_max_queries`, default 5 queries per rule, with each output
serialized to JSON and truncated to configurable `policy.discovery_max_output_bytes`, default 4096 bytes, with an
explicit `[TRUNCATED]` marker when exceeded).

**Evidence**: [PyPI 0.7.1](https://pypi.org/project/jsonata-python/0.7.1/) identifies a pure-Python distribution,
Python >=3.10, Apache-2.0 licensing, and release on 2026-10-10, older than the constitution's seven-day minimum.
The [JSONata existence function](https://docs.jsonata.org/boolean-functions) provides the missing-value distinction.
The binding is not installed in this workspace; release-specific conformance runs belong to implementation.

**Rationale**: JSONata is already an accepted design constraint. The envelope preserves the approved distinction
without implementing a parser. Using JSONata projections (`$keys($)`, `prs.items[0]`, `$type(prs.items)`) with a 5-query
budget and 4 KB output truncation solves scalable evidence discovery during creation without overflowing LLM token
limits when upstream tools return hundreds of records. Disable ambient-time/random/dynamic-evaluation builtins (`now`,
`millis`, `random`, `shuffle`, `eval`) in each expression's binding environment; reject their invocation, including
aliases. Do not register caller-supplied host functions. Time-dependent policies take explicit time parameters.

Use the binding's execution-time guard with the existing MCP call deadline of 60 seconds as the operational ceiling,
not a new performance target. Use the RE2 engine already required by CEL for regex evaluation, preventing an individual
regex from escaping the expression deadline.

**Alternatives considered**: Dumping raw evidence JSON or full inferred schemas into the creation prompt fails to scale
for large tool responses. Unbounded exploratory queries could stall creation or exceed token budgets.

## R3 — CEL Boolean Evaluation

**Decision**: Select `cel-python==0.5.0` (`celpy`) with its existing compiler/program and JSON conversion APIs. Compile
once per supplied control; evaluate with only `subject` and `parameters`. Accept only CEL boolean results; convert
CEL evaluation errors to subject ERROR. Never apply Python truthiness to an arbitrary result.

**Evidence**: [PyPI 0.5.0](https://pypi.org/project/cel-python/0.5.0/) and
[CEL Python documentation](https://cloud-custodian.github.io/cel-python/) describe a Python CEL implementation with
JSON conversion, boolean types, and evaluation errors. Release metadata supports Python >=3.10; the 2026-01-31 release
satisfies the age rule. The binding is not installed locally; 3.12/3.13 and null/error behavior require conformance
tests.

**Rationale**: Reuse the accepted expression language; do not implement policy evaluation with Python `eval`, custom
operators, or JSON Schema. The deterministic context contains no clock, model access, or external functions.

**Alternatives considered**: General Python execution violates the declarative boundary; an additional semantic engine
or JEV package is not needed to deliver the explicitly declared LLM exception.

## R4 — Step-by-Step Creation & Guided 2-Step Check LLM Flows (DSPy)

**Decision**: Reuse `LLMProviderManager.get_dspy_lm` and the existing backend registry. Structure LLM interactions into
focused, atomic DSPy Signatures (`src/policy/signatures/`):

1. **Creation Step 1 — `PolicyDecompositionSignature`**:
   - Decomposes `policy_text` and `construction_context` into atomic normative rules (`rule_id`, `description`,
     `excerpt`, `inferred_tags`), ignoring non-normative preamble/scope prose. If zero normative rules are found,
     creation fails immediately with `NO_NORMATIVE_RULES` (`stage: "decomposition"`).
   - Enforces the two splitting rules: (a) if two conditions can fail independently and a user wants to know which one
     failed, split them into separate rules; (b) if the policy author would consider them two distinct violations, split
     them into separate rules.
   - Derives each control's deterministic `id` as `f"{policy_id}-{rule_id}"` (or `rule_id` when already prefixed with
     `f"{policy_id}-"`, deduplicated within the creation output).
2. **Creation Step 2 — Two-Phase Orchestration with `SharedParameterCatalog`**:
   - **Phase A (All-Rules Step 2.1 Sweep — Collect-All `MISSING_CONSTRUCTION_CONTEXT`)**: Runs
     `RuleEvidenceAndParametersSignature` across all rules `r_1 ... r_N` first with the incremental
     `SharedParameterCatalog`, declaring only the minimum necessary parameters (`parameters: {}` when none are needed).
     `value_schema`, `construction_value`, and `default` must all match to share a parameter name; otherwise the
     constructor aligns to the catalog or uses a distinct name. Declarations already shared with other rules are
     immutable through later stages and retries; failed candidate additions are discarded. Step 0 and Phase A always
     use local `policy.evidence_profiles` (`arguments_schema` + `contract_reference`), without MCP startup, connection,
     or `list_tools`. With zero attempts, `construction_value` is optional and creation never contacts MCP.
     When `trial_attempts > 0`, any missing required `construction_value`s across all
     rules are aggregated and returned at once (`MISSING_CONSTRUCTION_CONTEXT`) before any MCP call occurs.
   - **Phase B (Per-Rule Steps 2.2–2.5 — Fail-Fast on Execution/Generation Failure)**:
     - **`EvidenceDiscoverySignature` (Step 2.2 — Bounded JSONata Exploration)**: After Phase A succeeds with enabled
       trials, connect to required servers and validate discovered `inputSchema` before evidence calls. Fetch
       declared evidence once in memory (cached per rule by `(server, tool, resolved_arguments)`) and runs up to
       `policy.discovery_max_queries` (default 5) JSONata expressions against the root context `{"evidence": ...,
       "parameters": ...}`, truncating each output to `policy.discovery_max_output_bytes` (default 4096 bytes +
       `[TRUNCATED]`). Query errors or `missing` results are returned as observations so the model can refine the next
       query; setting `done=True` closes the loop early without burning remaining queries.
     - **`RuleSelectionSignature` (Step 2.3)**: Can declare additional parameters (validated against
       `SharedParameterCatalog`) and generates `test_cases.selection_tests` and the JSONata `selection` expression.
       Validates `selection` immediately against all `selection_tests` (normalizing single-object JSONata outputs to
       `[obj]`, requiring non-whitespace `id`/`descriptive_id`, rejecting duplicate `id`s with `DUPLICATE_SUBJECT_ID`,
       and resolving parameters with `test_case.parameters` -> `default` -> `construction_value` priority) and against
       in-memory live evidence (when `trial_attempts > 0`).
     - **`RuleCriterionSignature` (Step 2.4)**: Can declare additional parameters (validated against
       `SharedParameterCatalog`) and generates `test_cases.criterion_tests` and `criterion` (`deterministic` CEL or
       justified `llm`). Validates `criterion` immediately against all `criterion_tests` (using CEL for
       `deterministic` and invoking `SemanticCriterionSignature` with `{{parameters.<name>}}` template substitution in
       `condition` for `mode: llm`, even when `trial_attempts == 0`) and against live trial subjects (when
       `trial_attempts > 0`).
     - **Full-Context Retry & Fail-Fast (Step 2.5)**: When `trial_attempts > 1` and a rule fails in Steps 2.2–2.4,
       re-executes from Step 2.1 carrying the full accumulated context and errors of all prior stages (`prior_attempt`),
       reusing in-memory fetched MCP evidence whenever `(server, tool, resolved_arguments)` are unchanged. If a rule
       exhausts its attempts, Phase B aborts immediately (**Fail-Fast**) without running Steps 2.2–2.5 for remaining
       rules.
3. **Runtime Semantic Evaluation — `SemanticCriterionSignature`**:
   - Singleton semantic subject judgment (`mode: llm`, 1 subject per call) with correlation token verification and
     deterministic `{{parameters.<name>}}` template substitution in `condition` prior to the LLM call (no separate
     `parameters` dict is sent to the LLM).
4. **Client Guided 2-Step Check Flow (`vatuta policy check "<prompt>"`)**:
   - Associates Steps 1, 2, and `PolicyService.execute` with a single `invocation_id` trace
     (`vatuta://traces/policy/execution/{invocation_id}`).
   - **Step 1 — `PolicyCheckRuleSelectionSignature`**: Iteratively queries `ClientControlQuery` with lightweight
     projection (`["id", "description", "tags", "source"]`, `include_merged_parameters=False`), governed by configurable
     `policy.check_max_filter_queries` (default 3 queries) and `policy.check_max_output_bytes` (default 4096 bytes,
     marked `[TRUNCATED]` if exceeded, exiting early when `done=True`), to select matching `control_id`s from
     `policy.controls_directory` (failing with `NO_CONTROLS_MATCHED` if empty). The client then deterministically loads
     those controls with `projection=["full"]` and `include_merged_parameters=True`.
   - **Step 2 — `PolicyCheckParameterExtractionSignature`**: Skipped when `merged_parameters` is empty (`parameters =
     {}`); otherwise receives the user's prompt and `merged_parameters` and extracts the explicit JSON `parameters` map
     present in the prompt (without inventing missing values) before the client calls `PolicyService.execute`.

Disable `ChatAdapter`'s JSON-adapter fallback for policy calls (`use_json_adapter_fallback=False`) and disable hidden
provider repair/retry paths for semantic judgments. Use roles `policy_creation`, `policy_evaluation`, and
`policy_check_agent`, keeping actual model identities outside YAML controls.

**Evidence**: Installed DSPy is 3.3.0. The existing provider caches by backend/role and adds role metadata for metrics.

**Rationale**: Decomposing creation into atomic steps with immediate synthetic unit testing (`selection_tests` and
`criterion_tests`) catches malformed JSONata selections and invalid CEL/LLM criteria at the exact step they occur, and
persisting `test_cases` in `ControlDefinition` makes every saved YAML rule self-testing. Passing an incremental
`SharedParameterCatalog` guarantees consistent parameter naming across rules of the same policy without a fragile
post-hoc renaming pass. Structuring `vatuta policy check "<prompt>"` into a deterministic 2-step pipeline (Rule
Selection -> deterministic `merged_parameters` load -> Parameter Extraction -> `PolicyService.execute`) avoids
unpredictable open-ended agent loops while scaling to large local control directories.

**Alternatives considered**: One-shot monolithic control generation struggles with complex evidence structures and
parameter consistency across rules. Free-form ReAct loops in `vatuta policy check` add latency and nondeterministic
tool invocations compared to the guided 2-step pipeline.

## R5 — Client Control Files, Filtering/Projection Tool, and Shared Parameters

**Decision**: Use Pydantic v2 models (`extra="forbid"`) with JSON-compatible types (`str`, `int`, `float`, `bool`,
`list`, `dict`, `Literal`, `Optional`) for `ControlDefinition` and all service requests/results. Include
`tags: list[str]` and `test_cases: ControlTestCases` (`selection_tests` and `criterion_tests`) on `ControlDefinition`.
Use self-contained JSON Schema 2020-12 (`jsonschema` 4.26.0) for parameter `value_schema`.

Manage YAML control files on the client side under configurable `policy.controls_directory` (default `data/controls`,
overridable via `--controls-dir`). Provide a client-side filtering and projection tool (`src/policy/files.py` and
`src/policy/parameters.py`) that:

- Safely loads YAML controls beneath `controls_directory` (resolving symlinks, enforcing root containment, and failing
  immediately with `INVALID_CONTROL_FILE` if any encountered YAML control file is malformed),
- Filters controls by `ids`, `policy_ids`, `rule_ids`, `tags`, `evaluation_mode`, `evidence_server`, `evidence_tool`,
  `parameter_names`, `text_query` (case-insensitive match in `description` and `source.excerpt`), `files`, or `glob`
  (combining distinct filter fields with logical **AND** and multi-value list fields with logical **OR**),
- Projects requested attributes (`projection`, e.g., `["id", "description", "tags", "source"]` or `["full"]`), and
- Accepts `include_merged_parameters: bool = True` (enabled by default) to return `merged_parameters` with
  deduplicated shared parameter schemas, required/default status, referencing `control_id`s, and conflict errors. Extra
  parameter keys in `request.parameters` not declared by any supplied control are permitted, passing each control only
  its own declared parameters.

**Rationale**: Keeping file storage, filtering, projection, and parameter inspection in the client allows both the CLI
(`vatuta policy execute`) and the 2-step prompt flow (`vatuta policy check "<prompt>"`) to query local rules flexibly
while `PolicyService` stays 100% stateless.

**Alternatives considered**: A server-side rule repository or a separate `policy_parameters` MCP tool was rejected to
keep the service/MCP boundary stateless.

## R6 — Evidence, Completeness, and Source Qualification

**Decision**: Keep started `MCPServer` containers alive across controls within a single `PolicyService.execute`
invocation and cache valid responses in memory by exact `(server, tool, resolved_arguments)`. Validate
`evidence_profile` existence, tool allowlist, `arguments_schema`, and discovered `inputSchema` per control in `Stage
E1`. Check `isError` first, then prefer structured content; otherwise parse exactly one complete JSON text block.
Validate evidence completeness using trusted deployment-owned profiles keyed by configured server and tool. A profile
contains its documented scope basis and a deterministic JSONata assertion over normalized `data`, resolved `arguments`,
and available response metadata. Only boolean true establishes completeness. False means incomplete; missing/invalid
proof means unknown. If any evidence declaration in a control fails validation, acquisition, conversion, or
completeness, fast-fail that control immediately without calling its remaining evidence declarations, Selection, or
Criterion. Generated controls cannot authorize or override a completeness profile. No automatic pagination,
provider-specific executor adapter, or count heuristic.

**Evidence**: The repository has no configured GitHub/Jira MCP reference schemas or response recordings; example
configuration only includes `everything` and `wikipedia`. Upstream
[GitHub pull-request tools](https://github.com/github/github-mcp-server/blob/main/pkg/github/pullrequests.go) show that
`list_pull_requests` returns a page of records while discarding REST pagination metadata. It cannot establish total
scope completeness merely by returning a short/empty array.

The same upstream repository declares `search_pull_requests` with the
[SearchIssuesOutput](https://github.com/github/github-mcp-server/blob/main/pkg/github/issues.go#L2035) fields
`total_count`, `incomplete_results`, and `items`. A positive proof requires page 1, an array, an explicitly false
incomplete flag, and a nonnegative integer total equal to the item count.

The upstream
[Jira search result model](https://github.com/sooperset/mcp-atlassian/blob/main/src/mcp_atlassian/models/jira/search.py)
serializes `total`, `start_at`, `max_results`, `issues`, and optional `next_page_token`. On Jira Server/DC offset
responses, require a nonnegative `total == $count(issues)` at `start_at = 0`; on Jira Cloud `/search/jql` cursor
responses with the documented `total == -1` sentinel, single-page exhaustion requires page 1 (`page_token` absent/null),
absent/null
`next_page_token`, and `$count(issues) < arguments.limit` under an explicit positive `limit`.

A valid nonnegative integer total always requires exact equality; a short response cannot override a contradictory
known total. Unknown-total inference is available only for the documented cursor contract, never for arbitrary
missing/wrong-type metadata. For example, `issues=[]`, `total=5`, `limit=50` must fail completeness.

**Rationale**: A configurable tool contract is the minimal provider-neutral implementation of FR-018. Reuse JSONata
rather than adding another language. Unknown tools remain explicit ERROR. Core tests use deterministic provider-neutral
fixtures in `tests/fixtures/policy/`; live GitHub and Jira MCP containers are used exclusively in the isolated temporary
blog demonstration ([demo-plan.md](demo-plan.md) / `pocs/policy_demo/`) and never in core `src/` or GitHub Actions CI.

**Alternatives considered**: Missing warnings, response length, guessed pagination flags, and model judgments do not
establish completeness.

## R7 — Security, Diagnostics, and Operational Bounds

**Decision**: Reuse hardened Docker source lifecycles and anchored tool allowlists. Require policy evidence profiles to
list authorized read-only tools explicitly. Container `read_only` protects its filesystem, not the remote API; source
credentials must also be read-only. Pin deployed source images by digest; do not modify unrelated server settings.

Reuse logging setup, external stream capture, and LLM metrics. Add a small policy trace writer because current logging
does not persist stage artifacts or redact arbitrary JSON payloads. Keep traces under `logs/policy/<invocation_id>/`,
with no expiry, and expose trace reading via `PolicyService.read_trace(uri)` following the operation contract's URI
templates.

**Rationale**: Apply a sanitization boundary before any persistence: redact credentials and configured sensitive JSON
paths, use consistent pseudonyms for identifying fields, and mask free-text prompt/evidence content unless its trace
projection is explicitly approved as non-sensitive. Record redaction markers instead of pretending retained data is raw.
Sequential calls, bounded creation attempts, bounded exploratory JSONata queries (<= 5 per rule, <= 4 KB output),
bounded check filter queries (<= 3 per check), existing token limits, source timeouts, and finite call budgets bound
work. Reuse the existing rate limiter for external calls.

**Alternatives considered**: Logging unsanitized arbitrary responses would violate existing privacy constraints. A
shared cache, background cleanup, parallel executor, new network service, or policy revision store is unnecessary.

## R8 — Confirmed Review Decisions (2026-11-11)

**Decision**: Keep the creation catalog local until Phase A succeeds; freeze declarations shared with other rules;
reject existing client output files; include `source` and effective per-control `parameters` in each `ControlResult`;
retain calculated results with global `TRACE_WRITE_FAILED`/exit `1` if execution trace persistence fails; and provide a
complete standalone configuration template for the optional demo. These six choices were explicitly confirmed by the
stakeholder and are recorded in the specification's 2026-11-11 clarification session.

**Rationale**: Preflight must not contact sources before missing construction context is reported. Immutable shared
declarations avoid invalidating other rules during repair. Collision rejection protects manually inspected controls.
Self-contained results preserve provenance and effective inputs even when their diagnostic trace is unavailable.
Standalone demo configuration makes the documented commands reproducible without a personal configuration file.

Use an internal per-invocation context to share the check/execution ID without changing public request schemas. A check
failure before execution returns that ID, empty results, and errors, and attempts to save the same trace. Invocation
diagnostic errors live at the top level; source/subject runtime errors remain attached to their controls. Trace failure
does not rewrite compliance states. Existing privacy redaction applies to results and diagnostics.

**Alternatives considered**: Early live schema discovery, mutating shared declarations with cascading revalidation,
automatic or opt-in overwriting, trace-only result attribution, discarding calculated results on trace-write failure,
and copying a personal configuration for the demo were not selected. No new lifecycle or persistence service is added.

## Evidence and Delivery Limits

Local probes confirmed installed MCP/Pydantic/JSON Schema APIs and inspected DSPy/provider behavior without source or
model calls. JSONata/CEL release metadata and upstream APIs were inspected, but neither package was installed or
executed. No dependency file, lockfile, production source, or runtime configuration changed during planning. All model
and external calls in unit/contract tests must be mocked. Deployment-source and binding qualification are explicit
implementation checks, not unresolved stakeholder behavior questions or claims that the feature already works.
