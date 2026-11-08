# Policy Validation Engine: Research

**Date**: 2026-11-08
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
     `excerpt`, `inferred_tags`).
   - Enforces the two splitting rules: (a) if two conditions can fail independently and a user wants to know which one
     failed, split them into separate rules; (b) if the policy author would consider them two distinct violations, split
     them into separate rules.
2. **Creation Step 2 (Per Rule, Sequential with Incremental `SharedParameterCatalog`)**:
   - **`RuleEvidenceAndParametersSignature`**: Given the current normative rule, `construction_context`, permitted MCP
     tool schemas, and the cumulative `SharedParameterCatalog` from earlier rules of the same policy, deduces parameters
     (reusing existing parameter names and schemas when the concept matches), `construction_value`s, optional explicit
     `default`s, and the `evidence` declarations (`server`, `tool`, `arguments`).
   - **`EvidenceDiscoverySignature` (Configurable Bounded JSONata Exploration)**: When `trial_attempts > 0`, fetches
     the declared evidence once in memory using `construction_value`s and lets the constructor issue up to
     `policy.discovery_max_queries` (default 5) JSONata discovery expressions against the in-memory evidence, receiving
     outputs truncated to at most `policy.discovery_max_output_bytes` (default 4096 bytes, marked `[TRUNCATED]` if
     exceeded).
   - **`RuleSelectionSignature`**: Generates `test_cases.selection_tests` (`mock_evidence` -> `expected_subjects` or
     `NOT_APPLICABLE`) and the JSONata `selection` expression. Validates `selection` immediately against all
     `selection_tests` (always mandatory, even when `trial_attempts == 0`) and against the in-memory live evidence
     (when `trial_attempts > 0`) to verify the resulting `Subject` objects.
   - **`RuleCriterionSignature`**: Given the verified `Subject` structure, generates `test_cases.criterion_tests`
     (`subject` -> `COMPLIANT` and `NON_COMPLIANT`) and the `criterion` (`deterministic` CEL or justified `llm`).
     Validates `criterion` immediately against all `criterion_tests` (always mandatory) and against the live trial
     subjects (when `trial_attempts > 0`).
   - **`RuleCandidateCorrectionSignature`**: Bounded repair signature invoked when a synthetic unit test or live trial
     fails within the configured attempt budget.
3. **Runtime Semantic Evaluation — `SemanticCriterionSignature`**:
   - Singleton semantic subject judgment (`mode: llm`) with correlation token verification.
4. **Client Guided 2-Step Check Flow (`vatuta policy check "<prompt>"`)**:
   - **Step 1 — `PolicyCheckRuleSelectionSignature`**: Uses `ClientControlQuery` with lightweight projection (`["id",
     "description", "tags", "source"]`, `include_merged_parameters=False`), governed by configurable
     `policy.check_max_filter_queries` (default 3 queries) and `policy.check_max_output_bytes` (default 4096 bytes,
     marked `[TRUNCATED]` if exceeded), to select matching `control_id`s from `policy.controls_directory`. The client
     then deterministically loads those controls with `projection=["full"]` and `include_merged_parameters=True`.
   - **Step 2 — `PolicyCheckParameterExtractionSignature`**: Receives the user's prompt and `merged_parameters` and
     extracts the explicit JSON `parameters` map present in the prompt (without inventing missing values) before the
     client calls `PolicyService.execute`.

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
- Safely loads YAML controls beneath `controls_directory` (resolving symlinks and enforcing root containment),
- Filters controls by `ids`, `policy_ids`, `rule_ids`, `tags`, `evaluation_mode`, `evidence_server`, `evidence_tool`,
  `parameter_names`, `text_query` (case-insensitive match in `description` and `source.excerpt`), `files`, or `glob`,
- Projects requested attributes (`projection`, e.g., `["id", "description", "tags", "source"]` or `["full"]`), and
- Accepts `include_merged_parameters: bool = True` (enabled by default) to return `merged_parameters` with
  deduplicated shared parameter schemas, required/default status, referencing `control_id`s, and conflict errors.

**Rationale**: Keeping file storage, filtering, projection, and parameter inspection in the client allows both the CLI
(`vatuta policy execute`) and the 2-step prompt flow (`vatuta policy check "<prompt>"`) to query local rules flexibly
while `PolicyService` stays 100% stateless.

**Alternatives considered**: A server-side rule repository or a separate `policy_parameters` MCP tool was rejected to
keep the service/MCP boundary stateless.

## R6 — Evidence, Completeness, and Source Qualification

**Decision**: Check `isError` first, then prefer structured content; otherwise parse exactly one complete JSON text
block. Validate evidence completeness using trusted deployment-owned profiles keyed by configured server and tool.
A profile contains its documented scope basis and a deterministic JSONata assertion over normalized `data`, resolved
`arguments`, and available response metadata. Only boolean true establishes completeness. False means incomplete;
missing/invalid proof means unknown. Both stop that control before Selection. Generated controls cannot authorize
or override a completeness profile. No automatic pagination, provider-specific executor adapter, or count heuristic.

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
responses (where `total` can be `-1`), single-page exhaustion requires page 1 (`page_token` absent/null), absent/null
`next_page_token`, and `$count(issues) < arguments.limit` under an explicit positive `limit`.

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

## Evidence and Delivery Limits

Local probes confirmed installed MCP/Pydantic/JSON Schema APIs and inspected DSPy/provider behavior without source or
model calls. JSONata/CEL release metadata and upstream APIs were inspected, but neither package was installed or
executed. No dependency file, lockfile, production source, or runtime configuration changed during planning. All model
and external calls in unit/contract tests must be mocked. Deployment-source and binding qualification are explicit
implementation checks, not unresolved stakeholder behavior questions or claims that the feature already works.
