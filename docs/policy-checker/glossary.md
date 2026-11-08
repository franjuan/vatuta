# Policy Validation Engine: Glossary

**Updated**: 2026-11-04
**Specification**: [spec.md](../../specs/002-policy-checker/spec.md)
**Current scope decision**: [ADR-019](adr-019-file-based-policy-operations.md)

## Policy and Control Artifacts

- **Policy / Policy Document**: Normative content supplied as a supported document or inline policy text. Initially
  supports UTF-8 plain text and Markdown. It may contain multiple obligations.
- **Policy ID**: Optional nonempty `policy_id` used as `source.policy` in generated controls and diagnostics. Defaults
  to the creation invocation ID; it does not address a registry entry, select files, or establish policy versions.
- **Normative Rule**: An atomic obligation extracted during policy decomposition. Conditions that can fail
  independently and require separate failure attribution, or that the policy author would consider distinct violations,
  are decomposed into separate normative rules.
- **Construction Context**: Optional descriptive `construction_context` used with `policy_text` to generate controls,
  infer tags, define parameters, and derive supported trial values. It may describe repositories, projects, time ranges,
  or examples without knowing future parameter names. It does not weaken policy obligations or establish implicit
  runtime defaults.
- **Control (`ControlDefinition`)**: An identifiable structured definition (and client-saved YAML file) declaring `id`,
  `description`, `tags`, `source` provenance, `parameters`, `evidence`, `selection`, `criterion` with an explicit
  deterministic or LLM evaluation mode, `test_cases` (`selection_tests` and `criterion_tests`), and `creation_check`.
- **Control ID**: Identity (`id`) used to attribute a control's definition and results. Within a single `execute`
  invocation, supplied `ControlDefinition` IDs must be unique. No identity matching across separate generations or
  version-management semantics are required.
- **Control Tags (`tags`)**: List of nonempty string labels on a `ControlDefinition`. During creation, the constructor
  generates relevant tags from the policy and `construction_context` and merges any caller-supplied `tags` from the
  creation request.
- **Control Test Cases (`ControlTestCases`)**: Synthetic unit tests generated, executed, and persisted in
  `ControlDefinition.test_cases` during creation: `selection_tests` (synthetic `mock_evidence` testing `selection` for
  populated subjects and `NOT_APPLICABLE` cases) and `criterion_tests` (synthetic `subject` objects testing `criterion`
  for `COMPLIANT` and `NON_COMPLIANT` states).
- **Control Set**: The complete list of `ControlDefinition` objects returned by creation, or a caller-supplied list of
  `ControlDefinition` objects passed to execution. This is not a published revision or a server-side registry entity.
- **Client Controls Directory (`policy.controls_directory`)**: Client-managed local directory (default
  `data/controls`, overridable via `--controls-dir`) where the Vatuta client saves `<id>.yaml` files after creation
  and queries them before execution. `PolicyService` does not read or manage this directory.
- **Client Filtering and Projection Tool**: Client-side function and agent tool (`ClientControlQuery`) that filters
  local YAML controls in `controls_directory` by metadata/text criteria (`ids`, `policy_ids`, `rule_ids`, `tags`,
  `evaluation_mode`, `evidence_server`, `evidence_tool`, `parameter_names`, `text_query`, `files`, `glob`), projects
  requested attributes (`projection`, including `["full"]`), and optionally returns `merged_parameters`
  (`include_merged_parameters=True` by default).
- **Manual Inspection**: A user reads the client-saved YAML files in `controls_directory`. This is not a managed
  review, approval, editing, or publication operation.

## Parameters and Invocation Preparation

- **Parameter Declaration**: A named input with semantics (`description`), expected value (`value_schema`), optional
  `construction_value`, and optional `default`, declared during creation from policy, construction context, and
  permitted tools. Trial values remain distinct from defaults.
- **Shared Parameter Catalog (`SharedParameterCatalog`)**: Incremental registry passed from one rule to the next during
  sequential per-rule creation so conceptually identical inputs across rules of the same policy reuse the exact same
  parameter name and compatible `value_schema`.
- **Shared Parameter**: Same-name declarations denote the same parameter across selected controls. One invocation value
  applies to every referencing control; per-control overrides are not required.
- **Parameter Semantics**: The meaning and intended use of an input, allowing an invoking agent to collect a suitable
  value from the user's prompt or context.
- **Construction Value / Trial Parameter Value**: A value derived during construction from the policy or construction
  context for generation or enabled trials, recorded as `construction_value`. It is not a caller-supplied named map or
  an invocation default unless the latter is explicitly declared.
- **Declared Default**: A value explicitly available when an invocation omits that input. A supplied value takes
  precedence. Without a supplied value or applicable default, a required parameter is missing.
- **Resolved Parameter Value**: The explicit invocation value or applicable declared default used in execution.
  It must satisfy the declaration, configured scope, and permissions.
- **Merged Parameters (`merged_parameters`)**: Deterministic aggregation of shared parameters across controls matched
  by the client's filtering and projection tool when `include_merged_parameters=True` (default). Reports parameter
  schemas, required/default status, referencing `control_id`s, and declaration conflicts without model or evidence
  calls. Replaces the earlier standalone `policy_parameters` MCP tool.
- **Policy Check Flow (`vatuta policy check "<prompt>"`)**: Guided 2-step client LLM pipeline (separate from the
  general RAG `vatuta ask` command): Step 1 uses `ClientControlQuery` with lightweight projection (bounded by
  configurable `policy.check_max_filter_queries`, default `3`, and `policy.check_max_output_bytes`, default `4096`
  bytes) to select applicable `control_id`s, then deterministically loads full controls and `merged_parameters`; Step 2
  extracts explicit parameter values from the prompt against `merged_parameters` and invokes `PolicyService.execute`.
- **Invocation Validation**: Checks the complete supplied `ControlDefinition` list (including unique `id`s) and its
  parameters before evidence acquisition. Missing mandatory values reject the whole invocation, identifying every
  missing input, meaning, and affected `control_id`. Incompatible same-name declarations are explicit input errors.

## Evidence, Selection, and Evaluation

- **Evidence Declaration**: A configured MCP server/tool call, evidence identifier, and explicit invocation arguments.
  Calls do not depend on evidence or control results. Connection settings and credentials stay outside controls.
- **Evidence**: Provider-neutral JSON obtained through the declared tools. Structured content takes precedence;
  otherwise exactly one text block containing a complete JSON value is accepted. Tool errors remain errors.
- **Exploratory Evidence Projection**: During creation when `trial_attempts > 0`, bounded JSONata queries (configurable
  via `policy.discovery_max_queries`, default `5`, each truncated to `policy.discovery_max_output_bytes`, default
  `4096` bytes, with a `[TRUNCATED]` marker when exceeded) run against in-memory source evidence so the constructor can
  inspect data structure, keys, types, and sample records without overflowing LLM context.
- **Evidence Completeness**: Whether the tool contract or response establishes complete evidence for the control's
  declared scope. Explicitly incomplete or unknown-completeness evidence produces ERROR before subject evaluation.
  No warning is not proof of completeness; automatic pagination is not implied.
- **Selection**: A JSONata transformation relating evidence into subjects. A no-result output or empty collection
  is NOT_APPLICABLE; root null is a contract ERROR. Scalar wrapping and invented identifiers are not allowed.
- **Subject**: A selected JSON object with `id` and `descriptive_id`; other fields may include null. A missing related
  record can be evaluated as a violation when complete evidence supports its absence.
- **Criterion**: One condition reused independently for each valid subject. Its definition declares deterministic
  CEL evaluation or exceptional LLM evaluation; true/false means compliant/non-compliant and failures mean ERROR.
- **Semantic Criterion**: A criterion requiring meaning-based judgment that cannot be resolved deterministically.
  Creation records the semantic condition, mode, and insufficiency reason; execution follows that declaration.
- **Semantic Batch**: Independent evaluations of one criterion grouped in a model request. Each subject retains its
  own evidence, judgment, and response mapping. No collective verdict or shared evidence inference is introduced.
- **Determinism**: Equivalent deterministic results for fixed control definitions, resolved inputs, and source
  responses. It does not promise unchanged sources or identical live LLM judgments between invocations.

## Operations and Results

- **Stateless Policy Service (`PolicyService`)**: Application service class implementing the exact MCP-ready interface
  (`create` / `policy_create`, `execute` / `policy_execute`, and `read_trace`) over JSON-serializable Pydantic models,
  with no file-system control repository. Invoked directly by the Vatuta client in this increment (`policy serve` is
  deferred).
- **Creation / Authoring (`PolicyService.create`)**: Decomposes `policy_text` into atomic normative rules and
  constructs each `ControlDefinition` sequentially (parameters/evidence with `SharedParameterCatalog` -> bounded
  in-memory JSONata evidence exploration -> `selection` + `selection_tests` -> `criterion` + `criterion_tests`).
  Returns structured `ControlDefinition` objects; the client may save them as YAML files.
- **Creation-Time Trial**: A test execution in the control's declared mode against live evidence when
  `trial_attempts > 0`, alongside mandatory synthetic `selection_tests` and `criterion_tests` unit tests (which always
  run, even when `trial_attempts == 0`). Explicit request `trial_attempts` overrides deployment setting
  `policy.trial_attempts` for one creation; when omitted, use deployment configuration, default one. One attempt has no
  repair retry; zero skips live source trials while keeping mandatory structural, permission, and synthetic unit-test
  checks.
- **Missing Construction Context**: A creation error explaining information needed for enabled trials, including
  parameter semantics/schema and affected controls. The caller can expand context in a new request. No controls,
  invented values, or deferred-trial state are returned.
- **Validation Exhaustion**: A control still fails after allowed creation attempts. The entire creation fails,
  retaining diagnostics but returning no reusable set or subset. Missing required trial values also fail creation.
- **Policy Execution Engine (`PolicyService.execute`)**: Receives `ControlDefinition` objects inline and explicit
  parameters, obtains declared evidence, selects subjects, evaluates the declared criterion mode, and reports results
  by `control_id`. It does not interpret the invocation prompt, read control files, choose modes, or repair controls at
  runtime.
- **Verification Invocation**: Execution of caller-supplied `ControlDefinition` objects with explicit parameters and
  applicable defaults. After invocation validation, failures remain per control, including source outages; independent
  controls continue.
- **Subject Result**: Technical ID, descriptive ID, and COMPLIANT, NON_COMPLIANT, or ERROR state for an evaluated subject.
- **Violation**: A NON_COMPLIANT subject result for a specific control definition.
- **Control Outcome**: ERROR for execution errors, subject errors, or invalid selected objects; NOT_APPLICABLE for
  successful empty selection. Error-free nonempty runs have no aggregate COMPLIANT or NON_COMPLIANT state.
- **Result Filter**: Invocation-level `result_filter=all` or `violations` (default). Both retain visible errors;
  `violations` limits findings to NON_COMPLIANT subjects without altering evaluation or its outcome.
- **Empty Findings List**: No returned NON_COMPLIANT findings. It does not imply empty selection or NOT_APPLICABLE.
- **Policy Report**: Results attributable to the executed `control_id`s, with visible errors and no policy-wide
  percentage.

## Diagnostics

- **Invocation Trace**: Stage inputs, outputs, source interactions, intermediate data, results, errors, and model
  interactions for creation or execution, saved under the configured log directory with required privacy redaction.
- **Trace Resource URI**: Address derived from the invoked operation and its returned `invocation_id`, using the
  [operation contract](../../specs/002-policy-checker/contracts/operations.md#diagnostic-trace-resources). It adds no
  additional response field. Clients use `PolicyService.read_trace` (and future MCP `resources/read`) to obtain
  persisted sanitized InvocationTrace JSON, or a resource error when the trace is unavailable. It is not a local path
  or policy identifier.
- **Diagnostic Artifact**: Material for investigating an invocation. Failed creation candidates may appear here;
  that does not make them a delivered reusable control set.
- **Trace Persistence**: Traces remain until explicitly deleted. This feature provides no configurable retention,
  automatic age-based cleanup, or policy history service.

## Historical or Excluded Concepts

Managed review, approval, publication, regeneration comparisons, normalized-definition comparison, stable identity
correspondence, policy/control versions, content hashes for revision comparison, historical retrieval, restoration,
service-side control file repositories (`--root`), and a standalone `policy_parameters` MCP tool are outside the
current scope. Earlier references to these concepts describe superseded decisions.

The earlier `INSUFFICIENT_EVIDENCE` state, evidence-coverage/compliance percentage formulas, automatic prerequisites,
and RASE runtime decomposition remain retired. Semantic uncertainty or technical failure is ERROR, not automatic
non-compliance. The earlier absolute model prohibition and unavailable-source invocation-abort rule are superseded.

## Decision Records

- [ADR-001: Creation validation failure](adr-001-creation-validation-failure.md) — trials retained; no publication gate.
- [ADR-002: Skipped authoring trials](adr-002-zero-attempt-verification.md) — retained without approval workflow.
- [ADR-003: Compliance and evidence coverage](adr-003-compliance-and-evidence-coverage.md) — historical formulas retired.
- [ADR-004: Per-control reporting](adr-004-per-control-reporting.md) — no overall percentage; old formulas retired.
- [ADR-005: Automatic prerequisites](adr-005-automatic-prerequisites.md) — superseded by independent evidence.
- [ADR-006: Technical model failures](adr-006-technical-model-failure.md) — historical fail-safe/abort rules retired.
- [ADR-007: Configurable retention](adr-007-configurable-artifact-retention.md) — superseded by ADR-019.
- [ADR-008: Supported policy formats](adr-008-policy-input-formats.md) — retained, with inline input under ADR-019.
- [ADR-009: Required trial parameters](adr-009-required-trial-parameters.md) — retained with no publication lifecycle.
- [ADR-010: Versions and hashes](adr-010-versioned-controls-and-policy-hashes.md) — outside current scope.
- [ADR-011: Authoring/execution separation](adr-011-authoring-execution-separation.md) — amended by ADR-018 and ADR-019.
- [ADR-012: Declarative standards](adr-012-declarative-control-standards.md) — retained with declared LLM exception.
- [ADR-013: Independent evidence](adr-013-independent-evidence-calls.md) — retained with uniform per-control errors.
- [ADR-014: Subject contract](adr-014-selection-subject-contract.md) — retained.
- [ADR-015: Document authority and regeneration](adr-015-document-authority-and-regeneration.md) — authority retained;
  maintenance, identity reconciliation, and comparison excluded by ADR-019.
- [ADR-016: Immutable revisions](adr-016-immutable-policy-revisions.md) — superseded by ADR-019.
- [ADR-017: Partial errors](adr-017-partial-subject-errors.md) — retained with source outages isolated per control.
- [ADR-018: Semantic criteria](adr-018-semantic-criterion-exception.md) — retained without approval/revision lifecycle.
- [ADR-019: Stateless policy service and LLM flows](adr-019-file-based-policy-operations.md) — current scope.
