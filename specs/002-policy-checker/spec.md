# Feature Specification: Policy Validation Engine

**Feature Branch**: `002-policy-checker`
**Created**: 2026-10-06
**Updated**: 2026-11-05
**Status**: Scope clarified — ready for technical planning
**Input**: Stakeholder scope, UX, and LLM flows refinement agreed on 2026-11-04 and 2026-11-05.

Vatuta provides a stateless policy service (`PolicyService`, structured 1:1 as an MCP-ready interface using
JSON-serializable Pydantic models) with two domain operations: create executable controls from a policy (`create` /
`policy_create`) and execute supplied controls (`execute` / `policy_execute`). The service manages no control files or
rule repository: creation returns structured `ControlDefinition` objects (including `tags` and embedded unit
`test_cases`), and execution receives `ControlDefinition` objects inline with explicit parameter values. The Vatuta
client manages local YAML control files in `policy.controls_directory` and provides a local filtering,
attribute-projection, and parameter-merging tool (`include_merged_parameters=True` by default). Users can run direct
CLI commands (`vatuta policy create`, `vatuta policy execute`) or a guided 2-step prompt-driven policy check flow
(`vatuta policy check "<prompt>"`) that selects controls via the client tool, deduces parameter values from the prompt,
and invokes execution. A `policy serve` MCP transport command is deferred to a follow-up increment.

This specification defines observable behavior. The [design](../../docs/policy-checker/design.md),
[glossary](../../docs/policy-checker/glossary.md), and [design inputs](design-inputs.md) carry technical decisions.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create Controls Step by Step from a Policy (Priority: P1)

As a manager, I want to supply policy text with descriptive construction context and optional tags and have the engine
decompose the policy into independent rules and construct each control step by step with embedded unit tests for
selection and criterion, so the client can save reproducible YAML controls for manual inspection and reuse.

**Why this priority**: Generated controls and their embedded unit tests are the reusable product of policy authoring.

**Independent Test**: Supply a policy with compound and independent obligations, descriptive construction context, and
optional tags; verify rule decomposition, shared parameter naming across rules, bounded JSONata evidence exploration,
embedded `test_cases` (`selection_tests` and `criterion_tests`), trial execution, and client-saved YAML files.

**Acceptance Scenarios**:

1. **Given** a policy containing obligations that can fail independently or that the policy author would consider
   distinct violations, **When** creation decomposes the policy, **Then** it splits them into separate normative rules
   and generates a distinct `ControlDefinition` for each rule.
2. **Given** multiple rules created sequentially from the same policy, **When** parameters are deduced for each rule,
   **Then** an incremental shared parameter catalog is passed from earlier rules to subsequent ones so conceptually
   identical inputs reuse the exact same parameter name and compatible `value_schema`.
3. **Given** enabled creation trials and a rule with resolved `construction_value` inputs, **When** the constructor
   inspects the source evidence structure and semantics, **Then** it fetches the evidence once in memory and runs a
   configurable bounded number of exploratory JSONata projections (`policy.discovery_max_queries`, default 5 queries
   per rule, with each output truncated to `policy.discovery_max_output_bytes`, default 4096 bytes) rather than dumping
   the raw evidence payload into the model context.
4. **Given** a rule's evidence and parameter definitions, **When** the constructor builds `selection` and `criterion`,
   **Then** it also generates and executes synthetic unit tests—`selection_tests` (validating subject transformation
   and `NOT_APPLICABLE` empty cases) and `criterion_tests` (validating at least one `COMPLIANT` and one `NON_COMPLIANT`
   subject)—and persists them in `ControlDefinition.test_cases` (`<id>.yaml`).
5. **Given** zero trial attempts (`trial_attempts=0`), **When** creation runs, **Then** live source trials are skipped
   (and recorded as `skipped`), but structural checks, permission checks, and the synthetic `selection_tests` and
   `criterion_tests` unit tests still execute and must pass.
6. **Given** rules that can be checked deterministically and a rule requiring semantic judgment, **When** creation
   selects evaluation modes, **Then** it prefers deterministic CEL criteria and declares LLM evaluation only for the
   semantic rule, including why deterministic evaluation is insufficient.
7. **Given** enabled creation trials, **When** any control exhausts its configured attempts, fails its synthetic unit
   tests, or lacks a required trial value, **Then** the whole creation fails with diagnostics and no reusable set or
   subset is returned. One attempt permits a trial without repair retries; additional attempts retain bounded
   correction behavior.
8. **Given** an obligation that cannot be represented using available evidence and either permitted evaluation mode,
   **When** creation cannot produce the complete control set, **Then** it fails explicitly with the reason and returns
   no reusable subset. It does not silently omit the rule.
9. **Given** unsupported document content or invalid UTF-8, **When** creation starts, **Then** it rejects the input
   before extraction and returns no controls.
10. **Given** enabled trials and passing synthetic unit tests, **When** a live trial completes without errors but
    selects no subjects, **Then** it passes with NOT_APPLICABLE and records that the live criterion was not executed.
    Creation may succeed without a populated live sample. A live trial that detects NON_COMPLIANT subjects also passes
    if it has no errors.
11. **Given** construction context describing a repository and project without generated parameter names, **When**
    creation defines its parameters, **Then** it derives supported trial values and validates their schemas/scope.
    Trial values do not become invocation defaults without an explicit basis in the policy or context.
12. **Given** enabled trials with insufficient context, **When** a required value cannot be resolved, **Then** creation
    returns no controls and an actionable `MISSING_CONSTRUCTION_CONTEXT` error describing the needed information,
    parameter semantics/schema, and affected controls. It invents no values and skips no required trial.
13. **Given** equivalent inline text and a supported CLI policy file, **When** the CLI reads it, **Then** it passes its
    contents to the same creation service operation. Creation accepts required `policy_text` and optional `policy_id`,
    `construction_context`, `tags`, and `trial_attempts`, without a file path or named construction-value map.
    Generated `source.policy` uses the supplied policy identifier or the creation invocation ID when omitted.
14. **Given** construction context that contradicts policy obligations or exceeds configured permissions, **When**
    controls are generated, **Then** the context cannot weaken those obligations or expand the permitted scope.

### User Story 2 - Filter Controls, Project Attributes, and Deduce Parameters in Check (Priority: P1)

As a user or policy agent in the Vatuta client, I want to filter local control files by metadata or text, project
their attributes (including merged shared parameters), and run a guided 2-step prompt check flow that selects the
applicable rules, deduces parameter values from the prompt, and invokes stateless execution.

**Why this priority**: Client-side filtering, projection, and the 2-step prompt pipeline let users and scripts select
rules and prepare invocation inputs scalably without coupling the stateless execution engine to a file repository.

**Independent Test**: Populate a local controls directory with YAML controls containing tags, shared parameters,
required inputs, defaults, and trial-only construction values. Filter and project controls with
`include_merged_parameters=True` and `False`, and invoke execution both directly and through `vatuta policy check`.

**Acceptance Scenarios**:

1. **Given** YAML control files in `policy.controls_directory` (or `--controls-dir`), **When** the client's filtering
   and projection tool runs with criteria (`ids`, `policy_ids`, `rule_ids`, `tags`, `evaluation_mode`,
   `evidence_server`, `evidence_tool`, `parameter_names`, `text_query`, `files`, or `glob`), **Then** it
   deterministically returns the matching controls projected to the requested attributes (`projection`, e.g.,
   `["id", "description", "tags", "source"]` or `["full"]`), without external evidence calls.
2. **Given** `include_merged_parameters=True` (the default), **When** the client tool projects control information,
   **Then** it also returns `merged_parameters` describing each shared parameter's schema, required or optional status,
   declared default, and referencing `control_id`s and descriptions. Setting `include_merged_parameters=False` omits
   the merged block.
3. **Given** a natural-language prompt in `vatuta policy check "<prompt>"`, **When** Step 1 (Rule Selection) runs,
   **Then** the LLM uses the client filtering/projection tool with lightweight projection (`["id", "description",
   "tags", "source"]`, bounded by configurable `policy.check_max_filter_queries`, default 3 queries, and output
   truncation `policy.check_max_output_bytes`, default 4096 bytes) to select the applicable `control_id`s, and the
   client deterministically loads those controls with `projection=["full"]` and `include_merged_parameters=True`.
4. **Given** the selected controls and their `merged_parameters`, **When** Step 2 (Parameter Deduction) of
   `vatuta policy check "<prompt>"` runs, **Then** the LLM extracts explicit parameter values present in the user's
   prompt (without inventing missing values) and the client invokes `PolicyService.execute` with the inline
   `ControlDefinition` list and resolved parameters. The stateless execution engine itself never infers parameters from
   free text.
5. **Given** several selected controls declaring the same parameter name, **When** their requirements are merged or
   validated for execution, **Then** that name denotes one shared parameter and one supplied value applies to every
   referencing control.
6. **Given** a required parameter with only a construction value, **When** execution omits its value,
   **Then** the construction value is not silently used. An explicitly declared default can satisfy an omitted value;
   an explicit invocation value takes precedence over that default.
7. **Given** missing mandatory values for any supplied controls, **When** execution validates the request,
   **Then** it rejects the entire invocation before obtaining evidence or evaluating any control and reports all
   missing values, their semantics, and affected `control_id`s.
8. **Given** an empty control selection, unreadable or malformed local YAML files, duplicate `control_id`s in an
   execution request, or incompatible same-name parameter declarations, **When** the client filter or execution
   validation checks the controls, **Then** it reports an explicit input error without evidence acquisition or
   silently selecting a conflicting default.

### User Story 3 - Execute Supplied Controls and Inspect Results (Priority: P1)

As a user or agent, I want to execute selected `ControlDefinition` objects with explicit parameters and receive
attributable findings and errors by `control_id`, so I can check compliance without repeating policy interpretation.

**Why this priority**: Execution is the second main operation and must remain useful when one independent control fails.

**Independent Test**: Execute structured `ControlDefinition` objects against fixed evidence and controlled semantic
responses through `PolicyService.execute`, `vatuta policy execute`, and `vatuta policy check`. Compare subject results,
error visibility, filters, and continued execution.

**Acceptance Scenarios**:

1. **Given** a nonempty list of valid `ControlDefinition` objects and complete valid parameters, **When** execution
   starts, **Then** it executes exactly those controls, obtains their independent evidence, selects subjects, and
   evaluates the declared criterion for each subject without analysing the policy again or reading control files.
2. **Given** a deterministic control checking pull requests 101, 102, and 103 with an approved linked issue, a pending
   issue, and no issue respectively, **When** execution completes, **Then** their states are COMPLIANT,
   NON_COMPLIANT, and NON_COMPLIANT, with no evaluation model calls. Fixed inputs and source responses reproduce them.
3. **Given** a control declaring LLM evaluation, **When** it executes, **Then** it uses the declared semantic
   criterion. Controlled true, false, and failed judgments produce COMPLIANT, NON_COMPLIANT, and ERROR respectively.
   A deterministic evaluation failure never switches to LLM evaluation or repairs the criterion at runtime.
4. **Given** successful complete evidence and a selection producing no result or an empty collection,
   **When** selection finishes, **Then** the control is NOT_APPLICABLE and no criterion runs. Root null is instead
   a selection contract ERROR. A valid subject with a null related-ticket field remains evaluable.
5. **Given** a mixture of valid subjects and objects missing mandatory identifiers, **When** evaluation runs,
   **Then** valid subjects are evaluated, invalid objects are reported without invented IDs, and the control returns
   global ERROR. An invalid-only collection is ERROR, not NOT_APPLICABLE.
6. **Given** multiple independent controls and a failure in source access, response conversion, selection, or
   criterion evaluation, **When** execution proceeds, **Then** the affected control reports ERROR and the others
   continue. An unavailable required source follows the same rule. Failed evidence or selection does not reach
   criterion evaluation; valid subject results already obtained remain available.
7. **Given** incomplete required evidence or completeness that neither the tool contract nor response establishes,
   **When** evidence is checked, **Then** the affected control reports ERROR before selection or criterion evaluation,
   with no subject verdicts. Other controls continue. Empty data with unknown completeness is not NOT_APPLICABLE.
8. **Given** a response with structured content and text, **When** evidence is converted, **Then** structured content
   is authoritative. Without it, exactly one text block containing a complete valid JSON value is accepted.
   Unsupported payloads, malformed JSON, and tool-reported errors remain errors rather than evidence.
9. **Given** the three reference pull-request results, **When** `result_filter=all`, **Then** all three subject results
   are returned; with `violations` or an omitted filter, only 102 and 103 appear as findings. Errors remain visible
   and distinguishable under either filter.
10. **Given** an error-free execution with at least one evaluated subject, **When** results are returned,
    **Then** there is no aggregate COMPLIANT or NON_COMPLIANT state. An empty violations list after evaluating only
    compliant subjects does not become NOT_APPLICABLE. Any subject error produces global control ERROR.
11. **Given** independent subjects evaluated together using one semantic criterion, **When** a batch response is
    processed, **Then** each judgment uses only its own subject inputs and maps unambiguously to that subject.
    Missing, conflicting, or unusable responses produce errors; valid mappings are preserved. A failed batch errors
    its affected subjects and does not stop other independent controls.
12. **Given** equivalent direct CLI and programmatic service requests, **When** they finish, **Then** their structured
    results, parameter validation, filtering, and error semantics agree.

### User Story 4 - Inspect Diagnostic Traces (Priority: P1)

As a user or developer, I want inspectable creation and execution traces, so I can understand failures and the
evidence behind results without a policy history system.

**Why this priority**: Diagnostics explain generated definitions and evaluations without adding a management workflow.

**Independent Test**: Inspect successful and failed creation and execution traces, including deterministic and semantic
evaluation, and verify attribution, intermediate data, and required redaction.

**Acceptance Scenarios**:

1. **Given** a creation or execution invocation, **When** stages run, **Then** an identifiable directory under the
   configured log location retains stage inputs, outputs, source interactions, intermediate data, results, and errors
   in readable, machine-reusable form, with credential and personal-data redaction.
2. **Given** model-assisted creation or declared semantic evaluation, **When** a model is called, **Then** traces
   retain prompts, responses, available model metadata, task context, and subject mappings where applicable.
   Deterministic criterion execution does not require model calls or generated rationales.
3. **Given** completed traces, **When** time passes, **Then** this feature does not delete them based on age or provide
   retention configuration. Their existence does not create policy revisions or a history retrieval operation.
4. **Given** interactive use, **When** stages advance, **Then** progress is shown separately from diagnostic logs and
   machine-readable results. Headless output remains suitable for programmatic consumption.
5. **Given** a creation or execution result, **When** a client needs diagnostics, **Then** it derives the trace resource
   URI from the invoked operation and returned `invocation_id`, following the operation contract without an additional
   trace field. Reading a persisted trace through the service trace reader (and future MCP `resources/read`) returns
   sanitized JSON, without model calls, evidence calls, or reevaluation.
6. **Given** an unknown, missing, deleted, malformed, or inaccessible trace resource, **When** a client reads it,
   **Then** the service returns a resource error without exposing arbitrary files. Trace-write failures remain visible
   through existing error semantics; deriving a URI does not guarantee that its trace is available.

### Edge Cases

- Empty control lists, duplicate `control_id`s in an execution request, unreadable/malformed client YAML files, or
  client filter selections that match no controls produce an explicit input error before evidence acquisition. They
  must not appear as successful compliance checks.
- Same-name parameters represent the same input. Incompatible declarations are input errors, not a reason to guess
  semantics, silently choose a conflicting default, or create per-control overrides.
- Missing trial values fail creation when trials are enabled. Zero attempts skips live source trials only; mandatory
  structural/permission checks and synthetic `selection_tests` and `criterion_tests` unit tests remain mandatory.
- Exploratory JSONata queries during creation and filter queries during `policy check` Step 1 are bounded by
  configurable query limits (`policy.discovery_max_queries`, default 5; `policy.check_max_filter_queries`, default 3)
  and configurable output truncation thresholds (`policy.discovery_max_output_bytes`, default 4096 bytes;
  `policy.check_max_output_bytes`, default 4096 bytes, marked `[TRUNCATED]` when exceeded) so large source responses or
  control directories never overflow the model context.
- An error-free live trial with an empty selection passes and records NOT_APPLICABLE and an unexecuted live criterion
  (while the control's synthetic `criterion_tests` still verify the criterion against mock subjects). Root null,
  incomplete evidence, and unknown completeness remain errors.
- Failed creation may retain diagnostic candidates, but must not deliver them as a reusable partial control set.
- Missing required invocation values stop the whole invocation before execution. Runtime failures stay with their
  affected controls; several controls using an unavailable source may each fail while independent controls continue.
- Incomplete or unusable evidence cannot establish the absence of a related record. No automatic pagination is added.
- Empty or invalid-only selections do not call either criterion evaluator. Null fields within valid subjects are
  evaluated normally. Technical errors and ambiguous model judgments never become non-compliance automatically.
- Different controls may express conflicting obligations; report them independently without a policy-wide percentage.

## Requirements *(mandatory)*

### Functional Requirements

Existing identifiers are retained where their responsibilities remain. FR-028 through FR-031 and FR-033 are retired
with the managed policy lifecycle; FR-016 and FR-020 were already absent. New parameter, tag, client, and LLM-flow
contracts use FR-038 through FR-047.

- **FR-001**: Creation (`PolicyService.create` / `policy_create`) MUST accept required nonempty `policy_text`, optional
  nonempty `policy_id` (defaulting to the creation `invocation_id`), optional descriptive `construction_context`
  (default empty), optional `tags: list[str]` (default empty), and optional nonnegative `trial_attempts` overriding
  deployment configuration `policy.trial_attempts` (default one) for that creation only. Support plain text and
  Markdown. The CLI MAY read a supported UTF-8 policy file and pass its contents to the same request; reject
  unsupported inputs before extraction. Do not require a predeclared parameter-name-to-value map.
- **FR-002**: Successful creation MUST return the complete set of structured `ControlDefinition` objects, each with an
  identifier (`id`), `description`, `tags` (combining constructor-inferred tags from policy/context and caller-supplied
  `tags`), `source` provenance to the supplied policy and its rule or section, and embedded `test_cases`. If an
  obligation cannot be represented, creation MUST fail with diagnostics and no reusable set or subset. Identity
  matching across generations is excluded.
- **FR-003**: Creation MAY use an LLM and MUST choose each control's evaluation mode at creation time. Prefer
  deterministic evaluation; declare LLM evaluation only when deterministic checking cannot resolve the obligation.
  Execution MUST follow the control definition without selecting, repairing, or replacing its criterion dynamically.
- **FR-004**: Execution (`PolicyService.execute` / `policy_execute`) MUST accept inline `ControlDefinition` objects,
  obtain declared evidence, transform it into subjects, and evaluate the declared criterion independently for each
  subject, keeping these responsibilities and their recorded inputs and outputs distinct.
- **FR-005**: Evidence calls MUST be independent. Arguments MUST NOT depend on other evidence or control results.
  Dependency graphs, automatic prerequisites, dynamic orchestration, and runtime planning are outside scope.
- **FR-006**: Deterministic criteria MUST produce equivalent results for the same definitions, invocation inputs,
  and source responses, without evaluation model calls. Semantic results MUST remain attributable; identical live
  model judgments are not promised. Fixed responses MUST map consistently to subject states and filters.
- **FR-007**: Subject states MUST be COMPLIANT, NON_COMPLIANT, or ERROR. A successful empty selection produces control
  NOT_APPLICABLE without criterion evaluation. Any subject error or invalid selected object produces global control
  ERROR. Error-free nonempty evaluations return individual states without an aggregate compliance state.
- **FR-008**: Evidence access MUST use configured provider-independent tools and remain read-only against external
  records. GitHub and Jira Cloud are reference sources, not executor-specific behavior.
- **FR-009**: Implement creation and execution on a stateless `PolicyService` class using JSON-serializable Pydantic
  request/result models that map 1:1 to MCP tool schemas (`policy_create`, `policy_execute`). The Vatuta client
  invokes `PolicyService` directly in this increment; `vatuta policy serve` is deferred. The client provides `create`,
  `execute`, and prompt-driven `check` CLI subcommands. `PolicyService.execute` MUST NOT use an LLM to infer missing
  invocation values.
- **FR-010**: Creation and execution MUST have identifiable trace directories under the configured log location,
  with subdirectories for executed stages. Clients MUST be able to derive the trace resource URI from the known
  operation and returned `invocation_id`, using the operation contract's templates without an additional trace field.
  Resolve persisted traces through `PolicyService.read_trace` (designed for MCP `resources/read`), returning sanitized
  InvocationTrace JSON (`application/json`) without model or evidence calls. Surface trace-write failures through
  existing error semantics. Unknown, missing, deleted, malformed, or inaccessible resources MUST produce resource
  errors.
- **FR-011**: Persist source interactions, intermediate data, results, and errors in readable, machine-reusable form,
  with credential and personal-data redaction. Distinguish creation model interactions from semantic evaluation.
- **FR-012**: `PolicyService.execute` MUST accept a nonempty list of `ControlDefinition` objects inline with unique
  `id` values per invocation. On the client side, the local control repository (`policy.controls_directory`,
  overridable via `--controls-dir`) MUST support filtering by `ids`, `policy_ids`, `rule_ids`, `tags`,
  `evaluation_mode`, `evidence_server`, `evidence_tool`, `parameter_names`, `text_query`, `files`, and `glob`, plus
  attribute `projection`. Execution MUST evaluate only the supplied controls without repeating policy analysis.
- **FR-013**: Each evidence declaration MUST identify its evidence, configured server, tool, and invocation arguments.
  Credentials and connection settings remain external. Prefer structured response content; otherwise accept exactly
  one text block whose entire text is valid JSON. Unsupported content or tool-reported errors MUST stop the affected
  control before selection and criterion evaluation. Do not extract embedded JSON from prose or Markdown.
- **FR-014**: Validate explicit invocation values and declared defaults against parameter declarations, configured
  scope, and permissions before evidence acquisition. Missing required values for any supplied control MUST reject
  the entire invocation and identify all missing values, their semantics, and affected `control_id`s. Execution MAY use
  valid values different from construction inputs without recreating controls.
- **FR-015**: Selection MUST return a JSON object or collection of objects with `id` and `descriptive_id` per subject.
  No result or an empty collection yields NOT_APPLICABLE; root null is ERROR. Null fields within valid subjects are
  allowed. Evaluate valid subjects even alongside invalid objects; report invalid objects without invented IDs or
  implicit scalar wrapping. An invalid-only collection is ERROR and invokes no criterion.
- **FR-017**: Results and errors MUST identify the executed control definition by `control_id`, source provenance,
  resolved invocation inputs, and evaluated subjects. Violations MUST identify their subjects unambiguously. Revision
  identifiers, file paths in service responses, generated explanations, and remedies are not required.
- **FR-018**: After request validation, source unavailability, conversion, selection, and criterion failures MUST be
  reported for the affected control while remaining independent controls continue. Required evidence that is
  incomplete or not established as complete by the tool contract or response MUST cause ERROR before selection and
  criterion evaluation, without subject verdicts. Preserve valid results and all errors; never classify technical
  failures as non-compliance or inapplicability. No automatic pagination or invocation-wide source-abort exception.
- **FR-019**: One declared criterion MUST be reused independently for every subject. Boolean true and false map to
  COMPLIANT and NON_COMPLIANT; non-boolean results and technical failures map to ERROR.
- **FR-021**: Accept invocation-level `result_filter` with `all` and `violations`, defaulting to `violations`.
  The former returns all subject results; the latter returns NON_COMPLIANT findings. Both MUST retain visible errors
  distinguished from violations. Subject results contain `id`, `descriptive_id`, and state. Filtering changes neither
  evaluation nor its status; the filter is not a control setting.
- **FR-022**: Interactive clients MUST show stage progress separately from diagnostic logs and machine-readable output.
- **FR-023**: Diagnostic records MUST identify components and stages and follow project logging and privacy standards.
- **FR-024**: Retain creation and semantic-evaluation model interactions as structured samples suitable for prompt
  evaluation export. Deterministic execution does not require model samples or generated rationales.
- **FR-025**: Preserve per-control result attribution, durations, errors, and available model usage measurements.
  Do not produce an overall policy compliance percentage or revive historical evidence-coverage formulas.
- **FR-026**: Preserve existing creation trials and bounded corrections: configurable attempts, one attempt without
  repair retries, and zero to skip live trials. Structural checks, permission checks, and synthetic unit tests
  (`selection_tests` and `criterion_tests`) remain mandatory even when `trial_attempts=0`. Missing required trial
  values or exhausted attempts fail the whole creation with diagnostics and no reusable subset. Record skipped live
  trials in both controls and traces. Successful creation requires no managed review, approval, or publication step.
  An error-free live trial passes even when it returns NON_COMPLIANT findings or NOT_APPLICABLE. For an empty live
  selection, record NOT_APPLICABLE and that the live criterion was not executed; do not require a populated live sample
  to finish creation.
- **FR-027**: Retain creation and execution traces in the log directory until explicitly deleted. Configurable
  retention periods, automatic age-based cleanup, and a trace-history management operation are outside this feature.
- **FR-032**: The supplied policy document or text is the normative source for generated controls. Manual inspection
  of the resulting YAML files is supported; managed editing, comparison, review, approval, and publication are excluded.
- **FR-034**: Each control MUST declare its deterministic or LLM evaluation mode and criterion. An LLM criterion
  MUST include its semantic condition and why deterministic evaluation is insufficient. Creation trials and later
  execution MUST use that same declared mode.
- **FR-035**: Semantic evaluations MAY batch independent subjects using the same criterion. Each judgment MUST use
  only that criterion and its own subject's inputs, with an unambiguous subject-response mapping and no shared verdict.
- **FR-036**: Missing, conflicting, ambiguous, non-boolean, or unusable semantic responses MUST produce ERROR for
  affected subjects. Preserve valid mappings and report unexpected responses as errors. A failed batch errors its
  affected subjects; other subjects and independent controls continue under FR-018.
- **FR-037**: Semantic traces MUST identify the criterion, evaluator and available model/version/settings, inputs,
  responses, batch membership, and subject mapping under existing privacy rules. Results identify the evaluation
  mode. Internal model reasoning is not required.
- **FR-038**: Creation MUST derive parameter declarations from the policy, descriptive construction context, and
  permitted tool capabilities, passing an incremental `SharedParameterCatalog` across rules created from the same
  policy so shared concepts reuse the same parameter name and compatible `value_schema`. Derive supported construction
  values without assuming the caller knows parameter names; never invent missing values. Trial inputs MUST NOT silently
  become invocation defaults. If enabled trials lack required values, fail creation with actionable
  `MISSING_CONSTRUCTION_CONTEXT` errors describing missing information, semantics/schema, and affected controls, with
  no reusable controls or implicit trial skipping. Context MUST NOT weaken policy obligations or configured
  permissions.
- **FR-039**: The client's local filtering and projection tool MUST deterministically filter YAML controls in
  `policy.controls_directory` (`--controls-dir`), project the requested control attributes, and when
  `include_merged_parameters=True` (the default) return the merged shared parameter descriptions (`name`,
  `value_schema`, `required`, `default`, and referencing `control_id`s with descriptions) without model or evidence
  calls. `policy_parameters` is not a separate service or MCP operation.
- **FR-040**: Same-name parameters MUST denote the same shared input across selected controls and use one resolved
  invocation value. An explicit value overrides its declared default. Without an explicit value or applicable declared
  default, the parameter is missing. Construction values MUST NOT silently become defaults.
- **FR-041**: `PolicyService.create` MUST return structured `ControlDefinition` objects directly. The client MAY
  serialize and save them as `<id>.yaml` files in `policy.controls_directory` (or `--output-dir`) for manual inspection
  and later selection; managing that client directory is not the service's responsibility.
- **FR-042**: Empty control lists, duplicate `control_id`s in an execution request, unreadable or malformed client YAML
  files, empty client filter matches, and incompatible same-name parameter declarations MUST produce explicit input
  errors before evidence acquisition. Both the client's `include_merged_parameters` aggregation and
  `PolicyService.execute` validation MUST expose declaration conflicts rather than inventing semantics or choosing
  between inconsistent defaults.
- **FR-043**: `ControlDefinition` MUST include `tags: list[str]`. During creation, the constructor MUST generate
  relevant tags from the policy and `construction_context` and merge any explicit `tags` supplied in the creation
  request.
- **FR-044**: Creation MUST first decompose the policy into atomic normative rules before constructing controls one by
  one. Two obligations MUST be split into separate rules whenever they can fail independently and a user needs to know
  which condition failed, or whenever the policy author would consider them distinct violations.
- **FR-045**: During per-rule creation when live trials are enabled (`trial_attempts > 0`), the constructor MAY inspect
  in-memory evidence fetched with `construction_value` inputs using bounded exploratory JSONata queries governed by
  configurable `policy.discovery_max_queries` (default 5 queries per control) and `policy.discovery_max_output_bytes`
  (default 4096 bytes per query output, marked `[TRUNCATED]` when exceeded) instead of injecting raw evidence payloads
  into LLM prompts.
- **FR-046**: Creation MUST generate, execute, and persist synthetic unit tests in `ControlDefinition.test_cases` for
  every control: `selection_tests` (testing `selection` against synthetic `mock_evidence` for both populated subjects
  and `NOT_APPLICABLE` cases) and `criterion_tests` (testing `criterion` against synthetic `Subject` objects for at
  least one `COMPLIANT` and one `NON_COMPLIANT` case). These unit tests MUST pass during creation even when
  `trial_attempts=0`.
- **FR-047**: `vatuta policy check "<prompt>"` MUST execute a guided 2-step LLM flow on the client: Step 1 uses the
  local `ClientControlQuery` tool with lightweight projection (`["id", "description", "tags", "source"]`), bounded by
  configurable `policy.check_max_filter_queries` (default 3 queries) and `policy.check_max_output_bytes` (default 4096
  bytes per query output, marked `[TRUNCATED]` when exceeded), to select applicable `control_id`s, followed by
  deterministic client loading of those controls with `projection=["full"]` and `include_merged_parameters=True`; Step
  2 extracts explicit parameter values from the user's prompt against `merged_parameters` (without inventing missing
  values) before invoking `PolicyService.execute`.

### Key Entities

- **Policy**: Supplied normative content, either a supported document or inline text, containing one or more rules.
- **Normative Rule**: An atomic obligation extracted during policy decomposition; conditions that can fail
  independently or represent distinct violations become separate rules.
- **Construction Context**: Descriptive information supplied with policy text to guide parameter definition, tag
  generation, evidence exploration, and trials.
- **Shared Parameter Catalog**: Incremental registry of parameter declarations (`name`, `description`, `value_schema`,
  `construction_value`, `default`) accumulated as rules are created one by one so shared inputs reuse identical names.
- **Control (`ControlDefinition`)**: An identifiable structured definition (serialized as YAML on the client) with
  `id`, `description`, `tags`, `source` references, `parameters` declarations, `evidence` acquisition, `selection`,
  `criterion` with its declared evaluation mode, `test_cases` (`selection_tests` and `criterion_tests`), and
  `creation_check`.
- **Control Test Cases (`ControlTestCases`)**: Embedded unit tests persisted on `ControlDefinition`, comprising
  `selection_tests` (`mock_evidence` -> `expected_subjects` or `NOT_APPLICABLE`) and `criterion_tests` (`subject` ->
  `COMPLIANT` or `NON_COMPLIANT`).
- **Parameter Declaration**: A named input with semantics, expected value schema, optional construction value, and
  optional default. Same-name declarations describe one shared input.
- **Construction Value**: A value derived while creating or testing a control; not automatically an invocation default.
- **Invocation Values**: Explicit values supplied by the agent or caller, completed by declared defaults where allowed.
- **Client Control Filter & Projection**: Client-side query criteria (`ids`, `policy_ids`, `rule_ids`, `tags`,
  `evaluation_mode`, `evidence_server`, `evidence_tool`, `parameter_names`, `text_query`, `files`, `glob`), attribute
  `projection`, and optional `include_merged_parameters` (default `True`) operating on `policy.controls_directory`.
- **Evidence**: Data obtained from declared external tools, independently of other evidence or control results.
- **Subject Result**: An evaluated subject's technical ID, descriptive ID, and COMPLIANT, NON_COMPLIANT, or ERROR state.
- **Invocation Trace**: Diagnostic inputs, outputs, intermediate artifacts, and errors; it is not a policy revision.
- **Trace Resource URI**: A resource address derived from the invoked operation and returned `invocation_id`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A reference policy supplied as a document or inline text is decomposed into independent normative rules
  and produces a complete set of structured `ControlDefinition` objects (with inferred and explicit `tags`, shared
  parameter names across rules, and embedded `test_cases`) that the client can save as `<id>.yaml`. An unrepresentable
  obligation produces an explicit failure and no reusable subset.
- **SC-002**: Creation fixtures cover policy decomposition, incremental shared parameter naming, configurable bounded
  JSONata evidence exploration (`discovery_max_queries`, default 5; `discovery_max_output_bytes`, default 4096),
  mandatory synthetic `selection_tests` and `criterion_tests`, zero attempts, one attempt, bounded corrections,
  exhausted attempts, and missing trial inputs. Error-free live trials with violations or empty selections pass; the
  latter record inapplicability and an unexecuted live criterion while synthetic `criterion_tests` still verify the
  criterion.
- **SC-003**: For equivalent local control selections, the client's filtering and projection tool returns the same
  projected attributes and (when `include_merged_parameters=True`) the same shared input descriptions without model
  calls or external evidence retrieval. Fixtures distinguish required inputs, declared defaults, and construction-only
  values; explicit invocation values replace defaults consistently.
- **SC-004**: An execution request missing mandatory values for multiple controls reports every missing value with its
  meaning and affected `control_id`s, and performs zero evidence acquisitions or subject evaluations.
- **SC-005**: The three reference pull requests yield the expected compliant/non-compliant outcomes, identities,
  and filtered findings. Fixed deterministic inputs reproduce those results without evaluation model calls.
- **SC-006**: Mixed independent-control fixtures covering source unavailability, response errors, incomplete evidence,
  selection failures, and criterion failures retain unaffected controls' results and all errors under both filters.
- **SC-007**: Empty-selection, root-null, mixed valid/invalid, invalid-only, null-field, and all-compliant fixtures
  distinguish errors, inapplicability, and an empty filtered findings list without invented identifiers or verdicts.
- **SC-008**: Controls for a deterministic rule and a semantic-only rule execute in their declared modes. Controlled
  true, false, and failed semantic judgments map to the correct subjects and outcomes, individually or in batches.
  Deterministic failures never trigger semantic evaluation.
- **SC-009**: Every reference creation and execution, successful or failed, retains attributable diagnostic artifacts
  with required redaction. Semantic cases include evaluator and subject-response attribution; traces do not expire
  automatically based on age.
- **SC-010**: Direct CLI (`vatuta policy execute`), guided 2-step prompt CLI (`vatuta policy check "<prompt>"`), and
  programmatic `PolicyService` calls produce consistent parameter validation, missing-input diagnostics, subject
  findings, and errors. A client can save, filter, project, and inspect controls in `policy.controls_directory`
  without the stateless service managing a control repository.

## Assumptions

- These are the stakeholder-approved 2026-11-04 scope, UX, and LLM-flow refinements. Compatible earlier evidence,
  selection, result, creation-trial, trace-URI, and semantic-evaluation decisions remain; earlier publication/history
  requirements and service-side file selectors do not.
- `PolicyService` defines the exact stateless contract (using JSON-serializable Pydantic models) that will back the
  MCP tools (`policy_create`, `policy_execute`) in a subsequent increment. `vatuta policy serve` is deferred for now.
- Local YAML storage (`policy.controls_directory`), filtering, attribute projection, `include_merged_parameters`, and
  the 2-step `vatuta policy check "<prompt>"` pipeline belong to the Vatuta client, not to `PolicyService`.
- Incompatible same-name declarations, duplicate `control_id`s in an execution request, and invalid client selections
  use explicit input errors. This is a minimal consistency safeguard, not a parameter reconciliation workflow.
- No model extraction accuracy, latency, throughput, batch size, or concurrency target is invented. Complete creation
  is the required outcome; tests use known policies rather than assuming infallible interpretation of arbitrary text.
- Creating again is an independent creation request. There is no update, identity correspondence, version comparison,
  policy/control history, restoration, approval, or publication operation. Users may inspect YAML files manually.
- Automatic trace cleanup and retention configuration are excluded. Diagnostics do not introduce policy versioning.

## Clarifications

### Session 2026-10-25 — Scope Simplification

- Keep two main operations, creation and execution, with manual file inspection and no managed approval gate.
- The invoking agent collects execution values from prompt/context using declared parameter semantics. The engine
  receives explicit values. Same names identify the same shared parameter across controls.
- Construction values may explicitly be defaults; otherwise they do not make required parameters optional.
- Missing mandatory invocation values reject the whole request before execution and report meanings and affected
  controls. Runtime failures, including unavailable sources, affect their controls while independent controls continue.
- Keep complete-or-error creation and existing trials unchanged. Fix deterministic/LLM evaluation mode in the control
  during creation; execution follows it without runtime mode switching or policy reinterpretation.
- Remove managed maintenance, review, approval, publication, comparisons, stable identity reconciliation, versions,
  and restoration. Keep diagnostic traces; remove configurable retention and automatic cleanup.
- Q: Can a creation trial pass when it selects no subjects? → A: Yes. An error-free trial passes with NOT_APPLICABLE
  and records that the criterion was not executed. NON_COMPLIANT findings alone also do not fail a trial.

### Session 2026-10-29 — Creation Inputs

- Define the creation request around `policy_text`, optional provenance identifier `policy_id`, optional descriptive
  `construction_context` defaulting to empty, and optional nonnegative `trial_attempts`. The CLI may read supported
  files into text; file paths and named construction-value maps are not creation service arguments.
- The constructor discovers parameter names and derives supported trial values from policy/context and permitted tools.
  Callers need not know the names in advance. Preserve construction values separately from explicit invocation defaults.
- Explicit request `trial_attempts` overrides deployment setting `policy.trial_attempts` for one creation; when omitted,
  use deployment configuration, default one. Zero attempts remains an explicit choice; missing context never disables
  required trials automatically.
- With enabled trials, insufficient context fails creation with `MISSING_CONSTRUCTION_CONTEXT`, missing-information
  semantics/schema, and affected controls, returning no reusable controls. The caller may expand context in a new
  request.
- Generated `source.policy` references the supplied `policy_id`, or the creation invocation ID when omitted, with
  rule references and excerpts retaining provenance. Context cannot weaken policy obligations or expand permissions.

### Session 2026-10-30 — Trace Resource URIs

- Clients derive trace URIs from the known operation and the engine-assigned `invocation_id`, without an additional
  trace field in requests or results. The templates distinguish construction and execution and are defined in
  [the operation contract](contracts/operations.md#diagnostic-trace-resources), which owns the URI documentation.
- Read them through the service trace reader (and future MCP `resources/read`) configured for the same trace storage;
  return sanitized InvocationTrace JSON without model calls or evidence acquisition.
- The convention applies to successful or failed invocations but does not guarantee trace availability. Surface
  trace-write failures. Unknown, missing, deleted, malformed, or inaccessible resources produce resource errors.

### Session 2026-11-04 — Stateless Service, Client Filtering Tool, and Prompt Check UX

- Make the core policy service (`PolicyService`, and the future MCP server it represents) 100% stateless regarding
  controls: it has no control repository, no `--root`, and no file/glob selectors. It exposes only two domain
  operations (`create` / `policy_create` and `execute` / `policy_execute`) plus trace reading (`read_trace`), using
  JSON-serializable Pydantic models ready for 1:1 MCP tool registration. Remove `policy_parameters` from the service
  and MCP boundary.
- Defer `vatuta policy serve` and the stdio MCP server wrapper to the next increment; in this increment, the Vatuta
  client invokes `PolicyService` directly.
- `PolicyService.create` accepts optional `tags: list[str]` in addition to `policy_text`, `policy_id`,
  `construction_context`, and `trial_attempts`. `ControlDefinition` includes `tags: list[str]`, combining tags inferred
  automatically by the constructor from the policy/context with any explicit request `tags`. `PolicyService.create`
  returns `controls: list[ControlDefinition]` directly; the client serializes them to YAML when saving to disk.
- `PolicyService.execute` accepts `controls: list[ControlDefinition]` inline (requiring unique `id` values within the
  invocation), `parameters: dict[str, Any]`, and `result_filter`. `ControlResult` and `ErrorDetail` identify controls
  by `control_id` without `control_file`.
- The Vatuta client manages local YAML control files under configurable `policy.controls_directory` (default
  `data/controls`, overridable via `--controls-dir`).
- Provide a client-side filtering and projection tool over `controls_directory` supporting criteria (`ids`,
  `policy_ids`, `rule_ids`, `tags`, `evaluation_mode`, `evidence_server`, `evidence_tool`, `parameter_names`,
  `text_query`, `files`, `glob`), attribute `projection` (including `["full"]`), and a configurable
  `include_merged_parameters: bool = True` parameter (enabled by default) that returns deduplicated, conflict-checked
  shared parameters (`merged_parameters`).
- Expose three CLI subcommands under `vatuta policy`: direct `create`, direct `execute` (using client filter flags),
  and prompt-driven `check "<prompt>"`.

### Session 2026-11-04 — Step-by-Step Creation & Check LLM Flows

- **Creation Step 1 (Policy Decomposition)**: Decompose the policy into atomic normative rules before building controls
  one by one. Split conditions into separate rules whenever they can fail independently and a user wants to know which
  one failed, or whenever the policy author would consider them distinct violations.
- **Creation Step 2 (Sequential Rule Construction & Shared Parameter Catalog)**: Construct rules one by one while
  passing an incremental `SharedParameterCatalog` from earlier rules so shared inputs reuse the same parameter name and
  compatible `value_schema`.
- **Scalable Evidence Discovery (Configurable Bounded JSONata Projections)**: During creation when `trial_attempts > 0`,
  fetch the source evidence once in memory using `construction_value` inputs and let the constructor inspect its
  structure and semantics through bounded exploratory JSONata queries governed by configurable
  `policy.discovery_max_queries` (default 5 queries per rule) and `policy.discovery_max_output_bytes` (default 4096
  bytes per output, marked `[TRUNCATED]` when exceeded) instead of dumping the full evidence JSON into the prompt.
- **Embedded Synthetic Unit Tests (`ControlDefinition.test_cases`)**: During creation, generate, execute, and persist
  two lists of synthetic unit tests inside `ControlDefinition.test_cases`:
  1. `selection_tests`: `mock_evidence` (and optional `parameters`) with `expected_outcome: "NOT_APPLICABLE"` or
     `expected_subjects` to unit-test `selection` before building `criterion`.
  2. `criterion_tests`: synthetic `subject` objects (and optional `parameters`) with `expected_state` (`COMPLIANT` and
     `NON_COMPLIANT`) to unit-test `criterion`.
  Synthetic unit tests are always mandatory (even when `trial_attempts == 0`), whereas live source trials remain
  governed by `trial_attempts`.
- **Guided 2-Step Check Flow (`vatuta policy check "<prompt>"`)**:
  1. **Step 1 (Rule Selection)**: The LLM uses `ClientControlQuery` with lightweight projection (`["id", "description",
     "tags", "source"]`, `include_merged_parameters=False`), governed by configurable `policy.check_max_filter_queries`
     (default 3 queries) and `policy.check_max_output_bytes` (default 4096 bytes per query output, marked `[TRUNCATED]`
     when exceeded), to select matching `control_id`s; the client then deterministically loads those controls with
     `projection=["full"]` and `include_merged_parameters=True`.
  2. **Step 2 (Parameter Extraction)**: A dedicated DSPy signature receives the user's prompt and `merged_parameters`
     and extracts explicit parameter values before calling `PolicyService.execute`.
