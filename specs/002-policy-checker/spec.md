# Feature Specification: Policy Checker Mechanism

**Feature Branch**: `002-policy-checker`
**Created**: 2026-10-06
**Updated**: 2026-10-20
**Status**: Draft — clarified
**Input**: [Policy checker notes](../../policy-checker.md) and the clarification session recorded below.

Managers need to turn formal policy documents into reusable atomic controls, then execute those controls against
workplace data to identify violations or assess compliance. Control creation and verification are separate operations
available through the existing Vatuta command client. Every decision must be explainable using its policy, evidence,
evaluation criteria, and recorded outcome.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Policy Validation Extraction (Priority: P1)

As a manager, I want to supply a policy document and execution scope so that Vatuta produces identifiable, reusable
controls with the evidence and evaluation criteria needed to check each policy.

**Why this priority**: Explicit controls are the foundation for repeatable policy verification.

**Independent Test**: Supply a policy document, scope, and available source capabilities; inspect the generated controls
without executing verification.

**Acceptance Scenarios**:

1. **Given** a formal natural-language policy document, **When** creation runs, **Then** Vatuta produces a
   normalized control definition that bridges the formal policy description and the deterministic instructions needed
   by the evaluation engine, containing:
   - A unique identifier
   - A detailed description of the control
   - Originating policy references, including document, section, and version where identified
   - The required evidence sources
   - Applicability conditions for evidence from sources
   - Transformations required on evidence from sources
   - The selection criteria to detect or identify violations (supporting multiple distinct transformation + selection
     paths)
   - The criteria governing allowed exceptions
   - Explicit parameters needed for verification.
2. **Given** creation instructions and a policy requiring parameters, **When** creation runs, **Then** the controls
   identify required parameters and the values supplied for the current operation. If creation-time trials are enabled,
   every parameter needed by those trials must have a value before creation can succeed; verification may later supply
   different valid values. With zero attempts, unresolved values remain explicit and must be supplied before use.
3. **Given** the configured data sources, **When** controls are created, **Then** Vatuta identifies the evidence needed,
   describes how it is collected and transformed, and selects an automated evaluation method, always preferring
   deterministic mechanisms over AI/LLM-based ones whenever deterministic evaluation is feasible.
4. **Given** a policy containing applicability conditions, alternatives, and exceptions, **When** controls are created,
   **Then** those conditions remain explicit rather than being lost in a single compliance statement.
5. **Given** completed creation, **When** the manager saves the controls, **Then** they can inspect them and use them in
   a later verification invocation without repeating document analysis.
6. **Given** control creation in progress, **When** a candidate control is generated, **Then** Vatuta validates its
   correctness by testing invocations against configured data sources, required transformations, and evaluation
   criteria mechanisms, and iteratively refines the control upon error up to a configurable maximum attempt threshold
   (which can be set to 1 or 0 to bypass retries or trial execution and save tokens, processing, and execution time).
7. **Given** creation-time validation is enabled and at least one generated control remains invalid, **When** its
   allowed validation attempts are exhausted, **Then** creation fails as a whole, retains diagnostic artifacts, and
   publishes no reusable control set, even if other controls passed validation.
8. **Given** creation-time validation attempts are set to zero and controls pass mandatory schema, dependency, and
   permission checks, **When** creation completes, **Then** it publishes reusable controls that record trial execution
   as skipped and are eligible for verification without a separate creation-time trial execution.
9. **Given** creation-time validation attempts are set to zero, **When** a control fails a mandatory schema, dependency,
   or permission check, **Then** it is rejected; disabling trial execution does not bypass these checks.
10. **Given** a UTF-8 plain-text or Markdown policy document, **When** creation runs, **Then** it accepts the document
    and preserves policy references as required by FR-002.
11. **Given** a policy document in an unsupported format or with invalid UTF-8 encoding, **When** creation starts,
    **Then** it rejects the input before policy extraction, identifies the input error and supported formats, and
    publishes no reusable control set. It does not silently interpret binary content as text or invoke OCR.
12. **Given** creation-time trials are enabled and a trial needs an unresolved parameter, **When** creation identifies
    the missing value, **Then** the entire creation invocation fails, identifies the parameter and affected trial,
    retains available diagnostics, and publishes no reusable control set or subset. It does not defer the trial or
    invent a value to allow creation to succeed.
13. **Given** creation-time trials succeed using supplied parameter values, **When** verification supplies different
    valid values within the configured scope and permissions, **Then** it evaluates the saved controls with those
    resolved values without requiring policy extraction again.
14. **Given** a saved control set for a policy, **When** creation succeeds again for that policy, **Then** it saves a
    separately identifiable version without overwriting earlier versions, and records a content hash of the supplied
    policy document with the new version.
15. **Given** a prior version for the same policy, **When** creation compares the current document hash with that
    version's recorded hash, **Then** it identifies whether document content changed and identifies the compared
    version. Unchanged content at a different path remains unchanged; different content at the same path is detected.
16. **Given** unchanged policy content, **When** creation is explicitly rerun and succeeds, **Then** the same document
    hash can be associated with a new control-set version; the document hash alone does not identify generated controls.

---

### User Story 2 - Policy Verification Execution (Priority: P1)

As a manager, I want to execute existing controls—whether running the entire set of controls, a single designated
control,
or a selected subset—against source data so that I can identify individual violations and assess policies whose outcomes
describe a whole project or reporting period.

**Why this priority**: Verification provides the actionable compliance results.

**Independent Test**: Execute predefined controls against controlled source data with known expected outcomes, without
running control creation.

**Acceptance Scenarios**:

1. **Given** saved controls, configuration, and required parameter values, **When** verification runs, **Then** Vatuta
   can evaluate the full list of controls, a single designated control by identifier, or a user-selected subset of
   controls,
   without requiring another extraction from the policy document.
2. **Given** work items whose policy requires a reference to an associated tracking record in a particular state,
   **When** verification runs, **Then** Vatuta correlates the work items to their referenced tracking records and
   identifies the specific violations with evidence, returning one or multiple (N) violations depending on the control's
   nature and the evaluated subjects.
3. **Given** a policy governing an aggregate metric over a population or reporting period, **When** verification
   runs, **Then** Vatuta returns that control's aggregate assessment supported by the calculated measurements and
   governing thresholds.
4. **Given** a control that uses another control's result, **When** verification runs, **Then** the prerequisite is
   automatically included and evaluated first, even if not explicitly selected, and its result is available as evidence
   with its origin preserved. The report distinguishes selected controls from automatically included prerequisites.
5. **Given** controls needing identical evidence in the same scope and invocation, **When** verification runs, **Then**
   they
   can reuse its collection and preparation while retaining separate evaluation results and traceability.
6. **Given** predefined fixtures for all four compliance outcomes, **When** verification runs, **Then** the report
   preserves
   the distinction between Compliant, Non compliant, Insufficient evidence, and Not applicable.
7. **Given** overlapping controls with conflicting requirements, **When** verification runs, **Then** both controls are
   evaluated independently and both results are reported.
8. **Given** a control with 80 Compliant, 10 Non compliant, 10 Insufficient evidence, and 20 Not applicable subjects,
   **When** its metrics are reported, **Then** compliance is 80/90 (approximately 88.9%) and evidence coverage is 90/100
   (90%), displayed together with outcome counts; the 20 Not applicable subjects are excluded from both denominators.
9. **Given** a control with only Insufficient evidence outcomes, **When** its metrics are reported, **Then** compliance
   is undefined and evidence coverage is 0%. If all subjects are Not applicable or the population is empty, both
   percentages are undefined and the outcome counts remain visible.
10. **Given** one control with 99 of 100 pull requests Compliant and another with one project Non compliant, **When**
    the policy report is produced, **Then** it shows each control's compliance, evidence coverage, and outcome counts,
    without averaging or pooling them into an overall policy percentage.
11. **Given** selected control A requires B and B requires C, with only A selected, **When** verification runs,
    **Then** it automatically includes B and C, evaluates C before B and B before A, and reports A as selected and B
    and C as prerequisites, all within the configured scope and permissions.
12. **Given** a required prerequisite cannot be evaluated within the configured scope or permissions, **When** the
    selection is checked, **Then** verification rejects the selection with the offending prerequisite identified,
    without silently widening scope, elevating permissions, or omitting the dependency.
13. **Given** verification requires a model call, **When** a timeout, service error, or unusable response persists after
    any permitted retries are exhausted, **Then** the entire invocation stops with an execution failure, preserving
    available diagnostics and completed results as incomplete work without publishing a completed assessment.
    The failed call produces no compliance outcome, including Non compliant or Insufficient evidence.

---

### User Story 3 - Programmatic Invocation Readiness (Priority: P2)

As an automation user, I want policy creation and verification to be usable outside an interactive terminal via headless
CLI or programmatic API contracts so that external systems and future MCP tool wrappers can easily invoke them.

**Why this priority**: The CLI is the primary user interface; programmatic readiness ensures future automated
integration without rework.

**Independent Test**: Invoke policy operations with explicit arguments in non-interactive/headless execution mode and
compare their results, output structure, and error semantics with equivalent interactive CLI invocations.

**Acceptance Scenarios**:

1. **Given** equivalent inputs and parameters, **When** the same policy creation or verification operation is invoked in
   a non-interactive/headless execution mode, **Then** it produces the same report structure, error semantics, and
   invocation trace as the interactive CLI operation.
2. **Given** headless execution completes, **When** outputs are inspected, **Then** the results are returned via
   structured data formats (e.g., JSON or machine-readable models) suitable for programmatic consumption and future MCP
   server wrapping.

---

### User Story 4 - Execution Traceability and Audit Logging (Priority: P1)

As an auditor or developer, I want to reconstruct a creation or verification invocation so that I can understand the
policy interpretation, source evidence, transformations, evaluation criteria, and final results, and track execution
progress clearly as it runs.

**Why this priority**: Developers need deep diagnostic visibility to improve Vatuta, while auditors need complete,
uncompromising auditability to analyze, challenge, or verify automated policy decisions.

**Independent Test**: Run creation, successful verification, and a verification that fails during source collection;
inspect the recorded artifacts without having to repeat the external calls, and confirm real-time client progress output
and module execution logging.

**Acceptance Scenarios**:

1. **Given** a creation or verification invocation, **When** processing begins, **Then** Vatuta creates an identifiable,
   unique directory under the configured log directory and a subdirectory for each executed workflow stage.
2. **Given** a stage that invokes a model or third-party system, **When** its trace is inspected, **Then** the inputs,
   outputs, and errors are human-readable and reusable by machines, subject to required privacy redaction.
3. **Given** a reported outcome, **When** an auditor follows its references, **Then** they can identify the policy
   control, evidence origins, intermediate transformations, applied criteria, and recorded explanation for the result.
4. **Given** an invocation that fails, **When** its trace is inspected, **Then** completed stages, the failure, and any
   incomplete work are identifiable without presenting the invocation as a completed compliance assessment.
5. **Given** an interactive execution in the client, **When** the workflow advances through its steps and stages,
   **Then**
   the client UX displays real-time progress updates focusing on how execution is advancing, formatted with rich text
   and descriptive emoticons.
6. **Given** execution across all modules and workflow stages, **When** operations run, **Then** the system records
   structured diagnostic logs adhering to project architectural standards (lazy formatting, module-scoped loggers,
   appropriate severity levels, and privacy/credential redaction), kept distinct from user-facing interactive progress
   output.
7. **Given** execution stages that invoke language models, **When** LLM calls are executed, **Then** the system captures
   and stores structured evaluation samples (prompts, completions, model metadata, and task context) in a standardized,
   machine-reusable format suitable for future model and prompt optimization and export to open-source MLOps tools.
8. **Given** creation or verification execution, **When** processing completes or terminates, **Then** Vatuta collects
   and reports both business metrics (violation counts, per-control compliance, evidence coverage, and outcome counts)
   and
   operational metrics (latencies, errors, token usage, and costs) measured end-to-end as well as broken down per stage
   and step.

9. **Given** retained traces and model interaction samples from successful and failed invocations, **When** no retention
   period is configured, **Then** those artifacts remain available regardless of age until explicitly deleted.
10. **Given** an explicitly configured retention period, **When** artifacts from a completed or failed invocation expire,
    **Then** they are automatically deleted, while active invocation artifacts and saved reusable control sets remain.

### Edge Cases

- **Repeated creation and document changes**: Successful creation saves a new control-set version and preserves prior
  versions. Compare hashes of policy document content, not file names or modification times. With no prior version,
  record an initial baseline rather than claiming that content changed. Failed creation publishes no new version and
  does not alter existing versions. Verification identifies the saved version and document hash it actually evaluates.
- **Missing creation-time trial parameters**: If trials are enabled, an unresolved value needed by any required trial
  fails the entire creation invocation with the missing parameter identified and no reusable controls published.
  Deferring that trial is not a successful creation path. Zero-attempt mode still skips trials, retains mandatory
  checks, and requires unresolved values before verification can use them.
- **Unsupported policy input**: The initial release accepts UTF-8 plain text and Markdown only. Unsupported formats
  and invalid UTF-8 are rejected before policy extraction with a clear input error and no reusable control set.
- **Artifact retention**: Without an explicitly configured retention period, invocation traces and model interaction
  samples remain until explicitly deleted. Configured expiry applies to retained artifacts from successful and failed
  invocations, without deleting active invocation artifacts or saved reusable control sets.
- **Creation-time validation exhaustion**: If any generated control still fails validation after all allowed attempts,
  fail the entire creation invocation. Retain diagnostic artifacts identifying the failed control and attempts;
  do not publish a reusable control set or a reusable subset from that invocation.
- **Trial execution skipped**: Zero creation-time validation attempts does not disable schema, dependency, or
  permission checks. Controls that pass these checks remain eligible for verification, with the skipped trial
  execution recorded. Verification execution errors remain execution failures, distinct from compliance outcomes.
- **Undefined percentages**: A percentage with a zero denominator is reported as undefined, never as 0% or 100%.
  Unknown outcomes remain visible in evidence coverage and outcome counts; they do not count as Non compliant.
- **Data source unavailability**: A required source being temporarily unavailable aborts the entire verification
  invocation
  immediately. Preserve the failure trace and identify unfinished work; do not publish a completed partial assessment.
- **Technical model failure**: A timeout, service error, or unusable response during verification aborts the entire
  invocation once any permitted retries are exhausted. Preserve available diagnostics and completed results as
  incomplete work. This is an execution failure, not an ambiguous semantic judgment or a compliance outcome.
- **Missing evidence from an available source**: A successful collection that cannot establish the facts needed for a
  control yields Insufficient evidence. This differs from a source access failure and from evidence of non-compliance.
- **Missing facts as violations**: When the policy explicitly requires a field or linked record, authoritative evidence
  that it is absent can establish Non compliant; inability to determine its presence yields Insufficient evidence.
- **Ambiguous semantic evaluation**: Preserve the prior fail-safe decision: an ambiguous semantic judgment is Non
  compliant.
  Record the uncertainty and distinguish it from a verified deterministic violation or unavailable evidence.
- **Conflicting rules**: Evaluate both rules independently and report both results, even when their requirements
  conflict.
- **Applicability and exceptions**: An established applicability mismatch or applicable exemption yields Not applicable.
  Missing facts needed to establish applicability or an exemption yield Insufficient evidence.
- **Alternative evidence**: Use the alternatives permitted by the policy. Do not infer that a missing optional
  alternative
  invalidates a sufficient allowed alternative, or invent equivalence between alternatives not authorized by the policy.
- **Invalid dependencies**: Reject cyclic dependencies or references to unknown controls before evaluation and identify
  the offending controls. A prerequisite's non-binary outcome remains explicit when used as downstream evidence.
- **Unselected prerequisites**: Automatically include all direct and indirect prerequisites of selected controls.
  Their execution remains within the configured scope and permissions. Reject selections whose prerequisites require
  access outside those boundaries, and distinguish selected controls from prerequisites in the report.
- **Missing parameters or unsupported sources**: Identify missing required inputs and unsupported source requirements
  before verification rather than silently widening the scope or substituting another source.
- **No violations found**: An empty violation list only supports a successful assessment when the applicable scope was
  evaluated completely. Insufficient evidence and invocation failures remain separately visible.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a `policy` command in the existing Vatuta client with separate creation and
  verification operations. A configuration file parameter MUST specify the policy document and execution scope.
  The initial release MUST accept policy documents in UTF-8 plain text and Markdown only. Unsupported formats and
  invalid UTF-8 MUST be rejected before policy extraction with an explicit input error identifying the supported
  formats, without publishing reusable controls. PDF, Word documents, scanned documents, and images are outside the
  initial input contract; automatic document conversion and OCR are not required.
- **FR-002**: Creation MUST extract and enumerate reusable atomic controls into a formalized and normalized control
  definition that acts as the authoritative bridge between the formal policy description and deterministic validation
  instructions. Each normalized control MUST encapsulate:
  - A unique identifier
  - A detailed description of the control
  - Policy references (document identifier, section/location, and version where identified)
  - Required evidence sources
  - Applicability conditions for evidence from sources
  - Required evidence transformations
  - Selection criteria to detect or identify violations, supporting multiple distinct paths (transformation + selection)
  - Criteria governing allowed exceptions
  - Required parameters and expected result structure, providing the evaluation engine with complete criteria to execute
    deterministic validation without re-reading the policy document.
- **FR-003**: Controls MUST support deterministic, semantic, and hybrid evaluation. When creating controls, the
  system MUST always prefer deterministic mechanisms over AI/LLM-based ones whenever the policy rule and available
  evidence
  can be evaluated deterministically; semantic (AI/LLM-based) evaluation MUST only be selected for criteria inherently
  dependent on natural-language meaning or semantic interpretation, and hybrid evaluation when both components are
  needed.
- **FR-004**: Verification MUST make source collection, evidence acquisition, facts normalization, evidence evaluation,
  and result evaluation identifiable. Collection obtains data; acquisition selects, combines, filters, projects, or
  aggregates relevant data; normalization produces consistent facts; evaluation applies the control; reporting records
  the result. These are logical responsibilities, not a mandated implementation topology.
- **FR-005**: Controls MUST be able to consume other controls' results as evidence. Verification MUST respect these
  dependencies, preserve result provenance, and reject unknown references and cycles before evaluation.
  Verification MUST automatically include all direct and indirect prerequisites of selected controls and evaluate
  each prerequisite before its dependents within the same invocation. This MUST NOT expand the configured scope or
  permissions; selections requiring prerequisites outside those boundaries MUST be rejected with the offending
  prerequisites identified.
- **FR-006**: Each control MUST identify its evaluation method and explicit decision criteria. Deterministic evaluation
  MUST return the same result for the same control, parameters, and evidence. Semantic evaluation MUST record the
  criteria
  and explanation supporting its outcome. Specific evaluator technologies are deferred to technical design.
- **FR-007**: Controls MUST support either a list of specific violations (returning 1 or N violations depending on
  the control's nature and evaluated subjects) or an aggregate assessment for that control, according to the policy.
  Compliance outcomes MUST distinguish Compliant, Non compliant, Insufficient evidence, and Not applicable. Reports MUST
  distinguish these outcomes from execution failures.
- **FR-008**: Initial validation MUST use one GitHub account and one Jira Free account through Vatuta's existing MCP
  source
  support. Controls MUST be able to combine evidence from those sources without changing external records.
- **FR-009**: Creation and verification MUST be usable without interactive terminal state (via headless CLI execution
  and programmatic API contracts) to ensure readiness for future MCP server wrapper integration, without requiring a
  bundled MCP service delivery in this release.
- **FR-010**: Every creation and verification invocation MUST have a unique, identifiable directory under the configured
  log directory, with subdirectories for its executed workflow stages, including every model-processing stage.
- **FR-011**: Stage traces MUST persist model prompts and responses, third-party requests and responses, errors, and
  intermediate data in human-readable, machine-reusable form. Credentials and sensitive personal data MUST be redacted
  according to the project constitution, and omissions MUST be identifiable without disclosing the redacted values.
- **FR-027**: Retention of invocation traces and model interaction samples MUST be configurable. If no retention period
  is explicitly configured, artifacts MUST remain until explicitly deleted, with no automatic deletion by age.
  An explicitly configured retention period MUST enable automatic expiry for artifacts from both successful and failed
  invocations. Active invocation artifacts and saved reusable control sets MUST NOT be deleted by this retention policy.
- **FR-022**: The client user interface MUST provide real-time visual progress updates across every workflow step and
  stage during execution, describing how the process is advancing using rich text and emoticons, ensuring the user is
  continually informed of system activity without exposing internal diagnostic noise.
- **FR-023**: All modules and execution workflows MUST generate structured diagnostic logs following the project's
  logging standards (module-scoped `logger = logging.getLogger(__name__)`, lazy printf-style formatting, semantic
  severity
  levels, centralized configuration, and privacy/credential redaction), maintaining strict architectural separation
  between internal diagnostic logs and user-facing interactive progress output.
- **FR-024**: All language model interactions (prompts, completions, model metadata, parameters, and task context) MUST
  be captured and persisted as structured evaluation samples. These samples MUST be formatted for future model and
  prompt
  optimization, benchmarking, and systematic integration into a widely adopted open-source MLOps platform, subject to
  applicable privacy and credential redaction.
- **FR-025**: The system MUST systematically collect, record, and report both business and operational metrics for every
  creation and verification invocation:
  - **Business metrics**: Total violations identified, violation breakdowns by subject and control, and compliance
    percentages, evidence coverage, and outcome counts computed per control. Reports MUST NOT average or pool controls'
    outcomes into an overall policy compliance percentage. A control evaluating a project, population, or reporting
    period MUST still report its own aggregate assessment and supporting measurements.
    For each control, let C, N, and I be the subject counts for Compliant, Non compliant, and Insufficient evidence.
    Compliance MUST equal 100 × C / (C + N), and evidence coverage MUST equal 100 × (C + N) / (C + N + I).
    Not applicable subjects MUST be excluded from both denominators. Reports MUST always present evidence coverage
    alongside compliance and retain all four outcome counts. A zero denominator MUST produce an undefined percentage.
    These calculations use the recorded outcomes, including the existing fail-safe Non compliant classification for
    ambiguous semantic judgments; evidence coverage does not imply certainty or correctness of those judgments.
  - **Operational metrics**: Execution duration, latencies, error counts and types, and model token consumption
    (prompt, completion, total tokens) along with associated operational costs, measured both end-to-end for the entire
    invocation and broken down granularly per stage and step.
- **FR-026**: During control creation, the system MUST validate the correctness of newly synthesized controls by
  executing test invocations against the relevant data sources, evidence transformation pipelines, or evaluation
  criteria
  mechanisms. If validation detects an error, malformed query/expression, or invalid evidence reference, the system MUST
  iteratively regenerate and revalidate the control up to a user-configurable maximum attempts threshold before
  reporting
  failure. Users MUST be able to configure this threshold to 1 (disabling retries upon failure) or 0 (disabling control
  validation invocations entirely) to conserve tokens, time, and computational resources.
  If any generated control still fails validation after all allowed attempts, creation MUST fail as a whole and retain
  diagnostic artifacts identifying the failed control and attempts. The failed invocation MUST NOT publish any reusable
  control set, including a subset of controls that passed validation. This exhaustion rule applies when validation is
  enabled; configuring zero validation attempts remains supported.
  Zero attempts MUST skip only creation-time trial execution. Schema, dependency, and permission checks MUST remain
  mandatory. Controls passing those checks MUST be eligible for verification without a separate creation-time trial
  execution, and both the saved controls and creation trace MUST record that trial execution was skipped. Verification
  MUST enforce its normal checks and report execution errors as execution failures, distinct from compliance outcomes.
- **FR-012**: Generated controls MUST be inspectable and reusable in an independent verification invocation.
  Verification MUST accept existing controls without requiring document analysis to run again, and MUST support
  evaluating either the complete list of controls, a single specific control by identifier, or a user-selected subset
  of controls. All direct and indirect prerequisite controls MUST be included automatically as required by FR-005.
- **FR-013**: Creation MUST identify evidence needs dynamically from the policy and available source capabilities. Each
  control MUST describe the subject, required evidence, source relationships, transformations, and normalization needed
  to evaluate it. Evidence requirements MUST NOT be limited to a fixed catalog of fields.
- **FR-014**: Creation and verification MUST accept relevant user instructions and parameters in addition to
  configuration.
  The system MUST expose required parameters and resolved values and identify missing required values before evaluation.
  When creation-time trials are enabled, every required trial MUST have its needed parameter values supplied. If any
  such value is missing, creation MUST fail as a whole, identify the missing parameter and affected trial, retain
  available diagnostics, and publish no reusable control set or subset. It MUST NOT defer the trial or invent a value
  to report successful creation. This requirement does not enable trials in zero-attempt mode or bypass mandatory
  checks. Verification MAY use different valid parameter values within its configured scope and permissions without
  repeating policy extraction; its resolved values MUST be recorded and required values supplied before use.
- **FR-015**: Controls MUST preserve the policy's applicability, permitted alternatives or selection conditions, and
  explicit
  exceptions. Verification MUST record which conditions and alternatives determined an outcome without adopting the
  example control syntax as a required format.
- **FR-016**: Multiple controls MUST be able to reuse equivalent evidence collection and preparation within the same
  invocation and scope. Reuse MUST preserve each control's evidence references and independent evaluation outcome.
- **FR-017**: Every outcome MUST be traceable to the control, originating policy reference, parameters, evidence
  sources,
  transformations, and evaluation criteria. Observed facts, model explanations, and final conclusions MUST remain
  distinct.
- **FR-018**: An unavailable required source MUST abort the entire verification invocation immediately. Otherwise,
  missing
  facts needed for evaluation MUST yield Insufficient evidence, unless their established absence violates the
  requirement.
  A technical model failure during verification, including a timeout, service error, or unusable response, MUST abort
  the entire invocation once any permitted retries are exhausted. The system MUST preserve available diagnostics and
  completed results as incomplete work, MUST NOT publish a completed assessment, and MUST NOT assign a compliance
  outcome to the failed call. The fail-safe semantic rule in FR-019 applies to ambiguous judgments, not technical
  model failures.
- **FR-019**: An ambiguous semantic judgment MUST yield Non compliant under the previously agreed fail-safe rule, with
  the
  ambiguity explicitly reported. Conflicting controls MUST still be evaluated and reported independently.
- **FR-021**: Reports MUST identify evaluated scope, per-control outcomes or violations, incomplete
  assessments, and invocation completion or failure. A violation MUST identify the affected subject and supporting
  evidence. An aggregate assessment MUST identify its population or time period and supporting measurements.
  Reports MUST identify which evaluated controls were explicitly selected and which were automatically included as
  prerequisites, preserving their dependency relationships and result provenance.

- **FR-028**: Every successful creation MUST save a separately identifiable control-set version without overwriting
  prior versions, which MUST remain available for verification. Failed creation MUST publish no new version and MUST
  NOT alter existing versions. Each saved version MUST include a hash of the exact supplied UTF-8 policy document
  content and the hash method. Creation MUST compare that hash with a recorded hash from a prior version of the same
  policy using the same hash method, identify the compared version, and report whether the content changed; without a
  prior version, it MUST record an initial baseline. File path and modification time MUST NOT substitute for content
  comparison. A matching document hash MUST NOT be treated as the identity of generated controls or as a reason to skip
  explicitly requested creation. Every verification MUST record the saved control-set version and associated document
  hash it evaluates, without requiring the original policy document to be reanalyzed or available.

### Key Entities

- **Control-set version**: A separately identifiable saved set of generated controls, associated with a policy document
  identity, the document's content hash and hash method, and its creation invocation. Prior versions remain available
  for verification. A document hash identifies source content, not the generated control-set version.
- **Policy configuration**: The document or document reference, source scope, and applicable parameter values supplied
  for
  policy operations. It remains the configuration-file entry point agreed in the earlier clarification.
- **Policy document**: The formal natural-language source of requirements, with document identity, section references,
  and version where available, supplied as UTF-8 plain text or Markdown in the initial release. Its exact supplied
  UTF-8 content is hashed to detect changes; this content hash is distinct from any version stated in the document.
- **Policy control (Normalized control)**: The formalized bridge linking the natural-language policy description
  to deterministic validation instructions. It encapsulates a unique identifier, detailed description, policy reference
  (document, section, version), evidence sources, source applicability conditions, evidence transformations, selection
  criteria (including multiple transformation + selection paths to detect violations), exception criteria, evaluation
  method, dependencies, and expected result form.
- **Subject**: The entity or population being evaluated, such as a pull request, linked issue, project, or sprint
  period.
- **Evidence**: Source observations, normalized facts, or prerequisite control results with provenance and
  transformation
  history sufficient to distinguish observed data from inferred conclusions.
- **Data source**: A configured external system and its available capabilities used to collect policy evidence.
- **Control result**: Specific violations (enumerating 1 or N individual violations depending on the control's nature
  and evaluated subjects) or an aggregate assessment for that control with supporting evidence and explanation.
  Execution
  failures are tracked separately from compliance outcomes.
- **Invocation trace**: The creation or verification record linking inputs, stage artifacts, external interactions,
  intermediate data, control results, and final completion or failure state.
- **Evaluation sample**: A standardized, persisted record of an LLM interaction—including rendered prompt, raw model
  completion, configuration metadata, latency/token telemetry, and evaluation task context—structured for machine
  consumption in model/prompt optimization workflows and ingestion into open-source MLOps platforms.
- **Policy execution metrics**: Quantitative business measurements (violations count, compliance percentages per
  control, and evidence coverage and outcome counts alongside compliance) and operational telemetry
  (latencies, error counts, token counts, and estimated cost)
  aggregated both end-to-end for the invocation and granularly per workflow stage and step.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a reference policy document with at least 20 independently annotated atomic rules, creation correctly
  captures at least 90% without manual correction, including their policy references and applicable RASE conditions.
- **SC-002**: In a controlled GitHub and Jira dataset with known violations and compliant subjects, verification
  identifies
  100% of expected deterministic violations and produces no unexpected deterministic violations.
- **SC-003**: Automated deterministic verification executes end-to-end against configured sources without hanging or
  unhandled failures, recording measured duration, step latencies, and operational telemetry.
- **SC-004**: Every creation and verification acceptance run, including a failed run, produces an identifiable stage
  trace
  containing its executed interactions and intermediate artifacts, subject to documented redaction.
- **SC-005**: Every operation in the programmatic invocation acceptance set produces the same report structure, data
  models, and error semantics as its interactive CLI equivalent when executed in headless mode.
- **SC-006**: Saved controls are successfully verified in a separate invocation without repeating policy extraction in
  every reuse acceptance scenario, covering runs of the full set of controls, a single selected control, and a subset
  of controls.
  Selection acceptance fixtures MUST cover direct and indirect prerequisites, distinguish selected controls from
  prerequisites in reports, and reject dependencies that would require expanding the configured scope or permissions.
- **SC-007**: Acceptance fixtures demonstrate all four compliance outcomes, conflicting rules, missing evidence,
  source unavailability, cross-source evidence, and dependency ordering.
  Fixtures for a model timeout, service error, and unusable response MUST demonstrate invocation failure after any
  permitted retries, retained incomplete work, and no compliance outcome attributed to the failed call.
- **SC-008**: In an acceptance run with at least two controls requiring identical evidence, one collection and
  preparation
  of that shared evidence supports both results, with complete provenance for each control.
- **SC-009**: Per-control metrics acceptance fixtures produce the FR-025 compliance and evidence coverage ratios for
  mixed outcomes, all-unknown outcomes, all-exempt outcomes, and empty populations. Every reported compliance percentage
  is accompanied by evidence coverage and outcome counts, with zero denominators represented as undefined.
  A report combining controls over different populations MUST retain their separate metrics and publish no overall
  policy compliance percentage.

- **SC-010**: Versioning acceptance fixtures MUST demonstrate that successful repeated creation preserves prior sets,
  unchanged document content produces matching hashes even at different paths, changed content is detected at the same
  path, and failed creation leaves existing versions intact. Verification of an earlier version MUST report that
  version and its associated document hash. A successful rerun with unchanged content MUST have a distinct control-set
  version even when its document hash matches the earlier version.

## Clarifications

### Session 2026-10-20

- Q: When creation runs again for the same policy, should a successful result replace the previous saved control set
  or create a new version? → A: Create a new version, retain earlier versions for verification, and record the version
  used by each run. Use a policy document content hash to detect whether its content changed.
- Q: When a required parameter will only be supplied during verification, how should creation handle trial executions
  that need its value? → A: If trials are enabled, fail creation until values are supplied for every required trial;
  verification may later use different valid values. Zero-attempt mode continues to skip trials.
- Q: Which policy document formats must control creation accept in the initial release? → A: UTF-8 plain text and
  Markdown only.
- Q: How long should invocation traces and model interaction samples be retained by default? → A: Retention is
  configurable; without an explicitly configured period, retain artifacts until explicitly deleted. Automatic expiry
  requires configuration and applies to successful and failed invocation artifacts, not saved reusable control sets.

### Session 2026-10-19

- Q: If a model call fails during verification—through a timeout, service error, or unusable response—should the entire
  invocation stop? → A: Yes, once any permitted retries are exhausted; preserve available diagnostics and completed
  results as incomplete work, publish no completed assessment, and assign no compliance outcome to the failed call.
- Q: When a user selects one control for verification, should Vatuta automatically execute prerequisite controls that
  were not selected? → A: Yes; execute all direct and indirect prerequisites within the configured scope and
  permissions,
  and distinguish selected controls from automatically included prerequisites in the report.
- Q: How should the report combine compliance across controls that assess different populations? → A: Publish no
  overall policy percentage; show each control's compliance, evidence coverage, and outcome counts separately.
- Q: How should a control's compliance percentage treat subjects with insufficient evidence? → A: Exclude them from
  the compliance denominator and always show evidence coverage alongside compliance; exclude Not applicable subjects
  from both calculations. For 80 Compliant, 10 Non compliant, and 10 Insufficient evidence, report approximately 88.9%
  compliance and 90% evidence coverage.
- Q: When the user sets creation-time validation attempts to zero, should the resulting controls be eligible for
  verification? → A: Yes; retain mandatory schema, dependency, and permission checks, record that trial execution was
  skipped, and handle verification execution errors normally.
- Q: If a generated control still fails validation after all allowed attempts, what should happen to the control set?
  → A: Fail creation as a whole; retain diagnostic artifacts but publish no reusable control set or subset.
- Q: Must the policy checker delivery include an operational MCP service or only readiness for later integration?
  (FR-009) → A: Readiness only: Expose capabilities via headless CLI and clean programmatic Python API contracts
  structured so an MCP server can wrap them in a future release.

### Session 2026-10-08

- Q: What specific data entities from Github and Jira will be extracted to act as "Evidence" for policy validation? → A:
  It is dynamic. The policy analysis phase will identify the required evidence, which will then be extracted and
  transformed.
- Q: How will the user specify the input policy document to the `policy` CLI command or MCP service? → A: Via a
  configuration file parameter that contains both the formal policy and its execution scope.
- Q: How should the system handle a situation where the semantic evaluation (LLM) returns an ambiguous compliance
  result? → A: Assume "Non compliant" (fail-safe).
- Q: How should the system handle a temporarily unavailable data source during policy verification? → A: Fail the entire
  policy verification execution immediately.
- Q: How should the system handle conflicting rules or overlapping applicabilities extracted from the policy document? →
  A: Evaluate both rules independently and report both results, even if conflicting.

## Reconciliation Notes

- **Control sets are versioned and policy changes are hashed**: Each successful creation produces a new saved version;
  earlier versions remain available. A policy document content hash detects source changes separately from the identity
  of generated controls. See [ADR-010](../../docs/policy-checker/adr-010-versioned-controls-and-policy-hashes.md).
- **Enabled trials require parameter values**: Creation fails if any enabled required trial lacks a needed value;
  deferral is not a successful creation path. Zero-attempt mode remains available, and later verification may supply
  different valid values. See [ADR-009](../../docs/policy-checker/adr-009-required-trial-parameters.md).
- **Policy input formats are bounded**: The initial release accepts UTF-8 plain text and Markdown; PDF, Word,
  scanned documents, and images are outside the input contract. See
  [ADR-008](../../docs/policy-checker/adr-008-policy-input-formats.md).
- **Retention is configurable and defaults to no expiry**: Invocation traces and model interaction samples are retained
  until explicitly deleted unless an expiry period is configured. Saved reusable controls are outside this policy.
  See [ADR-007](../../docs/policy-checker/adr-007-configurable-artifact-retention.md).
- **Technical model failures abort verification**: Once any permitted retries are exhausted, a model timeout, service
  error, or unusable response stops the invocation. This does not change the fail-safe classification of ambiguous
  semantic judgments. See [ADR-006](../../docs/policy-checker/adr-006-technical-model-failure.md).
- **Creation failure is atomic**: Exhausted control validation attempts fail the whole creation invocation; diagnostic
  artifacts remain available, but no reusable control set or subset is published. See
  [ADR-001](../../docs/policy-checker/adr-001-creation-validation-failure.md) and the
  [policy checker glossary](../../docs/policy-checker/glossary.md).
- **Zero attempts skips trial execution only**: Controls passing mandatory schema, dependency, and permission checks
  remain eligible for verification, with skipped trial execution recorded. See
  [ADR-002](../../docs/policy-checker/adr-002-zero-attempt-verification.md).
- **Compliance and evidence coverage are distinct**: Per-control compliance uses only Compliant and Non compliant
  outcomes, while evidence coverage also accounts for Insufficient evidence. Not applicable subjects are excluded
  from both. See [ADR-003](../../docs/policy-checker/adr-003-compliance-and-evidence-coverage.md).
- **No overall policy compliance percentage**: Per-control compliance, evidence coverage, and outcome counts replace
  the earlier requirement for a percentage across controls. Individual aggregate controls retain their assessments.
  See [ADR-004](../../docs/policy-checker/adr-004-per-control-reporting.md).
- **Selection includes prerequisites**: Verification automatically executes direct and indirect prerequisites within
  the configured scope and permissions, and reports them separately from explicitly selected controls. See
  [ADR-005](../../docs/policy-checker/adr-005-automatic-prerequisites.md).
- **MCP delivery scope resolved**: Per clarification on 2026-10-19, this delivery focuses on headless CLI execution and
  clean programmatic Python API contracts (readiness), without bundling an operational MCP service in this initial
  release.
- **Human review excluded from requirements**: Per user instruction, human review workflows or handoffs are not part of
  the requirements; policy evaluation is automated (deterministic, semantic, or hybrid).
- **Evaluator technologies are design inputs**: FR-006 now states required behavior rather than selecting an evaluator.
  Candidate technologies and the reasons to investigate them are retained in [design-inputs.md](design-inputs.md).
- **RASE examples are illustrative**: The concepts remain required, but example field names, expressions, graph layouts,
  and exact selection syntax or semantics are not an approved control contract.
- **Initial connectors remain bounded**: References to other source products in examples do not expand the GitHub and
  Jira Free validation scope.
- **Outcome distinctions preserve prior answers**: Missing evidence, an unavailable source, and an ambiguous semantic
  judgment are separate cases. The previously agreed fail-fast and fail-safe behaviors remain in force.
- **Audit completeness respects the constitution**: The notes request all intermediate information; recordings preserve
  auditability while applying the project's existing credential and personal-data redaction requirements.
- **Performance targets unbounded**: The previous artificial two-minute completion restriction has been removed;
  verification focuses on non-hanging completion and recording operational telemetry and duration without arbitrary SLA
  ceilings.

## Assumptions

- Users have the access permissions and existing MCP source configuration needed to read GitHub and Jira Free data.
- Verification is read-only against external sources; remediation and changes to source records are outside this
  feature.
- RASE supplies the conceptual decomposition. Requirements must preserve the meaning of the source policy; the technical
  design will define the representation and validate that it supports the required conditions and alternatives.
- Evidence reuse initially applies within one invocation and equivalent scope. Cross-invocation caching, freshness
  policies, and reuse across users are not requirements of this specification.
- Explicit exceptions yield Not applicable. Alternatives are determined by the source policy, not by interpreting the
  example documents as a universal selection algorithm.
- The input configuration remains the authoritative document and scope entry point. Supplemental instructions and
  parameters do not silently expand that scope.
- Audit explanations describe evidence, transformations, criteria, and recorded model outputs; they do not require
  access to a model provider's unavailable internal reasoning.

## Technical Design Inputs

Non-binding technology candidates and open design investigations are recorded separately in
[design-inputs.md](design-inputs.md). They are inputs to a later planning phase, not an implementation plan or decisions
that constrain acceptance of this specification.
