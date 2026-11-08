# Policy Validation Engine: Interview and Decision Record

**Updated**: 2026-10-30
**Status**: Current scope clarified; earlier answers retained below with supersession
**Specification**: [spec.md](../../specs/002-policy-checker/spec.md)
**Design**: [design.md](design.md)
**Terms and ADR index**: [glossary.md](glossary.md)

## Current Agreement — 2026-10-30

The stakeholder approved simplifying the existing feature to creation and execution of control files, plus deterministic
parameter discovery on 2026-10-25, then refined MCP creation inputs on 2026-10-29. Trace resource URIs were agreed on
2026-10-30. This section and
[ADR-019](adr-019-file-based-policy-operations.md) supersede incompatible earlier answers below. The current specification
is authoritative; older FR/SC/story references describe their earlier revision.

1. MCP creation accepts `policy_text`, optional provenance identifier `policy_id`, optional descriptive
   `construction_context`, and optional nonnegative `trial_attempts`, and returns individual YAML controls. The CLI may
   read supported files into text. Callers need no advance knowledge of parameter names; the constructor derives
   declarations and supported trial values. The client may save artifacts; MCP creation does not manage that directory.
   Generated `source.policy` references the supplied `policy_id`, or the creation invocation ID when omitted.
2. Select controls by file, file array, or glob. No catalog, published revision, approval, or managed review is needed.
3. MCP parameter discovery deterministically returns required inputs, semantics, defaults, and referencing controls.
   The invoking agent obtains values from prompt/context. Execution receives explicit values and does not infer them.
4. Same-name parameters are the same shared input and use one value across selected controls. Construction values may
   explicitly serve as defaults; a construction value alone does not make a parameter optional.
5. Validate all selected controls' mandatory inputs before execution. Missing values reject the complete request with
   their meanings and affected controls, before any evidence acquisition or evaluation.
6. Creation remains complete-or-error. An unrepresentable obligation fails the whole creation; no reusable partial set
   is returned. Explicit request `trial_attempts` overrides deployment setting `policy.trial_attempts` for that creation;
   when omitted, use deployment configuration, default one.
   Mandatory checks, bounded corrections, and zero-attempt behavior remain. Missing required trial information returns
   actionable `MISSING_CONSTRUCTION_CONTEXT` errors, allowing a new request with expanded context, without inventing
   values or implicitly skipping trials. Context cannot weaken policy obligations or expand configured permissions.
7. Declare deterministic or exceptional LLM evaluation during creation. The engine follows the YAML mode; deterministic
   failures do not cause runtime LLM fallback, mode selection, policy reinterpretation, or criterion repair.
8. Once request validation succeeds, control failures, including unavailable sources, do not stop independent controls.
   Preserve useful results and visible errors under both filters. This replaces the older source-abort exception.
9. Remove managed modification, review, approval, publication, comparison, stable identity reconciliation, versions,
   historical retrieval, and restoration. A later creation is an independent request.
10. Keep diagnostic traces and privacy protections in the log directory. Remove retention configuration and automatic
    age-based cleanup. Diagnostic persistence does not create policy or control version history.

## Working Method

Ask one concrete behavioral question at a time, explain its consequences, and recommend the simplest adequate option.
Recommendations remain proposals until answered. Update the specification, relevant ADR, glossary, and design when an
answer is accepted. Do not reopen the closed choices below without a verified contradiction or feasibility problem.

## Historical Decisions from the 2026-10-20 Stakeholder Brief

- Model-assisted Policy Authoring and Maintenance is separate from a model-free deterministic Policy Execution Engine.
- Each Control is an independent YAML artifact with Evidence, JSONata Selection, and a single boolean CEL Criterion.
- MCP calls in Evidence are independent; joins belong to Selection. No dependency graph or provider-specific engine
  logic.
- Subjects are JSON objects with `id` and `descriptive_id`; no implicit scalar wrapping or generated identifiers.
- Subject states are COMPLIANT, NON_COMPLIANT, and ERROR. Empty successful selection yields global NOT_APPLICABLE.
- Invocation filtering defaults to violations; errors remain visible with either filter.
- Policy Documents are the only normative source. Updated documents produce complete regenerated proposals.
- Review and approval precede publication. Stable IDs persist for the same obligation; uncertain mappings are explicit.
- Internal immutable incremental revisions contain the document and complete approved controls. No Git policy history
  or independent per-control version subsystem is required.
- No semantic search, incremental semantic impact engine, generated violation explanations, or arbitrary performance
  targets. No extra caching, retries, concurrency, or distributed mechanisms are inferred from examples.

## Earlier Interview Answers

### Question 1 — Partial Subject Errors

**Question**: If three subjects evaluate to true, false, and a technical error, how should the control execution finish?

**Recommendation**: Evaluate all subjects and return global ERROR if any fails, preserving all results.

**Alternatives presented**:

- A: Continue all subjects; any subject error makes the control's global result ERROR, with all results retained.
- B: Stop at the first subject error; global ERROR with only the results obtained so far.
- C: Continue all subjects and report results/errors without a global aggregate state for nonempty selections.

**Stakeholder answer**: A. Evaluate all subjects and return global ERROR if any fails, preserving every result.

**Recorded in**: FR-018, User Story 3, SC-003, [ADR-017](adr-017-partial-subject-errors.md), glossary, and design.
This decision is about criterion evaluation of valid subjects within one control, not cross-control continuation.

### Question 2 — Root Null from Selection

**Question**: How should the engine handle null as the root result of Selection?

**Recommendation presented**: A. No result or an empty collection yields NOT_APPLICABLE; root null is a contract
error. This preserves the object/collection contract and distinguishes no subjects from an invalid root value.
A valid subject containing null-valued fields remains evaluable under either option.

**Alternatives presented**:

- A: No result or empty collection means NOT_APPLICABLE; root null means a contract ERROR.
- B: No result, empty collection, or root null all mean NOT_APPLICABLE.

**Stakeholder answer (2026-10-21)**: A. No result or an empty collection means NOT_APPLICABLE; root null is a Selection
contract ERROR. CEL is not executed in any of these cases; a valid subject with null-valued fields remains evaluable.

**Recorded in**: FR-015, User Story 2, SC-003, [ADR-014](adr-014-selection-subject-contract.md), glossary, and design.

### Question 3 — Invalid Objects in a Selected Collection

**Question**: If Selection returns several objects and one lacks a mandatory subject identifier, should the engine
reject
this control's complete Selection output or evaluate the valid subjects and report the invalid object separately?

**Recommendation presented**: A. Reject this control's Selection output with a contract ERROR before CEL runs.
This keeps the stage contract simple and avoids manufacturing an identity for the malformed object. Diagnostic artifacts
still retain the output for investigation; this does not decide whether other controls in the invocation continue.

**Alternatives to discuss**:

- A: Reject the complete Selection output for this control; global ERROR and no criterion evaluation.
- B: Evaluate valid subjects and return global ERROR with separate diagnostics for malformed objects, without inventing
  IDs.

**Stakeholder answer (2026-10-21)**: B. Evaluate valid subjects and return global ERROR with separate diagnostics for
invalid objects, without inventing IDs. Both filters preserve errors. If a nonempty collection has only invalid objects,
there are no valid subjects for CEL, but the control still has contract errors and is not NOT_APPLICABLE.

**Recorded in**: FR-015, FR-018, User Story 2, SC-003, [ADR-014](adr-014-selection-subject-contract.md), glossary, and
design.

### Question 4 — Global Outcome Without Errors

**Question**: When a control has evaluated at least one subject without any errors, how should its global result be set?

**Recommendation presented**: A. NON_COMPLIANT if any subject is NON_COMPLIANT; otherwise COMPLIANT. This gives
the
consumer one predictable control outcome without changing individual results or the accepted ERROR precedence.

**Alternatives to discuss**:

- A: Any violation makes the global result NON_COMPLIANT; all subjects compliant means global COMPLIANT.
- B: Do not return an aggregate global outcome for error-free nonempty executions; return individual subject results
  only.

**Stakeholder answer (2026-10-21)**: B. Return individual results only, without an aggregate global state for error-free
nonempty executions. Global ERROR for errors and NOT_APPLICABLE for successful empty selection remain unchanged.
An empty violations-only findings list does not turn a nonempty evaluated population into NOT_APPLICABLE.

**Recorded in**: FR-018, User Story 3, SC-005, [ADR-017](adr-017-partial-subject-errors.md), glossary, and design.

### Question 5 — Evidence Explicitly Reported as Incomplete

**Question**: If a required MCP tool explicitly reports that its returned dataset is incomplete, should the affected
control stop before Selection and Criterion, or evaluate the returned subset while reporting incompleteness?

**Recommendation presented**: A. Report a control execution ERROR without evaluating Selection or Criterion.
A missing related issue in an incomplete dataset does not establish that the issue is absent. This option needs no
automatic pagination, dependent calls, or invented state. It does not decide other controls' continuation.

**Alternatives to discuss**:

- A: Report ERROR for the affected control without evaluating the incomplete data.
- B: Evaluate the returned subset, preserve individual results, and report incompleteness with global ERROR.

**Stakeholder answer (2026-10-21)**: A. Return ERROR for the affected control without executing Selection or Criterion
when required evidence is explicitly reported incomplete. Do not produce subject verdicts from that partial dataset
or add automatic pagination. At this point response indicators and unknown-completeness handling were unresolved;
Question 11 below subsequently resolves the behavior, while indicator detection remains technical investigation.

**Recorded in**: FR-018, User Story 2, SC-009, [ADR-013](adr-013-independent-evidence-calls.md), glossary, and design.

### Question 6 — Ambiguous Control Identity During Regeneration

**Question**: If the regenerated proposal cannot unambiguously map a control to an existing stable ID, should the
manager
resolve the correspondence during review or should the proposal be rejected and generated again?

**Recommendation presented**: A. Keep the proposal unpublished until the manager explicitly resolves ambiguous
correspondences, retaining an existing identity for the same obligation or confirming a new control. This preserves
useful generation work without a semantic reconciliation algorithm. The exact review interface is design work.

**Alternatives to discuss**:

- A: Resolve ambiguous identity mappings explicitly during review before approving and publishing the proposal.
- B: Reject the ambiguous proposal and require a new generation; provide no manual correspondence-resolution workflow.

**Stakeholder answer (2026-10-21)**: B. Reject the ambiguous proposal and request a new generation, without a manual
correspondence-resolution workflow. Identify the ambiguity, preserve published revisions and history, and do not publish
a subset. This adds no automatic regeneration loop or changes to approved trial attempt limits.

**Recorded in**: FR-030, User Story 4, SC-010, [ADR-015](adr-015-document-authority-and-regeneration.md), glossary, and
design.

### Question 7 — Restoring a Previous Configuration

**Question**: When restoring an earlier approved document and control set, should Vatuta publish a new revision with
that content or reactivate the earlier revision without creating another revision?

**Recommendation presented**: A. Publish a new incremental revision containing the selected historical document
and controls, without LLM regeneration. This records the restoration in revision history while preserving existing
revisions and stable control identities. Publication remains subject to the accepted approval requirement.

**Alternatives to discuss**:

- A: Publish a new revision copying the earlier approved document and controls, without regeneration.
- B: Reactivate the existing historical revision without creating a new one.

**Stakeholder answer (2026-10-21)**: A. Publish a new incremental revision copying the historical document and complete
control set without LLM regeneration. Preserve control identities and definitions and leave all existing revisions
unchanged. Review and approval remain required; restoration does not merely reactivate the historical revision.

**Recorded in**: FR-033, User Story 5, SC-007, [ADR-016](adr-016-immutable-policy-revisions.md), glossary, and design.

### Question 8 — Transformation or Criterion Errors Across Independent Controls

**Question**: In an invocation containing several independent controls, if one control returns ERROR from Selection or
CEL, should the remaining controls still execute or should the invocation stop before executing them?

**Recommendation presented**: A. Continue the other requested controls and retain each control's results and
errors.
They have no dependency on the failed control. This is separate from already accepted within-control subject handling
and does not reopen the earlier fail-fast rule for an unavailable required evidence source.

**Alternatives to discuss**:

- A: Continue the remaining independent controls and preserve each control's error visibility.
- B: Stop the invocation after the control error, retaining obtained results and identifying controls left unevaluated.

**Stakeholder answer (2026-10-21)**: A. Continue the remaining independent requested controls and preserve per-control
results and error visibility. This does not change the earlier invocation-abort rule for an unavailable required source,
introduce retries, or add an aggregate policy compliance state.

**Recorded in**: FR-018, User Story 2, SC-011, [ADR-017](adr-017-partial-subject-errors.md), glossary, and design.

### Question 9 — Accepted MCP Evidence Payloads

**Observed context**: The installed SDK distinguishes optional structured content from a list of text/non-text content
blocks and a tool-error flag. Vatuta currently forwards the tool result rather than converting it to uniform evidence.
The actual reference tools' payloads still need inspection.

**Question**: Should the initial evidence contract require structured JSON or one text block containing a complete valid
JSON value, or should it also attempt to extract JSON embedded in explanatory text?

**Recommendation presented**: A. Prefer structured content; otherwise accept exactly one text block whose complete
text parses as JSON. Other forms produce an explicit evidence-conversion error. This keeps a provider-neutral,
deterministic boundary without extracting fragments or guessing which representation the author intended.

**Alternatives to discuss**:

- A: Structured content, or one complete JSON text block; reject other payload forms explicitly.
- B: Also extract JSON fragments from surrounding explanatory text or Markdown formatting.

**Stakeholder answer (2026-10-21)**: A. Prefer structured content; otherwise accept exactly one text block whose
complete content is valid JSON. Reject other forms explicitly. An explicit tool error remains an error even when
parseable JSON accompanies it. Do not extract JSON fragments from prose or Markdown.

**Recorded in**: FR-013, User Story 2, [ADR-013](adr-013-independent-evidence-calls.md), glossary, design, design
inputs.

### Question 10 — Unusable Evidence Responses Across Controls

**Question**: If a required tool responds but its payload violates the accepted contract or explicitly reports an
incomplete dataset, should the invocation continue with its other independent controls?

**Recommendation presented**: A. Return ERROR for the affected control without Selection or Criterion, and
continue the other independent requested controls. Preserve every result and error. Source unavailability still
aborts the invocation under the existing decision; this question concerns responses that were received.

- A: Continue the remaining independent controls; the affected control reports ERROR without subject verdicts.
- B: Abort the invocation, retain completed results as incomplete work, and identify unevaluated controls.

**Stakeholder answer (2026-10-21)**: A. Unsupported or explicitly incomplete received evidence stops only the affected
control with ERROR, without Selection, Criterion, or subject verdicts. Continue other independent requested controls
and preserve each control's results and errors. Required-source unavailability still aborts the whole invocation.

**Recorded in**: FR-018, User Story 2, SC-011, ADR-013, ADR-017, glossary, design, and design inputs.

### Question 11 — Unknown Evidence Completeness

**Observed context**: Local MCP integration and tests do not establish completeness guarantees for the actual
GitHub/Jira reference tools. Valid JSON alone does not establish complete retrieval. The relevant tool contract or
response may provide a guarantee without a universal completeness flag; actual mechanisms require investigation.

**Question**: If neither the tool contract nor its response establishes that the required dataset is complete for the
control's declared scope, should the control evaluate it or return ERROR before evaluating subjects?

**Recommendation presented**: A. Return ERROR for the affected control without Selection or Criterion and
continue other independent requested controls. This avoids treating a linked ticket omitted by an unknown retrieval
limit as absent, or treating an empty partial population as NOT_APPLICABLE. It can prevent use of tools whose
completeness cannot be established. No automatic pagination or universal metadata field is proposed.

- A: Do not evaluate when completeness cannot be established; return ERROR and continue independent controls.
- B: Evaluate the returned data when no explicit incompleteness signal exists, accepting possible omitted records.

**Stakeholder answer (2026-10-21)**: A. If neither the tool contract nor response establishes completeness for the
control's declared scope, return ERROR before Selection and Criterion, without subject verdicts. Continue the other
independent requested controls and retain visible errors under both filters. No universal completeness flag,
automatic pagination, or new outcome state is required.

**Recorded in**: FR-018, User Story 2, SC-012, ADR-013, ADR-017, glossary, design, and design inputs.

## Interview Outcome

No stakeholder behavior question is currently pending. The known-unrepresentable-rule item is covered by existing
complete-proposal and no-subset publication requirements (FR-002, FR-026, FR-031; ADR-001 and ADR-015): failure to
produce the complete executable proposal means authoring failure with diagnostics and no publication. User Story 1
and SC-001 now make that consequence explicit. This is a consistency clarification of existing rules, not approval
for partial coverage, manual evaluation, or a new fallback. Published revisions remain unchanged.

The specification can enter technical planning; this does not claim implementation feasibility has been demonstrated.
No formal plan is generated. Reopen a stakeholder decision only if investigation demonstrates a material conflict,
unsupported required capability, or need for additional scope.

## Remaining Technical Investigations

These are technical investigation items, not outstanding stakeholder questions or permission to add capabilities.

1. Define exact error serialization during technical design. Continue after Selection/CEL failures and unsupported or
   explicitly incomplete received evidence. Required-source unavailability still aborts the invocation.
2. Inspect real retrieval-completeness limits and reference-tool compatibility with the accepted MCP payload contract.
Missing linked records cannot be declared
   absent from a collection known to be incomplete. Inspect actual tool behavior before proposing additional mechanisms.
3. Specify minimal provenance and identity-ambiguity detection during technical design; rejection and new generation
   are now accepted behavior, not an open choice of manual resolution workflow.
4. Implement the accepted publication/restoration behavior in the technical design: approval precedes publication,
   and restoration creates a new revision from historical content without regeneration. Exact metadata remains design
   work.
5. Specify how authoring validation and review expose known unrepresentable obligations under the existing
complete-proposal rule, without claiming infallible extraction or adding semantic execution.

Exact field names, Python binding selection, storage representation, and module layout are technical investigation,
not questions to ask the stakeholder without a material behavioral tradeoff.

## Supersession of Earlier Decisions

Earlier answers are preserved below as history. The explicit 2026-10-20 deterministic-engine brief takes precedence
over conflicting historical statements; those statements are not active requirements merely because they remain here.

- ADR-001 and ADR-002 remain applicable to authoring validation, with explicit review and approval before publication.
  Historical dependency checks do not require the retired prerequisite scheduler.
- ADR-003's compliance/coverage formulas are superseded by the new result model. Do not reuse their unknown-state or
  per-subject NOT_APPLICABLE semantics.
- ADR-004's no-overall-policy-percentage decision remains; its old per-control formulas are no longer required.
- ADR-005's automatic prerequisites are superseded by independent Evidence calls and the absence of dependency graphs.
- ADR-006's model failure handling during verification and the earlier semantic fail-safe rule no longer apply because
  verification has no model calls.
- ADR-007, ADR-008, and ADR-009 remain applicable to retention, input formats, and enabled authoring-trial parameters.
- ADR-010's automatic version-on-success workflow is superseded by approved immutable full revisions. Document hashing,
  retained history, and execution attribution remain under ADR-016.
- Headless readiness remains in scope; an operational MCP server remains outside the initial delivery.
- Human review of compliance outcomes remains excluded. Review of control proposals before publication is required.
- Required shared-evidence optimization and the historical RASE executable structure are not carried into initial scope.
- No two-minute target or other speculative numeric performance requirement is carried forward.

## Earlier Clarification Record — Historical

The following question-and-answer entries are preserved from the specification as it stood before this interview.
Apply the supersession notes above before interpreting them. Historical session dates do not distinguish the later
same-day deterministic-engine brief by themselves.

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
