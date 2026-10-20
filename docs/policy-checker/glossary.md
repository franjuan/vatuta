# Policy Checker Glossary

**Updated**: 2026-10-20
**Source of truth**: [Policy Checker Mechanism specification](../../specs/002-policy-checker/spec.md)

These terms describe the feature requirements and accepted decisions; they do not imply that the feature is implemented.

| Term | Meaning |
| ------ | --------- |
| Policy document | The natural-language source of requirements, identified by document, section, and version where available. The initial release accepts UTF-8 plain text or Markdown. |
| Supported policy formats | UTF-8 plain text and Markdown. PDF, Word documents, scanned documents, and images are outside the initial policy input contract. |
| Policy control (normalized control) | A reusable definition connecting a policy requirement to its evidence, applicability, transformations, criteria, exceptions, parameters, and evaluation method. |
| Control set | The collection of controls produced for reuse by a creation invocation. A failed creation invocation publishes neither a reusable set nor a reusable subset. |
| Control-set version | A separately identifiable saved control set produced by successful creation. Earlier versions remain available, and verification records which version it evaluates. |
| Policy document hash | A hash of the exact supplied UTF-8 policy content, stored with its hash method and control-set version to detect document changes. It identifies source content rather than generated controls; different control-set versions may share a document hash. |
| Initial hash baseline | The document hash recorded when there is no prior control-set version of the policy to compare against. It does not claim that the document changed. |
| Creation | The operation that analyzes a policy document and produces reusable control definitions. It is separate from verification. |
| Creation-time validation (trial execution) | Test invocations used during creation to check generated source queries, transformations, and evaluation mechanisms. Zero attempts skips these trials without bypassing mandatory checks. |
| Trial parameter value | A supplied value needed by an enabled creation-time trial. A missing value fails creation; the trial is not deferred to make creation succeed. Verification may later use a different valid value. |
| Resolved parameter value | The value actually used for a parameter in a particular creation or verification invocation, recorded for traceability and subject to its configured scope and permissions. |
| Mandatory checks | Schema, dependency, and permission checks that remain required even when creation-time trial execution is skipped. |
| Verification eligibility | Controls passing mandatory checks can be used for verification even if creation-time trial execution was skipped. Eligibility does not establish that trial execution succeeded. |
| Trial execution skipped | The recorded state of saved controls and their creation trace when zero attempts disables creation-time test invocations. |
| Validation exhaustion | A generated control still fails creation-time validation after all allowed attempts. Under ADR-001, this fails the whole creation invocation. |
| Verification | The operation that evaluates saved controls against evidence, using the supplied scope and parameters. |
| Selected control | A control explicitly requested for verification, individually, as part of a selected subset, or by selecting the full control set. |
| Prerequisite control | A control whose result is needed by another control, directly or indirectly. Unselected prerequisites are automatically included within the configured scope and permissions and identified in the report. |
| Subject | The entity or population evaluated, such as a pull request, linked issue, project, or reporting period. |
| Evidence | Source observations, normalized facts, or prerequisite control results with their origin and transformation history preserved. |
| Compliance outcome | Compliant, Non compliant, Insufficient evidence, or Not applicable; distinct from an execution failure. |
| Compliance percentage | For a control, 100 × Compliant / (Compliant + Non compliant), using subject counts. Always displayed with evidence coverage and outcome counts. |
| Evidence coverage | For a control, 100 × (Compliant + Non compliant) / (Compliant + Non compliant + Insufficient evidence). Excludes Not applicable subjects and does not measure confidence in semantic judgments. |
| Undefined percentage | A ratio whose denominator is zero. Reported as undefined rather than assigning 0% or 100%. |
| Aggregate control assessment | One control's evaluation of a project, population, or reporting period, supported by its measurements and governing criteria. It does not combine percentages from different controls. |
| Policy report | The invocation's report of evaluated scope, per-control outcomes, compliance, evidence coverage, and execution status. It contains no overall compliance percentage derived across controls. |
| Execution failure | An invocation that cannot complete its operation, such as creation-time validation exhaustion or an unavailable required source during verification. |
| Technical model failure | A model timeout, service error, or unusable response that provides no usable evaluation judgment. During verification, it aborts the whole invocation once any permitted retries are exhausted and produces no compliance outcome. |
| Ambiguous semantic judgment | A usable model judgment expressing uncertainty about the policy criterion. The existing fail-safe rule classifies it as Non compliant with the uncertainty disclosed; it is distinct from a technical model failure. |
| Diagnostic artifact | A retained record of inputs, outputs, errors, or intermediate work used to investigate execution. Retention does not publish a reusable control set. |
| Artifact retention | The configurable lifetime of invocation traces and model interaction samples. Without an explicitly configured period, artifacts remain until explicitly deleted. Saved reusable controls are outside this policy. |
| Automatic expiry | Deletion of aged artifacts under an explicitly configured retention period. Applies to successful and failed invocations, while protecting active invocation artifacts and saved reusable controls. |
| Invocation trace | The record connecting an invocation's inputs, stages, external interactions, intermediate artifacts, results, and completion or failure state. |
| RASE | Requirement, Applicability, Selection, and Exception: concepts for decomposing policy meaning, without prescribing a serialization format or universal evaluation algorithm. |

## Decision Records

- [ADR-001: Fail control creation when validation attempts are exhausted](adr-001-creation-validation-failure.md).
- [ADR-002: Allow verification when creation-time trial execution is skipped](adr-002-zero-attempt-verification.md).
- [ADR-003: Report compliance alongside evidence coverage](adr-003-compliance-and-evidence-coverage.md).
- [ADR-004: Report compliance per control without an overall policy percentage](adr-004-per-control-reporting.md).
- [ADR-005: Automatically execute prerequisites of selected controls](adr-005-automatic-prerequisites.md).
- [ADR-006: Abort verification on a technical model failure](adr-006-technical-model-failure.md).
- [ADR-007: Configure artifact retention with no automatic expiry by default](adr-007-configurable-artifact-retention.md).
- [ADR-008: Accept UTF-8 plain text and Markdown policy documents](adr-008-policy-input-formats.md).
- [ADR-009: Require parameter values for enabled creation-time trials](adr-009-required-trial-parameters.md).
- [ADR-010: Version saved control sets and hash policy documents](adr-010-versioned-controls-and-policy-hashes.md).
