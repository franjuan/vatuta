# ADR-017: Continue Subject Evaluation and Surface Any Error Globally

**Date**: 2026-10-20
**Updated**: 2026-10-21
**Status**: Accepted — uniform control-error continuation under ADR-019
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Current Applicability (2026-10-25)

Retain subject-error visibility, valid subject results, filtering, and existing control outcomes. After request
validation, source outages follow the same per-control continuation rule as other failures. The earlier unavailable-
source invocation-abort exception below is superseded. Missing required invocation parameters fail the whole request
before any control executes.

See [ADR-019](adr-019-file-based-policy-operations.md) and the current specification. The sections below
retain earlier decisions and their original requirement references; only compatible behavior remains active.

## Context and Problem

Valid selected subjects can yield mixed boolean results and criterion evaluation errors. A violations-only response
must not disguise a partially unsuccessful evaluation as a complete assessment.

## Decision

Evaluate all subjects even when individual criterion evaluations fail. Preserve every subject result. If any subject
evaluation fails, the control execution's global result is ERROR. Subject results remain COMPLIANT, NON_COMPLIANT, or
ERROR according to their individual evaluations.

With `result_filter=all`, return all subject results. With `result_filter=violations`, return NON_COMPLIANT findings and
still communicate errors, distinguished from violations. Filtering does not change execution or its global result.

On 2026-10-21 the stakeholder selected option B for error-free nonempty executions: return individual subject results
without an aggregate global state. Do not emit global COMPLIANT or NON_COMPLIANT. This does not remove global ERROR
when errors occur or NOT_APPLICABLE for a successful empty selection. Filtering compliant subjects out of a nonempty
evaluation does not make that execution NOT_APPLICABLE.

On 2026-10-21 the stakeholder selected option A for Selection/CEL failures across independent controls: execute the
remaining requested controls and retain each control's results and errors. A failed Selection does not execute its
Criterion, but does not prevent other controls from running. Keep the earlier fail-fast rule for an unavailable required
evidence source: abort the invocation, preserve diagnostics and completed results as incomplete work, and identify
controls not evaluated. No cross-control compliance aggregate or automatic repair/retry is introduced.

On 2026-10-21 the stakeholder also selected option A for unsupported received evidence payloads and explicitly
incomplete required responses. The affected control returns ERROR without Selection, Criterion, or subject verdicts;
remaining independent requested controls execute. Keep all errors visible under either filter. This extends the
continuation rule to those received-response cases and leaves the unavailable-source abort rule unchanged.

On 2026-10-21 the stakeholder selected the same continuation behavior for unknown evidence completeness. If neither
the tool contract nor response establishes completeness for the control's scope, that control returns ERROR without
Selection, Criterion, or subject verdicts; other independent controls continue and both filters retain errors.
See [ADR-013](adr-013-independent-evidence-calls.md). Source unavailability remains the invocation-abort exception.

## Alternatives Considered

- Stop on the first subject error and retain only results obtained so far: rejected in favor of retaining all useful
  subject evaluations.
- Omit the global state even when subject evaluations fail: rejected for erroneous executions because global ERROR
  makes failure explicit. This alternative does not apply to the subsequent decision for error-free executions.
- Return global NON_COMPLIANT for any violation and otherwise global COMPLIANT: rejected on 2026-10-21 in favor of
  individual subject results only for error-free nonempty executions.

- Stop the whole invocation when an independent control has a Selection or CEL error: rejected on 2026-10-21 in favor
  of preserving evaluations of the other requested controls. Required-source unavailability remains a separate abort
  case.

- Abort after an unsupported or explicitly incomplete received response: rejected on 2026-10-21; the affected
  control fails explicitly while other independent controls continue.

## Rationale

Keep useful results without allowing consumers to confuse partial evaluation failure with complete compliance checking.

## Consequences

- Positive: One subject's malformed criterion inputs do not suppress evaluation of other valid subjects.
- Positive: Errors remain visible even when the caller requests only violations.
- Negative: Global ERROR alone is not the whole report; consumers may need to inspect retained subject findings.
- Scope: This decision covers criterion evaluation of valid selected subjects within one control. The subsequent
  [ADR-014 update](adr-014-selection-subject-contract.md) covers malformed selected objects: evaluate valid subjects,
  report invalid objects separately without generated IDs, and return global ERROR. A failed JSONata expression is not
  repaired or resumed. The later multi-control decisions continue other requested controls after Selection/CEL errors,
  unsupported received payloads, or explicitly incomplete required responses, while preserving the source-unavailability
  abort rule.
- Positive: Error-free subject results are not reduced to an extra aggregate compliance judgment.
- Negative: Consumers needing an aggregate must derive it from sufficient individual results; the engine does not
  provide it. An empty violations list must still be distinguished from an empty selected population.
- Pending: Exact result/error serialization, including representation of the absence of an aggregate state.
- No retries, extra states, or generated explanations are introduced.

## Related Artifacts

See the [design](design.md), [glossary](glossary.md), and [interview record](interview.md).
