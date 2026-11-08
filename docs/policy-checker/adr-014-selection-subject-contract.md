# ADR-014: Selection Produces Identifiable JSON Subjects

**Date**: 2026-10-20
**Updated**: 2026-10-21
**Status**: Accepted for the stated contract; unresolved boundary cases remain pending
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Current Applicability (2026-10-25)

The subject identifiers, selection boundaries, boolean states, and invalid-object behavior remain applicable.
References below to CEL evaluation also apply to declared semantic criteria under
[ADR-018](adr-018-semantic-criterion-exception.md). Evaluation mode is fixed in YAML during creation, and errors remain
visible with independent controls continuing under [ADR-019](adr-019-file-based-policy-operations.md).

## Context and Problem

CEL needs identifiable subjects, and consumers need to distinguish a missing subject from a subject with missing data.
Implicitly wrapping scalar results or generating identifiers would hide mistakes in Selection.

## Decision

Selection produces a JSON object or a collection of JSON objects. Every subject includes `id` and `descriptive_id`.
Additional fields may contain any JSON value, including null. Selection is responsible for producing valid subjects;
the engine does not invent identifiers or wrap scalar values.

For each valid subject, a single CEL expression returns true or false, mapped to COMPLIANT or NON_COMPLIANT. Technical
evaluation errors are ERROR, not violations. A successful selection with no evaluable subjects produces global
NOT_APPLICABLE and does not execute CEL. NOT_APPLICABLE is never a subject outcome.

A pull request without a linked issue remains a subject; the absent link must not make that pull request disappear.

On 2026-10-21 the stakeholder selected option A for the root-result boundary: successful no-result output and an empty
collection yield NOT_APPLICABLE; root null is a Selection contract ERROR. CEL is not executed in any of those cases.
A valid subject with null-valued fields remains evaluable. This is an explicit contract rule, not implicit
normalization.

On 2026-10-21 the stakeholder selected option B for mixed valid/invalid selected objects: evaluate valid subjects and
report objects missing mandatory identifiers separately, without inventing IDs. Their presence makes the control's
global result ERROR regardless of valid subjects' outcomes. Errors remain visible with either result filter. CEL runs
only for valid subjects. A nonempty collection containing only invalid objects is ERROR, not NOT_APPLICABLE.

## Alternatives Considered

- Automatically wrap scalars and assign identifiers: rejected; it conceals transformation defects.
- Drop subjects whose related data is missing: rejected; it can suppress precisely the violations being checked.
- Separate expressions for compliance and violations: rejected; false from the single criterion already identifies
  non-compliance.

- Treat root null like an empty selection: rejected on 2026-10-21. It would conceal a value that does not satisfy the
  object/collection contract instead of distinguishing it from successful absence of subjects.

- Reject the whole selected collection before CEL when one object lacks an identifier: rejected on 2026-10-21 in favor
  of retaining useful evaluations of valid subjects and exposing malformed objects separately.

## Rationale

Keep the transformation output explicit and make failures distinguishable from both non-compliance and inapplicability.

## Consequences

- Positive: Every result can be traced to the intended subject through technical and human-readable identifiers.
- Positive: A nullable related field does not by itself change applicability.
- Negative: Authors must construct the subject shape explicitly and handle required domain nulls in their criterion.
- Positive: Malformed selected objects do not suppress valid subjects' evaluations; errors cannot be mistaken for an
  empty successful selection.
- Negative: Consumers must handle malformed-object diagnostics separately from identified subject results.
- Pending: Identifier types, duplicate IDs, and the exact malformed-object diagnostic representation.
  This ADR does not silently decide those cases or prescribe coercions.
- [ADR-017](adr-017-partial-subject-errors.md) resolves partial subject errors: continue all subjects, retain their
  results,
  and return global ERROR if any subject fails. The 2026-10-21 update to ADR-017 resolves error-free nonempty execution:
  return individual results without an
  aggregate global state. Representation details remain pending.
- [ADR-003](adr-003-compliance-and-evidence-coverage.md) is superseded because its subject states no longer exist.

## Related Artifacts

See the [design](design.md), [glossary](glossary.md), and [interview record](interview.md).
