# ADR-015: Maintain Policy Documents and Regenerate Complete Control Proposals

**Date**: 2026-10-20
**Updated**: 2026-10-21
**Status**: Partially superseded — managed maintenance excluded by ADR-019
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Current Applicability (2026-10-25)

The supplied policy document or inline text remains the normative source and creation remains complete-or-error.
Managed regeneration, comparison, identity reconciliation, review, approval, and publication are outside the current
scope. Creating again is an independent request; earlier lifecycle mechanics below are historical.

See [ADR-019](adr-019-file-based-policy-operations.md) and the current specification. The sections below
retain earlier decisions and their original requirement references; only compatible behavior remains active.

## Context and Problem

Managers need to change normative content without maintaining two competing sources of truth. Incremental semantic
impact analysis and automatic identity reconciliation add complexity and can misclassify restructured rules.

## Decision

The manager-edited Policy Document is the sole source of truth for normative content. Controls are derived artifacts.
Normal maintenance changes the document, not control YAML or conversational requests to an LLM to edit policy content.

For an updated document, authoring has the previous document and controls and generates a complete control proposal.
Compare it with the published controls to identify added, modified, removed, and unchanged controls. Review and approve
the proposal before publication. A document diff may be generation context; it does not decide what to regenerate.

Compare normalized definitions so YAML presentation changes alone do not count as functional changes. Do not assume
different JSONata or CEL expressions are semantically equivalent. Preserve identity when a control still represents
the same obligation; unresolved mappings must remain visible rather than being silently decided by a model.

On 2026-10-21 the stakeholder selected option B for ambiguous identity correspondence: reject the complete proposal,
identify the ambiguity, and request a new generation. Do not provide a manual correspondence-resolution workflow or
silently choose uncertain mappings. Published revisions and history remain unchanged; no subset of the rejected
proposal is published. This does not add an automatic regeneration loop or change existing authoring trial limits.

## Alternatives Considered

- Incremental regeneration driven by semantic impact analysis: excluded from the first implementation.
- Semantic search, embeddings, or a vector index for matching controls: excluded without a demonstrated need.
- Maintaining control YAML as an alternative normative source: rejected because it creates competing authorities.
- Letting an LLM always choose identity mappings: rejected because splits, merges, and rewordings can be ambiguous.

- Let the manager repair ambiguous mappings during proposal review: rejected on 2026-10-21 in favor of rejecting the
  proposal and requesting another generation, keeping manual identity reconciliation outside the initial workflow.

## Rationale

Complete proposals and explicit review provide a simpler update process with inspectable differences.

## Consequences

- Positive: Managers maintain one normative source, and reviewers see the whole proposed control set.
- Positive: Evidence, Selection, and Criterion changes can be reviewed separately for their behavioral implications.
- Negative: Regeneration may repeat work and introduce expression changes even for unchanged obligations.
- Positive: Identity ambiguity cannot be accepted through an unreviewed model guess or a manual mapping override.
- Negative: Rejecting the complete proposal can discard useful generated work and require additional generation effort.
  A subsequent generation is not guaranteed to resolve the ambiguity; it must satisfy the same acceptance conditions.
- Pending: The minimal detection and provenance contract for unambiguous correspondence. No complex semantic matching
  algorithm or claim of infallible model correspondence is introduced by this failure behavior.
- Successfully generating or validating a proposal is not publication. Earlier validation ADRs remain applicable to
  authoring checks, subject to this explicit publication gate.

## Related Artifacts

See [ADR-016](adr-016-immutable-policy-revisions.md), the [design](design.md), and the [interview record](interview.md).
