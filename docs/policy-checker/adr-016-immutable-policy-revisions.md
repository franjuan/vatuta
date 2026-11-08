# ADR-016: Version Approved Policy State with Immutable Revisions

**Date**: 2026-10-20
**Updated**: 2026-10-21
**Status**: Superseded by ADR-019 on 2026-10-25
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Current Applicability (2026-10-25)

No immutable policy revisions, control versions, revision numbering, comparison, historical retrieval, restoration,
or approval/publication subsystem is required. Creation returns control files and execution uses selected files.
Diagnostic traces preserve attribution without implementing the historical lifecycle described below.

See [ADR-019](adr-019-file-based-policy-operations.md) and the current specification. The sections below
retain earlier decisions and their original requirement references; only compatible behavior remains active.

## Context and Problem

Users must retrieve earlier documents and controls, compare changes, identify published definitions, and restore prior
configurations. A separate version system per control or a collaborative source-control model is unnecessary initially.

## Decision

Maintain internal immutable revisions with incremental identifiers. Each revision contains the Policy Document and
its complete corresponding approved control set. Do not use Git for policy versioning.

Keep stable control identifiers across revisions. Recover a control's historical definition through the complete
revision; no separate per-control version counter or storage lifecycle is required. Unchanged controls retain their
definition and identity even when the containing policy revision changes.

Record provenance from controls to the source document version and rule or section. Retain exact document-content
hashing from ADR-010 as change evidence, not as a substitute for revision or control identity. Generation, validation,
and approval precede publication. Failed authoring must not alter the published revision or previous history.

On 2026-10-21 the stakeholder selected option A for restoration: publish a new incremental revision copying the selected
historical document and complete control set, without LLM regeneration. Keep control IDs and definitions unchanged and
identify the source revision in the restoration record. Review and approval still precede publication. Neither the
source revision nor the previously current revision is modified or deleted; restoration does not merely reactivate
an old revision.

## Alternatives Considered

- Git branches, merges, and tags for policy storage: explicitly excluded.
- Independent control-version management: unnecessary for the first implementation; full revisions retain history.
- Overwrite the current set and rely on logs: rejected because retained immutable revisions are required independently
  of diagnostic-artifact expiry.
- Publish every successful generation automatically: superseded by mandatory proposal review and approval.

- Reactivate the existing historical revision without creating another revision: rejected on 2026-10-21 in favor of
  recording restoration as a new approved revision in the same immutable history.

## Rationale

A document and its approved controls form one reviewable policy configuration and one recoverable historical unit.

## Consequences

- Positive: Previous definitions remain available for verification, comparison, and audit.
- Positive: A control can be identified by its stable ID within a particular policy revision without another versioning
  subsystem.
- Negative: Full revisions can repeat unchanged control definitions; storage optimization is not an initial requirement.
- Positive: Restoration is represented in revision history without rerunning interpretation or changing historical
  control definitions and identities.
- Negative: Restoring older content creates another full snapshot, potentially repeating already stored content.
- Pending design work: Revision-number scope, exact provenance metadata including the restored-from reference, approval
  recording, and the concrete storage/interface representation. The restoration behavior itself is settled.
- Storage format and directory layout remain technical design decisions. No database, concurrency reconciliation, or
  distributed publication mechanism is mandated.
- [ADR-010](adr-010-versioned-controls-and-policy-hashes.md) is superseded as a version/publication strategy; compatible
  document hashing, retained history, and execution attribution are carried forward here.

## Related Artifacts

See the [design](design.md), [glossary](glossary.md), and [interview record](interview.md).
