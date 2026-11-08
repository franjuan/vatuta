# ADR-011: Separate Assisted Authoring from Deterministic Execution

**Date**: 2026-10-20
**Status**: Partially superseded by ADR-018 and ADR-019
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Current Applicability (2026-10-25)

Keep creation separate from execution. Creation selects deterministic or exceptional LLM evaluation and declares it
in YAML. Execution follows that mode without runtime repair or switching. The invoking agent collects parameters;
discovery and execution do not infer them through a model. Maintenance and managed approval/publication are
excluded. The absolute execution-time model prohibition below is superseded.

See [ADR-019](adr-019-file-based-policy-operations.md) and the current specification. The sections below
retain earlier decisions and their original requirement references; only compatible behavior remains active.

## Context and Problem

Earlier requirements allowed semantic and hybrid evaluation during verification. The stakeholder now requires reusable
declarative controls whose execution makes no model-dependent decisions. Authoring and execution have different trust
and reproducibility requirements.

## Decision

Policy Authoring and Maintenance may use an LLM to interpret source documents, generate complete control proposals,
and assist with maintenance. Policy Execution Engine never uses an LLM: it obtains declared evidence, transforms it,
evaluates each subject, and reports results deterministically.

Given the same control definition, invocation inputs, and source responses, execution must produce equivalent results.
This does not promise that external data remains unchanged between invocations. An authoring trial uses the same
deterministic execution responsibilities; any model-assisted repair happens in authoring, outside that execution.

## Alternatives Considered

- Semantic or hybrid execution: superseded; runtime judgments would violate the agreed deterministic boundary.
- Hand-authored controls only: rejected as the sole workflow because managers maintain natural-language source
  documents.

## Rationale

Keep interpretation in a reviewable authoring process and make repeated checks reproducible and independently testable.

## Consequences

- Positive: Execution can be tested with fixed source responses without model calls or hidden planning decisions.
- Positive: Model samples and prompt optimization remain authoring concerns.
- Negative: Policies that cannot be expressed using available evidence and deterministic expressions cannot be evaluated
  by adding an implicit semantic fallback. Their authoring treatment still needs clarification.
- No criterion-generated explanations, remediation recommendations, or human compliance-evaluation workflow are added.
  Human review of proposed controls before publication remains required.
- [ADR-006](adr-006-technical-model-failure.md) and the historical fail-safe rule for ambiguous semantic judgments no
  longer apply to verification. Technical errors must not become violations.

## Related Artifacts

See the [design](design.md), [glossary](glossary.md), and [interview record](interview.md).
