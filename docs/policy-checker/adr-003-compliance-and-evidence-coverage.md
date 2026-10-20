# ADR-003: Report Compliance Alongside Evidence Coverage

**Date**: 2026-10-19
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

Verification distinguishes four outcomes, but FR-025 previously required compliance percentages without defining how
Insufficient evidence affects the denominator. Counting unknown outcomes as failures conflates missing evidence with
non-compliance; excluding them without reporting coverage can conceal incomplete evidence.

## Decision

For a control, let C, N, and I denote subject counts for Compliant, Non compliant, and Insufficient evidence:

- Compliance percentage = 100 × C / (C + N).
- Evidence coverage percentage = 100 × (C + N) / (C + N + I).
- Exclude Not applicable subjects from both denominators.
- Always show evidence coverage alongside compliance, together with all four outcome counts.
- A zero denominator produces an undefined percentage, never a substituted 0% or 100%.

The stakeholder selected option B during clarification on 2026-10-19. The zero-denominator behavior follows from these
ratios and prevents a missing population from being presented as a compliance result.

## Examples

| Compliant | Non compliant | Insufficient evidence | Not applicable | Compliance | Evidence coverage |
| ----------- | --------------- | ----------------------- | ---------------- | ------------ | ------------------- |
| 80 | 10 | 10 | 20 | 80/90, approximately 88.9% | 90% |
| 0 | 0 | 10 | 0 | Undefined | 0% |
| 0 | 0 | 0 | 20 | Undefined | Undefined |
| 0 | 0 | 0 | 0 | Undefined | Undefined |

## Alternatives Considered

- Include unknown outcomes in the compliance denominator: rejected because the result measures confirmed compliance
  across the population rather than compliance among established outcomes.
- Suppress compliance whenever evidence is insufficient for any subject: rejected because paired coverage and outcome
  counts can communicate the limitation while preserving the information available from established outcomes.

## Consequences

- A high compliance percentage with low evidence coverage cannot be presented as an assessment of the full population.
- Percentages count subjects by their recorded outcomes, not individual violation records.
- The existing fail-safe rule still classifies ambiguous semantic judgments as Non compliant. Those outcomes count
  toward N, with uncertainty disclosed as already required; evidence coverage is not a measure of model confidence.
- Execution failures remain distinct from compliance outcomes and do not become Insufficient evidence for these ratios.
- This decision defines per-control ratios. [ADR-004](adr-004-per-control-reporting.md) resolves reporting across
  controls: retain their separate metrics without an overall policy compliance percentage.

## Terminology

See the [policy checker glossary](glossary.md) for compliance percentage, evidence coverage, and undefined percentage.
