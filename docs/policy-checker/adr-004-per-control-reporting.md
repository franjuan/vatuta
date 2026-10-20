# ADR-004: Report Compliance Per Control Without an Overall Policy Percentage

**Date**: 2026-10-19
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

FR-025 originally required compliance percentages per control and across the overall policy scope. Controls can assess
different populations, including individual pull requests and whole projects. An overall percentage would require an
unstated choice about how to weight these assessments.

For example, 99 compliant pull requests out of 100 and one non-compliant project produce 49.5% when averaging control
percentages, but approximately 98% when pooling subject-control assessments. These percentages communicate different
conclusions about the same findings.

## Decision

Publish no overall policy compliance percentage. Show each control's compliance percentage, evidence coverage, and
outcome counts separately, using the definitions in [ADR-003](adr-003-compliance-and-evidence-coverage.md).

This replaces the earlier overall-percentage requirement in FR-025. A control that evaluates an aggregate metric for
a project, population, or reporting period still reports its own assessment and supporting measurements.

The stakeholder selected option C during clarification on 2026-10-19. The decision is reflected in User Stories 2 and 4,
FR-007, FR-025, Key Entities, and SC-009 of the specification.

## Alternatives Considered

- Average control percentages, assigning each control equal weight: rejected because control importance is not
  established by counting controls.
- Pool subject-control outcomes, assigning each assessment equal weight: rejected because a control with a large
  population can dominate the headline and obscure a failed project-level requirement.

## Consequences

- Policy reports preserve the findings for every evaluated control without introducing a weighting rule.
- Per-control coverage remains visible alongside compliance and outcome counts.
- Total violation counts and operational metrics remain required; this decision removes only the overall compliance
  percentage derived by combining controls.
- Acceptance coverage must include controls with different populations and confirm that no overall policy compliance
  percentage is produced in either interactive or programmatic reports.

## Terminology

See the [policy checker glossary](glossary.md) for aggregate control assessment and policy report.
