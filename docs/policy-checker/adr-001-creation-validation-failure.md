# ADR-001: Fail Control Creation When Validation Attempts Are Exhausted

**Date**: 2026-10-19
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

Creation synthesizes reusable controls and can validate them through test invocations against evidence sources,
transformations, and evaluation mechanisms. FR-026 permits a configurable number of validation attempts, but did not
define whether a failed control prevents other controls from being published.

Publishing only successful controls could leave a policy requirement uncovered while allowing later verification to
proceed. Creation failures also need diagnostic artifacts so their cause can be investigated.

## Decision

When creation-time validation is enabled and any generated control still fails after all allowed attempts, the entire
creation invocation fails. Retain diagnostic artifacts identifying the failed control and attempts, but publish no
reusable control set or reusable subset from that invocation.

The stakeholder selected this behavior during clarification on 2026-10-19. It is reflected in User Story 1,
Edge Cases, and FR-026 of the specification.

## Alternatives Considered

- Publish successful controls as an explicitly incomplete, executable set: rejected because it permits verification
  against only part of the intended control set after creation fails.
- Save all controls with failed controls blocked, permitting selections that exclude them: rejected because it also
  permits a reusable subset from a failed creation invocation.

## Consequences

- Successful validation of some controls cannot turn a failed creation invocation into a reusable deliverable.
- Diagnostic artifacts remain available, subject to the existing credential and personal-data redaction requirements.
- Acceptance coverage must include a mixed set where some controls pass and one exhausts its attempts, confirming that
  creation fails and no reusable set or subset is published.
- This decision concerns creation-time validation failures, not compliance outcomes from verification.
- Configuring zero validation attempts remains supported by FR-026. The mandatory checks and verification eligibility
  for that mode are defined in [ADR-002](adr-002-zero-attempt-verification.md).

## Terminology

See the [policy checker glossary](glossary.md) for creation, verification, control set, and diagnostic artifact.
