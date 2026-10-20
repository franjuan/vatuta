# ADR-002: Allow Verification When Creation-Time Trial Execution Is Skipped

**Date**: 2026-10-19
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

FR-026 allows zero creation-time validation attempts to save tokens, time, and processing. The specification needs to
distinguish optional trial execution from mandatory structural and safety checks, and define whether controls produced
in this mode can be used for verification.

## Decision

Zero attempts skips creation-time trial execution only. Schema, dependency, and permission checks remain mandatory.
Controls passing those checks are eligible for verification without a separate creation-time trial execution.
Both saved controls and the creation trace must record that trial execution was skipped.

Verification continues to enforce its normal checks and error handling. An execution error is an execution failure,
not a compliance outcome. Skipping a trial does not grant permission to bypass source access restrictions or run
malformed controls.

The stakeholder selected this behavior during clarification on 2026-10-19. It is reflected in User Story 1,
Edge Cases, and FR-026 of the specification.

## Alternative Considered

Save controls for inspection only and require successful trial execution before verification: rejected because this
would prevent users from proceeding directly to verification after deliberately skipping creation-time trials.

## Consequences

- Users can avoid creation-time test calls while retaining reusable controls.
- Eligibility for verification does not establish that trial execution succeeded; artifacts explicitly show it was
  skipped, and execution errors can still occur during verification.
- Acceptance coverage must demonstrate successful reuse after skipping trials and rejection of controls that fail
  mandatory checks even when attempts are zero.
- [ADR-001](adr-001-creation-validation-failure.md) still governs exhausted validation attempts when trials are enabled.
- [ADR-009](adr-009-required-trial-parameters.md) requires supplied values for enabled trials. Zero-attempt mode still
  skips trials; unresolved values remain subject to mandatory checks and must be supplied before verification uses them.

## Terminology

See the [policy checker glossary](glossary.md) for trial execution, mandatory checks, and verification eligibility.
