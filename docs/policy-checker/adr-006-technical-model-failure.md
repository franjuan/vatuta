# ADR-006: Abort Verification on a Technical Model Failure

**Date**: 2026-10-19
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

The specification distinguishes execution failures from compliance outcomes and stops verification when a required
evidence source is unavailable. It also classifies ambiguous semantic judgments as Non compliant under an accepted
fail-safe rule. It previously left continuation behavior after a technical model failure unspecified.

A model timeout, service error, or unusable response provides no judgment that can be classified as a compliance
outcome. Continuing independent controls would require a separate policy for incomplete assessment delivery.

## Decision

If a technical model failure occurs during verification, stop the entire invocation once any permitted retries are
exhausted. Preserve available diagnostics and completed results as incomplete work, and do not publish a completed
assessment. Assign no compliance outcome to the failed call, including Non compliant or Insufficient evidence.

The existing fail-safe rule remains in force for an ambiguous semantic judgment. A technically unusable response is an
execution failure, whereas a usable judgment expressing semantic uncertainty remains subject to FR-019.

The stakeholder selected option A in the second clarification round on 2026-10-19. This decision is reflected in
User Story 2, Edge Cases, FR-018, and SC-007.

## Alternative Considered

Continue independent controls, block affected controls and their dependents, and return an incomplete assessment:
rejected in favor of stopping the whole invocation, consistent with the existing fail-fast evidence-source behavior.

## Consequences

- Completed control results remain attributable and available as incomplete work; their presence does not make the
  failed invocation a completed assessment.
- No fabricated compliance outcome is inserted for a failed model call or used to compute compliance metrics.
- Acceptance fixtures must cover model timeouts, service errors, and unusable responses, including failures after some
  controls have already completed.
- The decision applies to verification. It does not alter creation-time validation retry semantics or the existing
  immediate abort requirement for an unavailable evidence source.
- Retry counts, retryable error classification, and timeout values remain planning decisions; this decision does not
  introduce a retry entitlement or an arbitrary end-to-end latency target.

## Terminology

See the [policy checker glossary](glossary.md) for technical model failure and ambiguous semantic judgment.
