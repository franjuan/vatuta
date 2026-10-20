# ADR-005: Automatically Execute Prerequisites of Selected Controls

**Date**: 2026-10-19
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

Verification accepts a full control set, a single control, or a selected subset. Controls can depend on other controls'
results. Requiring dependency ordering alone does not specify what happens when a prerequisite is outside the user's
selection.

## Decision

Automatically include all direct and indirect prerequisites of selected controls. Evaluate each prerequisite before
its dependents within the same invocation, preserving result provenance and dependency relationships.

The report distinguishes explicitly selected controls from automatically included prerequisites. All execution remains
within the configured scope and permissions; reject selections requiring prerequisites outside those boundaries and
identify the offending prerequisites. Existing rejection of unknown references and cycles still applies.

The stakeholder selected option A during clarification on 2026-10-19. The decision is reflected in User Story 2,
Edge Cases, FR-005, FR-012, FR-021, and SC-006 of the specification.

## Alternative Considered

Reject a selection unless the user explicitly includes every prerequisite: rejected because selecting a control should
be sufficient to evaluate it when its dependencies are available within the configured scope and permissions.

## Consequences

- A single selected control can cause several controls to execute; reports make that additional work visible.
- A chain where A requires B and B requires C executes C before B before A, even when the user selects only A.
- Automatically including a prerequisite grants no additional source access and does not widen the configured scope.
- Prerequisite outcomes retain their original meaning, including Insufficient evidence and Not applicable, when
  consumed as downstream evidence.
- Acceptance coverage must include direct dependencies, indirect dependencies, report attribution, and rejection when
  prerequisites cannot be evaluated within the configured scope or permissions.

## Terminology

See the [policy checker glossary](glossary.md) for selected control and prerequisite control.
