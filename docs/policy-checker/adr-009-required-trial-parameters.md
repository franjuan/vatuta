# ADR-009: Require Parameter Values for Enabled Creation-Time Trials

**Date**: 2026-10-20
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

Controls declare parameters and can be reused in independent verification invocations. Creation also tests generated
controls when trial execution is enabled. A missing parameter value can prevent a required source query, transformation,
or evaluation mechanism from being tested, even if that value is intended to be supplied at verification time.

The earlier wording allowed unresolved parameters without explaining their effect on creation-time trials.

## Decision

When creation-time trials are enabled, require values for every parameter needed by those trials. If a needed value is
missing, fail the entire creation invocation, identify the missing parameter and affected trial, retain available
diagnostics, and publish no reusable control set or subset. Do not defer the trial or invent a value to make creation
succeed. Creation can be attempted again when the required values are supplied.

Verification may later use different valid values within its configured scope and permissions, without repeating
policy extraction. Record the values actually resolved for each invocation; successful trials with one set of values
do not guarantee successful verification with another.

The stakeholder selected option B in the second clarification round on 2026-10-20. The decision is reflected in
User Story 1, Edge Cases, and FR-014.

## Relationship to Earlier Decisions

- [ADR-001](adr-001-creation-validation-failure.md): creation still publishes no reusable subset when it fails.
  Missing trial inputs are an input failure; this decision does not require spending validation retries on missing data.
- [ADR-002](adr-002-zero-attempt-verification.md): zero attempts still skips trial execution. Unresolved values do not
  trigger a trial requirement in this mode, but mandatory checks remain enforced and verification requires its inputs
  before use. This decision does not permit missing values needed by mandatory checks.

## Alternative Considered

Run possible checks, defer parameter-dependent trials, and save the controls: rejected. With trials enabled, missing
inputs prevent successful creation even when those inputs could be provided during later verification.

## Consequences

- Creation-time inputs must support all enabled required trials; callers cannot obtain a reusable partial set by
  withholding values needed by some trials.
- Saved controls remain parameterized and reusable, rather than being restricted to the values used during trials.
- Acceptance coverage must include missing trial inputs, successful creation after supplying them, zero-attempt mode,
  and subsequent verification with different valid values.
- Trial parameter values must satisfy the existing configured scope and permission boundaries.

## Terminology

See the [policy checker glossary](glossary.md) for trial parameter value and resolved parameter value.
