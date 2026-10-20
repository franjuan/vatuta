# ADR-007: Configure Artifact Retention With No Automatic Expiry by Default

**Date**: 2026-10-20
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

Invocation traces and model interaction samples support audits, debugging, and future prompt evaluation. These
artifacts accumulate across successful and failed creation and verification invocations. The specification required
persistence but did not define when artifacts could expire.

The stakeholder requested configurable retention and then selected no automatic expiry as the default.

## Decision

Retention is configurable. Without an explicitly configured retention period, invocation traces and model interaction
samples remain until explicitly deleted. There is no automatic deletion based on age by default.

An explicitly configured retention period enables automatic expiry for artifacts from both successful and failed
invocations. Active invocation artifacts remain protected from retention cleanup. Saved reusable control sets are
outside this policy and must not be deleted when their originating invocation's artifacts expire.

This decision is reflected in User Story 4, Edge Cases, and FR-027 of the specification. All retained artifacts remain
subject to the existing credential and personal-data redaction requirements.

## Alternatives Considered

- Fixed indefinite retention without configuration: insufficient because the stakeholder explicitly requires a
  configurable retention policy.
- Default automatic expiry after 30 or 90 days: rejected because expiry must follow an explicitly configured period.

## Consequences

- Audit records and model samples remain available by default; storage consumption grows until artifacts are explicitly
  deleted or a retention period is configured.
- Configuring expiry deliberately bounds the history available for audit reconstruction and prompt evaluation.
- Retention cleanup does not remove reusable controls or interrupt active invocations.
- Acceptance coverage must verify indefinite retention when unconfigured, configured expiry for successful and failed
  invocations, and preservation of active artifacts and saved controls.
- Configuration naming, duration units, expiry timestamps, and cleanup scheduling remain planning details. This decision
  does not prescribe a new deletion command or a particular background service.

## Terminology

See the [policy checker glossary](glossary.md) for artifact retention and automatic expiry.
