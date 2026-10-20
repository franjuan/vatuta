# ADR-010: Version Saved Control Sets and Hash Policy Documents

**Date**: 2026-10-20
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

Creation can be repeated for the same policy. A later extraction may produce different controls because policy content,
instructions, source capabilities, or model behavior differ. Replacing the saved set would prevent earlier controls from
being reused directly and make comparisons dependent on retained invocation traces.

The stakeholder selected separate versions and additionally required a policy document hash to detect changes.

## Decision

Every successful creation saves a separately identifiable control-set version and preserves earlier versions for
verification. Failed creation publishes no new version and does not change existing versions.

Compute a content hash from the exact UTF-8 policy document content supplied to creation. Save the hash and its method
with the control-set version and its originating policy identity. Compare it with the hash recorded for a prior version
of the same policy, identifying the version used for comparison. If no prior version exists, record an initial baseline.

Compare content, not file paths or modification times. A renamed document with identical content is unchanged; edited
content at the same path is changed. Hash comparison uses the same hash method for both inputs. Changes to whitespace
or line endings count as content changes; the hash is not a semantic comparison of policy meaning.

A policy document hash and a control-set version are separate identifiers. Explicitly rerunning creation successfully
can produce a new control-set version with the same document hash. A hash match alone does not justify treating
generated controls as identical or skipping requested creation.

Every verification records the saved control-set version and its associated document hash. It evaluates that saved
version without requiring the original policy document to be reanalyzed or available. Its recorded hash must describe
the source of the evaluated controls, not a later revision of the document.

The decision is reflected in User Story 1, Edge Cases, FR-028, Key Entities, and SC-010 of the specification.

## Alternative Considered

Replace the previous reusable control set while retaining old definitions only in traces: rejected. Earlier saved sets
must remain available for verification independently of invocation-artifact retention.

## Consequences

- Creation accumulates saved versions. The invocation-artifact retention policy in
  [ADR-007](adr-007-configurable-artifact-retention.md) does not delete them.
- Verification remains attributable to the exact saved control set, including when policy content has since changed.
- Acceptance fixtures must cover repeated creation, unchanged content at a different path, changed content at the same
  path, failed creation preserving earlier versions, and verification of an earlier version.
- Hash algorithm choice, version identifier representation, storage layout, and selection syntax remain planning
  details. They must preserve the distinction between policy content identity and generated control-set identity.

## Terminology

See the [policy checker glossary](glossary.md) for control-set version, policy document hash, and initial hash baseline.
