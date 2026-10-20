# ADR-008: Accept UTF-8 Plain Text and Markdown Policy Documents

**Date**: 2026-10-20
**Status**: Accepted
**Feature**: [Policy Checker Mechanism](../../specs/002-policy-checker/spec.md)

## Context

Control creation requires a formal policy document, but the initial specification did not define supported formats.
Different formats require different extraction behavior and affect how policy references can be preserved.

## Decision

The initial release accepts UTF-8 plain text and Markdown policy documents only. The existing configuration remains
the entry point for the policy document and execution scope.

Reject unsupported formats and invalid UTF-8 before policy extraction. Report an explicit input error identifying the
supported formats, and publish no reusable control set. Do not silently interpret binary content as policy text.

PDF, Word documents, scanned documents, and images are outside the initial input contract. Automatic conversion and
optical character recognition (OCR) are not required. Users can supply policy content as a supported document instead.

The stakeholder selected option A in the second clarification round on 2026-10-20. This decision is reflected in
User Story 1, Edge Cases, FR-001, and the Policy document entity.

## Alternatives Considered

- Add PDFs containing selectable text: deferred to keep the initial input contract focused on text formats.
- Also accept Word documents: deferred with other document extraction capabilities.
- Also recognize text in scanned documents and images: deferred; OCR is outside the initial scope.

## Consequences

- Acceptance coverage must include successful creation from both supported formats and explicit rejection of
  unsupported formats and invalid UTF-8.
- The existing policy-reference requirements still apply, including document identity, section or location, and version
  where available. Supporting plain text does not require inventing section names or document versions.
- This decision concerns policy input documents, not the format of external evidence, saved controls, or reports.
- File discovery and format detection details remain planning decisions; renaming an unsupported binary file does not
  make its contents a valid text policy.

## Terminology

See the [policy checker glossary](glossary.md) for policy document and supported policy formats.
