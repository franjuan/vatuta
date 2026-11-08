# ADR-012: Use YAML, MCP, JSONata, and CEL

**Date**: 2026-10-20
**Status**: Accepted with ADR-018 semantic exception and ADR-019 parameter contract
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Current Applicability (2026-10-25)

Keep individual YAML controls, MCP evidence, JSONata selection, and deterministic CEL. Controls also declare
parameters and their semantics/defaults. ADR-018 permits an explicitly declared LLM criterion when deterministic
evaluation is insufficient. The CEL-only statements below no longer prohibit that exception.

See [ADR-019](adr-019-file-based-policy-operations.md) and the current specification. The sections below
retain earlier decisions and their original requirement references; only compatible behavior remains active.

## Context and Problem

Controls need an inspectable representation, provider-independent evidence access, JSON transformations, and boolean
evaluation. Earlier design inputs listed several technologies without a decision.

## Decision

Store each control in a separate YAML file with identifying and provenance metadata and three principal sections:
`evidence`, `selection`, and `criterion`.

- Evidence declares configured MCP server references, tool names, evidence identifiers, and invocation arguments.
- Selection is one JSONata expression that transforms and relates evidence into subjects.
- Criterion is one CEL expression reused for every subject and required to return a boolean.

MCP connection settings and credentials remain external to controls. The engine handles provider-neutral JSON data;
provider-specific field access and relationships belong in control expressions, not executor adapters.

## Alternatives Considered

- Substrait: rejected for this initial scope; it would introduce a relational representation and execution machinery
  when the inputs and transformations are naturally JSON-shaped.
- JSON Schema as the compliance evaluator or additional validation languages: not selected. Structural validation of
  controls remains necessary, but compliance evaluation uses CEL alone.
- A custom transformation or policy expression language: rejected because the selected standards cover those duties.

## Rationale

Reuse established languages and keep acquisition, transformation, and evaluation responsibilities distinct.

## Consequences

- Positive: Controls expose their source calls, transformation, and condition for review.
- Positive: Provider independence does not require a complete declared schema for every source response.
- Negative: Expressions must still know the actual field paths they use; provider independence does not make expressions
  automatically portable between differing source schemas.
- Python bindings and their language compatibility require investigation. No library has been selected.
- Response decoding and expression environments remain design questions. No implicit coercions or runtime extensions
  have been approved.

## Related Artifacts

See the [design](design.md) and [technical investigation register](../../specs/002-policy-checker/design-inputs.md).
