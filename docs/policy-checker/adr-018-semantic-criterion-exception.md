# ADR-018: Permit Semantic Criteria Only as a Last Resort

**Date**: 2026-10-22
**Updated**: 2026-11-11
**Status**: Accepted — evaluation mode confirmed; lifecycle assumptions removed by ADR-019
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Current Applicability (2026-11-11)

Choose deterministic or LLM mode during creation and declare it in YAML. Execution and creation trials follow that
mode without runtime fallback. Preserve independent judgments, attribution, and error handling. FR-035 now requires
one subject per model call; the batching decisions and rejection of singleton calls below are historical. Approval,
publication, revisions, configurable retention, and the source-unavailability abort exception are also superseded.

See [ADR-019](adr-019-file-based-policy-operations.md) and the current specification. The sections below
retain earlier decisions and their original requirement references; only compatible behavior remains active.

## Context and Problem

The stakeholder amended the absolute execution-time model prohibition. Some criteria concern meaning, such as whether
an item's description explains particular semantic aspects, and cannot be resolved using a deterministic expression.
Rejecting every such obligation would prevent otherwise useful complete control proposals.

## Decision

Prefer deterministic evaluation, using CEL for the criterion, for cost, efficiency, determinism, repeatability, and
interpretability. Only when deterministic evaluation cannot resolve the criterion may it use an LLM or JEV as a last
resort. The stakeholder named JEV as an evaluator option; its concrete integration is not selected by this decision.

Preserve one criterion evaluated independently for each subject. Semantic evaluations of the same criterion may be
grouped into batches for efficiency. Each judgment uses the criterion and that subject's selected inputs; another
subject's content or verdict is not evidence for it. Each submitted subject must have an unambiguous response mapping.
Batching does not create a collective verdict, evidence dependency, or new policy aggregation rule.

The deterministic path retains reproducibility for fixed definitions, inputs, and source responses. Live semantic
judgments are not guaranteed to repeat across invocations or batch layouts. Fixed evaluator responses must still
produce consistent subject attribution, state mapping, and filtering.

## Integration Assumptions

These defaults carry existing publication and error contracts into the authorized exception; they are not additional
stakeholder interview answers.

- Declare the evaluation mode, semantic condition, and reason deterministic evaluation is insufficient in the control
  proposal. Review and approval precede publication. Execution uses that definition rather than inventing a strategy.
- A deterministic runtime error remains ERROR; it does not authorize an automatic semantic fallback or expression
  repair.
- Acquisition and Selection remain deterministic. Empty/invalid-only selections or evidence failures do not call a
  model.
- Boolean true/false maps to COMPLIANT/NON_COMPLIANT. Missing, ambiguous, non-boolean, conflicting, or unusable
  judgments
  are ERROR. The historical fail-safe rule that classified semantic uncertainty as non-compliance is not restored.
- Retain unambiguous subject results from partial batch responses. Missing or conflicting mappings error their affected
  subjects; unexpected or unassignable responses remain visible as errors. A failed batch errors all affected subjects.
- Apply ADR-017: global control ERROR when evaluation has errors, preserve subject results, continue other subjects and
  independent controls, and keep errors visible under both filters. An evaluator failure does not revive ADR-006's
  historical invocation-wide abort. Required evidence-source unavailability retains its separate abort rule.
- Record evaluation mode, criterion, evaluator identity and available model/version/settings, inputs, outputs, batch
  membership, and response mapping under existing privacy and retention rules. Internal model reasoning is not required.
- Authoring trials use the declared evaluation mode. The same review, complete-proposal, and revision rules apply.

## Alternatives Considered

- Prohibit all semantic execution: superseded by the stakeholder's explicit exception.
- Use AI whenever convenient or after any deterministic failure: rejected by the last-resort constraint and approved
  control boundary.
- Require one model call per subject: unnecessary; independent judgments may share an efficient batch request.
- Combine subjects into one compliance judgment: incompatible with independent per-subject assessment.

## Consequences

- Positive: Semantic obligations can be represented without weakening the preference for deterministic checks.
- Positive: Optional batching can reduce call overhead while preserving attributable individual results.
- Tradeoff: Semantic criteria introduce model cost, variability, and model-specific validation work.
- Planning must select the minimal evaluator integration, declaration format, prompt contract, correlation mechanism,
  and batch limits. No fixed batch size, concurrency mechanism, retry policy, or new service is mandated.
- Semantic validation must include representative positive/negative descriptions and error/correlation fixtures.
  Live model quality and sensitivity to batch context need evaluation; no arbitrary accuracy target is introduced.
- Generated violation explanations, remediation, model-directed evidence acquisition, and semantic identity matching
  remain outside scope.

## Supersession

Partially supersedes [ADR-011](adr-011-authoring-execution-separation.md) and
[ADR-012](adr-012-declarative-control-standards.md): authoring/execution separation, YAML controls, MCP evidence,
JSONata Selection, and deterministic CEL remain; the absolute model prohibition and CEL-only restriction do not.
[ADR-006](adr-006-technical-model-failure.md) remains historical. Subject and batch failures follow
[ADR-017](adr-017-partial-subject-errors.md) and the current specification.

## Related Artifacts

See [design](design.md), [glossary](glossary.md), [interview](interview.md), and
[design inputs](../../specs/002-policy-checker/design-inputs.md).
