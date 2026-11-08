# ADR-013: Declare Independent Evidence Calls

**Date**: 2026-10-20
**Updated**: 2026-10-21
**Status**: Accepted — source-error continuation updated by ADR-019
**Feature**: [Policy Validation Engine specification](../../specs/002-policy-checker/spec.md)

## Current Applicability (2026-10-25)

Independent calls, response conversion, and evidence-completeness requirements remain. After invocation validation,
every control failure, including an unavailable required source, is reported for that control while independent
controls continue. The earlier invocation-abort exception below is superseded. Missing mandatory invocation
parameters instead reject the complete request before evidence acquisition.

See [ADR-019](adr-019-file-based-policy-operations.md) and the current specification. The sections below
retain earlier decisions and their original requirement references; only compatible behavior remains active.

## Context and Problem

Fetching records from one source and then using their identifiers in subsequent source calls would require dependency
resolution and orchestration. The stakeholder explicitly chooses broader independent retrieval to simplify execution.

## Decision

Every declared evidence call is independent. Its arguments cannot depend on another evidence result or a control result.
All relationships between returned collections are computed in Selection. Initial execution contains no evidence or
control dependency graph, automatic prerequisite evaluation, dynamic planner, or LLM-directed tool invocation.

For the reference case, retrieve open pull requests and relevant Jira issues independently, then relate them in JSONata.
Independent calls need not run concurrently; scheduling is not an added performance requirement.

On 2026-10-21 the stakeholder selected option A for explicitly incomplete required evidence: return ERROR for the
affected control before executing Selection or Criterion. Do not produce compliance results from the returned subset
or add automatic page retrieval. The error remains visible under either result filter. Continuation across controls is
now accepted below; an unmarked response is not thereby established as complete.

### Accepted MCP Response Boundary (2026-10-21)

The stakeholder selected option A: prefer `structuredContent` when present; otherwise accept exactly one text block
whose entire text is valid JSON. Reject other forms explicitly, without extracting JSON fragments from prose or
Markdown. An explicit tool error remains an acquisition error even if the response contains parseable JSON.
Structured content is authoritative when accompanying text differs. No provider-specific adapter is introduced.
Conversion errors stop Selection and Criterion for the affected control and remain visible under either filter.
On 2026-10-21 the stakeholder selected option A for received responses that violate the payload contract or explicitly
report incomplete data: stop only the affected control with ERROR before Selection and Criterion, without subject
verdicts, and continue other independent requested controls. Preserve results and visible errors under both filters.
The previously accepted invocation-abort rule for an unavailable required source remains unchanged. See
[ADR-017](adr-017-partial-subject-errors.md).

### Unknown Completeness (Accepted 2026-10-21)

The stakeholder selected option A: if neither the tool contract nor the response establishes that required evidence
is complete for the control's declared scope, return ERROR before Selection and Criterion, without subject verdicts.
Continue other independent requested controls and retain errors under either filter. Empty data with unknown
completeness is not NOT_APPLICABLE. No universal completeness flag, automatic pagination, or new state is required.

This prevents a retrieval limit from turning an omitted linked record into a false violation or an empty partial
population into false inapplicability. A tool contract can establish completeness without a response flag; absence
of an incompleteness warning alone cannot. Actual contract interpretation and detection remain technical research.

## Alternatives Considered

- Fetch only the Jira issues discovered from pull requests: rejected for this initial scope because it requires
  dependent
  calls and orchestration.
- Provider-specific join tools or executor adapters: rejected because source relationships belong in Selection.
- Automatically execute prerequisite controls: superseded; evidence is obtained from declared MCP tools, not from
  other controls' results.

- Evaluate an explicitly incomplete returned subset while marking the control ERROR: rejected on 2026-10-21 because
  missing related records could produce false violations. Useful-looking partial results do not justify that ambiguity.

- Extract JSON fragments from explanatory text or Markdown: rejected on 2026-10-21 because this adds interpretation
  rules and ambiguous payload selection. The initial boundary accepts explicit complete JSON representations.

- Abort all controls after an unsupported or explicitly incomplete received response: rejected on 2026-10-21 because
  the remaining controls are independent and can still produce useful results. Source unavailability remains the
  separately accepted invocation-abort case.

- Evaluate whenever no incompleteness warning exists: rejected on 2026-10-21 because silence does not establish
  complete retrieval and may cause false violations or false inapplicability.

## Rationale

Accept additional retrieval volume in exchange for a smaller declarative contract and executor.

## Consequences

- Positive: Inputs to each call can be inspected before any response is received.
- Positive: No dependency scheduler or reconciliation between dependent execution stages is needed.
- Negative: Broader datasets may exceed a tool's retrieval limits. Actual pagination and truncation behavior must be
  investigated before a missing related record can reliably be treated as absent.
- Positive: Explicit incompleteness cannot be mistaken for evidence that a linked record does not exist.
- Negative: A tool that returns only an incomplete dataset cannot support that control until the acquisition contract
  is resolved. The engine does not hide the limitation by evaluating the subset.
- Pending: Provider-neutral interpretation of actual tool contracts and completeness indicators. Unknown-completeness
  behavior is accepted above; its detection remains unproven for the reference tools.
- Negative: A tool whose completeness cannot be established cannot support subject evaluation for that control,
  even if it returns valid JSON. No automatic recovery or paging mechanism is added to overcome this limitation.
- No generic pagination loop, cache, retry mechanism, or cross-control collection optimization is approved here.
- [ADR-005](adr-005-automatic-prerequisites.md) is superseded. The previous required shared-evidence optimization is
  removed from the initial acceptance criteria.

- Positive: Payload precedence and failure behavior are explicit and independently testable.
- Negative: Tools returning only prose, fenced JSON, or multiple text blocks cannot supply evidence under this contract.
- Pending: Verify real reference-tool payloads and completeness indicators; no universal tool compatibility is claimed.

## Related Artifacts

See the [design](design.md), especially evidence completeness, and the [interview record](interview.md).
