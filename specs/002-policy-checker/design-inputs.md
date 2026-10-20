# Policy Checker: Technical Design Inputs

**Updated**: 2026-10-20
**Feature**: [Policy Checker Mechanism](spec.md)
**Source**: [Original notes](../../policy-checker.md)
**Status**: Non-binding input for future technical planning; no architecture or implementation plan has been selected.

## Purpose

Keep the user's technical suggestions available without treating exploratory examples as approved requirements.
The specification defines behavior and acceptance criteria; the later planning phase must compare alternatives and
justify any technology selection against those requirements and the Vatuta constitution.

## Candidates and Investigations

- **Deterministic evaluation**: CEL or JSON Schema. Compare expressiveness, reproducibility, validation, and safe
  execution for the required controls. They need not serve interchangeable purposes.
- **Semantic evaluation**: LLM evaluation or a JEV-type model from TypeSafe AI. Establish suitability, availability,
  evaluation criteria, reproducibility limits, and cost using representative policies. Preserve provider independence.
- **Hybrid controls**: Combine deterministic and semantic evaluation. Determine how partial evaluations compose while
  preserving outcome meanings and explanations.
- **Evidence transformations**: Substrait is explicitly a preference, not a decision. Evaluate whether it helps
  represent filtering, projection, aggregation, and normalization at acceptable complexity. Compare simpler
  alternatives.
- **Evidence relationships**: Operation sequences, evidence graphs, and query examples. Compare representations for
  cross-source references and prerequisite results without requiring a graph engine or query language.
- **Shared evidence**: Reuse collection and preparation for multiple controls. Define equivalence, invocation scope,
  provenance, and execution scheduling without assuming a cross-run cache.
- **Control representation**: Several illustrative YAML control shapes. Design a validated, inspectable contract after
  behavior is settled. Do not copy the examples into a schema as requirements.
- **Normalized control structure & execution engine**: The control creation stage outputs a normalized control that
  encapsulates unique ID, description, policy reference/section/version, evidence sources, source applicability,
  evidence transformations, selection criteria for violations (supporting multiple distinct transformation + selection
  paths), exception criteria, parameters, and evaluation method. The execution engine must consume this normalized
  definition as the bridge from policy text to deterministic execution instructions, holding all data, information,
  parameters, and criteria needed to run validation without re-reading the policy document.
- **Multi-path violation detection**: Support multiple alternative pathways (combinations of transformation +
  selection/criteria) within a single control to detect violations across different representations or mechanisms
  in source data.
- **Command interface**: Extend the existing Vatuta client with `policy`. Define creation and verification contracts,
  configuration inputs, parameter handling, reports, and errors.
- **Programmatic invocation**: Define headless CLI and programmatic API contracts suitable for a future MCP wrapper.
  An operational MCP service is outside this delivery, as already resolved in FR-009.
- **Automated evaluation**: Human review workflows and handoff artifacts are outside the agreed feature scope.
  Preserve the specification's automated evaluation and explicit failure behavior.
- **Audit artifacts**: One identifiable invocation directory with stage subdirectories. Choose names, formats,
  correlation, and redaction that support human inspection and machine reuse.
- **MLOps evaluation and prompt optimization platform**: Evaluate widely adopted open-source MLOps/LLMOps tools
  (e.g., MLflow, OpenLLMetry/Traceloop, Arize Phoenix, Langfuse) for importing and managing captured LLM interaction
  samples to support regression testing, prompt tuning (e.g., DSPy teleprompters/optimizers), and benchmark tracking
  without introducing tight vendor lock-in.
- **Languages and libraries**: The notes permit technologies beyond Python when useful. Retain the project's
  Python/Poetry baseline and assess interoperability, maintenance, and constitutional constraints before proposing
  additions.

## Boundaries of the Examples

- The `JIRA-027`, `JIRA-042`, and `PR-001` examples illustrate policy concepts, not a stable serialization contract.
- `one_of`, `evidence_graph`, `lateral_join`, interpolation syntax, field paths, and expression snippets are
  exploratory.
  Neither their exact syntax nor their evaluation semantics has been approved.
- RASE means Requirement, Applicability, Selection, and Exception. The eventual representation must preserve the
  policy's
  meaning; examples do not define a universal algorithm for selecting alternatives or applying exemptions.
- Azure DevOps appears only as an illustrative source. Initial validation remains limited to GitHub and Jira Free.
- The evidence-check pseudocode illustrates outcome distinctions, not a mandatory execution algorithm. The design must
  honor the spec's separate rules for missing evidence, applicability, exceptions, and source failures.
- Logical processing stages do not require separate services, a particular workflow framework, or one physical fetch per
  control. Shared collection must remain attributable to each consuming control.

## Existing Constraints to Carry into Planning

- Preserve [ADR-010](../../docs/policy-checker/adr-010-versioned-controls-and-policy-hashes.md): successful creation saves
  a new control-set version and preserves earlier versions. Hash the exact supplied UTF-8 policy content to detect
  changes independently of generated-control identity. Record the evaluated version and its document hash in each
  verification. Choose the hash algorithm, version identifiers, comparison baseline selection, and storage contract
  during planning without requiring policy reanalysis for verification.
- Preserve [ADR-009](../../docs/policy-checker/adr-009-required-trial-parameters.md): enabled creation-time trials require
  supplied parameter values; missing values fail creation rather than deferring trials. Zero-attempt mode remains
  available, and verification may use different valid values within its configured scope and permissions.
- Preserve [ADR-008](../../docs/policy-checker/adr-008-policy-input-formats.md): the initial policy input contract accepts
  UTF-8 plain text and Markdown only, with explicit rejection of unsupported formats and invalid encoding. Document
  conversion and OCR are outside the initial scope; preserve the existing policy-reference requirements.
- Preserve [ADR-007](../../docs/policy-checker/adr-007-configurable-artifact-retention.md): invocation artifact retention
  is configurable and defaults to no automatic expiry. Explicitly configured expiry covers successful and failed
  invocation artifacts while preserving active artifacts and saved controls. Define duration units, expiry timestamps,
  and cleanup scheduling during planning.
- Preserve [ADR-006](../../docs/policy-checker/adr-006-technical-model-failure.md): technical model failures during
  verification abort the invocation once any permitted retries are exhausted. Retain completed results as incomplete
  work and do not turn a failed call into a compliance outcome. Define bounded retry and timeout behavior during
  planning without changing the immediate abort rule for unavailable evidence sources.
- Preserve the five decisions from the 2026-10-19 clarification round, documented in
  [ADRs 001–005](../../docs/policy-checker/glossary.md#decision-records): fail creation when validation attempts are
  exhausted; allow verification after skipped trials while retaining mandatory checks; pair per-control compliance
  with evidence coverage; publish no overall policy compliance percentage; automatically execute prerequisites within
  the configured scope and permissions, distinguishing them from selected controls in reports.
- Preserve the five decisions recorded on 2026-10-08, including fail-fast source unavailability and fail-safe ambiguous
  semantic judgments. Do not reclassify either case as ordinary missing evidence.
- Preserve the existing CLI and configuration behavior outside the new policy capabilities.
- Follow the constitution's provider independence, structured model interaction, validated boundaries, and
  least-privilege
  source access requirements. Source content and model outputs are data, not authority to execute arbitrary actions.
- Preserve evidence provenance and keep observations, model explanations, and conclusions distinguishable.
- Record prompts, returned outputs, external interactions, transformations, and outcomes with the constitution's
  required
  credential and personal-data redaction. Do not require unavailable provider-internal reasoning.
- Treat new dependencies and non-Python components as proposals requiring justification against the existing project
  constraints, not as a blanket replacement for the established runtime or dependency-management rules.
- Include user and component documentation updates in the eventual implementation plan. This specification update does
  not change runtime functionality or publish a new CLI contract.

## Expected Later Outputs

After behavioral clarification, planning can produce a technical plan, research decisions, a data model, interface
contracts, and a validation guide. None of those outputs is generated or approved by this document.
