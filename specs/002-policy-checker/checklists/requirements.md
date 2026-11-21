# Specification Quality Checklist: Policy Validation Engine

**Purpose**: Validate specification completeness and quality before planning
**Created**: 2026-10-06
**Updated**: 2026-11-21
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] Technical constraints are explicit stakeholder decisions, not silently inferred scope
- [x] Focused on user value and business needs
- [x] Written in simple, accessible, schematic English without losing technical detail
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No `[NEEDS CLARIFICATION]` markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria verify behavior within approved technical constraints
- [x] All acceptance scenarios are defined (User Stories 1–5)
- [x] Edge cases are identified
- [x] Scope is clearly bounded (CLI + Agent Skill implemented; MCP server prepared/deferred)
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements (`FR-001`..`FR-051`) have clear acceptance criteria
- [x] User scenarios cover primary flows (creation, search/parameters/check, execution, traces, Agent Skill)
- [x] Feature meets measurable outcomes defined in Success Criteria (`SC-001`..`SC-011`)
- [x] Planning details remain in the plan, data model, research, and contracts

## Validation Notes (Schematic)

- **Format & Readability**: Rewritten in simple English with a schematic outline layout per stakeholder instruction on
  2026-11-19 while preserving every requirement, stage, error code, and edge case.
- **Templated Violation & Remediation Messages (`FR-019`, `FR-034`, `FR-046`, `FR-051`)**:
  - Adapted the mandatory `criterion.violation_message` format template to provide both the clear violation explanation
    and actionable remediation hints/instructions for an analyzing agent or user, populated from the `subject` (and
    control `parameters`) data context using safe Python `.format` / f-string style field placeholders (`{<field>}`,
    `{subject.<path>}`, `{parameters.<name>}`).
  - Validated during creation Step 2.4 against all `NON_COMPLIANT` `criterion_tests` (and live trial violations), and
    rendered deterministically into `SubjectResult.violation_message` for `NON_COMPLIANT` subjects at execution.
- **Agent Skill (`https://agentskills.io/`)**:
  - Added User Story 5, `FR-048`..`FR-050`, and `SC-011`.
  - Defines `SKILL.md` + 3 executable scripts under `scripts/` (combining rule search and parameter retrieval into the
    first script: `query`, `create`, `execute`).
  - Supports hierarchical folder trees under `policy.controls_directory` (one `<id>.yaml` per rule) and consistent rule
    selection across all scripts (file by file, by name/ID, by glob pattern, or by metadata filters).
- **Preserved Contracts**:
  - Stateless `PolicyService` (`create`, `execute`, `read_trace`) ready 1:1 for future MCP tools.
  - Two-phase creation (local catalog in Step 0 & Phase A, immutable `SharedParameterCatalog`, bounded JSONata
    discovery, mandatory `selection_tests` & `criterion_tests`, non-overwriting `<id>.yaml` saves).
  - Pre-flight `Stage E0` vs per-control `Stages E1–E3` isolation, MCP container reuse, exact evidence cache,
    `DUPLICATE_SUBJECT_ID` guard, singleton `llm` evaluation with `{{parameters.<name>}}` substitution, self-contained
    `ControlResult` provenance/parameters, and `TRACE_WRITE_FAILED` result preservation.

## Acceptance Coverage

| Requirements | Acceptance coverage | Success criteria |
| --- | --- | --- |
| FR-001, FR-002, FR-032, FR-041, FR-043, FR-044 | Story 1 scenarios 1, 8–9, 13, 17 | SC-001, SC-002, SC-010 |
| FR-026, FR-045, FR-046 | Story 1 scenarios 3–5, 7, 10, 12 | SC-001, SC-002 |
| FR-003, FR-034, FR-051 | Story 1 scenarios 4, 6; Story 3 scenarios 2–3, 14 | SC-002, SC-005, SC-008 |
| FR-038, FR-039, FR-040, FR-047 | Story 1 scenarios 2, 11, 14–16; Story 2 scenarios 1–6 | SC-002, SC-003, SC-010 |
| FR-014, FR-042 | Story 2 scenarios 7–8; input edge cases | SC-003, SC-004 |
| FR-009, FR-012 | Story 2 scenarios 1, 3–4; Story 3 scenarios 1, 12; Story 5 scenarios 1–5 | SC-003, SC-010, SC-011 |
| FR-004, FR-005, FR-006, FR-008 | Story 3 scenarios 1–3; source-scope edge cases | SC-005, SC-008 |
| FR-007, FR-015, FR-019 | Story 3 scenarios 2–5, 10, 14 | SC-005, SC-007, SC-008 |
| FR-013, FR-018 | Story 3 scenarios 6–8, 11 | SC-006 |
| FR-017, FR-021, FR-025 | Story 3 scenarios 2, 9–10, 13; Story 4 scenarios 1–2 | SC-005, SC-009 |
| FR-035, FR-036, FR-037 | Story 3 scenario 11; Story 4 scenario 2 | SC-008, SC-009 |
| FR-010, FR-023, FR-024 | Story 4 scenarios 1–2, 5–8 | SC-009 |
| FR-022, FR-027 | Story 4 scenarios 3–4 | SC-009, SC-010 |
| FR-048, FR-049, FR-050 | Story 2 scenarios 1–2; Story 5 scenarios 1–5 | SC-003, SC-010, SC-011 |
