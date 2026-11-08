# Specification Quality Checklist: Policy Validation Engine

**Purpose**: Validate specification completeness and quality before implementation planning
**Created**: 2026-10-06
**Updated**: 2026-11-04
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Validation Notes

- All 16 specification-quality checks pass. This evaluates the specification, not implementation or tool feasibility.
- The stateless `PolicyService` contract (`create` and `execute`), step-by-step creation LLM flow (policy
  decomposition, incremental `SharedParameterCatalog`, bounded JSONata evidence exploration, and mandatory persisted
  `selection_tests` and `criterion_tests`), `tags` on `ControlDefinition`, client-side YAML persistence
  (`policy.controls_directory`), client filtering/projection tool (`include_merged_parameters=True`), and the guided
  2-step `vatuta policy check "<prompt>"` pipeline are explicit stakeholder requirements agreed on 2026-11-04.
- User Stories 1–4 cover step-by-step creation, client filtering/projection and 2-step check parameter preparation,
  stateless execution, and diagnosis.
- FR-014 states that missing required values reject the entire invocation before evidence acquisition. FR-018
  isolates subsequent failures per control, including source outages.
- FR-038 through FR-047 distinguish construction values and declared defaults, add `tags` (inferred + explicit) and
  `test_cases` (`selection_tests` and `criterion_tests`), define policy decomposition and bounded JSONata exploration,
  move filtering/projection and `include_merged_parameters` to the client, and specify the 2-step `policy check`
  pipeline.
- FR-026 and FR-046 preserve creation trials and make synthetic `selection_tests` and `criterion_tests` mandatory even
  when `trial_attempts == 0`.
- FR-003 and FR-034 preserve the declared semantic exception.
- FR-027 removes configurable retention and automatic cleanup while retaining diagnostic artifacts and privacy rules.
- Retired lifecycle requirements FR-028 through FR-031 and FR-033 have no replacement implementation work. `vatuta
  policy serve` and the standalone `policy_parameters` MCP tool are excluded from this increment.

## Acceptance Coverage

| Requirements | Acceptance coverage | Success criteria |
| ------------ | ------------------- | ---------------- |
| FR-001, FR-002, FR-032, FR-041, FR-043, FR-044 | Story 1 scenarios 1, 8–9, 13 | SC-001, SC-002, SC-010 |
| FR-026, FR-045, FR-046 | Story 1 scenarios 3–5, 7, 10, 12 | SC-001, SC-002 |
| FR-003, FR-034 | Story 1 scenario 6; Story 3 scenario 3 | SC-008 |
| FR-038, FR-039, FR-040, FR-047 | Story 1 scenarios 2, 11, 14; Story 2 scenarios 1–6 | SC-002, SC-003, SC-010 |
| FR-014, FR-042 | Story 2 scenarios 7–8; input edge cases | SC-003, SC-004 |
| FR-009, FR-012 | Story 2 scenarios 1, 3–4; Story 3 scenarios 1, 12 | SC-003, SC-010 |
| FR-004, FR-005, FR-006, FR-008 | Story 3 scenarios 1–3; source-scope edge cases | SC-005, SC-008 |
| FR-007, FR-015, FR-019 | Story 3 scenarios 2–5, 10 | SC-005, SC-007, SC-008 |
| FR-013, FR-018 | Story 3 scenarios 6–8, 11 | SC-006 |
| FR-017, FR-021, FR-025 | Story 3 scenarios 2, 9–10; Story 4 scenarios 1–2 | SC-005, SC-009 |
| FR-035, FR-036, FR-037 | Story 3 scenario 11; Story 4 scenario 2 | SC-008, SC-009 |
| FR-010, FR-011, FR-023, FR-024 | Story 4 scenarios 1–2, 5–6 | SC-009 |
| FR-022, FR-027 | Story 4 scenarios 3–4 | SC-009, SC-010 |
