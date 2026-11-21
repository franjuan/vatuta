# Policy Validation Engine: Design Inputs Register (Schematic)

**Updated**: 2026-11-21
**Specification**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Demo Plan**: [demo-plan.md](demo-plan.md)
**Research**: [research.md](research.md) | **Scope Decision**: [ADR-019](../../docs/policy-checker/adr-019-file-based-policy-operations.md)

## 1. Role & Status

- The scope and behavior decisions through 2026-11-21 are reflected in the planning artifacts.
- Phase 0 research and Phase 1 design pass the constitution check.
- Library installation (`jsonata-python`, `cel-python`) and runtime implementation happen in `/speckit-implement`.

## 2. Resolved Design Decisions

| Topic | Decision | Reference |
| --- | --- | --- |
| Stateless service & MCP boundary | `PolicyService` exposes `create`, `execute`, and `read_trace` over JSON-serializable Pydantic models (`policy serve` deferred). | Research R1 |
| Lazy LLM startup | `query` and deterministic `execute` skip eager LLM preflight; LLM backends initialize only for `create`, `check`, or `mode: llm`. | Research R1, R4 |
| Creation pipeline | Local tool catalog in Step 0 & Phase A; immutable `SharedParameterCatalog`; collect all `MISSING_CONSTRUCTION_CONTEXT` before MCP; bounded JSONata discovery (5 queries, 4096 bytes) in Phase B. | Research R4, R8 |
| Embedded unit tests (`test_cases`) | Mandatory `selection_tests` (`[obj]` normalization, `DUPLICATE_SUBJECT_ID` check) and `criterion_tests` (CEL or LLM with `{{parameters.<name>}}` substitution) even when `trial_attempts == 0`. | Research R4; control-format |
| Hierarchical folder tree & selection | Store 1 `<id>.yaml` per rule in nested subfolders under `policy.controls_directory`. Select by `--control` (`files`), wildcard `--id` (`ids`), path/filename `--glob` (`glob`), or metadata filters. | Research R5, R9 |
| Duplicate IDs across subfolders | Allow same `control.id` in different folders on disk; fail with `DUPLICATE_CONTROL_ID` (Exit `2`) if the active filter/selection matches two files with the same `control.id`. | Research R5 |
| Agent Skill (`https://agentskills.io/`) | Tracked at `skills/vatuta-policy/` (`SKILL.md` + 3 PEP 723 scripts in `scripts/`: `query.py`, `create.py`, `execute.py`), sharing client handlers with CLI `vatuta policy query\|create\|execute --format json`. | Research R9 |
| Guided `policy check` | Confirm within the query budget and verify each selected path and ID. | Research R4 |
| Pre-flight vs per-control isolation | `Stage E0` validates non-empty unique controls and parameter schemas/presence (extra keys in `request.parameters` allowed and scoped per control); `Stages E1–E3` isolate failures per control (`outcome="ERROR"`). | Operations contract |
| Evidence caching & fast-fail | Reuse started `MCPServer` containers and exact `(server, tool, resolved_arguments)` evidence cache across controls in one `execute` call; fast-fail a control on its first evidence error. | Research R6 |
| Self-contained results & traces | Every `ControlResult` includes `source` and effective per-control `parameters`. `TRACE_WRITE_FAILED` preserves calculated `results` and exits `1`. | Research R8 |
| Testing & demo boundary | Hermetic unit and contract tests. Demo uses a separate config. | [Plan](plan.md) |

## 3. Implementation Qualification Checks

During `/speckit-implement`:

- Lock `jsonata-python==0.7.1` and `cel-python==0.5.0` via Poetry (`poetry run pip-audit`, Python 3.12 & 3.13).
- Verify JSONata existence envelope, disabled builtins (`now`, `millis`, `random`, `shuffle`, `eval`), `google-re2`
  regex guard, byte truncation (`[TRUNCATED]`), and CEL strict boolean evaluation.
- Verify `ClientControlQuery` recursive folder scan, root containment, wildcard `ids`, path/filename `glob`,
  `DUPLICATE_CONTROL_ID`, `INVALID_CONTROL_FILE`, and `merged_parameters`. Open only YAML in the path scope.
- Verify that guided checks stop before evidence acquisition on incomplete or invalid selection. Confirm that a
  same-ID variant in an unselected folder does not affect the result.
- Verify `skills/vatuta-policy/SKILL.md` and PEP 723 scripts (`query.py`, `create.py`, `execute.py`) share the exact
  handlers with `vatuta policy query|create|execute --format json`.
- Keep all GitHub/Jira live MCP profiles strictly inside `pocs/policy_demo/` and out of `src/` and CI.

## 4. Next Step

Use [plan.md](plan.md) as the input to `/speckit-tasks`.
