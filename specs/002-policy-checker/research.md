# Policy Validation Engine: Research (Schematic)

**Date**: 2026-11-21
**Scope**: [spec.md](spec.md) — stateless service, CLI, Agent Skill (`agentskills.io`), LLM flows, and hermetic tests
**Outcome**: Technical decisions and the constitution check are recorded.

## R10 — Clarified Behavior and Constitution Check (2026-11-21)

- **Shared parameters**: Use one resolver in query and execution. A declaration without a default is required
  unless it sets `required: false`. Track whether the default was declared so a JSON `null` remains distinct from
  an omitted default. Pydantic can preserve this distinction with field presence. The JSON Schema must allow
  `null` for it to be valid. Reject conflicting declared values or schemas. Requiring equal presence of defaults
  would reject the approved mixed-default case.
- **Creation retries**: Apply one per-rule budget to synthetic and live failures. Zero runs the synthetic tests
  once without MCP. One runs the synthetic tests and the live trial without a retry. Higher settings retry from
  Step 2.1 with the accumulated error context and reuse evidence only for an identical cache key. Separate retry
  budgets would exceed the configured limit.
- **YAML saving**: Keep the pre-check for all collisions. If a later write fails, preserve complete files and
  report failed and pending paths. Publish each target without replacing existing files or exposing partial YAML.
  Batch rollback was rejected by the scope decision.
- **Violation messages**: Resolve only JSON mapping paths. Use the same renderer for creation tests, live trials,
  and findings. Python object traversal and evaluation would expose behavior beyond template substitution.
- **Evidence completeness**: Define `complete_when` in the trusted evidence profile. A control names the tool and
  supplies arguments, so it cannot weaken the profile's completeness check.
- **Trace safeguards**: Redact credentials and tokens in diagnostic payloads. Keep ordinary progress and status
  logs separate from the trace payloads and machine-readable output.

## R1 — Stateless Service Boundary and MCP-Ready Design

- **Decision**:
  - Implement a stateless `PolicyService` (`src/policy/service.py`) with 3 methods: `create`, `execute`, `read_trace`.
  - Use JSON-serializable Pydantic v2 request/result models with `Field(description=...)` (no `Path` or host disk
    state).
  - Client (`src/client/policy.py`) and Agent Skill scripts (`skills/vatuta-policy/scripts/`) call `PolicyService`
    directly in Python.
  - Defer the MCP stdio server transport (`vatuta policy serve`) to a later increment.
  - Access upstream evidence sources through existing `src/mcp/server.py:MCPServer`.
  - Skip eager LLM preflight in `src/client/client.py` for the `policy` command group; initialize LLMs lazily only when
    `create`, `check`, or `mode: llm` execution runs.
- **Rationale**:
  - Keeping `PolicyService` 100% free of local file paths and directory state allows 1:1 wrapping as MCP tools
    (`policy_create`, `policy_execute`) and `resources/read` later.
  - Lazy LLM setup allows deterministic operations (`query` and deterministic `execute`) to run without LLM credentials.
- **Alternatives considered**:
  - File/glob selectors inside `PolicyService` (rejected: couples service to local disk).
  - Routing through `vatuta ask` (rejected: mixes RAG chat with structured policy validation).

## R2 — JSONata Selection, Exploratory Projections, and Completeness Expressions

- **Decision**:
  - Use `jsonata-python==0.7.1` (`jsonata.Jsonata`).
  - Wrap `selection` in an internal existence envelope:
    `($v := (<expression>); {"exists": $exists($v), "value": $v})`.
  - Distinguish missing output (`exists=false` -> `NOT_APPLICABLE`), empty list (`[]` -> `NOT_APPLICABLE`), and explicit
    root null (`exists=true, value=null` -> `ERROR`).
  - Reuse guarded JSONata during Creation Step 2.2 for bounded in-memory evidence exploration:
    - Max queries per rule: `policy.discovery_max_queries` (default `5`).
    - Max output per query: `policy.discovery_max_output_bytes` (default `4096` bytes, appending `[TRUNCATED]`).
  - Disable non-deterministic builtins (`now`, `millis`, `random`, `shuffle`, `eval`) including aliases.
  - Enforce the 60-second execution ceiling and use `google-re2` for regex safety.
- **Rationale**:
  - Prevents large MCP tool payloads from overflowing the LLM context window during control creation.
  - Keeps selection deterministic and side-effect free.
- **Alternatives considered**:
  - Dumping raw evidence JSON into prompts (rejected: fails on large responses).
  - Unbounded exploration queries (rejected: risks runaway latency and token usage).

## R3 — CEL Boolean Evaluation

- **Decision**:
  - Use `cel-python==0.5.0` (`celpy`).
  - Compile expression once per control; evaluate with `{subject, parameters}` activation only.
  - Require strict boolean return values (`true` -> `COMPLIANT`, `false` -> `NON_COMPLIANT`); any non-boolean or
    evaluation exception becomes subject `ERROR`.
- **Rationale**:
  - Declarative, safe, deterministic evaluation without Python `eval` or ambient state.
- **Alternatives considered**:
  - Python `eval` or custom expression parsers (rejected: security and maintenance risk).

## R4 — Step-by-Step Creation & Guided 2-Step Check LLM Flows (DSPy)

- **Decision**:
  - Use atomic DSPy Signatures (`src/policy/signatures/`). Get the LM from `LLMProviderManager.get_dspy_lm` and
    use a per-call DSPy context with `ChatAdapter(use_json_adapter_fallback=False)`.
    1. **Creation Step 1 (`PolicyDecompositionSignature`)**:
       - Split `policy_text` into atomic normative rules (`rule_id`, `description`, `excerpt`, `inferred_tags`).
       - Split conditions whenever they can fail independently or represent separate violations.
       - Fail with `NO_NORMATIVE_RULES` (`stage: "decomposition"`) if zero rules are extracted.
       - Assign deterministic control slug `id`: `f"{policy_id}-{rule_id}"` (or `rule_id` if already prefixed).
    2. **Creation Step 2, Phase A (`RuleEvidenceAndParametersSignature` across all rules `r_1..r_N`)**:
       - Use local `policy.evidence_profiles` only (no MCP container startup or `list_tools`).
       - Pass incremental `SharedParameterCatalog`. Two rules share a parameter name only when `value_schema`,
         `construction_value`, and `default` match. Shared entries are immutable across later stages and retries.
       - Declare minimum necessary parameters (`parameters: {}` allowed).
       - When `trial_attempts > 0`, collect and return **all** `MISSING_CONSTRUCTION_CONTEXT` errors across all rules at
         once before starting any MCP server.
    3. **Creation Step 2, Phase B (Per-Rule Steps 2.2–2.5 with Fail-Fast)**:
       - **Step 2.2 (`EvidenceDiscoverySignature`, when `trial_attempts > 0`)**: Connect to MCP, validate live
         `inputSchema`, fetch evidence once in memory (cached by `(server, tool, resolved_arguments)`), and run up to
         `discovery_max_queries` (default `5`) JSONata queries over `{"evidence", "parameters"}`. Return syntax/runtime
         errors or `missing` as observations within budget; exit early on `done=True`.
       - **Step 2.3 (`RuleSelectionSignature`)**: Build `selection` and `test_cases.selection_tests` (at least 1
         populated case and 1 `NOT_APPLICABLE` case). Normalize single object `{...}` to `[{...}]`, require
         non-whitespace `id`/`descriptive_id`, reject duplicate `id`s (`DUPLICATE_SUBJECT_ID`), resolve test parameters
         (`test_case.parameters` -> `default` -> `construction_value`), and run `selection_tests` (always mandatory) +
         live selection (`trial_attempts > 0`).
       - **Step 2.4 (`RuleCriterionSignature`)**: Build `criterion` (`deterministic` CEL or justified `llm`) and
         `test_cases.criterion_tests` (at least 1 `COMPLIANT` and 1 `NON_COMPLIANT` subject). Run `criterion_tests`
         (always mandatory, using CEL or `SemanticCriterionSignature` with `{{parameters.<name>}}` substitution) + live
         subjects (`trial_attempts > 0`).
       - **Step 2.5 (Full-Context Retry & Fail-Fast)**: If `trial_attempts > 1` and Steps 2.2–2.4 fail, retry from Step
         2.1 with full accumulated stage context and errors, reusing cached in-memory MCP evidence if bindings did not
         change. Abort creation immediately (Fail-Fast) if any rule exhausts its attempts.
    4. **Runtime Semantic Criterion (`SemanticCriterionSignature`)**:
       - Evaluate 1 subject per LLM call (`mode: llm`) with correlation token `subject_ref` and deterministic
         `{{parameters.<name>}}` template substitution in `condition` (no raw `parameters` dict sent to LLM).
    5. **Client Guided 2-Step Check (`vatuta policy check "<prompt>"`)**:
       - Share one `invocation_id` trace (`vatuta://traces/policy/execution/{invocation_id}`) across Steps 1, 2, and
         `execute`.
       - **Step 1 (`PolicyCheckRuleSelectionSignature`)**: Query `ClientControlQuery` (`projection=["id",
         "description", "tags", "source"]`, `include_merged_parameters=False`) up to `policy.check_max_filter_queries`
         (default `3`). Limit each response to `policy.check_max_output_bytes` (default `4096` bytes). Keep the
         complete `(file, id)` pairs shown to the model for this invocation. An intermediate no-match response
         lets the model refine the filters within the budget.
       - Accept a final selection only when the model sets `done=True`. It may confirm after seeing the last
         allowed query result. If it needs another query at that point, return `CONTROL_SELECTION_INCOMPLETE`
         (Exit `1`). A confirmed empty selection returns `NO_CONTROLS_MATCHED` (Exit `2`).
       - Check the confirmed pairs against those shown to the model. Reuse the local loader to open just those
         paths and verify each file's ID before merging parameters. An unseen pair or changed ID fails with
         `INVALID_CONTROL_SELECTION` (Exit `1`). Two selected files with the same verified ID fail with
         `DUPLICATE_CONTROL_ID` (Exit `2`).
       - **Step 2 (`PolicyCheckParameterExtractionSignature`)**: Skip if `merged_parameters` is empty
         (`parameters={}`); otherwise extract explicit parameter values from the prompt without guessing missing values,
         then call `PolicyService.execute`.
- **Rationale**:
  - Step-by-step construction with embedded unit tests catches expression bugs immediately and makes saved `.yaml` files
    self-testing.
  - Guided 2-step `policy check` gives deterministic CLI UX without open-ended ReAct loops.
  - Confirmation keeps incomplete searches from producing a compliance result. The relative path preserves the
    selected file when another folder contains a control with the same ID. The service still receives inline controls.
- **Alternatives considered**:
  - One-shot monolithic rule generation (rejected: fragile on complex tool schemas and shared parameters).
  - Running provisional choices at the limit could produce a partial review. Reloading by ID across the directory
    could lose the selected variant.

## R5 — Client Control Directory Tree, Universal Selection, and Shared Parameters

- **Decision**:
  - Store one YAML file per rule (`<id>.yaml`) under `policy.controls_directory` (default `data/controls`, overridable
    via `--controls-dir`), supporting nested subdirectories organized by user use case.
  - Implement `ClientControlQuery` (`src/policy/files.py`, `src/policy/parameters.py`):
    - Select candidate paths under `controls_directory` before opening files. Use `files`, `glob`, or their
      intersection if both are set. With neither selector, every `.yaml` file under the root is a candidate.
      Enforce root containment, including symlink resolution, before reading candidates.
    - Fail with `INVALID_CONTROL_FILE` (Exit `2`) for invalid candidate YAML. Apply ID and metadata
      filters after the candidates are read and validated.
    - Combines distinct filter fields with logical **AND** and list values with logical **OR**:
      - `files` (`--control`): specific `.yaml` file paths relative to or contained in `controls_directory`.
      - `ids` (`--id`): exact `control.id` or glob/wildcard patterns on `control.id` (e.g., `change-mgmt-*`, using
        `fnmatchcase`).
      - `glob` (`--glob`): glob pattern matched against relative paths (`team-a/**/*.yaml`) or bare `<id>.yaml`
        filenames (`*github*.yaml`).
      - Metadata filters: `policy_ids`, `rule_ids`, `tags`, `evaluation_mode`, `evidence_server`, `evidence_tool`,
        `parameter_names`, `text_query`.
    - Allows two `.yaml` files in different subfolders on disk to share the same `control.id`, but fails immediately
      with `DUPLICATE_CONTROL_ID` (Exit `2`) if the active filter/selection matches two files with the same
      `control.id`.
    - Projects requested attributes (`projection`, plus relative `file` path in projected dicts) and computes
      conflict-checked `merged_parameters` when `include_merged_parameters=True` (default).
- **Rationale**:
  - Supports flexible folder hierarchies per user use case and uniform selection across CLI and Agent Skill scripts while
    keeping `PolicyService` stateless.
  - Applying path selectors before YAML validation limits errors to the files in scope and avoids unnecessary reads.
- **Alternatives considered**:
  - Flat-only control directory or global duplicate-ID rejection across unselected subfolders (rejected: prevents users
    from organizing separate rule variants in subfolders).
  - Parsing the full directory tree before filtering by path would let an unrelated invalid file block a scoped check.

## R6 — Evidence Acquisition, Caching, Completeness, and Source Independence

- **Decision**:
  - Keep started `MCPServer` containers alive across controls within a single `PolicyService.execute` call.
  - Cache valid evidence responses in memory by exact `(server, tool, resolved_arguments)`.
  - Validate `evidence_profile`, tool allowlist, `arguments_schema`, and discovered live `inputSchema` per control in
    `Stage E1`.
  - Check `isError` first, prefer `structuredContent`, otherwise parse a single valid JSON text block.
  - Verify completeness via the profile's JSONata `complete_when` expression (`True` = complete; `False` = incomplete
    ERROR; other/missing = unknown completeness ERROR).
  - Fast-fail the affected control immediately on its first evidence error while other controls continue.
  - Keep `src/policy/` and `tests/policy/` 100% independent of GitHub/Jira; live GitHub/Jira profiles exist only in
    `pocs/policy_demo/`.
- **Rationale**:
  - Prevents false compliance results on truncated pages and avoids redundant MCP container startups.
- **Alternatives considered**:
  - Guessing completeness from short arrays or warning absence (rejected: unsafe when tools omit pagination metadata).

## R7 — Security, Diagnostics, and Operational Bounds

- **Decision**:
  - Reuse Docker `MCPServer` with validated policy profiles. Require the existing read-only root and mounts,
    non-root user, capability drop, CPU and memory limits, digest pinning, and an explicit tool allowlist.
    Network access follows the existing default-deny rule and requires a deployment-approved exception.
  - Scope the MCP subprocess log sink for policy calls to safe status messages. The reused transport currently sends
    raw MCP stderr to `LoggerWriter`. Do not forward that raw text to ordinary policy logs.
  - Save sanitized stage traces under `policy.trace_directory` (`logs/policy/<invocation_id>/`) with secret
    redaction, no automatic age expiry, and read access via `PolicyService.read_trace(uri)`.
  - On execution trace write failure, keep calculated `results`, append top-level `TRACE_WRITE_FAILED`, and exit `1`.
- **Rationale**:
  - Satisfies Constitutional security, provenance, and observability principles without adding a database.
- **Alternatives considered**:
  - Discarding execution results when trace saving fails (rejected by stakeholder review).

## R8 — Confirmed Review Decisions (2026-11-11)

- **Decision**:
  1. Local tool catalog through Step 0 and Phase A; connect to MCP only in Phase B when `trial_attempts > 0`.
  2. Shared parameter declarations already used by earlier rules are immutable through Phase B and retries.
  3. Reject existing `<id>.yaml` target files with `CONTROL_FILE_EXISTS` (Exit `1`); check all targets before writing.
  4. Include `source` and effective per-control `parameters` in every `ControlResult` (and trace).
  5. Preserve execution `results` on `TRACE_WRITE_FAILED` (Exit `1`).
  6. Provide a complete standalone `pocs/policy_demo/vatuta.demo.yaml` template for the optional blog demo.
- **Rationale**:
  - Protects existing YAML files, avoids unnecessary MCP startups, and makes `ControlResult` self-contained.

## R9 — Agent Skill (`https://agentskills.io/`) & PEP 723 Scripts (2026-11-20)

- **Decision**:
  - Implement a tracked Agent Skill package at `skills/vatuta-policy/` following the `https://agentskills.io/`
    standard:
    - `skills/vatuta-policy/SKILL.md`: YAML frontmatter (`name: vatuta-policy`, `description`) + concise, schematic
      instructions for AI agents.
    - `skills/vatuta-policy/scripts/`: 3 executable Python scripts with **PEP 723 inline script metadata**
      (`# /// script ... # ///`):
      1. `query.py`: Search YAML rules across the folder tree, project attributes, and retrieve `merged_parameters`
         (combining rule search and parameter inspection into one script).
      2. `create.py`: Create rules from a policy text/file and save `<id>.yaml` files in `--output-dir`.
      3. `execute.py`: Select `.yaml` rules and execute them with explicit `--parameters`.
  - Expose the matching `vatuta policy query` CLI subcommand alongside `create`, `execute`, and `check`.
  - Share the exact same Python client handlers (`src/client/policy.py`) between `vatuta policy
    query|create|execute --format json` and the 3 skill scripts (`query.py`, `create.py`, `execute.py`).
- **Rationale**:
  - External AI agents (Claude Code, Cursor, Antigravity, Codex, etc.) already have their own reasoning LLM. With the
    3 skill scripts, the agent searches rules/parameters (`query.py`), creates rules (`create.py`), and executes rules
    (`execute.py`) directly with 100% deterministic JSON output and exit codes (`0`, `1`, `2`), while reusing the CLI
    implementation with zero code duplication.
- **Alternatives considered**:
  - 4 separate scripts (`search` + `parameters` split) (rejected: `query.py` already handles both via
    `--include-merged-parameters` / `--no-merged-parameters` and `--projection`).
  - Untracked skill in `.agents/skills/` (rejected: `.agents/` is gitignored for personal IDE setup; `skills/` is
    tracked and distributable).
