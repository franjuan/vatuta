# Vatuta Constitution

**Core Philosophy**: Build the simplest solution that satisfies current requirements and can evolve cleanly later.
Every increase in complexity MUST be justified, never assumed.

These rules govern all SpecKit phases (`/speckit-specify`, `/speckit-clarify`, `/speckit-plan`, `/speckit-tasks`,
`/speckit-implement`, `/speckit-analyze`, `/speckit-converge`) and all repository code and documentation.

## Core Principles

### I. Simplicity as a First-Class Architectural Principle

Simplicity is a mandatory quality attribute:

- **Fewer Moving Parts**: Prefer fewer components, abstractions, dependencies, config options, and code paths.
- **No Premature Generalization**: Do not add speculative extensibility or extra architectural layers for future ideas.
- **No Cargo-Culting**: Do not add patterns, frameworks, or infrastructure just because they are "industry practice".
- **Convention Reuse**: Reuse existing project patterns and components before creating new ones.
- **Justified Complexity**: Every complexity increase MUST link to an explicit, approved requirement.
- **Pragmatic Readability**: Optimize for readability, maintainability, testability, and easy understanding.
- **Simplicity by Default**: When multiple designs work, pick the simplest one unless a complex option proves a clear,
  current benefit.

*Rationale: Complexity grows over time, increasing cognitive load, bugs, and maintenance cost.*

### II. Requirements Traceability and Scope Control

Every requirement MUST trace to a clear source and belong to one of three categories:

- **A. Explicit Requirements**: Stated directly by the user or approved specs. This is the mandatory scope.
- **B. Essential Inferred Requirements**: Needed for correctness, consistency, security, or execution. The AI MUST
  explain why they are needed and ask confirmation when they affect scope, complexity, or visible behavior.
- **C. Optional Recommended Requirements**: Improvements or extra features not needed for the current use case. Present
  them strictly as proposals; MUST NOT implement without explicit user approval.
- **No Silent Promotion**: Never silently turn recommendations, assumptions, or edge cases into mandatory requirements.

*Rationale: Uncontrolled scope creep dilutes core deliverables and adds unwanted complexity.*

### III. AI Initiative with Human Approval

The AI assistant should spot risks, gaps, and improvements, but identifying an idea does not authorize adding it:

- **Clear Distinction**: Separate verified requirements from advisory recommendations.
- **Transparent Trade-offs**: Explain practical pros and cons of including or excluding major recommendations.
- **Approval Gate**: Ask explicit user approval before adding scope or significant complexity.
- **Consolidated Queries**: Group related questions together to avoid long clarification loops.
- **Autonomous Trivial Details**: Resolve minor implementation details automatically using project conventions.
- **Reversible Defaults**: When uncertain and risk is low, pick the simpler, reversible option.
- **Core Axiom**: *The AI recommends; the user decides on scope.*

*Rationale: Combines proactive AI technical analysis with strict human control over scope and architecture.*

### IV. Proportional Edge-Case Handling

Handle edge cases by their real-world impact across three tiers:

- **Essential Cases**: Primary workflows and core operating conditions -> MUST implement and test.
- **Relevant Edge Cases**: Realistic situations whose omission causes functional errors, security risks, or data
  corruption -> MUST implement and test.
- **Speculative Edge Cases**: Unlikely or hypothetical cases -> MUST NOT implement without explicit user approval.
- **Architecture Stability**: A single unlikely scenario MUST NOT distort an otherwise simple design.
- **Simple Error Handling**: Prefer direct, explicit error handling over complex recovery frameworks.

*Rationale: Defending against hypothetical failures bloats codebases with fragile, untested recovery paths.*

### V. Non-Functional Requirements Must Be Justified

Never invent quantitative non-functional requirements (NFRs):

- **No Arbitrary Limits**: Do not invent execution time, latency, throughput, scalability, concurrency, availability,
  or memory targets without concrete justification.
- **No Speculative Mechanisms**: Do not add retry queues, caches, or complex fault-tolerance layers unless required by
  approved constraints.
- **Justification Workflow** (when performance or another NFR is a real concern):
  1. Explain the concern and its operational context.
  2. Check whether a concrete constraint currently exists.
  3. Propose a target only if technically justified.
  4. Get explicit user approval before adding a quantitative target to acceptance criteria.
- **Core Safeguards Intact**: Never sacrifice basic correctness, required security, or data integrity for simplicity,
  but keep safeguards proportional to real risk.

*Rationale: Invented NFRs trigger premature optimization and unnecessary infrastructure.*

### VI. Incremental Development and Avoidance of Overengineering

Each feature increment MUST deliver the smallest coherent, useful, and verifiable slice:

- **Present Needs First**: Build what is needed now, not everything that might be needed later.
- **No Speculative Consumers**: Do not build generalized abstractions for hypothetical future callers.
- **No Unused Extension Points**: Do not add config options or hooks without current, verifiable use.
- **Evolvable Over Anticipatory**: Prefer clean designs that are easy to change later over rigid designs that try to
  predict every future requirement.
- **Proportional Maturity**: Keep architecture and operational complexity proportional to feature maturity.
- **Deferred Improvements**: Record future ideas separately (e.g., backlog notes) outside active scope.
- **Balanced Delivery**: Avoid both overengineering and sloppy shortcuts that hurt correctness or maintainability.

*Rationale: Iterative delivery reduces work-in-progress and prevents wasted effort on unused abstractions.*

### VII. Specification and Planning Discipline

Every SpecKit phase MUST enforce scope, simplicity, and schematic clarity:

- **Across All Phases (`/speckit-*`)**:
  - Write all artifacts in simple, accessible English using a schematic outline format without losing detail (see
    Principle XX).
- **Specification & Clarification (`/speckit-specify`, `/speckit-clarify`)**:
  - Describe required behavior (WHAT and WHY) without prescribing unapproved implementation details.
  - Do not expand functional scope to cover hypothetical scenarios.
  - Do not add quantitative NFRs without user approval.
  - List assumptions and optional proposals separately from approved requirements.
  - Make acceptance criteria concise, schematic, precise, and testable.
- **Planning & Task Generation (`/speckit-plan`, `/speckit-tasks`)**:
  - Pick the least complex architecture that satisfies approved specs.
  - Do not add mechanisms that solve unapproved requirements.
  - Avoid unnecessary interfaces, adapters, orchestration layers, and config options.
  - Keep tasks strictly focused on delivering approved behavior.
- **Implementation & Review (`/speckit-implement`, `/speckit-analyze`)**:
  - Validate implementation strictly against approved requirements.
  - Cover primary flows, relevant edge cases, and meaningful failures in tests without introducing unapproved scope.
  - Flag scope expansion or extra complexity during review rather than implementing it silently.
  - Remove obsolete or redundant complexity when practical.

*Rationale: Keeps quality, simplicity, readability, and scope aligned from spec to code review.*

### VIII. Constitution Enforcement and Decision Priority

Evaluate every design and requirement decision against 6 questions:

1. Is it required by an explicit, approved requirement?
2. Is it essential for correctness, security, or data integrity?
3. Does it solve a realistic problem in the current scope?
4. Is there a simpler alternative?
5. Does it add complexity disproportionate to its benefit?
6. If it adds new scope, has the user approved it?

If a decision cannot be justified by these questions, omit it or present it as an optional proposal.

**Priority Order When Principles Conflict**:

1. **Correctness, essential security, and data integrity**.
2. **Explicit user-approved requirements**.
3. **Simplicity and long-term maintainability**.
4. **Reasonable extensibility backed by current needs**.
5. **Optional enhancements and speculative future capabilities**.

Never use simplicity to excuse bugs, and never use hypothetical bugs to excuse unnecessary complexity.

*Rationale: Clear decision questions and an explicit priority hierarchy resolve trade-offs consistently.*

### IX. Python 3.12 & Poetry Standard Environment

- Manage the project environment exclusively with **Poetry** targeting **Python 3.12** (`>=3.12,<3.14`).
- Execute all tools (testing, linting, formatting, type checking, and scripts) via `poetry run` or inside an active
  Poetry virtual environment.
- Global ad-hoc package installs and untracked dependencies are strictly prohibited.

*Rationale: Guarantees identical, hermetic environments across developer machines and CI pipelines.*

### X. Mandatory Quality Gates & Static Verification

All code and documentation changes MUST pass every static gate before merging:

- **Formatting & Imports**:
  - **Black**: 120-character line limit, target `py312`.
  - **isort**: `profile = "black"`, line length 120, treating `mcp` as third-party so it never shadows local `src/mcp`.
- **Linting & Docstrings**:
  - **Ruff**: Rules `E, W, F, I, C, B`, 120-character line limit.
  - **pydocstyle**: Google docstring convention on all public modules, classes, and functions.
- **Static Type Checking**:
  - **Mypy (Strict Mode)**: `python_version = "3.12"`, `disallow_untyped_defs = true`, `check_untyped_defs = true`,
    `warn_return_any = true`.
  - Add type hints to all function parameters and return values; avoid `Any` unless technically unavoidable.
- **Complexity & Maintainability**:
  - **Xenon**: Max absolute Grade D (`<= 30`), max module average Grade B (`<= 10`), max total average Grade B
    (`<= 10`).
  - Refactor functions longer than 30 lines; target complexity `<= 5` (refactor when `> 7`, tighten progressively
    from Grade D toward Grade C and Grade B).
  - **Radon**: Track Cyclomatic Complexity (CC) and Maintainability Index (MI).
- **Duplication Detection**:
  - **jscpd**: Minimum 50 tokens and 5 lines threshold.
- **Security & Vulnerability Scanning**:
  - **Bandit** (`-q -ll`), **Semgrep** (`p/ci`), **GitHub CodeQL**, **detect-secrets** (`.secrets.baseline`), and
    **pip-audit** (`.pip-audit-ignore`).
- **Repository & File Hygiene**:
  - Block trailing whitespace, missing final newlines, merge conflict markers, debug statements (`breakpoint()`,
    `pdb`), and added files `> 2000 KB` (`check-added-large-files`).
  - **codespell**: Spell check across code, tests, and docs.
  - **check-yaml & yamllint**: Validate YAML syntax and style.
  - **markdownlint-cli2**: Enforce Markdown style and 120-character line limit (`.markdownlint.json`).
- **Developer Certificate of Origin (DCO)**:
  - Every commit MUST include `Signed-off-by: Full Name <email>` (`check-signed-off-by` hook and CI check).
- **Approved Exclusion Scopes**:
  - Linters and gates respect project exclusion boundaries: `pocs/`, local IDE agent configs (`.agents/`, `.cursor/`,
    etc.), and `.specify/`.

*Rationale: Automated static gates catch bugs, complexity spikes, and security flaws before code runs.*

### XI. LLM Provider Independence & Isolation

Keep LLM providers replaceable and isolated from domain logic, data models, and storage:

- **Modular LLM Mediation**: Route LLM calls through modular abstractions (such as DSPy LM interfaces).
- **Atomic Prompt Isolation (DSPy)**:
  - Do not scatter ad-hoc prompt strings or string concatenations inside business logic.
  - Formalize prompts declaratively using atomic DSPy Signatures and Modules (`dspy.Signature`, `dspy.Predict`,
    `dspy.ChainOfThought`).
- **Config-Driven Backends**:
  - Configure router and generator model backends (AWS Bedrock, Google Gemini, Anthropic Claude) in
    `config/vatuta.yaml` without code changes.
- **Domain Decoupling**:
  - Adding, swapping, or tuning an LLM provider or prompt strategy MUST NOT break domain models or pipelines.

*Rationale: Prevents vendor lock-in and prompt drift while keeping business logic clean and testable.*

### XII. Determinism, Cost/Value Pragmatism & Core Validation

Balance strict determinism against pragmatic LLM usage based on error tolerance and cost/value:

- **Zero-Tolerance Operations (Deterministic)**:
  - Report calculations, financial metrics, permission checks, and schema parsing MUST be 100% deterministic and
    unit-tested because error tolerance is zero.
- **Heuristic & Error-Tolerant Tasks (LLM Allowed)**:
  - Where false positives/negatives have low operational impact (semantic classification, policy violation heuristics,
    advisory checks), an LLM solution MAY be used if flexibility and delivery value outweigh token and latency costs.
- **Deterministic Core Rules**:
  - Validate all domain inputs and configs with Pydantic models.
  - Parse retrieval constraints (dates, source filters, ticket keys) into deterministic query filters.

*Rationale: Uses deterministic code where exactness is mandatory and LLMs where semantic judgment adds value.*

### XIII. Provenance, Lineage & Fact-Conclusion Separation

The origin, transformations, and decision criteria of all facts and conclusions MUST be traceable:

- **Source Provenance & Grounding (RAG)**:
  - Every synthesized statement MUST cite its originating sources (source ID, document ID, title, URL, timestamp).
  - Ingestion and chunking pipelines MUST preserve this metadata.
- **Fact Lineage (Reports)**:
  - Treat every discrete metric or data point as an auditable Fact with traceable raw origin, transformations, and
    aggregations.
- **Evaluative Transparency (Policy Reviews)**:
  - Expose raw evidence origin, selection transformations, and the exact decision criteria or rules used to reach each
    compliance verdict.
- **Separate Evidence, Reasoning, and Conclusions**:
  - Keep raw factual evidence architecturally distinct from internal reasoning traces (Chain of Thought, router
    rationales) and final verdicts.
- **Honest Bounds & Non-Speculation**:
  - Explicitly report when data or context is missing or incomplete instead of guessing or extrapolating.

*Rationale: Transparency over facts and decision rules builds trust, enables audits, and prevents hallucinations.*

### XIV. Zero-Trust, Container Isolation & Least Privilege

External tools, connectors, and Model Context Protocol (MCP) servers MUST run under Zero-Trust isolation:

- **Zero-Trust for External Elements**:
  - Treat all external tools, MCP servers, third-party APIs, and remote data as untrusted.
  - Continuously validate, bound, and sanitize all inputs, capabilities, and returned payloads.
- **Isolated Container Execution**:
  - Execute external tools, plugins, and MCP servers inside isolated, ephemeral sandbox containers that protect the
    host filesystem and OS.
- **Strict Least-Privilege Policy (PoLP)**:
  - **Default-Deny Resources**: Deny network egress, filesystem writes, OS capabilities, and extra resources by
    default; grant only explicit, validated exceptions.
  - **Whitelisted Tools Only**: Expose only explicitly allowed tools and operations required for the task.
  - **Immutable Artifacts**: Pin container images and tool dependencies to immutable cryptographic digests.

*Rationale: Ensures buggy or compromised external tools cannot escape to the host or exfiltrate data.*

### XV. Safeguards for Side-Effecting Operations

Any operation that changes state or writes to external services MUST include defensive safeguards:

- **Idempotency & Checkpoints**: Persistent workflows MUST be idempotent and finish in known states.
- **Caching**: Data MAY be cached to reduce redundant network and compute calls.
- **Read-Only Default**: External tool definitions MUST default to `read_only: true`.
- **Dry-Run Support**: Tools SHOULD support a `DRY-RUN` mode for safe auditing.
- **Approval & Audit**: Destructive, mutating, or external write actions MUST require validation, audit logging, and
  explicit user approval before execution.
- **Atomicity**: Multi-step operations SHOULD stop immediately if a preceding step fails.

*Rationale: Protects production systems (Jira, Slack, GitHub) from unintended modifications.*

### XVI. Behavioral Preservation & Public Contract Stability

Do not break existing public CLI commands or configuration schemas unintentionally:

- Maintain backwards compatibility for `vatuta` CLI subcommands and `config/vatuta.yaml` schemas.
- Breaking changes MUST be deliberate, reviewed, documented in `RELEASE.md`, and bump the MAJOR version.
- Isolate experimental proofs-of-concept under `pocs/` until ready for production.

*Rationale: Keeps operator workflows and automated scripts reliable across releases.*

### XVII. Comprehensive Test Coverage & CI Compliance

All new features, connectors, and bug fixes MUST include automated tests that pass CI:

- **Unit Tests Required**: Every public function and class MUST have unit tests under `tests/`.
- **Pragmatic TDD & Testing Pyramid**:
  - **Unit Tests (Primary)**: Fast, isolated, deterministic tests covering business logic, parsing, transformations,
    relevant edge cases, and failure handling. Mock or fake external services, LLMs, and containers.
  - **Contract Tests (Boundary Validation)**: Lightweight, self-contained schema and interface checks (MCP tool
    schemas, Pydantic data contracts, Agent Skill script interfaces) running in CI without external services.
  - **Integration Tests (Minimized)**: Add ONLY when providing clear value beyond unit and contract tests.
- **Hermetic, Low-Infrastructure CI**:
  - GitHub Actions tests MUST run fast and deterministically without external credentials, live third-party SaaS,
    databases, or Docker container orchestration unless explicitly justified and approved.
  - Live infrastructure integration tests stay optional and local.
- **CI Gates & Coverage Target**:
  - Keep Python 3.12 and 3.13 matrix tests, SAST scans, and CycloneDX SBOM generation green.
  - Maintain the current 25% legacy coverage floor while progressively raising coverage toward a **90% overall target**
    (new code and critical domain modules SHOULD target `>= 80%` coverage).
- **Exploratory PoC Validation**:
  - Temporary feature demos and exploratory scripts MAY live under `pocs/` outside formal CI suites.

*Rationale: Fast, hermetic unit and contract tests catch regressions reliably without flaky CI infrastructure.*

### XVIII. Standard Protocols & Ecosystem Reuse Over Custom Wheels

Prefer established standards, protocols, and mature open-source libraries over custom code:

- **Reuse Proven Libraries**: Integrate well-maintained, widely adopted libraries rather than writing custom engines.
- **Standard Agent & Tool Protocols**: Use **Model Context Protocol (MCP)** for inter-system tool servers and the
  **Agent Skills standard (`https://agentskills.io/` + PEP 723 inline script metadata)** for reusable agent skills.
- **Data Models**: Use Pydantic v2 and standard format serializers (YAML, JSON, JSON Schema).
- **Core Stack**: Leverage Qdrant, spaCy, Sentence Transformers, Typer, and Rich.
- **Dependency Hygiene**: Audit any new dependency for security, Apache-2.0 compatible licensing, and active
  maintenance.

*Rationale: Reusing standard protocols and mature libraries cuts maintenance cost and security surface area.*

### XIX. IDE & AI-Agent Agnosticism

The project MUST NOT depend on any specific IDE, editor, or proprietary AI coding assistant:

- **CLI-First Operations**: All canonical operations (install, build, test, lint, format, CI) MUST run from standard
  shell and CLI tools (`poetry run`, `just`, `git`).
- **Local Assistant Configs Untracked**: Personal IDE and AI assistant folders (`.agents/`, `.cursor/`, `.claude/`,
  `.gemini/`, `.clinerules/`) and local Spec-Kit state (`.specify/integration.json`, `.specify/integrations/`) are
  local developer assets and MUST NOT be tracked in git. *(Note: Product-delivered Agent Skills shipped by Vatuta, such
  as `skills/vatuta-policy/`, are tracked project deliverables, not personal IDE settings.)*
- **Optional Productivity Tools**: External wrappers (such as RTK or Spec-Kit) are recommended helpers, never mandatory
  build or test prerequisites.

*Rationale: Guarantees frictionless onboarding across any editor or CLI workflow and keeps CI hermetic.*

### XX. Simple English & Schematic Format Without Detail Loss

All project specifications, plans, task breakdowns, design documents, and AI explanations MUST be easy to read:

- **Simple, Accessible English**:
  - Always write in English using short sentences, common everyday words, and direct active voice.
  - Avoid dense academic, legalistic, or overly verbose phrasing so non-native English readers can read quickly and
    clearly.
- **Schematic Outline Structure**:
  - Format documents as structured outlines using bullet points, numbered steps, compact tables, and clear
    `input -> action -> output` summaries rather than walls of formal prose.
- **100% Completeness & Detail Preserved**:
  - Writing simply and schematically MUST NEVER omit functional requirements, edge cases, data fields, error codes,
    stage transitions, or validation rules. Keep every technical detail intact in a compact, scannable layout.

*Rationale: Schematic, plain-English documents are faster to read, easier to review for non-native speakers, and less
prone to hiding critical requirements inside long paragraphs.*

## Security, Isolation & Tool Constraints

1. **Credential Hygiene & Least Privilege**:
   - Store all credentials and API tokens (Jira, Confluence, Slack, GitHub, LLM keys) exclusively in `.env` or secure
     environment variables—never in version control or config files.
   - Use read-only service accounts and tokens by default.
2. **Supply Chain Security & Immutability**:
   - Audit dependencies continuously (`pip-audit`) and lock with `poetry.lock`.
   - New package releases MUST be at least **7 days old** before adoption unless explicitly overridden.
   - Pin container images and external binaries to cryptographic SHA-256 digests (`@sha256:...`), never mutable tags
     like `latest`.
3. **Container Hardening & Ephemeral Execution**:
   - Launch MCP containers with non-root users, dropped capabilities (`--cap-drop=ALL`), read-only root filesystems,
     and memory/CPU limits.
   - Tear down and rebuild containers fresh between sessions so no state persists.
4. **Network Exposure & Verified Endpoints**:
   - Enforce authentication and TLS/HTTPS on any exposed API listener (`just api`) and all outbound API calls.
   - Never bind unauthenticated listeners to public interfaces.
5. **Boundary Validation & Payload Sanitization**:
   - Validate all data crossing trust boundaries (LLMs, external APIs, MCP tools, YAML files) with strict Pydantic
     schemas.
   - Sanitize ingested external content to strip embedded scripts.
6. **LLM Boundaries & Data Minimization**:
   - Strictly separate trusted system prompts from untrusted user/external data to mitigate Prompt Injection.
   - Send only the minimum required data chunks or projections to the LLM.
7. **Graceful Degradation**:
   - Degrade gracefully on external service errors instead of crashing.
8. **Auditability & Safe Telemetry**:
   - Log security-sensitive operations, tool calls, and retrievals with appropriate severity. Redact secrets.
9. **Vulnerability Disclosure Policy**:
   - Maintain `SECURITY.md` with a private channel for coordinated vulnerability disclosure.

## Observability: Tracing, Logging & Metrics

Observability serves both end users and developers across three pillars: **Tracing**, **Logging**, and **Metrics**.

1. **Execution Tracing, Graph Reasoning & Prompt Auditing**:
   - **End-to-End Trace ID**: Every operation or pipeline run MUST generate a unique invocation/trace ID segmented by
     discrete stages or graph nodes.
   - **Internal Decisions & External Calls**: Capture stage inputs, decision branches, data queries, and MCP tool calls
     (inputs, arguments, outputs, errors).
   - **Prompt & Response Persistence**: Save submitted prompts and raw/structured model responses inside the trace for
     auditing, regression testing, and prompt evaluation, with strict redaction of credentials, tokens, and secrets.
2. **Logging Principles & Standards**:
   - **User Progress**: Keep users informed during long or multi-step runs without noisy log dumps.
   - **Developer Diagnostics**: Log state transitions, causal sequences, and errors clearly for fast debugging.
   - **Module-Level Logger**: Define `logger = logging.getLogger(__name__)` in every logging module.
   - **Severity Discipline**: Use `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` consistently.
   - **Lazy Formatting Only**: Always use printf-style lazy formatting (`logger.info("Loaded %d items", count)`),
     **NEVER f-strings**, inside logging calls.
   - **Centralized Setup**: Configure logging only at application entry points (`setup_logging()` with
     `config/logging.yaml`); never call `logging.basicConfig()` at module level.
   - **Subprocess Stream Capture**: Route external process/container `stderr` streams into structured logging via
     `LoggerWriter(logging.getLogger(f"external.mcp.{server_name}"))`.
   - **Secret Redaction**: Never log tokens, passwords, or other credentials.
3. **Console Output & Interactive User Experience (UX)**:
   - **Rich CLI Output**: Use Rich tables, panels, progress bars, colors, and emoticons in interactive CLI views
     (`src/client/`).
   - **Stream Separation**: Keep diagnostic logs/progress (`stderr`) separate from interactive UI and machine-readable
     JSON output (`stdout`). Never use raw `print()` in domain modules.
4. **Metrics & Telemetry (Prometheus / OpenTelemetry)**:
   - Use standard counters, gauges, and histograms with labeled dimensions.
   - Measure user workflows, stage durations, error rates, and model token usage to guide architectural and prompt
     benchmarking with empirical data.

## Development Workflow & Quality Gates

1. **Local Pre-Flight Checks**:
   - Run `just check` (`just format`, `just lint`, `just test`) and `pre-commit run --all-files` before committing.
2. **Code Structure & Complexity Rules**:
   - Keep functions small and focused: target complexity `<= 5`, refactor if `> 7`, hard fail if `> 30`.
   - Refactor functions longer than 30 lines into smaller helpers.
   - Define at most **one class per file**.
   - Follow PEP 8 with a **120-character max line length** for both Python code and Markdown docs.
3. **Import Ordering**:
   - Standard library -> third-party packages (`mcp` treated as third-party) -> local `src` imports.
4. **Commit Discipline**:
   - Include `Signed-off-by: Full Name <email>` on every commit (`git commit -s`).
5. **Release Requirements** (execute in lockstep):
   - **Update `README.md`**: Reflect new features, commands, skills, or config changes.
   - **Update `RELEASE.md`**: Document changes under a version header and release date (`Added`, `Changed &
     Refactored`, `Fixed`, `Removed`, `Security`).
   - **Bump Version**: Update `[project] version = "X.Y.Z"` in `pyproject.toml` (Semantic Versioning) and tag in git.

## Governance

This Constitution is Vatuta's binding architectural and workflow policy. It supersedes ad-hoc practices.

**Core Governing Rule**: Build the simplest correct, maintainable solution for requirements that actually exist, and
document it in simple, schematic English without losing detail.

- **SpecKit Workflow Application**:
  - Apply all constitutional principles across `/speckit-specify`, `/speckit-clarify`, `/speckit-plan`,
    `/speckit-tasks`, `/speckit-implement`, `/speckit-analyze`, and `/speckit-converge`.
  - Keep extra requirements as optional proposals until explicitly approved by the user.
  - Document and obtain user approval for any exception that materially increases complexity.
  - Do not reinterpret user-approved decisions without confirmation.
- **Amendment Procedure**:
  - Any change to core principles, security rules, quality gates, or release procedures requires an explicit proposal,
    justification, and update to affected `docs/` files.
  - Review and remove the temporary top-of-file HTML `Sync Impact Report` comment before merging constitution commits.
- **Versioning Policy**:
  - **MAJOR**: Incompatible governance changes, principle removals, or fundamental policy redefinitions.
  - **MINOR**: New principles, new operational sections, or material expansion of rules.
  - **PATCH**: Wording clarifications, formatting improvements, and typo fixes.
- **Compliance Review**:
  - All pull requests, code reviews, releases, and CI runs MUST enforce compliance with this Constitution.

**Version**: 3.0.0 | **Ratified**: 2026-09-11 | **Last Amended**: 2026-11-21
