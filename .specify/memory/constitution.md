<!--
Sync Impact Report
- Version change: 1.13.0 → 1.14.0 (Minor Version Bump)
- Modified principles & sections:
  - Security, Isolation & Tool Constraints: Completely overhauled and expanded to include enterprise-grade security rules: Least Privilege, Supply Chain Security (7-day rule, pinning), Container Hardening, Network Exposure (TLS), Boundary Validation (Pydantic), LLM Boundaries, Rate Limiting, Auditability, and Vulnerability Disclosure.
- Added sections: None
- Removed sections: None
- Follow-up TODOs: None
-->

# Vatuta Constitution

## Core Principles

### I. Python 3.12 & Poetry Standard Environment

The project environment MUST be managed exclusively using Poetry with Python 3.12 as the primary target runtime
(`>=3.12,<3.14`). All tool invocations (testing, linting, formatting, type checking, and development scripts) MUST
be executed via `poetry run` or within an active Poetry virtual environment. Ad-hoc global package installations or
untracked dependencies are strictly prohibited.

*Rationale: Ensures identical, hermetic runtime and development environments across developer machines and automated
CI pipelines, eliminating dependency drift and environment contamination.*

### II. Mandatory Quality Gates & Static Verification

All codebase modifications MUST pass every established static verification check and quality gate prior to merging:

- **Formatting & Import Ordering**: Strict Black formatting (120 character line limit, target Python 3.12) and `isort`
  import classification (`profile = "black"`, treating `mcp` as third-party to prevent shadowing local packages).
- **Linting & Docstrings**: Ruff (rules `E, W, F, I, C, B` with line length 120) and `pydocstyle` strictly enforcing
  the Google docstring convention on all public classes, functions, and modules.
- **Static Type Checking**: Mypy under strict mode (`python_version = "3.12"`, `disallow_untyped_defs = true`,
  `check_untyped_defs = true`, `warn_return_any = true`). Type annotations are required for all function parameters
  and return values; usage of `Any` is prohibited unless technically unavoidable.
- **Complexity & Maintainability**: Xenon cyclomatic complexity enforcement (Grade D max absolute <= 30, Grade B module
  and overall average <= 10; functions exceeding 30 lines MUST be refactored) and Radon CC/MI tracking. While the
  current Grade D baseline is intentionally permissive during active migration, subsequent iterations MUST
  progressively tighten complexity thresholds toward Grade C and Grade B ceilings.
- **Duplication Detection**: `jscpd` copy/paste detector enforcing a threshold of minimum 50 tokens and 5 lines.
- **Security & Vulnerability Scanning**: Bandit SAST (`-q -ll`), Semgrep static analysis (`p/ci`), GitHub CodeQL
  security analysis, secret detection via `detect-secrets` against `.secrets.baseline`, and dependency vulnerability
  scanning via `pip-audit` against `.pip-audit-ignore`.
- **Repository & Code Hygiene**:
  - Pre-commit hooks MUST block trailing whitespace, missing final newlines, merge conflict artifacts, and committed
    Python debug statements (`breakpoint()`, `pdb`).
  - Pre-commit file size bounds MUST reject added files exceeding 2000 KB (`check-added-large-files`).
  - Spell checking MUST pass via `codespell` across all source code, tests, and documentation.
  - Configuration files MUST pass syntax validation via `check-yaml` and style rules via `yamllint`.
  - Documentation MUST satisfy `markdownlint-cli2` style rules (120 character limit configured in `.markdownlint.json`).
- **Developer Certificate of Origin (DCO)**: Every git commit MUST contain a valid `Signed-off-by: Full Name <email>`
  trailer, enforced locally via the `check-signed-off-by` commit-msg hook and remotely in CI PR workflows.
- **Approved Exclusion Scopes**: Quality gates and linters MUST respect documented project exclusion boundaries:
  experimental proof-of-concept code (`pocs/`), local AI coding-agent configurations (`.agents/`, `.cursor/`, etc.),
  and Spec-Kit local integration assets (`.specify/`).

*Rationale: Multi-layered static analysis catches defects, regressions, complexity spikes, and security vulnerabilities
before execution, preserving long-term code quality, security posture, and developer velocity.*

### III. LLM Provider Independence & Isolation

Language model providers MUST remain replaceable and strictly isolated from business logic, data models, and storage:

- **Modular LLM Mediation**: Interactions with LLMs MUST be mediated through modular interfaces (such as DSPy language
  model abstractions).
- **Structured & Atomic Prompt Isolation (DSPy)**: Prompts MUST NOT be implemented as ad-hoc, free-form string
  concatenations or scattered across business logic. Prompt interactions MUST be formalized atomically and
  declaratively through structured DSPy Signatures and Modules (e.g., `dspy.Signature`, `dspy.Predict`,
  `dspy.ChainOfThought`), completely isolating prompt input/output contracts, task specifications, and optimization
  from both the underlying model provider and domain code.
- **Config-Driven Backend Providers**: Router and generator model backends (e.g., AWS Bedrock, Google Gemini,
  Anthropic Claude) MUST be independently configurable via `config/vatuta.yaml` without modifying application source
  code.
- **Resilient Domain Decoupling**: Adding, replacing, or deprecating an LLM provider or tuning prompt strategies MUST
  NOT break domain logic, entity management, or retrieval pipelines.

*Rationale: Decoupling model providers prevents vendor lock-in, enables task-specific model optimization, and shields
the core application from upstream provider API deprecations or disruptions. Leveraging DSPy to isolate prompts
atomically into structured signatures prevents prompt drift, enables systematic compiler optimization, and preserves
clean boundaries between business logic and LLM prompting.*

### IV. Determinism, Cost/Value Pragmatism & Simplicity

Architecture decisions MUST balance strict determinism against pragmatic cost/value trade-offs and simplicity:

- **Cost/Value Evaluation for LLM vs. Determinism**: The applicability of an LLM versus a deterministic implementation
  MUST be decided through a cost/value and operational impact analysis of the real-world use case:
  - *Zero-Tolerance Operations*: Calculations in reports, financial metrics, permission boundaries, and schema parsing
    MUST be deterministic and verifiable by unit tests because error tolerance is zero.
  - *Heuristic & Error-Tolerant Tasks*: Where false positives or false negatives carry low operational impact
    (e.g., policy violation heuristics, semantic classification, advisory checks), an LLM solution MAY be justified
    if delivery speed, flexibility, and business value exceed the operational and token costs.
- **Inherent Simplicity & Requirement Negotiation**: Simplicity is a paramount quality attribute. Engineers SHOULD
  prioritize the simplest viable solution. If an overly complex requirement threatens system maintainability or
  introduces disproportionate fragility, the requirement itself MAY be challenged, reshaped, or simplified subject to
  analyst or stakeholder validation.
- **Deterministic Core Rules**: Domain inputs and configurations MUST be validated using Pydantic models.
  Retrieval query constraints (dates, source filters, ticket keys) MUST be parsed into deterministic query filters.

*Rationale: Blindly applying LLMs to exact computations leads to hallucinations, while dogmatically building complex
deterministic engines for fuzzy heuristic problems leads to unmaintainable bloat. Grounding decisions in cost/value and
simplicity ensures robust, economical, and agile architecture.*

### V. Provenance, Lineage & Fact-Conclusion Separation

The origin, processing transformations, and underlying criteria of all conclusions, insights, and factual assertions
MUST be known, traceable, and communicable across all assistant subsystems and capabilities:

- **Source Provenance & Grounding (e.g., RAG)**: In retrieval-augmented workflows, every synthesized insight or
  statement MUST maintain explicit, verifiable references and citations to its originating factual sources (including
  source identifiers, document IDs, titles, URLs, and timestamps). Ingestion pipelines MUST retain this metadata across
  all chunking operations.
- **Fact Lineage & Processing Traceability (e.g., Reports)**: In analytical outputs, summaries, and generated reports,
  every discrete data point MUST be treated as an auditable Fact. The system MUST maintain complete lineage for each
  fact, enabling full traceability of its raw origin, transformations, aggregations, and intermediate processing.
- **Evaluative Transparency & Decision Criteria (e.g., Policy Reviews)**: When conducting automated evaluations,
  compliance assessments, or policy reviews, the system MUST be capable of exposing the raw data origin, every
  transformation applied, and the precise decision criteria, rules, or rubrics used to reach the final evaluation
  or verdict.
- **Strict Separation of Evidence, Logic, and Inferences**: Factual evidence, internal reasoning traces (e.g., Chain of
  Thought, router rationales, intermediate calculations), and synthesized conclusions MUST be architecturally decoupled
  and distinguishable, ensuring that evidence is not conflated with model deduction.
- **Honest Bounds & Non-Speculation**: Systems and models MUST explicitly acknowledge when context or data is
  insufficient, incomplete, or missing rather than speculating, extrapolating without basis, or generating ungrounded
  assertions.

*Rationale: Vatuta serves as an authoritative source of truth for engineering leadership and operational teams.
Absolute transparency regarding where facts originate, how they are processed, and what criteria produce conclusions
is essential to establish trust, ensure auditability, and eliminate misleading hallucinations.*

### VI. Zero-Trust, Container Isolation & Least Privilege

External tools, data connectors, and Model Context Protocol (MCP) servers MUST operate under an uncompromising
Zero-Trust security model, strict containerized isolation, and least-privilege boundaries:

- **Zero-Trust Philosophy for External Elements**: All external tools, MCP servers, third-party services, and remote
  data sources MUST be treated as inherently untrusted execution targets and potential vulnerability vectors. No
  external component is implicitly trusted; all operations, capabilities, and returned data MUST be subject to
  continuous verification, boundaries, and sanitization.
- **Isolated & Containerized Execution**: External tools, plugins, and MCP servers MUST execute strictly within
  isolated, containerized, and ephemeral sandbox environments. The runtime environment MUST enforce impenetrable
  boundaries that shield the host system, preventing container escapes, unauthorized host filesystem access, and host
  compromise.
- **Strict Least-Privilege Policy (PoLP)**: External components MUST be granted solely the minimum privileges and
  capabilities strictly necessary to fulfill their explicit operational scope:
  - Default-deny resource posture: network egress, filesystem write privileges, operating system capabilities, and
    system resources MUST be denied by default and granted only on an explicitly declared, validated exception basis.
  - Granular tool and operation exposure: Agents MUST only have access to strictly whitelisted tools and operations
    relevant to their authorized tasks.
  - Immutability and provenance verification: Container execution runtimes and tool dependencies MUST be pinned to
    immutable, verifiable artifacts to prevent unauthorized runtime tampering or supply-chain drift.

*Rationale: External tools and protocol servers operate outside the application's immediate trust boundary. Applying
Zero-Trust, containerized isolation, and strict least-privilege ensures that buggy, malicious, or compromised external
components cannot compromise the host environment, exfiltrate sensitive data, or access unauthorized assets.*

### VII. Safeguards for Side-Effecting Operations

Any operation that alters state, writes to external services, or triggers side effects MUST provide robust safeguards:

- Persistent workflows MUST be idempotent, leveraging checkpoints and finishing in known states to ensure data integrity.
- Data MAY be cached to optimize network, processing, and data utilization.
- External tool definitions MUST default to read-only operation (`read_only: true`).
- Tools SHOULD support a DRY-RUN mode to allow process auditing without impacting persistent data.
- Destructive, state-mutating, or externally visible actions MUST support validation, audit logging, and explicit user approval before execution.
- Multi-step operations SHOULD guarantee atomicity. Subsequent steps MUST NOT execute unless preceding steps complete successfully.

*Rationale: Autonomous agent capabilities interacting with production tooling (e.g., Jira, Slack, Git) risk
unintended modification or disruption without defensive operational safeguards.*

### VIII. Behavioral Preservation & Public Contract Stability

Existing public behavior, command-line interfaces, and configuration contracts MUST NOT be changed unintentionally:

- The public CLI surface (`vatuta` subcommands) and `config/vatuta.yaml` schemas MUST maintain backwards compatibility.
- Breaking modifications MUST be deliberate, reviewed, documented in `RELEASE.md`, and accompanied by a MAJOR version
  increment adhering to the formal release process.
- Experimental features or proofs-of-concept MUST be isolated under `pocs/` or clearly marked as non-production
  integrations until stabilized.

*Rationale: Predictable behavior ensures that automated workflows, continuous ingestion, and operator usage remain
dependable across updates.*

### IX. Comprehensive Test Coverage & CI Compliance

New functionality, connectors, and bug fixes MUST be accompanied by automated tests that satisfy all CI requirements:

- Every public function and component MUST have corresponding unit tests under `tests/`.
- External service calls, LLM invocations, and container lifecycles MUST be mocked in unit test suites to ensure
  reliable, hermetic execution.
- CI pipeline suites (matrix testing for Python 3.12 and 3.13, minimum 25% test coverage floor, SAST security reports,
  and CycloneDX SBOM generation) MUST remain fully green.
- **Coverage Evolution Target**: While the baseline CI failure threshold is currently set at a 25% coverage floor to
  accommodate legacy modules, future development iterations MUST progressively increase coverage with the objective of
  getting as close as possible to a 90% comprehensive test coverage target. New code and critical domain modules SHOULD
  target high test coverage (>= 80%) upon introduction.
- **Exploratory & Feature Validation Tests**: Tests aimed at validating new features, interactive workflows, or iterative development proofs-of-concept MAY be placed under `pocs/`. These validation scripts confirm correct feature behavior during active development but are not strictly required to be integrated into formal unit testing suites or CI regression pipelines.

*Rationale: Automated regression testing is the principal defense against subtle functional breakage across complex
ingestion and retrieval components. An ambitious coverage trajectory ensures long-term system reliability.*

### X. Standard Protocols & Ecosystem Reuse Over Custom Wheels

The system architecture MUST prefer established standards, protocols, and mature open-source libraries over custom
reinventions:

- **Prefer Proven Ecosystem Solutions**: If a well-maintained, stable, widely adopted, and suitable library, dependency,
  or service exists with reliable guarantees, integrating it is strictly preferred over writing custom in-house code
  or reinventing existing wheels.
- Inter-system tool communication MUST adopt standard specifications (Model Context Protocol).
- Data models and serialization MUST rely on Pydantic and standard format serializers.
- Vector search, NLP chunking, and CLI interfaces MUST leverage mature libraries (Qdrant, spaCy, Sentence Transformers,
  Typer, Rich) rather than in-house alternatives.
- Any newly introduced third-party dependency MUST be audited for security, license compatibility (Apache 2.0
  compatible), and active maintenance.

*Rationale: Reusing proven ecosystem components reduces maintenance overhead, limits novel security surface area, and
focuses project effort on core RAG capabilities.*

### XI. IDE & AI-Agent Agnosticism

The project MUST NOT depend on or mandate any specific integrated development environment (IDE), editor, or AI coding
assistant:

- Canonical project operations—including dependency installation, builds, testing, linting, formatting, and CI
  pipelines—MUST be fully executable from standard shell and CLI interfaces (`poetry run`, `just`, git)
  without requiring any specific IDE, GUI, or proprietary extension.
- AI assistant configurations (e.g., `.agents/`, `.cursor/`, `.claude/`, `.gemini/`, `.clinerules/`) and
  agent-specific local integration state (such as `.specify/integration.json` and `.specify/integrations/`) are
  strictly local developer assets and MUST NOT be tracked in version control or enforced as prerequisites.
- External assistant tooling, command wrappers (such as RTK), and specification workflows (such as Spec-Kit) are
  recommended productivity enhancements for contributors, but MUST NEVER be mandatory dependencies or gating factors
  for building, testing, or contributing to the codebase.

*Rationale: Universal CLI-first accessibility prevents developer lock-in, guarantees frictionless onboarding across
diverse contributor toolchains (VS Code, Cursor, Claude Code, Antigravity, Neovim, CLI), and ensures CI/CD hermeticity.*

## Security, Isolation & Tool Constraints

1. **Credential Hygiene & Least Privilege**: Credentials and API tokens (Jira, Confluence, Slack, LLM keys) MUST reside exclusively in `.env` or secure environment variables. Secrets MUST NEVER be committed to configuration files or source control. Service accounts and API tokens MUST be granted the absolute minimum permissions required (e.g., read-only scopes by default).
2. **Supply Chain Security & Immutability**: All third-party dependencies MUST be continuously audited for vulnerabilities (e.g., `pip-audit`). Lockfiles (`poetry.lock`) MUST be used for deterministic builds. To mitigate supply chain attacks, newly published dependency versions MUST have a minimum age of 7 days before adoption unless explicitly overridden. Base container images and external executables MUST be pinned to specific cryptographic hashes (e.g., SHA-256 digests) rather than mutable tags (like `latest`).
3. **Container Hardening & Ephemeral Execution**: MCP containers and sandboxes MUST be launched with strict process constraints, non-root users, and memory quotas. Furthermore, execution environments MUST be strictly ephemeral and stateless; they MUST be torn down and rebuilt fresh between distinct sessions to prevent state contamination and payload persistence.
4. **Network Exposure & Verified Endpoints**: Any exposed API endpoints or network listeners (e.g., `just api`) MUST enforce strict authentication and TLS/HTTPS encryption. Binding to public interfaces without authorization is strictly prohibited. Likewise, all outbound calls to external endpoints, services, and APIs MUST exclusively use verified and secure channels (TLS/HTTPS).
5. **Boundary Validation & Payload Sanitization**: All data crossing trust boundaries (from LLMs, external APIs, or MCP tools) MUST be strictly validated against formal schemas (e.g., Pydantic). Untyped parsing of external payloads is prohibited. Ingested external content MUST be sanitized to neutralize embedded scripts.
6. **LLM Boundaries & Data Minimization**: Interactions with LLMs MUST strictly demarcate trusted system prompts from untrusted user/external data to mitigate Prompt Injection. The retrieval pipelines MUST adhere to strict data minimization, sending only the precise document chunks required to fulfill the task.
7. **Rate Limiting & Graceful Degradation**: The system MUST implement strict rate limiting, usage quotas, and circuit breakers for external API and LLM invocations to prevent "Denial of Wallet" loops. External service failures MUST trigger graceful degradation or fallback mechanisms rather than causing unhandled crashes.
8. **Auditability & Observability**: Security-sensitive operations, tool invocations, and retrieval actions MUST be logged with appropriate severity levels and monitored via telemetry without logging sensitive payloads or credentials.
9. **Vulnerability Disclosure Policy**: The project MUST maintain a `SECURITY.md` file detailing a coordinated vulnerability disclosure process, providing a private channel to report vulnerabilities before public disclosure.

## Observability: Tracing, Logging & Metrics

Observability is a fundamental architectural discipline rather than a collection of ad-hoc print statements or raw
telemetry dumps. It fulfills a dual role, deliberately serving two primary stakeholders: the end user and the system
developer. Telemetry MUST be intentional, contextual, and structured across three synergistic pillars: Tracing,
Logging, and Metrics:

1. **Execution Tracing, Graph Reasoning & Prompt Auditing**:
   - **End-to-End Operation & Graph State Tracing**: Every user interaction, workflow run, or pipeline execution MUST
     generate a unified execution trace distinguished by a unique operation or invocation identifier. Within each
     trace, execution MUST be granularly segmented by every discrete step, node, and state transition of the execution
     graph, allowing complete reconstruction of the workflow's state evolution.
   - **Internal Reasoning & External Call Inspection**: Traces MUST capture the full internal reasoning process,
     decision branches, and all interactions with external elements—including data source queries and MCP tool calls
     (recording inputs, arguments, outputs, and error states). This comprehensive traceability ensures developers can
     systematically evaluate internal reasoning and pinpoint exact failure points during debugging and review.
   - **Prompt & Completion Telemetry and Persistence**: Prompts submitted to models and their raw/structured
     responses MUST be captured and persistently stored within the trace context. Retaining prompt histories is
     mandatory to enable quantitative prompt evaluation, benchmark model alternatives, run regression tests, and
     drive continuous prompt optimization while strictly observing credential and personal data redaction.

2. **Logging Principles & Architectural Standards**:
   - **User Transparency & Progress Awareness**: Logging MUST keep users continuously informed of system activity,
     particularly during long-running, asynchronous, or multi-step operations. Systems MUST never behave as opaque or
     frozen black boxes. User progress communication MUST remain timely, intelligible, and reassuring without noise.
   - **Developer Diagnostics & Incident Analysis**: Logging MUST provide developers with comprehensive visibility
     into internal state transitions, causal sequences, operational incidences, errors, and bugs, enabling rapid,
     reliable reproduction, root-cause investigation, and systematic debugging.
   - **Source Attribution & Component Hierarchy**: Every log record MUST explicitly identify its origin. Loggers MUST
     be scoped to their respective component, class, or module hierarchy so callers and causality are unmistakable.
   - **Severity Levels Discipline**: Log statements MUST employ appropriate semantic severity levels (DEBUG, INFO,
     WARNING, ERROR, CRITICAL) consistently across all modules to reflect genuine operational significance.
   - **Lazy Formatting Philosophy**: All logging calls MUST adhere to a lazy formatting philosophy (deferred string
     evaluation) rather than eager string interpolation (e.g., f-strings). String formatting cost MUST only be
     incurred when the target logging level is actively enabled, eliminating unnecessary runtime overhead.
   - **Centralized Configuration**: Logging behavior, formatting, handlers, and filters MUST be configured
     centrally at application entry points. Invoking ad-hoc logging configuration or altering handlers at module or
     class level is strictly prohibited.
   - **External Component & Subprocess Integration**: When Vatuta manages external processes or runtimes (such as MCP
     servers, containerized tools, or external CLI subprocesses), their I/O streams and error channels MUST be captured
     and routed into Vatuta's structured logging system to maintain unified operational context and error handling.
   - **Information Privacy & Credential Redaction**: Neither user-facing nor diagnostic logs may ever compromise
     privacy, leak sensitive personal information, or emit confidential tokens and credentials.

3. **Console Output & Interactive User Experience (UX)**:
   - **Rich Text & Structured Presentation**: Console output destined for end users MUST prioritize clarity,
     visual hierarchy, and readability. Interactive CLIs and clients SHOULD leverage Rich text content—incorporating
     tasteful colors, structured tables, visual panels, and dynamic progress formatting whenever it enhances user
     understanding, status transparency, or overall UX. Emoticons usage is encouraged.
   - **Strict Separation of Diagnostic Logs and UI Output**: Diagnostic system logging MUST remain architecturally
     distinct from user-facing interactive presentation; diagnostic telemetry MUST never pollute interactive console
     interfaces, and presentation formatting MUST NOT replace structured system logs.

4. **Metrics Principles & Telemetry Standards (Prometheus / OpenTelemetry)**:
   - **Standardized Multi-Dimensional Telemetry**: System instrumentation SHOULD follow modern metrics conventions
     compatible with Prometheus and OpenTelemetry standards (such as counters, gauges, histograms, and labeled
     dimensions), enabling consistent metric scraping, dashboarding, and alerting.
   - **User Process & Behavioral Analysis**: Metrics MUST capture and quantify user processes, actions, and end-to-end
     workflows. This empirical telemetry enables deep understanding of how the system is actually utilized, revealing
     adoption patterns, operational friction points, latency bottlenecks, and practical user needs.
   - **Architectural Evaluation & Empirical Benchmarking**: Metrics MUST provide the quantitative foundation for
     evaluating, comparing, and benchmarking alternative architectural solutions, models, and prompt strategies.
     Decisions regarding alternative models, prompt configurations, retrieval designs, and throughput optimizations
     SHOULD be backed by verifiable, empirical metric baselines rather than subjective intuition.

## Development Workflow & Quality Gates

1. **Local Pre-Flight Checks**: Developers SHOULD run `just check` (or `just lint`, `just format`, `just test`) and
   ensure `pre-commit run --all-files` succeeds before submitting code.
2. **Code Structure & Complexity Evolution**:
   - Maintain small, focused functions: target complexity <= 5, refactor if > 7, hard reject if > 30.
   - Progressively tighten project-wide Xenon complexity gates in subsequent iterations, transitioning from the
     current transitional Grade D baseline toward stricter Grade C and Grade B ceilings.
   - Functions longer than 30 lines MUST be refactored into smaller helpers where feasible.
   - Do not define more than one class per file.
   - Adhere strictly to PEP 8 with a 120-character maximum line length for both code and documentation.
3. **Import Ordering**: Use `isort`-compatible ordering: standard library first, third-party packages second (with
   `mcp` explicitly treated as third-party to prevent shadowing), and local `src` imports last.
4. **Commit Discipline**: All commits MUST include `Signed-off-by: Full Name <email>` to comply with DCO verification.
5. **Release Requirements**: Executing a project release requires completing three mandatory operations in lockstep:
   - **Update `README.md`**: Review and update `README.md` to reflect any new features, architectural updates,
     prerequisite modifications, or configuration adjustments introduced since the last release.
   - **Complete `RELEASE.md`**: Document all user-facing changes, enhancements, refactorings, and fixes under a
     dedicated version header with the release date, organized using standard changelog categories (Added, Changed &
     Refactored, Fixed, Removed, Security).
   - **Version Bump**: Increment the project version in `pyproject.toml` (`[project] version = "X.Y.Z"`) in accordance
     with Semantic Versioning rules, and mirror the release version in git tags upon publication.

## Governance

This Constitution is the foundational architectural policy of the Vatuta project. It supersedes all informal
guidelines, documentation conventions, and ad-hoc practices.

- **Amendment Procedure**:
  - Any amendment to core principles, security baselines, quality gates, or release procedures requires an explicit
    proposal, architectural justification, and update to relevant documentation in `docs/`.
  - Temporary Sync Impact Reports MUST be reviewed and validated prior to merging constitution amendments.
- **Versioning Policy**:
  - **MAJOR**: Incompatible governance updates, principle removals, or fundamental policy redefinitions.
  - **MINOR**: Addition of new principles, new operational sections, or material expansion of guidelines.
  - **PATCH**: Clarifications, non-semantic wording improvements, and typo corrections.
- **Compliance Review**:
  - All pull requests, code reviews, releases, and CI pipelines MUST enforce compliance with this Constitution.
  - Deviations or exceptions MUST be rejected unless formally ratified via the amendment process.

**Version**: 1.14.0 | **Ratified**: 2026-09-11 | **Last Amended**: 2026-09-17
