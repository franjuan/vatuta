<!--
Sync Impact Report
- Version change: 1.14.0 → 1.15.0 (Minor Version Bump)
- Modified principles & sections:
  - Principles I-XI re-indexed to IX-XIX, preserving existing environment, quality, security, and runtime policies.
  - Principle IV updated to "Determinism, Cost/Value Pragmatism & Core Validation" to integrate with Simplicity rules.
  - Governance section updated with core governing rule, SpecKit workflow governance, and documented justification.
- Added sections:
  - Core Principles I-VIII added: Simplicity as a First-Class Architectural Principle, Requirements Traceability and
    Scope Control, AI Initiative with Human Approval, Proportional Edge-Case Handling, Non-Functional Requirements
    Must Be Justified, Incremental Development & Anti-Overengineering, Specification & Planning Discipline, and
    Constitution Enforcement & Decision Priority.
- Removed sections: None
- Follow-up TODOs: None
-->

# Vatuta Constitution

**Core Philosophy**: Build the simplest solution that correctly satisfies current requirements, while keeping future
evolution reasonably possible. Complexity must be justified, not assumed.

These principles govern all SpecKit phases—including specification, clarification, planning, task generation,
implementation, and review—as well as general codebase architecture.

## Core Principles

### I. Simplicity as a First-Class Architectural Principle

Simplicity is a fundamental quality attribute, not merely a preference for prototypes:

- **Fewer Moving Parts**: Solutions MUST prefer fewer components, abstractions, dependencies, configuration options,
  and execution paths.
- **No Premature Generalization**: Developers and AI assistants MUST avoid premature generalization, speculative
  extensibility, and unnecessary architectural layers.
- **No Cargo-Culting**: Solutions MUST NOT introduce design patterns, frameworks, or infrastructure solely because
  they represent industry best practices.
- **Convention Reuse**: Implementations MUST prefer existing project conventions and components over introducing
  new ones.
- **Justified Complexity**: Every significant increase in complexity MUST have a concrete justification linked to an
  actual, approved requirement.
- **Pragmatic Comprehension**: Designs MUST optimize for readability, maintainability, testability, and ease of
  understanding, not theoretical completeness.
- **Simplicity by Default**: When multiple solutions satisfy the requirements, the simplest one MUST be chosen unless
  a more complex alternative offers a demonstrable and relevant benefit.

*Rationale: Complexity compounds over time, increasing cognitive load, bug frequency, and maintenance costs.
Prioritizing simplicity keeps the system agile, reliable, and easily comprehensible.*

### II. Requirements Traceability and Scope Control

Every requirement MUST originate from an identifiable source. Requirements MUST be classified into three distinct
categories:

- **A. Explicit Requirements**: Directly expressed by the user or established in previously approved project
  specifications. These form the authoritative, mandatory scope.
- **B. Essential Inferred Requirements**: Not explicitly stated, but necessary for correctness, consistency,
  security, or successful execution. The AI MUST explain their necessity and request confirmation when they materially
  affect scope, complexity, or observable behavior.
- **C. Optional Recommended Requirements**: Potential improvements, enhancements, optimizations, or additional
  capabilities that are not required by the current use case. These MUST be presented strictly as proposals and
  MUST NOT be incorporated without explicit user approval.
- **No Silent Promotion**: The AI and contributors MUST NOT silently promote recommendations, assumptions, or
  potential edge cases into mandatory requirements.

*Rationale: Uncontrolled scope expansion dilutes core deliverables and introduces unwanted complexity. Categorizing
requirements maintains clear boundaries and ensures the user retains full governance.*

### III. AI Initiative with Human Approval

The AI assistant is expected to actively identify omissions, risks, ambiguities, and improvement opportunities;
however, identifying a potential requirement does not authorize adding it:

- **Clear Distinction**: The AI MUST clearly distinguish verified requirements from advisory recommendations.
- **Transparent Trade-offs**: The AI MUST explain the practical consequences of including or excluding significant
  recommendations.
- **Approval Gate**: The AI MUST request explicit user approval before introducing additional scope or substantial
  complexity.
- **Consolidated Queries**: Related questions MUST be consolidated into structured queries to avoid excessive,
  burdensome clarification cycles.
- **Trivial Details Autonomous**: The AI MUST NOT ask for confirmation of trivial implementation details that can be
  resolved through existing project conventions.
- **Reversible Defaults**: When uncertainty remains and no material risk is involved, the AI MUST prefer the simpler,
  reversible alternative.
- **Core Axiom**: *The AI recommends; the user decides on scope.*

*Rationale: Proactive AI analysis provides valuable technical insight while keeping architectural and scope decisions
strictly under human authority.*

### IV. Proportional Edge-Case Handling

Exhaustive edge-case coverage MUST NOT be confused with engineering quality. Edge cases MUST be classified and handled
according to three distinct tiers:

- **Essential Cases**: The primary workflows and core operating conditions necessary for correct execution. MUST be
  fully implemented and verified.
- **Relevant Edge Cases**: Realistic situations whose omission would cause meaningful functional errors, security
  vulnerabilities, or data integrity issues. MUST be implemented and verified.
- **Speculative Edge Cases**: Hypothetical or exceptionally unlikely situations without demonstrated relevance.
  MUST NOT be implemented with elaborate handling without explicit user approval.
- **Architecture Stability**: A single unlikely scenario MUST NOT dictate the architecture of an otherwise simple
  solution unless its verified consequences justify doing so.
- **Simple Error Handling**: Systems MUST favor simple, explicit error handling over generalized recovery mechanisms
  when the latter are unnecessary.

*Rationale: Defending against improbable or speculative failure modes distorts architectures and bloats codebases with
fragile, untested recovery paths.*

### V. Non-Functional Requirements Must Be Justified

Quantitative non-functional requirements (NFRs) MUST NEVER be invented. Arbitrary constraints or targets MUST NOT be
introduced without concrete, documented justification:

- **No Arbitrary Limits**: Arbitrary execution time limits, latency targets, throughput goals, scalability
  objectives, concurrency targets, availability goals, or resource consumption limits MUST NOT be assumed.
- **No Speculative Mechanisms**: Retry queues, caching layers, or complex fault-tolerance mechanisms MUST NOT be added
  unless specifically required by approved constraints.
- **Justification Workflow**: When performance or another non-functional attribute is of legitimate concern:
  1. Explain the potential concern and its operational context.
  2. Determine whether a concrete constraint currently exists.
  3. Propose an appropriate requirement only if technically justified.
  4. Obtain explicit user approval before incorporating a new quantitative target into acceptance criteria.
- **Core Safeguards Intact**: Basic correctness, necessary security safeguards, and data integrity MUST NEVER be
  sacrificed in the name of simplicity; their implementations MUST nevertheless remain proportional to actual risks.

*Rationale: Unjustified non-functional targets trigger premature optimization, complex asynchronous machinery, and
unnecessary infrastructure dependencies.*

### VI. Incremental Development and Avoidance of Overengineering

Each feature increment MUST deliver the smallest coherent, useful, and verifiable increment:

- **Present Needs First**: Implementations MUST deliver what is needed now, not everything that might be needed later.
- **No Speculative Consumers**: Generalized solutions MUST NOT be built for hypothetical future consumers.
- **No Unused Extension Points**: Configuration parameters or extension hooks MUST NOT be introduced without current,
  verifiable justification.
- **Evolvable Over Anticipatory**: Solutions MUST prefer designs that can evolve cleanly over architectures that
  attempt to anticipate every future requirement.
- **Proportional Maturity**: Functionality, architecture, and operational complexity MUST remain proportional to the
  maturity of the feature.
- **Deferred Improvements**: Future enhancements SHOULD be documented separately (e.g., in backlog notes) without
  being included in the active implementation scope.
- **Balanced Delivery**: Both overengineering and shortcuts that would make the current implementation incorrect or
  difficult to maintain MUST be avoided.

*Rationale: Iterative delivery minimizes work in progress, accelerates validation, and prevents wasted effort on
speculative abstractions that may never be used.*

### VII. Specification and Planning Discipline

Every SpecKit workflow phase MUST adhere to strict scope, complexity, and verification discipline:

- **During Specification and Clarification (`/speckit-specify`, `/speckit-clarify`)**:
  - The specification MUST describe required behavior without unnecessarily prescribing implementation details.
  - Functional scope MUST NOT be expanded to cover hypothetical scenarios.
  - Non-functional acceptance criteria MUST NOT be added without concrete justification.
  - Significant assumptions and optional recommendations MUST be surfaced separately.
  - The specification MUST make explicit which requirements have been approved by the user.
  - Acceptance criteria MUST be concise, precise, and verifiable.
- **During Planning and Task Generation (`/speckit-plan`, `/speckit-tasks`)**:
  - Architecture plans MUST select the least complex solution that satisfies approved specifications.
  - Plans MUST NOT introduce implementation mechanisms that solve unapproved requirements.
  - Unnecessary interfaces, adapters, orchestration layers, and configuration options MUST be avoided.
  - Task breakdowns MUST remain strictly focused on delivering approved behavior.
  - Scope MUST NOT be silently expanded to achieve theoretical completeness.
- **During Implementation and Review (`/speckit-implement`, `/speckit-analyze`)**:
  - Implementation correctness MUST be validated strictly against approved requirements.
  - Test suites MUST cover primary workflows, relevant edge cases, and meaningful failure conditions.
  - Tests MUST NOT effectively introduce new, unapproved requirements.
  - Scope expansion or unnecessary complexity MUST be flagged during review rather than implemented automatically.
  - Obsolete or redundant complexity MUST be removed when reasonably possible.

*Rationale: Enforces quality, simplicity, and scope discipline at every milestone from initial specification to code
review.*

### VIII. Constitution Enforcement and Decision Priority

The principles of this Constitution are binding architectural and specification constraints. Every design and
requirement decision MUST be evaluated against six questions:

1. Is this required by an explicit, approved requirement?
2. Is it essential for correctness, security, or data integrity?
3. Does it solve a realistic problem in the current scope?
4. Is there a simpler alternative?
5. Does it introduce complexity disproportionate to its benefit?
6. If it adds new scope, has the user approved it?

If a requirement or design decision cannot be justified through these questions, it MUST be omitted or presented as
an optional proposal.

When constitutional principles appear to conflict, the following priority order MUST strictly govern:

1. **Correctness, essential security, and data integrity**.
2. **Explicit user-approved requirements**.
3. **Simplicity and long-term maintainability**.
4. **Reasonable extensibility backed by current needs**.
5. **Optional enhancements and speculative future capabilities**.

Simplicity MUST NOT be used to excuse incorrectness, nor MAY hypothetical correctness concerns be used as an excuse
for unnecessary complexity.

*Rationale: Explicit decision questions and a clear hierarchy resolve ambiguity, ensuring consistent architectural
trade-offs across the team.*

### IX. Python 3.12 & Poetry Standard Environment

The project environment MUST be managed exclusively using Poetry with Python 3.12 as the primary target runtime
(`>=3.12,<3.14`). All tool invocations (testing, linting, formatting, type checking, and development scripts) MUST
be executed via `poetry run` or within an active Poetry virtual environment. Ad-hoc global package installations or
untracked dependencies are strictly prohibited.

*Rationale: Ensures identical, hermetic runtime and development environments across developer machines and automated
CI pipelines, eliminating dependency drift and environment contamination.*

### X. Mandatory Quality Gates & Static Verification

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

### XI. LLM Provider Independence & Isolation

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

### XII. Determinism, Cost/Value Pragmatism & Core Validation

Architecture decisions MUST balance strict determinism against pragmatic cost/value trade-offs and simplicity:

- **Cost/Value Evaluation for LLM vs. Determinism**: The applicability of an LLM versus a deterministic implementation
  MUST be decided through a cost/value and operational impact analysis of the real-world use case:
  - *Zero-Tolerance Operations*: Calculations in reports, financial metrics, permission boundaries, and schema parsing
    MUST be deterministic and verifiable by unit tests because error tolerance is zero.
  - *Heuristic & Error-Tolerant Tasks*: Where false positives or false negatives carry low operational impact
    (e.g., policy violation heuristics, semantic classification, advisory checks), an LLM solution MAY be justified
    if delivery speed, flexibility, and business value exceed the operational and token costs.
- **Deterministic Core Rules**: Domain inputs and configurations MUST be validated using Pydantic models.
  Retrieval query constraints (dates, source filters, ticket keys) MUST be parsed into deterministic query filters.

*Rationale: Blindly applying LLMs to exact computations leads to hallucinations, while dogmatically building complex
deterministic engines for fuzzy heuristic problems leads to unmaintainable bloat. Grounding decisions in cost/value and
simplicity ensures robust, economical, and agile architecture.*

### XIII. Provenance, Lineage & Fact-Conclusion Separation

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

### XIV. Zero-Trust, Container Isolation & Least Privilege

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

### XV. Safeguards for Side-Effecting Operations

Any operation that alters state, writes to external services, or triggers side effects MUST provide robust safeguards:

- Persistent workflows MUST be idempotent, leveraging checkpoints and finishing in known states to ensure
  data integrity.
- Data MAY be cached to optimize network, processing, and data utilization.
- External tool definitions MUST default to read-only operation (`read_only: true`).
- Tools SHOULD support a DRY-RUN mode to allow process auditing without impacting persistent data.
- Destructive, state-mutating, or externally visible actions MUST support validation, audit logging, and explicit user
  approval before execution.
- Multi-step operations SHOULD guarantee atomicity. Subsequent steps MUST NOT execute unless preceding steps complete
  successfully.

*Rationale: Autonomous agent capabilities interacting with production tooling (e.g., Jira, Slack, Git) risk
unintended modification or disruption without defensive operational safeguards.*

### XVI. Behavioral Preservation & Public Contract Stability

Existing public behavior, command-line interfaces, and configuration contracts MUST NOT be changed unintentionally:

- The public CLI surface (`vatuta` subcommands) and `config/vatuta.yaml` schemas MUST maintain backwards compatibility.
- Breaking modifications MUST be deliberate, reviewed, documented in `RELEASE.md`, and accompanied by a MAJOR version
  increment adhering to the formal release process.
- Experimental features or proofs-of-concept MUST be isolated under `pocs/` or clearly marked as non-production
  integrations until stabilized.

*Rationale: Predictable behavior ensures that automated workflows, continuous ingestion, and operator usage remain
dependable across updates.*

### XVII. Comprehensive Test Coverage & CI Compliance

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
- **Exploratory & Feature Validation Tests**: Tests aimed at validating new features, interactive workflows, or
  iterative development proofs-of-concept MAY be placed under `pocs/`. These validation scripts confirm correct
  feature behavior during active development but are not strictly required to be integrated into formal unit testing
  suites or CI regression pipelines.

*Rationale: Automated regression testing is the principal defense against subtle functional breakage across complex
ingestion and retrieval components. An ambitious coverage trajectory ensures long-term system reliability.*

### XVIII. Standard Protocols & Ecosystem Reuse Over Custom Wheels

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

### XIX. IDE & AI-Agent Agnosticism

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

1. **Credential Hygiene & Least Privilege**: Credentials and API tokens (Jira, Confluence, Slack, LLM keys) MUST reside
   exclusively in `.env` or secure environment variables. Secrets MUST NEVER be committed to configuration files or
   source control. Service accounts and API tokens MUST be granted the absolute minimum permissions required
   (e.g., read-only scopes by default).
2. **Supply Chain Security & Immutability**: All third-party dependencies MUST be continuously audited for
   vulnerabilities (e.g., `pip-audit`). Lockfiles (`poetry.lock`) MUST be used for deterministic builds. To mitigate
   supply chain attacks, newly published dependency versions MUST have a minimum age of 7 days before adoption unless
   explicitly overridden. Base container images and external executables MUST be pinned to specific cryptographic
   hashes (e.g., SHA-256 digests) rather than mutable tags (like `latest`).
3. **Container Hardening & Ephemeral Execution**: MCP containers and sandboxes MUST be launched with strict process
   constraints, non-root users, and memory quotas. Furthermore, execution environments MUST be strictly ephemeral and
   stateless; they MUST be torn down and rebuilt fresh between distinct sessions to prevent state contamination and
   payload persistence.
4. **Network Exposure & Verified Endpoints**: Any exposed API endpoints or network listeners (e.g., `just api`) MUST
   enforce strict authentication and TLS/HTTPS encryption. Binding to public interfaces without authorization is
   strictly prohibited. Likewise, all outbound calls to external endpoints, services, and APIs MUST exclusively use
   verified and secure channels (TLS/HTTPS).
5. **Boundary Validation & Payload Sanitization**: All data crossing trust boundaries (from LLMs, external APIs, or
   MCP tools) MUST be strictly validated against formal schemas (e.g., Pydantic). Untyped parsing of external payloads
   is prohibited. Ingested external content MUST be sanitized to neutralize embedded scripts.
6. **LLM Boundaries & Data Minimization**: Interactions with LLMs MUST strictly demarcate trusted system prompts from
   untrusted user/external data to mitigate Prompt Injection. The retrieval pipelines MUST adhere to strict data
   minimization, sending only the precise document chunks required to fulfill the task.
7. **Rate Limiting & Graceful Degradation**: The system MUST implement strict rate limiting, usage quotas, and circuit
   breakers for external API and LLM invocations to prevent "Denial of Wallet" loops. External service failures MUST
   trigger graceful degradation or fallback mechanisms rather than causing unhandled crashes.
8. **Auditability & Observability**: Security-sensitive operations, tool invocations, and retrieval actions MUST be
   logged with appropriate severity levels and monitored via telemetry without logging sensitive payloads or
   credentials.
9. **Vulnerability Disclosure Policy**: The project MUST maintain a `SECURITY.md` file detailing a coordinated
   vulnerability disclosure process, providing a private channel to report vulnerabilities before public disclosure.

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

**Core Governing Rule**: The goal is not to build the most comprehensive solution imaginable. The goal is to build
the simplest correct, maintainable solution for the requirements that actually exist.

- **SpecKit Workflow Application**:
  - Apply these principles consistently across all SpecKit workflows (`/speckit-specify`, `/speckit-clarify`,
    `/speckit-plan`, `/speckit-tasks`, `/speckit-implement`, `/speckit-analyze`, `/speckit-converge`).
  - Ensure all specification, planning, and task artifacts respect these principles.
  - Treat additional requirements as proposed until explicitly approved by the user.
  - Any exception that materially increases complexity MUST include a documented, user-approved justification.
  - Do not reinterpret existing user-approved decisions without confirmation.
  - Preserve existing constitution principles that do not conflict with these rules.
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

**Version**: 1.15.0 | **Ratified**: 2026-09-11 | **Last Amended**: 2026-10-20
