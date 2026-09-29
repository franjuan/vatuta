# Feature Specification: Provider-Independent LLM Abstraction

**Feature Branch**: `001-llm-provider-abstraction`

**Created**: 2026-09-24

**Status**: Draft

**Input**: User description: "Add a provider-independent LLM abstraction so Vatuta can switch between Gemini,
Claude and OpenAI without changing business logic. LLM model should be selected in config/vatuta.yaml.
LiteLLM is recommended although I am open to other solutions. Keep DSPy abstraction"

## Clarifications

### Session 2026-09-24

- Q: How should the scope for non-Gemini providers (Claude and OpenAI) be adjusted given that you do not want to test them with simulated responses? → A: Restrict the initial implementation and testing exclusively to Gemini; defer Claude and OpenAI implementation and configuration to future iterations.
### Session 2026-09-25

- Q: When and how should LiteLLM handle credential and authentication validation? → A: Delegate credential management to LiteLLM without custom per-provider key checks; capture LiteLLM errors within Vatuta and act accordingly.
- Q: How should Vatuta act when capturing LiteLLM errors across startup and runtime? → A: Distinct startup vs runtime handling: halt cleanly with actionable diagnostics on startup config/auth errors; return a user-friendly error message on runtime failures without crashing the session.
- Q: When an LLM provider call fails during execution, how should Vatuta terminate the operation? → A: Abort the operation cleanly: log diagnostics, report an informative error message to the user, and exit the command with an error status; automatic retries and multi-provider failover are out of scope.

### Session 2026-09-26

- Q: How should the new LiteLLM-based configuration schema coexist with the existing `llm_backends` configuration? → A: Replace in-place; the new LiteLLM-native schema replaces the existing `llm_backends` structure entirely, with migration guidance documented in RELEASE.md.
- Q: Should LiteLLM serve as the unified LM backend for both DSPy and LangGraph, or only replace one layer? → A: Unified single initialization path; LiteLLM serves as the LM backend for both DSPy (`dspy.LM`) and LangGraph (`ChatLiteLLM`).
- Q: Should this release include Prometheus metrics for LLM provider calls, or defer instrumentation? → A: Full metrics in this release; include Prometheus counters and histograms for LLM call latency, token usage, and error rates.

### Session 2026-09-27

- Q: How broad and detailed should the initial pre-flight check of configured models be? → A: The initial evaluation must be broad, general, and basic without model-specific custom probes, network pings, or dummy test prompts; it relies strictly on LiteLLM's built-in validation mechanisms (such as model identifier format verification and standard provider environment credential presence).
- Q: How should runtime LLM transient errors be categorized and treated in system logs and execution flow? → A: Any runtime LLM failure that cannot be remediated during execution is treated as a fatal, unrecoverable error: it must be logged at ERROR/CRITICAL severity level and immediately cancel/abort the entire execution process cleanly with a non-zero exit status.
- Q: Is the 2-second startup threshold required for pre-flight model validation? → A: No; remove the arbitrary 2-second requirement. Pre-flight check is purely local, static, and synchronous during initialization (verifying model string format and environment variable presence without network calls), and reports configuration or credential defects immediately to the operator before starting processing.
- Q: Is a custom BaseChatModel adapter (ChatLiteLLM) needed for LangGraph/LangChain or does DSPy LM via LiteLLM suffice? → A: Discard the custom BaseChatModel adapter (ChatLiteLLM). All assistant prompting and routing in Vatuta are structured through DSPy signatures and modules running on dspy.LM (which uses LiteLLM internally). A redundant BaseChatModel wrapper is unnecessary.
- Q: What is the architectural role of Gemini versus the general LiteLLM provider abstraction? → A: The architecture is strictly provider-independent: LiteLLM is adopted specifically to avoid any vendor lock-in or provider-specific coupling in business logic. Gemini is chosen purely as the concrete validation and testing target for this release because the user has an active subscription and API key, but the codebase and configuration design remain fully generic.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Configure and Switch Model Providers via Central Configuration (Priority: P1)

As an assistant operator or developer, I want to select and configure the active language model provider (initially
Gemini) by updating settings in the configuration file (`config/vatuta.yaml`) using an extensible provider
abstraction, so that I can adapt to different operational cost, latency, or capability requirements without modifying
application code.

**Why this priority**: Enabling configuration-driven provider selection is the core capability that unlocks provider
independence and eliminates hardcoded model dependencies across the assistant.

**Independent Test**: Can be tested independently by running live end-to-end queries against the Gemini provider
(for which live credentials are provided) across different Gemini models or parameters, verifying that configuration
changes in `config/vatuta.yaml` alter model selection without application code changes. No simulation or mocking of
other providers is required.

**Acceptance Scenarios**:

1. **Given** a valid configuration specifying a Gemini model backend with valid environment credentials,
   **When** the assistant is initialized and receives a query,
   **Then** it routes and answers the query using the live Gemini API without raising errors.
2. **Given** an existing configured Gemini backend,
   **When** the operator updates the active backend model or parameters in `config/vatuta.yaml`,
   **Then** the assistant uses the newly selected configuration on next startup without code changes.
3. **Given** a missing or invalid model configuration,
   **When** the assistant initializes,
   **Then** it halts with a descriptive error message indicating the configuration defect.

---

### User Story 2 - Independent Model Assignment for Routing and Generation (Priority: P2)

As a system operator, I want to configure different model backends or tiers for query routing versus content
generation (for example, a fast and cost-effective Gemini Flash model for routing and a reasoning-heavy Gemini Pro
model for synthesized answers), so that I can optimize both system latency and operational costs.

**Why this priority**: Decoupling the routing stage from the generation stage allows fine-grained cost-performance
tradeoffs tailored to distinct pipeline tasks.

**Independent Test**: Can be tested independently by setting the routing backend to one model tier (e.g., Gemini Flash)
and the generator backend to another (e.g., Gemini Pro), executing a query, and verifying that each stage executes
against its designated backend.

**Acceptance Scenarios**:

1. **Given** a configuration where the routing backend is assigned to Model Tier A and the generator backend is
   assigned to Model Tier B,
   **When** a user submits a retrieval question,
   **Then** tool selection and query routing use Model Tier A and answer synthesis uses Model Tier B.
2. **Given** a configuration where both routing and generator roles point to the same model backend,
   **When** a user submits a retrieval question,
   **Then** both stages execute successfully using that single backend.

---

### User Story 3 - Clean Failure Termination and Error Reporting (Priority: P3)

As an operator or user, I want Vatuta to catch LiteLLM exceptions, log detailed diagnostics, report a clear and
actionable error message explaining why the model call failed, and abort the operation cleanly with an error exit
status, so that failures are transparent, immediate, and do not hang or terminate with unhandled stack traces.

**Why this priority**: If a configured model cannot be reached or executed, the requested operation cannot proceed.
Aborting cleanly with clear diagnostics prevents indefinite hangs, unhandled crashes, or undefined states.

**Independent Test**: Can be tested independently by:
1. Attempting to initialize Vatuta with missing required credentials, verifying that Vatuta catches LiteLLM's error
   and halts cleanly with an actionable diagnostic.
2. Simulating a runtime rate limit or upstream failure during a query, verifying that Vatuta catches the exception,
   logs diagnostic details, displays an informative error message, and exits with a non-zero status.

**Acceptance Scenarios**:

1. **Given** a selected provider backend whose required credentials are not set in the environment,
   **When** Vatuta initializes the LiteLLM backend at startup,
   **Then** Vatuta catches the authentication/credential error and halts initialization cleanly with an actionable
   diagnostic message identifying the issue.
2. **Given** a provider backend configured with an unrecognized or malformed model identifier,
   **When** Vatuta initializes the model backend,
   **Then** Vatuta catches the LiteLLM configuration error and halts with a clear explanation of the format expectation.
3. **Given** an active provider experiencing rate limits or upstream service errors during runtime execution,
   **When** the request fails,
   **Then** Vatuta catches the LiteLLM exception, logs the diagnostic context, reports an informative error message to
   the user, and aborts the operation cleanly with an error exit status without an unhandled crash.

---

### Edge Cases

- **Missing Provider Credentials**: What happens when an operator selects a model but required credentials are not
  present in the environment? LiteLLM handles credential resolution; Vatuta catches the resulting authentication
  error at startup, logs diagnostic details, and halts cleanly with an actionable message.
- **Unsupported or Invalid Model Names**: What happens when an operator enters an invalid model string in
  `config/vatuta.yaml`? LiteLLM raises a configuration or unsupported model error; Vatuta catches it at startup and
  halts with an informative diagnostic.
- **Provider API Outage or Network Disconnection**: How does the system handle upstream provider downtime or HTTP 5xx
  errors? Vatuta catches LiteLLM connection exceptions, logs full diagnostic details at ERROR level, displays an
  informative failure message, and aborts the operation cleanly with an error exit status.
- **Rate Limit or Quota Exhaustion (HTTP 429)**: How does the system react when a provider's rate limit is reached?
  Vatuta catches LiteLLM's `RateLimitError`, logs the event, informs the user that provider rate limits have been
  exceeded, and aborts the operation cleanly with an error exit status.
- **Varying Parameter Constraints**: How does the system accommodate differences in temperature bounds or token limits?
  The model configuration must validate parameters within acceptable universal boundaries and pass them appropriately
  to the target provider.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support selecting the active language model backend for the assistant exclusively through
  configuration in `config/vatuta.yaml`.
- **FR-002**: System MUST implement a provider-independent LLM abstraction using LiteLLM as the unified LM backend
  for DSPy (`dspy.LM`), designed for multi-provider extensibility, with initial implementation and runtime support
  strictly focused on the Gemini model family.
- **FR-003**: System MUST allow configuring separate model backends or tiers for query routing and answer generation.
- **FR-004**: System MUST preserve existing structured prompt declarations and reasoning signatures without modifying
  domain logic when configuring model backends.
- **FR-005**: System MUST delegate credential discovery and authentication handling directly to LiteLLM, relying on
  LiteLLM to read provider credentials from environment variables without hardcoding secrets in configuration files
  or maintaining custom per-provider key registries in Vatuta.
- **FR-006**: System MUST perform a broad, general, and non-intrusive pre-flight evaluation of configured model
  backends upon startup (validating model string format and provider environment variable presence via LiteLLM's
  native mechanisms, without executing live network probes or test prompts), halting cleanly with actionable diagnostics.
- **FR-007**: System MUST support configuring standard model hyperparameters (including temperature and maximum
  token limits) per backend in `config/vatuta.yaml`.
- **FR-008**: System MUST replace the existing `rag.llm_backends` configuration schema in-place with a new
  LiteLLM-native schema, documenting migration steps and breaking changes in `RELEASE.md`.
- **FR-009**: System MUST treat any unrecoverable runtime LLM failure (including authentication errors, rate limits,
  and network failures) as a fatal execution event, logging diagnostic context at ERROR/CRITICAL severity level,
  displaying an informative failure message, and immediately canceling the entire operation with a non-zero exit status.
- **FR-010**: System MUST expose Prometheus metrics for LLM provider calls, including call latency histograms,
  token usage counters (prompt and completion tokens), and error rate counters, labeled by provider and pipeline role.

### Key Entities

- **Model Backend Configuration**: Represents configuration parameters for a specific language model service,
  including provider identifier, model name, temperature, and token limits.
- **Provider Adapter**: LiteLLM acting as the unified intermediary that translates model requests into
  provider-specific invocations via `dspy.LM`, maintaining isolation from business logic.
- **Pipeline Role Binding**: The association between a functional pipeline role (such as query routing or synthesis
  generation) and a configured model backend.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An operator can configure and switch Gemini model configurations in `config/vatuta.yaml` with zero
  changes to application code, utilizing an extensible provider abstraction.
- **SC-002**: 100% of core assistant retrieval, routing, and response capabilities execute successfully with Gemini in
  live environment validation, with no mock or simulated tests required for non-Gemini providers in this release.
- **SC-003**: Startup configuration and credential defects caught from LiteLLM's local validation are reported
  with actionable diagnostic guidance immediately to the operator upon initialization before executing commands.
- **SC-004**: Runtime provider failures (rate limits, network outages) caught from LiteLLM produce an informative
  failure message and terminate execution cleanly with an error exit status without unhandled exceptions or hangs.
- **SC-005**: Updating model configuration incurs zero regression in existing automated test suites and maintains
  existing structured prompt contracts.
- **SC-006**: LLM provider call metrics (latency histograms, token usage counters, error rates) are observable via
  Prometheus endpoints after every provider invocation, labeled by provider and pipeline role.

## Assumptions

- Live environment credentials and interactive validation are available for Gemini (`GEMINI_API_KEY`), which is
  used strictly as the concrete validation and testing target for this release. The core architecture, configuration
  schema, and provider abstraction (via LiteLLM) are completely provider-agnostic and free of Gemini-specific
  assumptions. Other providers (Claude, OpenAI, Bedrock, etc.) can be used simply by configuring their models and
  supplying their standard environment variables without application code modifications.
- No mock or simulated responses for non-Gemini providers are required or included in this feature.
- LiteLLM manages provider credential resolution and authentication requirements; Vatuta does not implement or
  maintain custom lists of required environment variables per provider.
- Automatic retries on transient errors and multi-provider runtime failover are explicitly out of scope for this
  release; failed requests abort cleanly with descriptive diagnostic messaging.
- Prompt optimization and provider-specific prompt tuning are handled at the prompt definition layer without requiring
  provider-specific branching in business logic.
- Supported providers communicate via secure outbound HTTPS connections conforming to standard network isolation
  policies.
