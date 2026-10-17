# Feature Specification: Policy Checker Mechanism

**Feature Branch**: `002-policy-checker`

**Created**: 2026-10-06

**Status**: Draft

**Input**: User description: "As a manager, I want to create a mechanism for reviewing management procedure policies in the context of my work. We will use Vatuta's MCP support to add data sources..."
**Update 1**: "The user interface (CLI) is the current command client for Vatuta. But prepare the development so that it can be invoked as an MCP (Model Context Protocol) service. If new technologies or libraries other than Python are needed, do not hesitate to use them. The interpretability of the whole process is very important. For each execution, in the log directory, in a directory per invocation such that its name allows identifying the invocation, all information must be left to follow and audit the reasoning and all intermediate data. Inside that directory, each stage of the LLM flow must be an additional directory. Inside each directory, prompts and their responses, invocations to third-party systems and their responses must be stored in a human-readable but machine-reusable way."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Policy Validation Extraction (Priority: P1)

As a manager, I want to provide a formal policy document written in natural language, so the system can extract atomic validations and the mechanisms to verify them.

**Why this priority**: It is the foundation for automating policy checking. Without extracting the rules, they cannot be verified automatically.

**Independent Test**: Can be fully tested by providing a sample policy document and verifying that the extracted validation controls cover the expected rules correctly structured in the RASE methodology.

**Acceptance Scenarios**:

1. **Given** a natural language policy document, **When** the extraction is executed, **Then** the system identifies all policies and outputs atomic validation controls following the RASE (Requirement, Applicability, Selection, Exception) methodology.
2. **Given** a policy that requires specific parameters (e.g., project name), **When** the extraction runs, **Then** the system identifies these parameters and includes them in the control definition.

---

### User Story 2 - Policy Verification Execution (Priority: P1)

As a manager, I want to execute a list of validation controls against my project data sources (e.g., Github, JIRA Free) and obtain a list of violations or a compliance assessment.

**Why this priority**: This delivers the core value of the feature by actually performing the automated checks.

**Independent Test**: Can be fully tested by providing a set of predefined validation controls and a mock/real data source, and checking if the system correctly reports compliance or violations.

**Acceptance Scenarios**:

1. **Given** a set of deterministic validation controls and accessible data sources, **When** the verification is executed, **Then** the system accurately returns a list of violations or a compliance status (Compliant, Non compliant, Insufficient evidence, Not applicable).
2. **Given** a control that depends on the output of a previous control, **When** the verification runs, **Then** the system executes them in the correct order, passing the required evidence.

---

### User Story 3 - MCP Client Invocation (Priority: P2)

As an external system or agent, I want to invoke the policy checker as an MCP service, so that I can programmatically trigger policy validations within my own workflows.

**Why this priority**: It expands the reach of the policy checker beyond the CLI, enabling automation and integration with other AI agents or systems.

**Independent Test**: Can be fully tested by connecting an MCP client to the Vatuta MCP server and successfully executing a policy verification request.

**Acceptance Scenarios**:

1. **Given** the policy checker is running as an MCP server, **When** a client requests a policy validation, **Then** the system executes the validation and returns the results via the MCP protocol.

---

### User Story 4 - Execution Traceability & Audit Logging (Priority: P1)

As an auditor or developer, I want to inspect a detailed log of every policy verification execution, so that I can understand the reasoning, LLM prompts, and third-party data retrieved.

**Why this priority**: Interpretability is explicitly required to trust the automated policy decisions and audit the LLM reasoning process.

**Independent Test**: Can be fully tested by executing a validation and verifying that a structured, stage-by-stage log directory is created containing human-readable and machine-reusable files.

**Acceptance Scenarios**:

1. **Given** a policy verification execution, **When** the execution completes, **Then** a unique directory (e.g., by timestamp/UUID) is created containing subdirectories for each logical stage of the flow.
2. **Given** a stage that invokes an LLM or third-party system, **When** inspecting its log directory, **Then** the exact prompts, responses, requests, and payloads are stored in a human-readable and machine-reusable format (e.g., JSON or structured Markdown).

---

### Edge Cases

- **Data Source Unavailability**: If a required data source (e.g., Jira or Github) is temporarily unavailable during execution, the system MUST fail the entire policy verification execution immediately rather than proceeding with partial data.
- **Conflicting Rules**: If multiple extracted policy controls have overlapping applicabilities or conflicting requirements, the system MUST evaluate both rules independently and report both results, even if conflicting.
- **Ambiguous Semantic Evaluation**: If an LLM-based semantic control returns an ambiguous or uncertain compliance result, the system MUST default to a "Non compliant" (fail-safe) state.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a new command named `policy` in the Vatuta client (and corresponding MCP service operation) that accepts a configuration file parameter containing the formal policy and its execution scope.
- **FR-002**: The system MUST extract atomic validation controls from the natural language policy document embedded or referenced in the configuration file, following the RASE methodology.
- **FR-003**: The system MUST support different types of controls: Deterministic, Semantic, Hybrid, and Human review, prioritizing them in that order.
- **FR-004**: The system MUST execute verification in a defined sequence: Source collection -> Evidence acquisition -> Facts normalization -> Evidence evaluation -> Result evaluation.
- **FR-005**: The system MUST allow controls to depend on the results of previously executed controls, managing the execution order appropriately.
- **FR-006**: The system MUST evaluate deterministic controls using Common Expression Language (CEL) or JSON schema, and semantic controls using LLM or JEV-type models.
- **FR-007**: The system MUST output validation results as either a list of specific violations or a compliance status (Compliant, Non compliant, Insufficient evidence, Not applicable).
- **FR-008**: The system MUST integrate with Github and JIRA Free data sources using Vatuta's MCP support.
- **FR-009**: The system MUST expose the policy checking capabilities as an MCP service, in addition to the CLI command.
- **FR-010**: The system MUST create a unique, identifiable directory for every execution within the configured log directory, structured with subdirectories for each stage of the LLM flow.
- **FR-011**: The system MUST serialize and store all LLM prompts/responses and third-party invocations/responses within the corresponding stage directory in a human-readable and machine-reusable format.

### Key Entities

- **Policy Configuration**: A configuration file containing the formal policy documentation and defining the scope (e.g., target repositories, Jira projects) where the policy applies.
- **Policy Control**: An atomic validation rule, structured using RASE (Requirement, Applicability, Selection, Exception) and defined by its evaluation mechanism (CEL, JSON schema, LLM).
- **Evidence**: Normalized data facts dynamically identified during policy analysis and extracted from data sources, used to evaluate applicability and requirements.
- **Data Source**: An external system (e.g., Github, JIRA) that provides raw data to be transformed into evidence.
- **Invocation Trace**: A recorded instance of an execution, containing structured logs (prompts, responses, payloads) organized by logical stages.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The system correctly structures at least 90% of clearly stated rules from a sample natural language document into RASE controls without manual intervention.
- **SC-002**: Verification execution across Jira and Github correctly identifies 100% of deterministic violations in a controlled test environment.
- **SC-003**: Execution of the `policy` command provides a complete compliance report in under 2 minutes for a typical repository/project scan.
- **SC-004**: 100% of policy verification executions generate a structured trace log directory containing all intermediate prompts and third-party interactions.
- **SC-005**: An MCP client can successfully trigger a policy verification and receive the compliance report via the MCP protocol.

## Clarifications

### Session 2026-10-08

- Q: What specific data entities from Github and Jira will be extracted to act as "Evidence" for policy validation? → A: It is dynamic. The policy analysis phase will identify the required evidence, which will then be extracted and transformed.
- Q: How will the user specify the input policy document to the `policy` CLI command or MCP service? → A: Via a configuration file parameter that contains both the formal policy and its execution scope.
- Q: How should the system handle a situation where the semantic evaluation (LLM) returns an ambiguous compliance result? → A: Assume "Non compliant" (fail-safe).
- Q: How should the system handle a temporarily unavailable data source during policy verification? → A: Fail the entire policy verification execution immediately.
- Q: How should the system handle conflicting rules or overlapping applicabilities extracted from the policy document? → A: Evaluate both rules independently and report both results, even if conflicting.

## Assumptions

- Initial implementation is strictly limited to Github and JIRA Free data sources for validation purposes.
- Users have the necessary access permissions (MCP configurations) to retrieve data from Github and JIRA.
- The RASE methodology is sufficient to model the target policies.
- While Substrait might be considered for Evidence Collection, the specification assumes a generic, technology-agnostic pipeline for source collection and facts normalization.
