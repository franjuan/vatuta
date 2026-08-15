# Release Notes

All notable changes to the Vatuta project are documented in this file.

## [0.3.0] - 2026-08-13

### Added

- **Model Context Protocol (MCP) Client Integration**: Enables the `RAGAgent` to dynamically discover, register, and
  invoke external tools provided by MCP servers within the DSPy ReAct routing loop. Input argument schemas are
  generated at runtime directly from server JSON Schema metadata using the `json-schema-to-pydantic` library to
  safely support complex nested structures. Includes end-to-end unit test coverage for container lifecycle and
  tool wrappers.
- **MCP Tool Filtering (Whitelisting)**: Added regex-based tool whitelisting (`allowed_tools`) per MCP server
  configuration, allowing fine-grained restriction of which server tools are exposed to the agent.
- **Hardened Docker Container Execution**: Implemented an isolated stdio container runner for MCP servers using the
  Python Docker SDK. Enforces least-privilege security policies (read-only root filesystem, non-root user execution,
  dropped Linux capabilities, network isolation, resource limits) with pre-run image digest resolution.
- **Sync/Async Execution Bridge**: Introduced a background event loop manager (`AsyncLoopThread`) to safely bridge
  synchronous LangGraph nodes and DSPy ReAct routing loops with asynchronous MCP container stdio sessions without
  blocking or corrupting event loops.
- **MCP Environment Variable Passthrough (`env_passthrough`)**: Added support for forwarding host environment
  variables (such as API tokens from `.env`) securely to MCP Docker containers via `-e VAR_NAME` without exposing
  secrets in YAML files. Supports `${VAR}` expansion in container `args`.
- **MCP Tool Execution Timeout & Exception Protection**: Enforced a 60-second execution timeout on MCP tool calls to
  prevent infinite hangs when external containers encounter network or DNS failures.
- **Agent Lifecycle & CLI Context Management**: Extended `RAGAgent` with context manager support (`__enter__` /
  `__exit__`) for automatic container initialization, dynamic tool discovery, and graceful resource teardown,
  integrated directly into the `vatuta ask` CLI workflow.

### Changed & Refactored

- **Poetry 2.0 Migration & Dependencies**: Upgraded `pyproject.toml` configuration to Poetry 2.0 specification
  standards (`[project]` table format) and added `mcp` (`^1.28.1`) and `docker` (`^7.2.0`) dependencies. Configured
  `isort` and `ruff` (`known_third_party = ["mcp"]`) to prevent import shadowing between PyPI `mcp` and local
  project modules.
- **Logging Hygiene & Architecture**: Replaced ad-hoc `print()` and f-string logging across data sources
  (`slack.py`, `qdrant_manager.py`) with centralized `setup_logging` and lazy string formatting, adhering strictly to
  new code quality guidelines.
- **Python Version Pinning**: Restricted max Python version to `<3.14` in `pyproject.toml` and documentation due to
  compatibility considerations.
- **Default LLM & MCP Image Configurations**: Updated default Gemini LLM model identifier in configuration
  templates to `gemini/gemini-3.7-flash` and updated the default Wikipedia MCP server image reference to
  `mcp/wikipedia-mcp`.
- **Documentation & Operational Troubleshooting**: Added comprehensive MCP architecture and security documentation
  (`docs/mcp.md`), and updated `docs/qdrant_setup.md` with troubleshooting steps for Docker container storage
  permission issues (`data/qdrant` ownership) and API key configuration.

### Fixed

- **Agent Tool Calling Heuristics**: Adjusted DSPy ReAct routing prompt to explicitly emphasize the usage of
  external tools, preventing the LLM from aggressively skipping tool execution when no internal RAG filters were
  needed.

---

## [0.2.1] - 2026-06-13

### Added

- **Embedding Configuration**: Support for configurable embedding prefixes (`embeddings_query_prefix` and
`embeddings_document_prefix`) and L2 normalization (`embeddings_normalize`) in the Qdrant manager.

### Fixed

- **Document Deletion Count**: Fixed a limitation in Qdrant document manager where it would only report up to 10,
000 deleted documents by replacing pagination-based scroll count with native Qdrant `count` API.

---

## [0.2.0] - 2026-06-04

### Added

- **Hybrid Search**: Support for hybrid Qdrant search combining dense embeddings with BM25 sparse vectors.
- **Dynamic Routing**: Implement dynamic routing in RAG and routing summary to outputs, generator input, and client.
- **Multiple Sources**: Allow configuring multiple sources and enable filtering by source type or ID.
- **Configuration & CLI**: Add `--max-tokens` CLI argument to override LLM generation limits.
- **Metrics**: Instrument Jira ingestion with detailed Prometheus metrics.
- **CI/CD & Workflows**: Update weekly/security workflows to generate, parse, and upload JSON pip-audit reports.

### Fixed

- Update search result labeling for hybrid search.

### Changed & Refactored

- Update default embedding model to `multilingual-e5-small` with dynamic max sequence length detection.
- Automatically cap `chunk_max_size_chars` based on embedding model capacity.
- Formalize embedding model configurations and add validation in Qdrant manager.
- Introduce `AgentTool` base class for modular RAG agent tools and simplify agent state management.
- Remove unused chunking parameters from Confluence source.
- Externalize ignored vulnerabilities to `.pip-audit-ignore` and update CI workflows.

---

## [0.1.0] - 2026-04-18

### Added

- **Core Engine**: Initial CLI client, RAG system, and data sources for Slack, Jira, and Confluence.
- **Vector Database**: Integrate Qdrant vector database for RAG with configurations and POCs.
- **LLM Support**: Add Gemini LLM backend support with configurable settings.
- **Chunking Strategy**: Advanced comment chunking (Jira) and structural/semantic chunking (Confluence).
- **Quality & Security**: Integrate SBOM generation, code quality, complexity, and security analysis tools.
- **Repository Setup**: Add Github social preview image and pre-commit Signed-off-by trailer hook.

### Fixed

- Update Qdrant filter key assertion in test from `source` to `metadata.source`.
- Address various mypy type-checking issues.

### Changed & Refactored

- Rename Jira module to `jira_source` and improve content formatting tests.
- Improve type checking by removing mypy ignores and adding pre-commit dependencies.

### Documentation

- Rewrite README with comprehensive architecture, tech stack, quick start guide, and badges.
- Establish licensing, contribution guidelines, and legal notices.
