# Release Notes

All notable changes to the Vatuta project are documented in this file.

## [0.2.1] - 2026-06-13

### Added

- **Embedding Configuration**: Support for configurable embedding prefixes (`embeddings_query_prefix` and
`embeddings_document_prefix`) and L2 normalization (`embeddings_normalize`) in the Qdrant manager.

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
