# Integrations

Vatuta supports multiple data source integrations for importing content into a searchable knowledge base. This document covers setup and usage for each integration.

## JIRA Integration

Vatuta includes powerful JIRA integration capabilities for importing tickets into a searchable knowledge base.

### Setup JIRA Integration

1. Configure your JIRA credentials in the `.env` file:

```bash
# JIRA Configuration
JIRA_USER=your_jira_email@company.com
JIRA_API_TOKEN=your_jira_api_token_here
JIRA_INSTANCE_URL=https://yourcompany.atlassian.net
JIRA_CLOUD=True
```

2. Install required spaCy model:

```bash
just install-spacy
```

### Using JIRA Integration

The JIRA source can be configured in `config/vatuta.yaml`. See the [JIRA source documentation](sources/jira.md) for detailed configuration options.

To query the assistant with JIRA data:

```bash
just assistant query="your question here" k="20"
```

For more details on JIRA source configuration and implementation, see [docs/sources/jira.md](sources/jira.md).

## Confluence Integration

Vatuta can import Confluence pages into the same searchable knowledge base.

### Setup Confluence Integration

1. Configure your Confluence credentials in the `.env` file:

```bash
# Confluence Configuration
CONFLUENCE_ROOT=your_confluence_root_page_id
```

Note: Confluence uses the same JIRA credentials (same Atlassian instance).

### Using Confluence Integration

The Confluence source can be configured in `config/vatuta.yaml`. See the [Confluence source documentation](sources/confluence.md) for detailed configuration options.

To query the assistant with Confluence data:

```bash
just assistant query="your question here" k="20"
```

For more details on Confluence source configuration and implementation, see [docs/sources/confluence.md](sources/confluence.md).

## Slack Integration

Vatuta can import Slack channels, messages and conversations into the searchable knowledge base.

### Setup Slack Integration

1. Add the following variables to your `.env`:

```bash
# Slack Configuration
SLACK_BOT_TOKEN=xoxb-your_bot_token
# Optional
SLACK_CHANNEL_TYPES=public_channel,private_channel,im,mpim
SLACK_OLDEST_TIMESTAMP= # e.g. 1712816400.000000 (epoch seconds as float str)
```

2. Install dependencies (if not already):

```bash
poetry install
```

### Using Slack Integration

The Slack source can be configured in `config/vatuta.yaml`. See the [Slack source documentation](sources/slack.md) for detailed configuration options including:

- Channel filtering by ID or name patterns
- Date range filtering with `initial_lookback_days`
- Channel type filtering (public, private, DMs, group DMs)
- Incremental updates with checkpointing

To query the assistant with Slack data:

```bash
just assistant query="your question here" k="20"
```

For more details on Slack source configuration and implementation, see [docs/sources/slack.md](sources/slack.md).

## GitLab Integration (PoC)

Import GitLab Issues and Merge Requests into the knowledge base.

> [!NOTE]
> GitLab integration is currently in Proof of Concept (PoC) stage and may have limited functionality.

### Setup GitLab Integration

Add the following variables to your `.env`:

```bash
# GitLab Configuration
GITLAB_URL=https://gitlab.com
GITLAB_TOKEN=your_gitlab_access_token
# Comma-separated numeric project IDs
GITLAB_PROJECT_IDS=123,456
```

### Using GitLab Integration

The GitLab source is currently implemented as a PoC script. Configuration and usage details are being developed.

## Knowledge Base Features

All integrations share these common features:

- **Vector Search**: Documents are stored with embeddings for semantic search
- **Persistent Storage**: Documents are saved locally and persist between sessions
- **Metadata Tracking**: Each document includes source, timestamps, and metadata
- **Text Splitting**: Large documents are automatically split into manageable chunks
- **Incremental Updates**: Sources support checkpointing to avoid re-fetching data
- **Multi-source Support**: Query across JIRA, Confluence, Slack, and GitLab simultaneously

## Configuration

All sources are configured via `config/vatuta.yaml`. Each source can be enabled/disabled individually and supports specific configuration options. See the individual source documentation in `docs/sources/` for detailed configuration schemas.

Example configuration structure:

```yaml
sources:
  - id: my-jira
    type: jira
    enabled: true
    # ... JIRA-specific config

  - id: my-slack
    type: slack
    enabled: true
    # ... Slack-specific config
```

For detailed configuration options for each source, refer to:

- [JIRA Configuration](sources/jira.md)
- [Confluence Configuration](sources/confluence.md)
- [Slack Configuration](sources/slack.md)
