# Vatuta - Virtual Assistant for Task Understanding, Tracking & Automation

<div align="center">
  <img src="vatuta.png" alt="Vatuta Logo" width="200"/>
</div>

An intelligent personal assistant built with LangChain that can help you with various daily tasks.

## Features

- 🤖 Intelligent conversational assistant
- 🔗 Integration with multiple APIs and services
- 📊 Data analysis and report generation
- 🗄️ Personal information management
- 🔍 Document search and processing
- 📝 Content generation
- 🎫 JIRA ticket import and management
- 📄 Confluence page integration
- 🧠 Vector-based knowledge base

## Installation

1. Clone the repository:

```bash
git clone git@github.com:franjuan/vatuta.git
cd vatuta
```

1. Install Poetry (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

1. Install Just (if not already installed):

```bash
# On macOS with Homebrew
brew install just

# On Linux
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin

# On Windows with Chocolatey
choco install just
```

1. Install direnv (optional but recommended for automatic environment loading):

```bash
# On macOS with Homebrew
brew install direnv

# On Linux
curl -sfL https://direnv.net/install.sh | bash

# On Windows with Chocolatey
choco install direnv
```

1. Allow direnv to load the environment (if using direnv):

```bash
direnv allow
```

**Note**: If you get `layout_poetry: command not found`, you have two options:

- **Option A**: Use the basic direnv setup (recommended)
- **Option B**: Install direnv Poetry plugin: `pip install direnv-poetry`

1. Install pre-commit hooks (optional but recommended):

```bash
just pre-commit-install
```

## Usage

### Run the assistant

```bash
just run
# or
just assistant query="My query" k="20"
```

### Development

Setup development environment:

```bash
just setup
```

Run tests:

```bash
just test
```

Format code:

```bash
just format
```

Lint code:

```bash
just lint
```

Run all checks:

```bash
just check
```

Audit dependencies:

```bash
just audit
```

### Environment Management

The project supports automatic environment loading with **direnv**:

- **Automatic .env loading**: Environment variables are loaded automatically when you enter the project directory
- **Poetry integration**: Virtual environment is activated automatically
- **No manual activation**: No need to run `poetry shell` or `source .env` manually
- **Cross-directory support**: Works seamlessly across different project directories

With direnv enabled:

- Environment variables from `.env` are loaded automatically
- Poetry virtual environment is activated automatically
- Simply `cd` into the project directory and everything is ready

### Code Quality

The project uses pre-commit hooks to ensure code quality and consistency:

- **Automatic formatting**: Code is automatically formatted with Black
- **Linting**: Code is checked with ruff and mypy
- **Import sorting**: Imports are automatically organized with isort
- **Basic checks**: Trailing whitespace, file endings, and merge conflicts are detected

Install pre-commit hooks:

```bash
just pre-commit-install
```

Run pre-commit hooks manually:

```bash
just pre-commit
```

For more commands, run:

```bash
just --list
```

### Quick Start

1. Configure environment variables:

   ```bash
   cp env.example .env
   # Edit the .env file with your API keys (Jira, Confluence, Slack, Qdrant)
   ```

1. Start Qdrant Vector Database:

   ```bash
   just qdrant-start
   ```

1. Configure sources in `config/vatuta.yaml` (see `config/vatuta.yaml.example`)
1. Install dependencies: `just install`
1. Query the assistant: `just assistant query="your question" k="20"`

## Project Structure

```text
vatuta/
├── src/                    # Main source code
│   ├── sources/           # Data source integrations (Slack, Jira, Confluence)
│   ├── models/            # Data models and schemas
│   ├── rag/               # RAG (Retrieval-Augmented Generation) components
│   ├── client/            # Client interfaces
│   ├── metrics/           # Metrics and monitoring
│   └── utils/             # Utility functions
├── pocs/                   # Proof-of-concept scripts and experiments
├── config/                 # Configuration files (vatuta.yaml)
├── data/                   # User data and cached source data
├── logs/                   # Log files
├── tests/                  # Unit tests
└── docs/                   # Documentation
    └── sources/           # Source-specific documentation
```

## Configuration

## Integrations

Vatuta supports multiple data source integrations including:

- **JIRA**: Import tickets and issues
- **Confluence**: Import pages and documentation
- **Slack**: Import channels and conversations
- **GitLab**: Import issues and merge requests (PoC)

All sources support:

- Vector-based semantic search
- Incremental updates with checkpointing
- Configurable filtering and date ranges
- Unified query interface

For detailed setup instructions and usage examples, see [docs/integrations.md](docs/integrations.md).

## Contributing

1. Fork the project
1. Create a feature branch (`git checkout -b feature/AmazingFeature`)
1. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
