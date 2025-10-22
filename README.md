# Vatuta - Personal AI Assistant

An intelligent personal assistant built with LangChain that can help you with various daily tasks.

## Features

- 🤖 Intelligent conversational assistant
- 🔗 Integration with multiple APIs and services
- 📊 Data analysis and report generation
- 🗄️ Personal information management
- 🔍 Document search and processing
- 📝 Content generation

## Installation

1. Clone the repository:
```bash
git clone git@github.com:franjuan/vatuta.git
cd vatuta
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install Just (if not already installed):
```bash
# On macOS with Homebrew
brew install just

# On Linux
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/bin

# On Windows with Chocolatey
choco install just
```

4. Install direnv (optional but recommended for automatic environment loading):
```bash
# On macOS with Homebrew
brew install direnv

# On Linux
curl -sfL https://direnv.net/install.sh | bash

# On Windows with Chocolatey
choco install direnv
```

5. Install dependencies:
```bash
poetry install
```

6. Configure environment variables:
```bash
cp env.example .env
# Edit the .env file with your API keys
```

7. Allow direnv to load the environment (if using direnv):
```bash
direnv allow
```

8. Install pre-commit hooks (optional but recommended):
```bash
just pre-commit-install
```

## Usage

### Run the assistant

```bash
just run
# or
just assistant
```

### REST API

```bash
just api
```

### Development server

```bash
just api-dev
```

### Development

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

Setup development environment:
```bash
just setup
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
- **Linting**: Code is checked with flake8 and mypy
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

## Project Structure

```
vatuta/
├── src/                    # Main source code
│   ├── assistant/         # Assistant module
│   ├── utils/            # Utilities
│   ├── integrations/     # External integrations
│   └── api/              # REST API
├── config/               # Configuration files
├── data/                 # User data
├── logs/                 # Log files
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Configuration



## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
