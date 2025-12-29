# Cursor Instructions for Vatuta Personal AI Assistant

## Project Context

You are working on **Vatuta** (Virtual Assistant for Task Understanding, Tracking &
 Automation), a Personal AI Assistant built with LangChain and DSPy. This is a modern Python
 project with comprehensive development tooling and best practices.

## Key Technologies

- **LangChain**: For AI assistant functionality and conversation management
- **DSPy**: For document understanding and retrieval
- **Poetry**: For dependency management
- **Pydantic**: For data validation and settings
- **pytest**: For testing
- **Just**: For task automation (replaces Make)
- **Ruff**: Fast Python linter and formatter (replaces flake8)

## Agent Behavioral Rules

### 1. Poetry Wrapper Rule

**Crucial**: This is a Poetry-managed project.
You **MUST** always invoke Python tools (pytest, mypy, ruff, black, python scripts, etc.) using `poetry run`.

- **Incorrect**: `pytest tests/`
- **Correct**: `poetry run pytest tests/`
- **Incorrect**: `python src/script.py`
- **Correct**: `poetry run python src/script.py`

### 2. Documentation First Rule

**Crucial**: Documentation is a first-class citizen in this project.

- **Plans**: Your `implementation_plan.md` MUST include a section for documentation updates if you are modifying functionality.
- **Execution**: You MUST update the relevant documentation in `docs/` (e.g., `docs/integrations.md`, component docs)
*during* the execution phase, not as an afterthought.
- **Verification**: If documentation was not updated when functionality changed, the task is incomplete.
- **Quality**: Documentation is checked by **pydocstyle** (Google convention) and **markdownlint**.

## Development Environment

### Setup Commands

```bash
# Install dependencies
poetry install

# Setup development environment
just setup

# Allow direnv (if using direnv)
just direnv-allow

# Activate virtual environment (IMPORTANT)
source .venv/bin/activate
# OR invoke python directly from the venv
poetry run python script.py

# IMPORTANT: Always use `poetry run` when executing python scripts to ensure dependencies are loaded correctly.
# Example: `poetry run python tests/reproduce_jira_source.py`
```

### Common Development Tasks

```bash
# Run the assistant
just run

# Run tests
just test

# Format code
just format

# Lint code
just lint

# Run all checks
just check
```

## Code Style Guidelines

### Python Standards

- Always use English for comments and documentation, even when my prompt is in Spanish
- Use **type hints** for all function parameters and return values
- Follow **PEP 8** style guidelines
- Use **f-strings** for string formatting
- Keep functions **small and focused** (max 20 lines)
- Use **descriptive names** for variables and functions
- Add **docstrings** for all public functions and classes

```python
# Standard library imports
import os
import sys
from typing import Optional, Dict, Any

# Third-party imports
from fastapi import FastAPI, HTTPException
from langchain import ChatOpenAI
from pydantic import BaseModel

# Local imports
from src.assistant import PersonalAssistant
from src.utils import get_settings
```

### Code Generation Formatting Rules

When generating or modifying code, always ensure:

1. **Import Ordering**: Use `isort` compatible ordering (standard library → third-party → local)
2. **Trailing Whitespace**: Remove all trailing whitespace from lines
3. **End of File**: Ensure files end with a single newline character
4. **Line Endings**: Use Unix-style line endings (LF, not CRLF)
5. **Mypy Compliance**: Write type-safe code that passes strict mypy checks

These rules prevent linting errors and ensure consistency across the codebase.

### Mypy Compliance Guidelines

This project uses strict mypy configuration. Always follow these rules:

1. **Type Hints Required**: Add type hints to all function parameters and return values

   ```python
   # Correct
   def process_data(items: list[str], count: int) -> dict[str, Any]:
       return {"items": items, "count": count}

   # Incorrect - missing type hints
   def process_data(items, count):
       return {"items": items, "count": count}
   ```

2. **Avoid `Any` When Possible**: Use specific types instead of `Any`

   ```python
   # Preferred
   def get_config() -> dict[str, str]:
       return {"key": "value"}

   # Avoid (unless truly necessary)
   def get_config() -> dict[str, Any]:
       return {"key": "value"}
   ```

3. **Use Pydantic Models Instead of Dicts**: When unpacking dictionaries, use Pydantic model instantiation

   ```python
   # Correct - Direct instantiation
   config = JiraConfig(
       url="https://jira.example.com",
       projects=["TEST"],
       id="jira-main"
   )

   # Incorrect - Dict unpacking (causes mypy errors)
   config_dict = {"url": "https://jira.example.com", "projects": ["TEST"], "id": "jira-main"}
   config = JiraConfig(**config_dict)
   ```

4. **Handle Optional Types Properly**: Check for None before accessing attributes

   ```python
   # Correct
   def get_name(user: Optional[User]) -> str:
       if user is None:
           return "Unknown"
       return user.name

   # Incorrect - mypy error: Item "None" has no attribute "name"
   def get_name(user: Optional[User]) -> str:
       return user.name
   ```

5. **Type Collections Properly**: Use modern type hints (Python 3.12+)

   ```python
   # Correct (Python 3.12+)
   from typing import Optional

   def process(items: list[str]) -> dict[str, int]:
       return {item: len(item) for item in items}

   # Avoid (old style)
   from typing import List, Dict

   def process(items: List[str]) -> Dict[str, int]:
       return {item: len(item) for item in items}
   ```

6. **Use `type: ignore` Comments Sparingly**: Only when absolutely necessary, with specific error codes

   ```python
   # Acceptable when needed
   data: dict[str, Any] = {"items": []}  # type: ignore[var-annotated]

   # Not acceptable - too broad
   data = get_external_data()  # type: ignore
   ```

### Error Handling Pattern

```python
try:
    result = some_operation()
    return {"success": True, "data": result}
except SpecificException as e:
    logger.error(f"Specific error: {e}")
    return {"success": False, "error": str(e)}
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return {"success": False, "error": "Internal server error"}
```

## Project Structure

### Main Directories

- `src/` - Source code for the application
- `tests/` - Test suite (unit and integration tests)
- `config/` - Application configuration files
- `data/` - Local data storage and checkpoints
- `docs/` - Project documentation
- `logs/` - Application logs

### Core Modules

- `src/sources/` - Data ingestion and connectors
  - `slack.py`, `jira.py`, `confluence.py` - Source implementations
  - `source.py` - Base classes and protocols
- `src/rag/` - RAG and Retrieval logic
  - `document_manager.py` - Document handling
  - `engine.py` - Retrieval engine
- `src/models/` - Data models and configuration
  - `documents.py` - Document and chunk definitions
  - `config.py` - Configuration models
- `src/metrics/` - Observability and metrics
- `src/utils/` - Utility functions
- `src/client/` - Client implementations

### Configuration Files

- `pyproject.toml` - Poetry configuration and dependencies
- `justfile` - Task automation commands
- `.envrc` - direnv configuration for automatic environment loading
- `env.example` - Environment variables template
- `pyrefly.toml` - Pyrefly LSP configuration
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `.secrets.baseline` - detect-secrets baseline file

## Testing Guidelines

### Test Structure

- One test file per module (`test_*.py`)
- Use **descriptive test names**
- Test both **success and failure cases**
- **Mock external dependencies**
- Test **edge cases and error conditions**

### Test Example

```python
def test_assistant_chat_functionality(assistant):
    """Test basic chat functionality."""
    with patch.object(assistant.conversation_chain, 'predict', return_value="Test response"):
        response = assistant.chat("Hello")
        assert response == "Test response"
```

## Environment & Configuration

### Environment Variables

- Use `.env` files for local development
- Document all required environment variables in `env.example`
- Use **Pydantic Settings** for configuration management
- Validate configuration on startup

### Settings Pattern

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    database_url: Optional[str] = None
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Use structured logging
logger.info(f"User {user_id} started chat session")
logger.error(f"API call failed: {error}", extra={"user_id": user_id})
```

## Quick Commands Reference

```bash
# Development
just setup          # Setup development environment
just run            # Run the assistant
just api            # Run API server
just test           # Run tests
just check          # Run all quality checks

# Code Quality
just format         # Format code with Black, Ruff, and isort
just lint           # Lint with Ruff, Mypy, Pydocstyle, Detect-secrets
just cc             # Run cyclomatic complexity checks (Radon)
just audit          # Audit dependencies (pip-audit)
just pre-commit     # Run all pre-commit hooks

# Git Operations
just git-init       # Initialize git repository
just commit "msg"   # Add and commit changes
just push           # Push to remote repository

# Environment
just direnv-allow   # Allow direnv to load environment
just env            # Show environment info
just packages       # Show installed packages
```

## When Making Changes

1. **Follow the established patterns** in the codebase
2. **Add type hints** to all new functions
3. **Write tests** for new functionality
4. **Update documentation** if needed
5. **Run quality checks** before committing (`just check` or `just pre-commit`)
6. **Consider security implications** of changes
7. **Think about performance** impact
8. **Maintain backward compatibility** when possible
9. **Update documentation** at `docs/` when needed

## Code Quality & Pre-commit Hooks

The project uses `pre-commit` to enforce strict code quality standards. The following tools are configured:

### 1. Hygiene & Formatting

- **Black**: Code formatting
- **Isort**: Import sorting
- **Ruff**: Fast linting (replaces flake8)
- **Yamllint**: YAML validation
- **Markdownlint-cli2**: Markdown style validation
- **Trailing whitespace & End-of-file**: Standard hygiene

### 2. Static Analysis & Type Checking

- **Mypy**: Static type checking (ignore-missing-imports enabled)

### 3. Security

- **Bandit**: Security analysis for Python
- **Semgrep**: Static analysis for security and bugs
- **Detect-secrets**: Secrets detection
- **Pip-audit**: Dependency vulnerability auditing

### 4. Code Quality & Maintenance

- **Pydocstyle**: Docstring validation (Google convention)
- **Codespell**: Spell checking
- **Xenon**: Cyclomatic complexity assertions
- **Jscpd**: Copy/paste detector (duplication check)

## Python Project Rules

### Code Style

- Follow PEP8 strictly.
- Max line length: **100**.
- Use `snake_case` for functions and variables.
- Use `PascalCase` for classes.

### Function Documentation

- Every public function must include a docstring in **Google style**.
- Private helpers may omit docstring if self-explanatory.

#### Function Documentation Example

"""
Short summary.

Args:
    param_name (type): Explanation.

Returns:
    type: Explanation.
"""

### Comments

- Use comments only to clarify *why*, not *what*.
- Avoid redundant comments.

### Structure

- Do not define more than one class per file.
- Functions longer than 30 lines should be refactored.

### Testing

- Every function added must include a unit test under `tests/`.

### Cyclomatic Complexity Rules (Xenon)

We use **Xenon** to strictly enforce cyclomatic complexity. The configuration is:

- **Max Absolute (Per Function)**: Grade D (Complexity <= 30)
- **Max Modules (Average per Module)**: Grade B (Complexity <= 10)
- **Max Average (Total Average)**: Grade B (Complexity <= 10)

If you exceed these limits, the build will fail.

**Development Guidelines:**

- **Target Complexity**: Keep functions simple (Complexity ≤ 5).
- **Refactoring Threshold**: Consider refactoring if complexity > 7.
- **Strict Limit**: Any function > 30 will be rejected by CI.

If you encounter high complexity:

- Extract helper functions.
- Use dictionaries for dispatch instead of `if/elif` chains.
- Use early returns to reduce nesting levels.

### Code Duplication (Jscpd)

We use **jscpd** to detect duplicated code.

- **Threshold**: Min 50 tokens, Min 5 lines.
- **Action**: Refactor duplicated code into shared helper functions or base classes.
