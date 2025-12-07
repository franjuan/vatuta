# Cursor Instructions for Vatuta Personal AI Assistant

## Project Context

You are working on **Vatuta** (Virtual Assistant for Task Understanding, Tracking & Automation), a Personal AI Assistant built with LangChain and DSPy. This is a modern Python project with comprehensive development tooling and best practices.

## Key Technologies

- **LangChain**: For AI assistant functionality and conversation management
- **DSPy**: For document understanding and retrieval
- **Poetry**: For dependency management
- **Pydantic**: For data validation and settings
- **pytest**: For testing
- **Just**: For task automation (replaces Make)

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

### Import Organization

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
just format         # Format code with Black
just lint           # Lint code with flake8 and mypy
just pre-commit     # Run pre-commit hooks

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
5. **Run quality checks** before committing
6. **Consider security implications** of changes
7. **Think about performance** impact
8. **Maintain backward compatibility** when possible
9. **Update documentation** at `docs/` when needed

## Common Issues & Solutions

### Import Errors

- Check if the module is in the correct directory
- Verify `__init__.py` files exist
- Use relative imports when appropriate

### Type Checking Errors

- Add proper type hints
- Use `Optional` for nullable values
- Use `Union` for multiple types
- Use `List[Type]` for lists

### Testing Issues

- Mock external dependencies
- Use fixtures for common test data
- Test edge cases and error conditions
- Ensure tests are isolated and repeatable

Remember: This is a **modern Python project** with **comprehensive tooling**. Always follow **best practices** and maintain **high code quality**.

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

### Cyclomatic complexity rules (Python)

When writing or modifying Python code, follow these cyclomatic complexity constraints:

- Prefer very simple functions:
  - Target cyclomatic complexity **≤ 5** for most functions.
- Soft limit per function:
  - Functions with cyclomatic complexity **> 7** should be refactored if possible.
- Hard limit per function:
  - Functions with cyclomatic complexity **> 10** are **not allowed** in new or heavily modified code.
  - If you detect such a function, propose a refactor:
    - Extract smaller helper functions.
    - Replace nested `if/elif` with early returns or mapping/dictionary dispatch.
    - Consider polymorphism / strategy pattern instead of large `if` chains.

Additional guidelines:

- Avoid deeply nested control flow:
  - Prefer early returns (`guard clauses`) to reduce nesting.
  - Avoid more than **3 nested levels** of `if/for/while/try` inside the same function.
- For legacy functions that already exceed these limits:
  - Do not make huge refactors unless explicitly requested.
  - Instead, add comments or TODOs suggesting a future refactor and keep complexity from increasing.

When suggesting changes, always:

- Explain briefly how the change reduces cyclomatic complexity.
- Prefer readability over "clever" tricks that make the code harder to understand.
