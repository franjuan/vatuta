# Vatuta Personal Assistant - Justfile
# Task automation for development

# Default recipe
default:
    @just --list

# Install dependencies
install:
    poetry install

# Install development dependencies
dev:
    poetry install --with dev

# Run tests
test:
    poetry run pytest

# Run tests with verbose output
test-verbose:
    poetry run pytest -v

# Run tests with coverage
test-coverage:
    poetry run pytest --cov=src --cov-report=html --cov-report=term

# Lint code
lint:
    poetry run flake8 src tests
    poetry run mypy src

# Format code
format:
    poetry run black src tests
    poetry run isort src tests

# Check code formatting
format-check:
    poetry run black --check src tests
    poetry run isort --check-only src tests

# Clean up temporary files
clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    rm -rf build/
    rm -rf dist/
    rm -rf .pytest_cache/
    rm -rf .mypy_cache/
    rm -rf htmlcov/

# Run the assistant
run:
    poetry run python -m src.main

# Run the assistant with the vatuta command
assistant:
    poetry run vatuta

# Run the API server
api:
    poetry run uvicorn src.api.main:app --reload

# Run the API server with specific host and port
api-dev host="0.0.0.0" port="8000":
    poetry run uvicorn src.api.main:app --host {{host}} --port {{port}} --reload

# Build the package
build:
    poetry build

# Publish to PyPI
publish:
    poetry publish

# Update dependencies
update:
    poetry update

# Run all checks (lint, format-check, test)
check:
    just lint
    just format-check
    just test

# Setup development environment
setup: install dev
    @echo "Development environment setup complete!"
    @echo "Run 'just run' to start the assistant or 'just api' to start the API server"

# Show help
help:
    @echo "Available commands:"
    @just --list

# Initialize git repository
git-init:
    git init
    git add .
    git commit -m "Initial commit: Vatuta Personal Assistant"

# Add all files and commit
commit message:
    git add .
    git commit -m "{{message}}"

# Push to remote repository
push:
    git push origin main

# Create a new branch
branch name:
    git checkout -b {{name}}

# Switch to main branch
main:
    git checkout main

# Show git status
status:
    git status

# Show git log
log:
    git log --oneline -10

# Run pre-commit hooks
pre-commit:
    poetry run pre-commit run --all-files

# Install pre-commit hooks
pre-commit-install:
    poetry run pre-commit install

# Show project structure
tree:
    tree -I '__pycache__|*.pyc|.git|.pytest_cache|.mypy_cache|htmlcov|build|dist'

# Show environment info
env:
    poetry env info

# Show installed packages
packages:
    poetry show

# Update lock file
lock:
    poetry lock

# Export requirements
export:
    poetry export -f requirements.txt --output requirements.txt

# Run with specific Python version
python-version:
    poetry env use python3.12

# Create virtual environment
venv:
    poetry env create

# Remove virtual environment
venv-remove:
    poetry env remove python

# Show virtual environment path
venv-path:
    poetry env info --path

# Run shell in virtual environment
shell:
    poetry shell

# Run command in virtual environment
run-cmd cmd:
    poetry run {{cmd}}

# Show help for a specific command
help-cmd cmd:
    @just --list | grep "{{cmd}}"

# Show all available commands
list:
    @just --list

# Allow direnv to load environment
direnv-allow:
    direnv allow

# Deny direnv (remove environment loading)
direnv-deny:
    direnv deny

# Show direnv status
direnv-status:
    direnv status

# Reload direnv environment
direnv-reload:
    direnv reload
