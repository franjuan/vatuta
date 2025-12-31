# Vatuta Personal Assistant - Justfile
# Task automation for development

# Default recipe
default:
    @just --list

# Install dependencies
install: install-spacy
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
    poetry run ruff check src tests
    poetry run mypy src
    poetry run pydocstyle src tests --convention=google
    poetry run detect-secrets scan

# Format code
format:
    poetry run black src tests
    poetry run ruff check src tests --fix
    poetry run isort src tests

# Check code formatting
format-check:
    poetry run black --check src tests
    poetry run isort --check-only src tests

# Run cyclomatic complexity check
cc:
    poetry run radon cc src -a -na
    poetry run radon mi src


# Audit dependencies for vulnerabilities
audit:
    poetry run pip-audit --desc


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
    poetry run vatuta

# Run the assistant with query and stats (k defaults to 20)
assistant query k="20":
    poetry run vatuta --query "{{query}}" --k {{k}} --show-stats --show-sources

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
setup: install dev install-spacy
    @echo "Development environment setup complete!"
    @echo "Run 'just run' to start the assistant"

# Show help
help:
    @echo "Available commands:"
    @just --list

# Add all files and commit
commit message:
    git add .
    git commit -m "{{message}}"

# Push to remote repository (current branch)
push:
    git push origin $(git branch --show-current)

# Push to remote repository with specific branch
push-branch branch:
    git push origin {{branch}}

# Create a new branch
branch name:
    git checkout -b {{name}}

# Switch to main branch
main:
    git checkout main

# Switch to a specific branch
switch branch:
    git checkout {{branch}}

# List all branches
branches:
    git branch -a

# Delete a branch (local)
delete-branch branch:
    git branch -d {{branch}}

# Show git status
status:
    git status

# Show git log
log:
    git log --oneline -10

# Pull latest changes from remote
pull:
    git pull origin $(git branch --show-current)

# Pull from specific branch
pull-branch branch:
    git pull origin {{branch}}

# Merge a branch into current branch
merge branch:
    git merge {{branch}}

# Rebase current branch onto main
rebase:
    git rebase main

# Show current branch
current-branch:
    git branch --show-current

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

# Install spaCy model (required for text splitting)
install-spacy:
    poetry run python -m spacy download en_core_web_sm

# Qdrant Development Commands

# Start Qdrant Docker container
qdrant-start:
    @echo "🚀 Starting Qdrant..."
    docker pull qdrant/qdrant:latest
    docker run -d --name vatuta-qdrant -p 6333:6333 -p 6334:6334 -e QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY} -v {{justfile_directory()}}/data/qdrant:/qdrant/storage:z qdrant/qdrant:latest
    @echo "✅ Qdrant started at http://localhost:6333"
    @echo "📊 Dashboard: http://localhost:6333/dashboard"

# Stop Qdrant Docker container
qdrant-stop:
    @echo "🛑 Stopping Qdrant..."
    -docker stop vatuta-qdrant
    -docker rm vatuta-qdrant
    @echo "✅ Qdrant stopped"

# Check Qdrant status
qdrant-status:
    @echo "📊 Qdrant Status:"
    @docker ps -a --filter "name=vatuta-qdrant" --format "table {{'{{'}}.Names{{'}}'}}\t{{'{{'}}.Status{{'}}'}}\t{{'{{'}}.Ports{{'}}'}}"

# View Qdrant logs
qdrant-logs:
    @echo "📋 Qdrant Logs:"
    docker logs vatuta-qdrant

# Restart Qdrant
qdrant-restart: qdrant-stop qdrant-start

# Open Qdrant dashboard in browser
qdrant-dashboard:
    @echo "🌐 Opening Qdrant dashboard..."
    xdg-open http://localhost:6333/dashboard || open http://localhost:6333/dashboard || echo "Open http://localhost:6333/dashboard in your browser"
