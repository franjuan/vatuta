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

# Run tests (pass args like '-v' for verbose output)
test *args:
    poetry run pytest {{args}}

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
pip-audit:
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
    -docker rm -f vatuta-qdrant 2>/dev/null
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

# Display detailed information about a dependency and trace its reverse dependency tree
# showing which packages in the 'main' group require it.
find-dep dependency:
    poetry show {{dependency}}
    poetry show --tree --why --only main {{dependency}}

dep_path_script := '''
import sys
import subprocess
try:
    import tomllib
except ImportError:
    tomllib = None

target = sys.argv[1].lower()
lines = sys.stdin.read().splitlines()

# Load direct dependencies from pyproject.toml to filter noise
direct_deps = set()
if tomllib:
    try:
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        poetry = config.get("tool", {}).get("poetry", {})
        for pkg in poetry.get("dependencies", {}):
            direct_deps.add(pkg.lower())
        for group in poetry.get("group", {}).values():
            for pkg in group.get("dependencies", {}):
                direct_deps.add(pkg.lower())
    except Exception:
        pass

def is_direct(pkg_name):
    if not direct_deps:
        return True # Fallback if parsing fails
    if pkg_name in direct_deps:
        return True
    for dep in direct_deps:
        if dep.startswith(pkg_name) or pkg_name.startswith(dep):
            return True
    return False

# Get the current installed version of the target package
installed_version = "unknown"
try:
    out = subprocess.check_output(["poetry", "show", target], text=True)
    for line in out.splitlines():
        if line.strip().startswith("version"):
            installed_version = line.split(":", 1)[1].strip()
            break
except Exception:
    pass

stack = []
found = False

for raw in lines:
    if not raw.strip():
        continue
    # Skip poetry warnings
    if raw.startswith("The lock file") or raw.startswith("Upgrade Poetry"):
        continue

    line = raw.replace("│   ", "    ")
    line = line.replace("├── ", "    ")
    line = line.replace("└── ", "    ")
    
    indent = len(line) - len(line.lstrip(" "))
    depth = indent // 4

    content = line.lstrip()
    parts = content.split()
    name = parts[0].lower()
    version = parts[1] if len(parts) > 1 else ""

    display_name = f"{name} ({version})" if version else name

    if len(stack) <= depth:
        stack.extend([None] * (depth - len(stack) + 1))
    
    # Store a tuple of (name, display_name) to easily check direct deps by name
    stack[depth] = (name, display_name)
    stack = stack[:depth + 1]

    if name == target:
        root_name = stack[0][0]
        if is_direct(root_name):
            path_str = " -> ".join([s[1] for s in stack])
            print(f"{path_str} [current: {installed_version}]")
            found = True

if not found:
    sys.exit(f"No direct paths found for {target} from pyproject.toml")
'''

# Find and show the shortest dependency paths from direct project dependencies to a specific package.
# Usage:
#   just dep-path numpy          # Search across all dependency groups
#   just dep-path numpy main     # Search exclusively within the 'main' group
dep-path package group="all":
    @if [ "{{group}}" = "all" ]; then \
        poetry show --tree; \
    else \
        poetry show --tree --only "{{group}}"; \
    fi | python -c '{{dep_path_script}}' "{{package}}"