# Contributing to Vatuta

Thanks for your interest in contributing! This document explains how to get started,
the standards the project follows, and what to expect from the contribution process.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)

---

## Getting Started

### Prerequisites

| Tool | Notes |
| ---- | ----- |
| [Python](https://www.python.org/) >= 3.12 | Required |
| [Poetry](https://python-poetry.org/) | Dependency management |
| [Just](https://github.com/casey/just) | Task runner |
| [Docker](https://www.docker.com/) | Required for Qdrant (integration tests) |
| [direnv](https://direnv.net/) | Optional but recommended |

### Set up

Please refer to the [Installation](README.md#installation) section in the `README.md`
for complete instructions on how to fork the repository, install development dependencies,
and configure your local environment.

---

## Development Workflow

```bash
just format          # Auto-format code (Black, Ruff, isort)
just lint            # Run all linters (Ruff, mypy, pydocstyle, detect-secrets)
just test            # Run the test suite
just check           # lint + format-check + test (run before opening a PR)
just pre-commit      # Run all pre-commit hooks on all files
just --list          # See all available commands
```

Always run `just check` before pushing. The CI pipeline runs the same checks
and your PR will not be merged if they fail.

---

## Code Standards

### Style & Polishing

- **Formatter**: Black with a 120-character line length.
- **Linter**: Ruff (rules E, W, F, I, C, B).
- **Imports**: isort (Black-compatible profile).
- **Type hints**: Required on all public functions and methods (strict mypy, Python 3.12+).
- **Docstrings**: Google convention via `pydocstyle`, required on all public classes/functions.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Spelling**: Enforced globally by `codespell`.
- **Documentation**: Markdown linting enforced by `markdownlint-cli2`, YAML by `yamllint`.

### Complexity (Xenon/Radon)

- Functions should be ≤ 30 lines; prefer ≤ 20.
- Cyclomatic complexity is enforced strictly in both pre-commit and CI:
  - **Max Absolute (Per Function):** Grade D (Complexity ≤ 30)
  - **Max Modules (Average per Module):** Grade B (Complexity ≤ 10)
  - **Max Average (Total Average):** Grade B (Complexity ≤ 10)

### Code Duplication (jscpd)

- Code duplication is strictly monitored by `jscpd` in pre-commits and CI.
- **Thresholds**: Minimum of 50 tokens and minimum of 5 lines triggers an error.
- Refactor duplicated code into shared helper functions or base classes.

### One class per file

Each module file should define at most one main class. Helpers and small data
classes that directly support that class are allowed in the same file.

### Type safety

The project uses strict mypy. Key rules:

- Use specific types instead of `Any` where possible
- Handle `Optional` types explicitly (check for `None` before attribute access)
- Use modern Python 3.12+ type syntax (`list[str]`, `dict[str, int]`, not `List`, `Dict`)

```python
# Good
def get_documents(source_id: str, limit: int = 20) -> list[Document]:
    ...

# Bad — missing type hints
def get_documents(source_id, limit=20):
    ...
```

### Security

- **Secrets**: Never hardcode secrets or credentials. Pre-commit runs `detect-secrets scan` against
  `.secrets.baseline` to prevent accidental leaks.
- **SAST**: Static analysis for security and bugs is enforced by both `Bandit` (Python security) and `Semgrep`.
- **Dependencies**: New dependencies must not introduce known CVEs. Audited automatically by `pip-audit`.

---

## Testing

- Every new function or class must have at least one corresponding unit test
- Tests live in `tests/`, mirroring the `src/` structure
  (e.g. `src/rag/engine.py` → `tests/rag/test_engine.py`)
- Use `pytest` fixtures and `unittest.mock` to mock external dependencies
- Minimum coverage enforced by CI: 25% overall (aim higher for new code)

```bash
just test                  # Run all tests
just test tests/rag/       # Run a specific sub-suite
just test-coverage         # Generate HTML coverage report in htmlcov/
```

### Test naming

Use descriptive names that describe **what** is being tested and **under what condition**:

```python
def test_collect_documents_returns_empty_when_no_projects_configured() -> None:
    ...

def test_entity_manager_resolves_same_email_to_single_global_id() -> None:
    ...
```

---

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(optional scope): <short summary>

[optional body]

Signed-off-by: Your Name <your@email.com>
```

Common types:

| Type | When to use |
| ---- | ----------- |
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that is neither a fix nor a feature |
| `test` | Adding or fixing tests |
| `chore` | Tooling, dependencies, CI |
| `perf` | Performance improvement |

**All commits must be signed off** (see [DCO](#developer-certificate-of-origin-dco) below).

```bash
git commit -s -m "feat(slack): add support for filtering by channel name pattern"
```

---

## Pull Request Process

1. **Create a branch** from `main` with a descriptive name:

   ```bash
   git checkout -b feat/slack-channel-filter
   ```

2. **Keep changes focused** — one logical change per PR. Large changes are harder to review and slower to merge.

3. **Update documentation** — if your change affects behaviour, update the relevant files in `docs/`.

4. **Ensure CI passes** — run `just check` locally before pushing.

5. **Open a Pull Request** with:
   - A clear title following the commit convention
   - A description of *what* changed and *why*
   - Reference to any related issues (`Closes #123`)

6. **Respond to review comments** — the project maintainer may request changes before merging.

---

## Developer Certificate of Origin (DCO)

By contributing, you certify that you wrote the code or have the right to submit it,
as described in the [DCO](DCO) file. All commits must include a `Signed-off-by` line.

Add it automatically with the `-s` flag:

```bash
git commit -s -m "fix: correct chunk size calculation"
```

The CI pipeline checks that every commit in a PR is signed off.
If you forgot to sign off previous commits, you can amend them:

```bash
# Amend the last commit
git commit --amend -s

# Or rebase all commits in your branch to add sign-off
git rebase --signoff main
```
