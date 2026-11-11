# Temporary Blog Demonstration Plan: GitHub Free & Jira Cloud Free via MCP

**Date**: 2026-11-11
**Feature**: [002-policy-checker](spec.md) | **Master Plan**: [plan.md](plan.md)
**Target Location (when executed)**: `pocs/policy_demo/` (strictly isolated from `src/`, `tests/`, and CI)

## 1. Purpose, Scope, and Non-Goals

This document designs a **one-time, temporary local demonstration** exclusively to showcase Vatuta's Policy Validation
Engine in a technical blog post with realistic compliance outcomes and terminal screenshots.

- **Not a product dependency**: GitHub and Jira are **demonstration evidence sources only**. Neither `src/policy/` nor
  `src/client/` may contain GitHub-specific or Jira-specific logic, schemas, adapters, or tool names.
- **Not a maintained environment or test suite**: The demo dataset is not used for regression testing and requires no
  ongoing synchronization, provisioning framework, or automated teardown.
- **Strict CI isolation**: GitHub Actions workflows (`.github/workflows/`) MUST NOT run demo scripts, start GitHub/Jira
  MCP containers, or require `GITHUB_*` / `JIRA_*` secrets.
- **No custom UI**: All outputs use Vatuta's standard Rich CLI formatting (`vatuta policy create`, `check`, `execute`)
  alongside standard GitHub and Jira web pages.

---

## 2. Demonstration Infrastructure & Source-Independent MCP Setup

### 2.1 Selected Services & Existing MCP Servers

Reuse Vatuta's unchanged `MCPServer` Docker-stdio runtime (`src/mcp/server.py`) and `MCPContainerConfig`
(`src/mcp/config.py`) with two established open-source MCP server images—no custom MCP servers or wrappers:

| Component | Choice | Justification & Hardening |
| --- | --- | --- |
| Source code host | **GitHub Free** (`<github_owner>/vatuta-payments-demo`) | Free tier supports branches, commits, open/draft PRs, and fine-grained read-only PATs. |
| Issue tracker | **Jira Cloud Free** (project `PAY`, `<site>.atlassian.net`) | Free tier supports Scrum/Kanban statuses, priorities, issue types, JQL, and API tokens. |
| GitHub MCP server | `ghcr.io/github/github-mcp-server@sha256:<digest>` | Official GitHub stdio container; supports `GITHUB_READ_ONLY=1`, `search_pull_requests`, and `list_commits`. |
| Jira MCP server | `ghcr.io/sooperset/mcp-atlassian@sha256:<digest>` | Mature stdio container supporting Jira Cloud `JIRA_API_TOKEN`, `READ_ONLY_MODE=true`, and `jira_search`. |

Before running the demo locally, pull both images once (`docker pull ...`), inspect their `RepoDigests`, and pin the
exact `@sha256:...` digests in `pocs/policy_demo/vatuta.demo.yaml` per Vatuta's container security policy.

### 2.2 Local Credentials (`.env` Only — Never Committed)

Store demo credentials exclusively in the local `.env` file at the repository root (git-ignored):

```dotenv
# Credentials for the model selected in the standalone demo configuration
GEMINI_API_KEY="<local-gemini-api-key>"

# GitHub Free fine-grained PAT (Read-only: Metadata, Contents, Pull requests on vatuta-payments-demo)
GITHUB_PERSONAL_ACCESS_TOKEN="<github-fine-grained-pat>"
GITHUB_READ_ONLY="1"

# Jira Cloud Free API token (scoped to demo Atlassian site)
JIRA_URL="https://<demo-site>.atlassian.net"
JIRA_USERNAME="<demo-email@example.com>"
JIRA_API_TOKEN="<jira-api-token>"
READ_ONLY_MODE="true"
```

### 2.3 Isolated Demo Configuration (`pocs/policy_demo/vatuta.demo.yaml`)

Provide the following complete standalone template as `pocs/policy_demo/vatuta.demo.yaml` during implementation.
It must not depend on copying a personal configuration or on implicit merging: `--config` loads one complete file.
Replace `<model-name>` with a Gemini model available to the configured account and replace both `<pinned-digest>`
values with the verified image digests. Keep actual credentials only in `.env`. Another supported LLM provider can be
selected by changing the backend model identifier and documenting its required credential variable.

The required `rag`, `qdrant`, and `sources` sections satisfy the existing configuration schema. Policy commands use
only the selected LLM backend and MCP evidence sources; they must not start RAG ingestion, download embeddings, or
connect to Qdrant. No Qdrant server is needed for this demo. Source-specific profiles remain isolated here.

```yaml
rag:
  llm_backends:
    demo:
      model: "gemini/<model-name>"
      temperature: 0.2
      max_tokens: 2000
  router_backend: "demo"
  generator_backend: "demo"

qdrant:
  url: "http://localhost:6333"
  collection_name: "unused_policy_demo"
  embeddings_model: "intfloat/multilingual-e5-small"
  sparse_embeddings_model: "Qdrant/bm25"

sources: {}

mcp_servers:
  github:
    image: "ghcr.io/github/github-mcp-server@sha256:<pinned-digest>"
    auto_pull: false
    allow_network: true
    read_only: true
    memory: "256m"
    memory_swap: "256m"
    pids_limit: 128
    env_passthrough:
      - GITHUB_PERSONAL_ACCESS_TOKEN
      - GITHUB_READ_ONLY
    allowed_tools:
      - "^search_pull_requests$"
      - "^list_commits$"

  jira:
    image: "ghcr.io/sooperset/mcp-atlassian@sha256:<pinned-digest>"
    auto_pull: false
    allow_network: true
    read_only: true
    memory: "256m"
    memory_swap: "256m"
    pids_limit: 128
    env_passthrough:
      - JIRA_URL
      - JIRA_USERNAME
      - JIRA_API_TOKEN
      - READ_ONLY_MODE
    allowed_tools:
      - "^jira_search$"

policy:
  creation_backend: "demo"
  evaluation_backend: "demo"
  controls_directory: "pocs/policy_demo/controls"
  trace_directory: "logs/policy_demo"
  trial_attempts: 1
  evidence_profiles:
    github.search_pull_requests:
      contract_reference: "github-mcp-server SearchIssuesOutput single-page completeness"
      arguments_schema:
        type: object
        required: ["query", "page", "perPage"]
        properties:
          query: {type: string, minLength: 1}
          owner: {type: string, minLength: 1}
          repo: {type: string, minLength: 1}
          page: {type: integer, const: 1}
          perPage: {type: integer, minimum: 1, maximum: 100}
      complete_when: >-
        arguments.page = 1 and $type(data.items) = "array" and
        $type(data.total_count) = "number" and data.total_count >= 0 and
        $floor(data.total_count) = data.total_count and $type(data.incomplete_results) = "boolean" and
        data.incomplete_results = false and $count(data.items) = data.total_count
      trace_redact_paths: []
      trace_safe_text_paths: ["/items/*/title", "/items/*/body", "/items/*/html_url"]

    github.list_commits:
      contract_reference: "github-mcp-server list_commits page-1 under-limit completeness"
      arguments_schema:
        type: object
        required: ["owner", "repo", "sha", "page", "perPage"]
        properties:
          owner: {type: string, minLength: 1}
          repo: {type: string, minLength: 1}
          sha: {type: string, minLength: 1}
          page: {type: integer, const: 1}
          perPage: {type: integer, minimum: 1, maximum: 100}
      complete_when: >-
        arguments.page = 1 and $type(data) = "array" and
        $count(data) < arguments.perPage
      trace_redact_paths: []
      trace_safe_text_paths: ["/commit/message", "/html_url"]

    jira.jira_search:
      contract_reference: "mcp-atlassian JiraSearchResult single-page JQL completeness"
      arguments_schema:
        type: object
        required: ["jql", "limit"]
        properties:
          jql: {type: string, minLength: 1}
          fields: {type: string}
          limit: {type: integer, minimum: 1, maximum: 50}
      complete_when: >-
        ($not($exists(arguments.page_token)) or arguments.page_token = null) and
        ($not($exists(arguments.start_at)) or arguments.start_at = 0) and
        $type(data.issues) = "array" and
        ($not($exists(data.start_at)) or data.start_at = 0) and
        ($not($exists(data.next_page_token)) or data.next_page_token = null) and
        $type(data.total) = "number" and $floor(data.total) = data.total and
        ((data.total >= 0 and $count(data.issues) = data.total) or
         (data.total = -1 and $count(data.issues) < arguments.limit))
      trace_redact_paths: []
      trace_safe_text_paths: ["/issues/*/key", "/issues/*/summary", "/issues/*/description"]
```

---

Validate the pinned Jira tool contract before enabling the `total = -1` branch: it is a documented cursor-response
sentinel, not a general fallback. A known nonnegative total must equal the issue count even when the response is shorter
than the limit. For example, `issues=[]`, `total=5`, `limit=50` is incomplete and must produce ERROR, never
NOT_APPLICABLE. Missing/wrong-type totals or unknown pagination semantics fail qualification until the profile has a
documented positive completeness proof. Exercise these cases using local profile fixtures before taking screenshots.

The standalone template and its setup instructions are planned artifacts, not a claim that policy runtime support or
live source qualification already exists. Keep the generic loader unchanged; add the approved optional `policy` model
during production implementation. Document the model ID, pinned images, demo repository/site, and `.env` variable names
needed to reproduce the run from a clean checkout.

## 3. Realistic Software Development Scenario (`vatuta-payments-demo` + `PAY`)

The scenario models a small engineering team maintaining a **Payments Service** (`vatuta-payments-demo`) tracked in Jira
project **`PAY`**. It requires only **5 Jira issues**, **3 commits on `main`**, and **4 Pull Requests**—small enough to
create manually in ~15 minutes (or via a simple `gh` CLI shell snippet in `pocs/policy_demo/README.md`).

### 3.1 Jira Issues (Project `PAY`)

| Key | Type | Priority | Status | Summary | Description Highlights |
| --- | --- | --- | --- | --- | --- |
| `PAY-100` | Story | Medium | `Done` | Bootstrap payments service skeleton | Initial FastAPI service layout and health endpoint. |
| `PAY-101` | Story | `High` | `In Review` | Enforce idempotency keys on `POST /charges` | Prevent duplicate customer charges on client network retries. |
| `PAY-102` | Bug | `High` | `Backlog` | Retry transient 502/503 webhook delivery failures | Untriaged backlog item; engineering work not yet approved to start. |
| `PAY-103` | Task | Medium | `Done` | Enforce TLS 1.2+ on outbound gateway HTTP client | Security hardening task completed and merged to `main`. |
| `PAY-104` | Story | Low | `In Progress` | Add Prometheus latency histogram for `/charges` | Early spike currently in a draft PR. |

*(Note: Zero `Bug` issues of priority `Highest` are `Done`, intentionally exercising `NOT_APPLICABLE` on Rule 4.)*

### 3.2 GitHub `main` Branch Commits (`<github_owner>/vatuta-payments-demo`)

Create the repository with three direct linear (non-merge) commits on `main`:

1. `PAY-100: bootstrap payments service structure` — **Compliant** with commit-ticket hygiene.
2. `PAY-103: enforce TLS 1.2+ on outbound gateway client` — **Compliant** with commit-ticket hygiene.
3. `hotfix: bump gateway read timeout to 15s` — **Non-compliant** (untracked hotfix pushed directly to `main` with no
   `PAY-<digits>` ticket key).

### 3.3 GitHub Pull Requests (`PR #1` – `PR #4`)

| PR | Branch | State / Draft | Title | Body Content | Linked Jira Issue |
| --- | --- | --- | --- | --- | --- |
| `#1` | `feat/pay-101-idempotency` | Open (`draft: false`) | `feat(PAY-101): validate idempotency keys on charge requests` | Explains change + concrete `## Rollback Plan`: *"Disable `PAYMENTS_IDEMPOTENCY_ENFORCE=false` via config map and restart pods (<2 min); new `idempotency_key` column is nullable and backward-compatible, requiring no schema rollback."* | `PAY-101` (`In Review`, `High`) |
| `#2` | `fix/pay-102-webhook-retry` | Open (`draft: false`) | `fix(PAY-102): retry transient webhook failures` | Explains retry loop + superficial `## Rollback Plan`: *"TBD - we will revert or figure out a fix if webhooks break in prod."* | `PAY-102` (`Backlog`, `High`) |
| `#3` | `chore/timeout-constants` | Open (`draft: false`) | `chore: tune default connection pool constants` | *"Quick adjustment to pool constants observed during local testing."* (No Jira key anywhere.) | None (`null`) |
| `#4` | `wip/pay-104-metrics` | Open (`draft: true`) | `wip(PAY-104): draft latency histogram` | *"Early WIP draft, do not review yet."* | `PAY-104` (`In Progress`, `Low`) |

---

## 4. Proposed Example Policy and Rules

### 4.1 Example Policy Document (`pocs/policy_demo/policy.md`)

```markdown
# Software Change Management & Release Traceability Policy

1. **Active Jira Traceability for Pull Requests**: Every open non-draft pull request in the repository must reference a
   valid Jira issue key from the project (such as `PAY-123`) in its title or description, and that linked Jira issue
   must currently be in `In Progress` or `In Review` status (never `Backlog` or unlinked).
2. **Main Branch Commit Ticket Attribution**: Every non-merge commit on the `main` branch must include a project Jira
   issue key (`<PROJECT>-<number>`) in the first line of its commit message.
3. **Actionable Rollback Plan for High-Priority Changes**: Every open non-draft pull request linked to a `High` or
   `Highest` priority Jira issue must document a concrete, actionable operational rollback or mitigation procedure in
   its pull request description (placeholder notes such as "TBD" or "just revert" do not satisfy this requirement).
4. **Postmortem Documentation on Critical Resolved Bugs**: Every `Bug` issue with `Highest` priority in `Done` status
   must include a `Postmortem:` link or document reference in its Jira description.
```

### 4.2 Detailed Rule Breakdown

#### Rule 1: `pr-jira-traceability` — Cross-Source PR-to-Jira Status Verification

- **Identifier & Description**: `pr-jira-traceability` — Every open non-draft PR must link to an existing Jira issue
  currently in `In Progress` or `In Review`.
- **Business / Governance Objective**: Prevent untracked "shadow changes" and premature code reviews on unapproved
  `Backlog` items.
- **Applicability Criteria**: Open non-draft pull requests (`draft = false`) in `<github_owner>/<github_repo>`.
- **Selection Criteria (JSONata)**: Filter `evidence.prs.items[draft = false]`, extract `<jira_project>-<digits>` from
  `title` or `body`, join against `evidence.issues.issues` by `key`, and project `{id, descriptive_id, jira_key,
  jira_status}` (preserving `null` when absent).
- **Evidence Required & Source**: **Both GitHub and Jira** — `github.search_pull_requests` (`is:pr is:open`) and
  `jira.jira_search` (`project = <jira_project>`).
- **Evaluation Mode**: **Deterministic (CEL)** —
  `subject.jira_key != null && (subject.jira_status == "In Progress" || subject.jira_status == "In Review")`.
- **Compliant Scenario**: `PR #1` links `PAY-101`, whose Jira status is `In Review` → `COMPLIANT`.
- **Non-Compliant Scenarios**: `PR #2` links `PAY-102` in `Backlog` → `NON_COMPLIANT`; `PR #3` has no Jira key
  (`jira_key == null`) → `NON_COMPLIANT`. (`PR #4` is `draft: true` and excluded from selection.)
- **Limitations / Ambiguities**: If a PR references multiple Jira keys, Selection deterministically binds the first
  project key matched in `title` before `body`.

#### Rule 2: `commit-ticket-prefix` — Default-Branch Commit Hygiene

- **Identifier & Description**: `commit-ticket-prefix` — Every non-merge commit on `main` must cite `<PROJECT>-<digits>`
  in its summary line.
- **Business / Governance Objective**: Preserve a clean, auditable production commit log tied to engineering tickets.
- **Applicability Criteria**: Non-merge commits (`$count(parents) <= 1`) on branch `main`.
- **Selection Criteria (JSONata)**: Filter `evidence.commits[$count(parents) <= 1]` and project `{id: sha,
  descriptive_id: html_url, summary: $split(commit.message, "\n")[0]}`.
- **Evidence Required & Source**: **GitHub only** — `github.list_commits` (`sha: "main"`, `page: 1`, `perPage: 50`).
- **Evaluation Mode**: **Deterministic (CEL)** —
  `subject.summary.matches("(^|[^A-Z0-9])" + parameters.jira_project + "-[0-9]+([^0-9]|$)")`.
- **Compliant Scenario**: Commits `PAY-100: bootstrap...` and `PAY-103: enforce TLS 1.2+...` → `COMPLIANT`.
- **Non-Compliant Scenario**: Commit `hotfix: bump gateway read timeout to 15s` → `NON_COMPLIANT`.
- **Limitations / Ambiguities**: `list_commits` lacks a total-count envelope; completeness holds on page 1 only while
  commit count `< perPage` (50). Larger repositories would require an explicit `since`/`until` release window parameter
  or produce an incomplete-evidence `ERROR` rather than silent partial compliance.

#### Rule 3: `high-priority-rollback-plan` — Semantic Rollback Readiness Review

- **Identifier & Description**: `high-priority-rollback-plan` — Open non-draft PRs linked to `High` or `Highest`
  priority Jira issues must include a concrete operational rollback/mitigation procedure.
- **Business / Governance Objective**: Reduce MTTR on high-risk changes by requiring real operational recovery steps
  before merge, catching boilerplate headings that fool regexes.
- **Applicability Criteria**: Open non-draft PRs linked to a Jira issue whose `priority.name` is `High` or `Highest`.
- **Selection Criteria (JSONata)**: Join non-draft PRs from `evidence.prs.items` with `evidence.issues.issues`, retain
  matches with `priority in ["High", "Highest"]`, and project `{id, descriptive_id, pr_title, pr_body, jira_key,
  jira_priority}`.
- **Evidence Required & Source**: **Both GitHub and Jira** — `github.search_pull_requests` and `jira.jira_search`.
- **Evaluation Mode**: **Semantic (`mode: llm`)** — `deterministic_insufficiency`: *"Keyword or section-header checks
  cannot distinguish a concrete operational mitigation (feature flags, schema compatibility, safe revert steps) from
  placeholder text such as 'Rollback Plan: TBD'."*
- **Compliant Scenario**: `PR #1` (`PAY-101`, `High`) specifies feature-flag toggle `PAYMENTS_IDEMPOTENCY_ENFORCE=false`
  and nullable backward-compatible column → `COMPLIANT`.
- **Non-Compliant Scenario**: `PR #2` (`PAY-102`, `High`) contains a `## Rollback Plan` header but body says `"TBD - we
  will revert or figure out a fix..."` → `NON_COMPLIANT`.
- **Limitations / Ambiguities**: Live LLM semantic judgment can vary on borderline prose, though clear concrete vs
  `"TBD"` cases evaluate reliably and record full prompt/response attribution in the invocation trace.

#### Rule 4: `critical-incident-postmortem-link` — Critical Bug Postmortem Verification (`NOT_APPLICABLE` Showcase)

- **Identifier & Description**: `critical-incident-postmortem-link` — Resolved (`Done`) `Bug` issues of priority
  `Highest` must include a `Postmortem:` reference in their description.
- **Business / Governance Objective**: Guarantee root-cause documentation on severity-1 bugs while demonstrating honest
  `NOT_APPLICABLE` reporting when zero qualifying incidents exist.
- **Applicability Criteria**: Jira issues in `<jira_project>` with `issuetype = Bug`, `priority = Highest`, and
  `status = Done`.
- **Selection Criteria (JSONata)**: Project matching issues from `evidence.critical_bugs.issues` into `{id: key,
  descriptive_id: key, summary: summary, description: description}`. Empty array `[]` yields `NOT_APPLICABLE`.
- **Evidence Required & Source**: **Jira only** — `jira.jira_search`.
- **Evaluation Mode**: **Deterministic (CEL)** —
  `subject.description != null && subject.description.contains("Postmortem:")`.
- **Compliant / Non-Compliant Unit Tests**: Exercised automatically during `vatuta policy create` via synthetic
  `test_cases.criterion_tests` (`COMPLIANT` with `Postmortem: https://...`, `NON_COMPLIANT` without it).
- **Live Demo Outcome**: Zero `Highest`-priority `Done` bugs exist in `PAY`, so Selection returns `[]` on complete
  evidence and Vatuta reports control outcome **`NOT_APPLICABLE`** (with `criterion` not executed).
- **Limitations / Ambiguities**: Checks presence of a `Postmortem:` reference string in Jira; does not fetch Confluence
  to judge postmortem quality.

---

## 5. How Vatuta Executes the Demo End-to-End (Zero Custom Engine Logic)

1. **Authoring (`vatuta policy create`)**:
   - Reads `pocs/policy_demo/policy.md` and construction context (`"Repository <owner>/vatuta-payments-demo; Jira
     project PAY"`).
   - Decomposes the 4 obligations into 4 rules, reuses shared parameter names (`github_owner`, `github_repo`,
     `jira_project`) via `SharedParameterCatalog`, explores live MCP responses via bounded JSONata projections, verifies
     every rule's `selection_tests` and `criterion_tests` plus 1 live trial, and saves 4 `<id>.yaml` controls into
     `pocs/policy_demo/controls/`.
2. **Prompt Check (`vatuta policy check "<prompt>"`) & Direct Execution (`vatuta policy execute`)**:
   - Step 1 selects all 4 controls from `pocs/policy_demo/controls/` via `ClientControlQuery` and merges shared
     parameters deterministically.
   - Step 2 extracts `github_owner`, `github_repo`, and `jira_project` from the natural-language prompt and calls
     `PolicyService.execute`.
   - `PolicyService.execute` invokes `github` and `jira` Docker MCP containers read-only, proves completeness via
     `policy.evidence_profiles`, runs JSONata selections, runs deterministic CEL for Rules 1, 2, 4 and DSPy semantic
     judgment for Rule 3, and renders Rich tables and trace URIs.

### Expected Demo Outcomes Summary

| Rule | Mode | Evaluated Subjects (`--result-filter all`) | Default Output (`--result-filter violations`) |
| --- | --- | --- | --- |
| `pr-jira-traceability` | `deterministic` | `PR #1`: `COMPLIANT`; `PR #2`: `NON_COMPLIANT`; `PR #3`: `NON_COMPLIANT` (`PR #4` draft excluded) | Findings: `PR #2`, `PR #3` |
| `commit-ticket-prefix` | `deterministic` | `c1 (PAY-100)`: `COMPLIANT`; `c2 (PAY-103)`: `COMPLIANT`; `c3 (hotfix)`: `NON_COMPLIANT` | Finding: `c3 (hotfix)` |
| `high-priority-rollback-plan` | `llm` | `PR #1`: `COMPLIANT`; `PR #2`: `NON_COMPLIANT` | Finding: `PR #2` |
| `critical-incident-postmortem-link` | `deterministic` | `[]` → Control outcome `NOT_APPLICABLE` | Outcome: `NOT_APPLICABLE` (0 findings, 0 errors) |

---

## 6. Execution Steps for Capturing Blog Screenshots

Run locally from the repository root after implementation is complete and `.env` + `pocs/policy_demo/vatuta.demo.yaml`
are configured:

1. **Screenshot 1 — Real Development Context (Jira + GitHub UI)**:
   - Capture the Jira `PAY` board/issues (`PAY-100`..`PAY-104` showing `PAY-101` `In Review` vs `PAY-102` `Backlog`)
     side by side with GitHub's Pull Requests list (`PR #1`..`PR #4`) and `main` commit history.
2. **Screenshot 2 — Natural-Language Policy & Step-by-Step Control Creation**:
   - Display `pocs/policy_demo/policy.md` and run:

     ```bash
     poetry run vatuta --config pocs/policy_demo/vatuta.demo.yaml policy create \
       --policy-file pocs/policy_demo/policy.md \
       --policy-id change-governance-demo \
       --construction-context "GitHub repository <owner>/vatuta-payments-demo and Jira project PAY." \
       --tag blog-demo --trial-attempts 1
     ```

   - Capture Rich stage progress and the summary of 4 generated controls (`3 deterministic`, `1 llm`, unit tests
     passed).
3. **Screenshot 3 — Generated Self-Testing YAML Control**:
   - Display one generated YAML file in `pocs/policy_demo/controls/` highlighting `parameters`, `evidence`, JSONata
     `selection`, CEL/`llm` `criterion`, and embedded `test_cases`.
4. **Screenshot 4 — Natural-Language Compliance Check (`vatuta policy check` — Violations View)**:
   - Run:

     ```bash
     poetry run vatuta --config pocs/policy_demo/vatuta.demo.yaml policy check \
       "Check change-governance compliance for GitHub repo <owner>/vatuta-payments-demo and Jira project PAY"
     ```

   - Capture Vatuta selecting the 4 controls, extracting parameters, fetching MCP evidence, and rendering actionable
     `NON_COMPLIANT` findings (`PR #2`, `PR #3`, `hotfix` commit) plus `NOT_APPLICABLE` on Rule 4.
5. **Screenshot 5 — Full Audit View (`--result-filter all`)**:
   - Run `vatuta --config pocs/policy_demo/vatuta.demo.yaml policy execute --policy-id change-governance-demo
     --parameters '{"github_owner":"<owner>","github_repo":"vatuta-payments-demo","jira_project":"PAY"}'
     --result-filter all` to show both `COMPLIANT` (`PR #1`, `PAY-100`, `PAY-103`) and `NON_COMPLIANT` subjects.
6. **Screenshot 6 — Auditable Sanitized Trace Inspection**:
   - Inspect the execution trace under `logs/policy_demo/<invocation_id>/` showing recorded MCP calls, completeness
     proofs, and Rule 3's semantic evaluation prompt/verdict correlation.
