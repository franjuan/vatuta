# Global Entity Management System

The Global Entity Management System is a core component of Vatuta designed to unify user identities across multiple data
sources (Slack, Jira, Confluence). It assigns a unique `global_id` to each user and enables consistent tagging of content,
facilitation of cross-reference, and identity resolution.

## 1. Core Architecture

The system is implemented in the `src.entities` package.

### Models (`src/entities/models.py`)

- **`Entity`**: Represents a unique, global user identity.
  - `global_id`: Unique UUID for the global entity.
  - `names`: List of known display names for the user.
  - `emails`: List of known email addresses (primary key for linking).
  - `source_refs`: List of `SourceReference` objects linking to specific source identities.
  - `metadata`: Aggregated metadata (e.g., job title, timezone).

- **`SourceReference`**: Links a global entity to a specific user account on a source platform.
  - `source`: The source name (e.g., 'slack', 'jira').
  - `source_user_id`: The ID used by that source (e.g., Slack User ID `U12345`, Jira Account ID `557058...`).

### EntityManager (`src/entities/manager.py`)

The `EntityManager` class is the central entry point. It handles:

1. **Persistence**: Loads and saves the entity registry to a JSON file (default: `data/entities.json`).
2. **Resolution**: The `get_or_create_user` method is the primary API.
    - It accepts source-specific user data (ID, name, email).
    - It attempts to link the incoming user to an existing global entity via **Email Address**.
    - If no match is found, it creates a new `Entity`.
    - It updates existing entities with new source references or metadata if found.
3. **Caching**: It maintains an in-memory cache of source-ID-to-global-ID mappings for fast lookups.

## 2. Configuration

The path to the entities storage file is configured in `config/vatuta.yaml`:

```yaml
entities_manager:
  storage_path: "data/entities.json"
```

If not specified, it defaults to `data/entities.json`.

## 3. Integration with Sources

The `EntityManager` is injected into each Source instance. Sources use it to resolve users and tag content.

### General Tagging Strategy

- **`user:<global_id>`**: This system tag is applied to documents and chunks.
- **Authorship**: The creator of a document (Slack message, Jira issue, Confluence page) is tagged.
- **Mentions**: Users mentioned within the content are scanned, resolved, and tagged on the specific chunk where the
  mention occurs.

### Slack Integration (`src/sources/slack.py`)

- **Resolution**:
  - Uses the local Slack user cache (built from `users.list`) to hydrate User IDs (`U...`) into profiles (Email, Real Name).
  - Registers these profiles with `EntityManager` to obtain global IDs.
- **Mentions**:
  - Scans message text for `<@U.......>` patterns.
  - Resolves the mentioned ID to a global entity and tags the chunk.

### Jira Integration (`src/sources/jira_source.py`)

- **Resolution**:
  - Uses the `accountId`, `displayName`, and `emailAddress` fields from Jira User objects.
- **Tagging**:
  - **Issues**: Reporter and Assignee are tagged.
  - **Comments**: Comment authors are tagged.
  - **History**: Changelog authors are tagged.
- **Mentions**:
  - Scans text for `[~accountid:...]` patterns (Jira Cloud format).
  - Resolves the account ID to a global entity and tags the chunk.

### Confluence Integration (`src/sources/confluence.py`)

- **Resolution**:
  - Uses `accountId` and user details from page version history (`version.by`).
- **Tagging**:
  - **Page Author**: Tagged on the document system tags.
- **Mentions**:
  - Scans naming storage format (HTML) for `<ri:user ri:accountId="...">` tags.
  - Resolves the account ID to a global entity and tags the document.

## 4. Developer Usage

To use the `EntityManager` in a new component:

```python
from src.entities.manager import EntityManager

# Instantiate (usually done in main/client and passed down)
em = EntityManager(storage_path="data/entities.json")

# Resolve a user
user_data = {
    "source_id": "slack",
    "source_user_id": "U123456",
    "email": "jane@example.com",
    "display_name": "Jane Doe"
}
global_id = em.get_or_create_user(**user_data)

# Lookup by source ID (fast, if already known)
entity = em.get_user_by_source_id("slack", "U123456")
if entity:
    print(f"Found global user: {entity.global_id}")
```
