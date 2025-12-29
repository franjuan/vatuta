"""Entity manager for handling global user registry and linking."""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional

from .models import Entity, SourceReference

logger = logging.getLogger(__name__)


class EntityManager:
    """Manages the lifecycle and linking of global user entities."""

    def __init__(self, storage_path: str = "data/entities.json"):
        """Initialize the entity manager with a storage path.

        Args:
            storage_path: Path to the JSON file where entities are persisted.
        """
        self.storage_path = Path(storage_path).expanduser().resolve()
        self._entities: Dict[str, Entity] = {}  # global_id -> Entity
        self._source_map: Dict[SourceReference, str] = {}  # SourceReference -> global_id
        self._email_map: Dict[str, str] = {}  # email -> global_id
        self._load()

    def _load(self) -> None:
        """Load entities from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    entity = Entity.from_dict(item)
                    self._register_entity_in_maps(entity)
        except Exception as e:
            logger.error(f"Failed to load entities from {self.storage_path}: {e}")

    def _save(self) -> None:
        """Save entities to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = [entity.to_dict() for entity in self._entities.values()]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save entities to {self.storage_path}: {e}")

    def _register_entity_in_maps(self, entity: Entity) -> None:
        """Register entity in internal lookup maps."""
        self._entities[entity.global_id] = entity
        for ref in entity.source_refs:
            self._source_map[ref] = entity.global_id
        for email in entity.emails:
            self._email_map[email.lower()] = entity.global_id

    def get_user_by_source_id(self, source_type: str, source_id: str, source_user_id: str) -> Optional[Entity]:
        """Retrieve an entity by its source reference."""
        ref = SourceReference(source_type, source_id, source_user_id)
        global_id = self._source_map.get(ref)
        if global_id:
            return self._entities.get(global_id)
        return None

    def get_or_create_user(
        self,
        source_type: str,
        source_id: str,
        source_user_id: str,
        user_data: Optional[Dict] = None,
    ) -> Entity:
        """Get an existing entity or create a new one based on source info and metadata.

        Strategies:
        1. Check if source reference exists -> return entity.
        2. Check if provided email matches any existing entity -> link and return.
        3. Create new entity.

        Args:
            source_type: e.g., "slack", "jira"
            source_id: Unique ID of the source instance (workspace/tenant ID)
            source_user_id: User's ID in that source (e.g., U123)
            user_data: Dict containing metadata (email, name, real_name, etc.)

        Returns:
            The linked Entity object.
        """
        user_data = user_data or {}
        ref = SourceReference(source_type, source_id, source_user_id)

        # 1. Direct Lookup
        if ref in self._source_map:
            global_id = self._source_map[ref]
            entity = self._entities[global_id]
            self._update_entity(entity, user_data, ref)  # Update metadata if needed
            self._save()
            return entity

        # 2. Email Linking
        email = user_data.get("email")
        if email and email.lower() in self._email_map:
            global_id = self._email_map[email.lower()]
            entity = self._entities[global_id]
            self._update_entity(entity, user_data, ref)  # Link new source ref
            self._save()
            return entity

        # 3. Create New
        new_global_id = str(uuid.uuid4())
        entity = Entity(global_id=new_global_id)
        self._update_entity(entity, user_data, ref)
        self._save()
        return entity

    def _update_entity(self, entity: Entity, user_data: Dict, ref: SourceReference) -> None:
        """Update entity fields and maps with new data."""
        # Update Source Refs
        entity.source_refs.add(ref)
        self._source_map[ref] = entity.global_id

        # Update Names
        # Collect possible name fields
        possible_names = [
            user_data.get("name"),
            user_data.get("real_name"),
            user_data.get("display_name"),
            user_data.get("displayName"),
        ]
        for name in possible_names:
            if name:
                entity.names.add(name)

        # Update Emails
        email = user_data.get("email")
        if email:
            normalized_email = email.lower()
            entity.emails.add(normalized_email)
            self._email_map[normalized_email] = entity.global_id

        # Update Metadata (merge)
        # We prefix keys with source type to avoid collisions if needed,
        # or just merge flat if that's preferred. For now, flat merge.
        # But maybe we want to keep some source specific info?
        # Let's keep it simple: merge provided data.
        for k, v in user_data.items():
            if v and k not in ["email", "name", "real_name", "displayName"]:
                # Only overwrite if not present? Or always overwrite?
                # Let's assume enrichment: always overwrite/add.
                if isinstance(v, (str, int, float, bool)):
                    entity.metadata[k] = str(v)

        # Re-register maps (safe to call repeatedly)
        self._register_entity_in_maps(entity)
