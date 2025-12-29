"""Entity models for global user management."""

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class SourceReference:
    """Reference to a user entity in a specific source system."""

    source_type: str  # e.g., "slack", "jira", "confluence"
    source_id: str  # The unique ID of the source instance (e.g., workspace ID, tenant ID)
    source_user_id: str  # The user ID within that source (e.g., "U12345", "account-123")

    def __hash__(self) -> int:
        """Return hash of the source reference."""
        return hash((self.source_type, self.source_id, self.source_user_id))


@dataclass
class Entity:
    """Global user entity that can be linked across multiple sources."""

    global_id: str
    names: Set[str] = field(default_factory=set)
    emails: Set[str] = field(default_factory=set)
    source_refs: Set[SourceReference] = field(default_factory=set)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize entity to dictionary for persistence."""
        return {
            "global_id": self.global_id,
            "names": list(self.names),
            "emails": list(self.emails),
            "source_refs": [
                {
                    "source_type": ref.source_type,
                    "source_id": ref.source_id,
                    "source_user_id": ref.source_user_id,
                }
                for ref in self.source_refs
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Entity":
        """Deserialize entity from dictionary."""
        return cls(
            global_id=data["global_id"],
            names=set(data.get("names", [])),
            emails=set(data.get("emails", [])),
            source_refs={
                SourceReference(
                    source_type=ref["source_type"],
                    source_id=ref["source_id"],
                    source_user_id=ref["source_user_id"],
                )
                for ref in data.get("source_refs", [])
            },
            metadata=data.get("metadata", {}),
        )
