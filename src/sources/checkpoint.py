"""Checkpoint management for data sources.

Provides base checkpoint functionality for tracking incremental data collection.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

from src.models.source_config import BaseSourceConfig

T = TypeVar("T", bound="Checkpoint")
TConfig = TypeVar("TConfig", bound=BaseSourceConfig)


@dataclass
class Checkpoint(Generic[TConfig]):
    """Checkpoint for tracking source data collection progress.

    A checkpoint is a point in time when the source was last checked.
    It is used to track the progress of the source and to avoid
    collecting the same documents twice.

    Parameters:
        config: Configuration for the checkpoint's source (e.g. workspace, filters).
        state:  Checkpoint state (e.g. last_ts per channel).
    """

    config: TConfig
    state: Dict[str, Any] = field(default_factory=dict)

    def update(self, state: Dict[str, Any]) -> None:
        """Update the checkpoint with the given state.

        Typical pattern: merge / overwrite fields.
        """
        self.state.update(state)

    def save(self, path: Path) -> None:
        """Save the checkpoint to the given path.

        We store config + state to poder validar cambios de config.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.config.model_dump(),
            "state": self.state,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls: Type[T], path: Path, config: Optional[TConfig] = None) -> T:
        """Load the checkpoint from the given path.

        If `config` is provided, it overrides the stored config
        If `config` is not provided, the stored config is used.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        file_config = data.get("config", {})
        state = data.get("state", {})

        effective_config = config if config is not None else file_config

        # Attempt to convert dict config to Pydantic model if the subclass expects it
        if isinstance(effective_config, dict):
            # Inspect the type hint of 'config' in the class
            try:
                from typing import get_type_hints

                hints = get_type_hints(cls)
                cfg_type = hints.get("config")
                # Check if it's a Pydantic model class
                if cfg_type and isinstance(cfg_type, type) and issubclass(cfg_type, BaseModel):
                    effective_config = cfg_type(**effective_config)
            except Exception:
                # If inspection fails or valid type not found, proceed with dict
                pass

        return cls(config=effective_config, state=state)
