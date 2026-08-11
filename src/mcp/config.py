"""Configuration and path validation for MCP containers."""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

DEFAULT_FORBIDDEN_EXACT_PATHS: set[str] = {
    "/",
    "/etc",
    "/root",
    "/sys",
    "/proc",
    "/dev",
    "/boot",
    "/home",
    "/var",
}

DEFAULT_FORBIDDEN_PATH_PREFIXES: tuple[str, ...] = (
    "/etc/",
    "/root/",
    "/sys/",
    "/proc/",
    "/dev/",
    "/boot/",
    "/var/run/",
    "/var/lib/docker",
    "/var/log",
    "/var/backups",
)


def is_forbidden_host_path(
    src: str,
    forbidden_exact: Optional[set[str]] = None,
    forbidden_prefixes: Optional[tuple[str, ...]] = None,
) -> bool:
    """Validate that the source path is not a forbidden host path.

    Args:
        src: The host path to validate.
        forbidden_exact: Set of strictly forbidden exact paths.
        forbidden_prefixes: Tuple of strictly forbidden path prefixes.

    Returns:
        bool: True if the path is forbidden, False otherwise.
    """
    if not src:
        return True

    # Use defaults if not provided
    exact_paths = forbidden_exact if forbidden_exact is not None else DEFAULT_FORBIDDEN_EXACT_PATHS
    prefixes = forbidden_prefixes if forbidden_prefixes is not None else DEFAULT_FORBIDDEN_PATH_PREFIXES

    try:
        # Resolve path to catch '..' and symlinks
        resolved = str(Path(src).resolve())
    except Exception:
        # If path resolution fails (e.g. invalid chars), deny it
        return True

    if resolved in exact_paths:
        return True

    if resolved == "/var/run/docker.sock":
        return True

    for prefix in prefixes:
        # Ensure prefix match happens at a directory boundary
        # For example, prefix "/var/run/" matches "/var/run/docker.sock"
        if resolved == prefix.rstrip("/") or resolved.startswith(prefix if prefix.endswith("/") else prefix + "/"):
            return True

    return False


class MCPContainerConfig(BaseModel):
    """Configuration model for an MCP Docker container execution."""

    name: str = Field(..., description="Unique name/ID for this MCP server instance.")
    image: str = Field(..., description="The Docker image to execute.")
    auto_pull: bool = Field(default=True, description="Whether to automatically check and pull the image.")
    allow_network: bool = Field(default=False, description="Whether the container has network access.")
    read_only: bool = Field(default=True, description="Whether the container filesystem is strictly read-only.")
    tmpfs_options: str = Field(default="rw,noexec,nosuid,nodev,size=64m", description="Tmpfs mount for /tmp.")
    user: str = Field(default="1000:1000", description="User and group IDs for container execution.")
    cap_drop: list[str] = Field(default_factory=lambda: ["ALL"], description="Linux capabilities to drop.")
    pids_limit: int = Field(default=64, description="Maximum number of PIDs.")
    memory: str = Field(default="128m", description="Memory limit.")
    memory_swap: str = Field(default="128m", description="Swap limit.")
    cpus: str = Field(default="0.25", description="CPU quota.")
    nofile: str = Field(default="128:128", description="Ulimit configuration for open files.")

    # Mount configuration: list of (host_path, container_path, mode)
    mounts: list[tuple[str, str, str]] = Field(default_factory=list, description="Volume bind mounts.")

    # Security configuration
    forbidden_exact_paths: set[str] = Field(
        default_factory=lambda: DEFAULT_FORBIDDEN_EXACT_PATHS.copy(),
        description="Exact host paths forbidden from being mounted.",
    )
    forbidden_path_prefixes: tuple[str, ...] = Field(
        default=DEFAULT_FORBIDDEN_PATH_PREFIXES,
        description="Path prefixes forbidden from being mounted.",
    )

    @model_validator(mode="after")
    def validate_mounts(self) -> "MCPContainerConfig":
        """Validate all configured mounts against security rules."""
        for mount in self.mounts:
            if len(mount) != 3:
                raise ValueError(f"Mount must be a 3-tuple (src, dst, mode), got: {mount}")
            src, _, mode = mount
            if mode != "ro":
                raise ValueError(f"All MCP mounts must be read-only ('ro'), got mode: {mode}")

            if is_forbidden_host_path(src, self.forbidden_exact_paths, self.forbidden_path_prefixes):
                raise ValueError(f"Host path '{src}' is forbidden by security rules.")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPContainerConfig":
        """Create a configuration instance from a dictionary.

        Args:
            data: The configuration dictionary (e.g. from YAML).

        Returns:
            MCPContainerConfig: Validated configuration object.
        """
        return cls(**data)
