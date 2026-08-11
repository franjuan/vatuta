"""MCP Server lifecycle and execution management."""

import asyncio
from typing import Any, Optional

import docker
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.mcp.config import MCPContainerConfig


def check_image_exists(image: str) -> bool:
    """Check if the Docker image exists locally."""
    try:
        client = docker.from_env()
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False
    except docker.errors.APIError as e:
        raise RuntimeError(f"Docker API error while checking image '{image}': {e}") from e


def pull_docker_image(image: str) -> None:
    """Pull the specified Docker image from the registry."""
    try:
        client = docker.from_env()
        print(f"Pulling image {image}...")
        client.images.pull(image)
    except docker.errors.APIError as e:
        raise RuntimeError(f"Docker API error while pulling image '{image}': {e}") from e


def get_image_digest(image: str) -> str:
    """Get the RepoDigest of a Docker image to pin it securely."""
    try:
        client = docker.from_env()
        img = client.images.get(image)
        digests = img.attrs.get("RepoDigests", [])
        if digests:
            return str(digests[0])
        # Fallback to provided image tag/name if no digest is found (e.g. locally built)
        return image
    except docker.errors.APIError as e:
        raise RuntimeError(f"Docker API error while getting digest for '{image}': {e}") from e


def ensure_image_available(config: MCPContainerConfig) -> str:
    """Ensure image is available locally and return its resolved RepoDigest."""
    # If the image is already pinned with a digest, just ensure it exists
    if "@sha256:" in config.image:
        if not check_image_exists(config.image):
            if config.auto_pull:
                pull_docker_image(config.image)
            else:
                raise RuntimeError(f"Image {config.image} not found locally and auto_pull is disabled.")
        return config.image

    # For tag-based images, check if it exists or pull it, then resolve digest
    if config.auto_pull:
        pull_docker_image(config.image)
    elif not check_image_exists(config.image):
        raise RuntimeError(f"Image {config.image} not found locally and auto_pull is disabled.")

    return get_image_digest(config.image)


def build_docker_mcp_params(config: MCPContainerConfig, resolved_image: str) -> StdioServerParameters:
    """Build the Docker parameters for the stdio_client MCP connection."""
    args = [
        "run",
        "-i",
        "--rm",
        "--pull=never",
    ]

    # Name for container uniqueness
    args.append(f"--name=vatuta-mcp-{config.name}")

    if not config.allow_network:
        args.append("--network=none")

    if config.read_only:
        args.append("--read-only")
        args.append(f"--tmpfs=/tmp:{config.tmpfs_options}")

    args.append(f"--user={config.user}")

    for cap in config.cap_drop:
        args.append(f"--cap-drop={cap}")

    args.extend(
        [
            "--security-opt=no-new-privileges=true",
            "--security-opt=seccomp=builtin",
            f"--pids-limit={config.pids_limit}",
            f"--memory={config.memory}",
            f"--memory-swap={config.memory_swap}",
            f"--cpus={config.cpus}",
            f"--ulimit=nofile={config.nofile}",
        ]
    )

    # Mounts
    for src, dst, mode in config.mounts:
        args.append(f"--mount=type=bind,src={src},dst={dst},readonly={'true' if mode == 'ro' else 'false'}")

    args.append(resolved_image)

    return StdioServerParameters(command="docker", args=args, env=None)


class MCPServer:
    """Main class for managing an MCP server lifecycle and client session."""

    def __init__(self, config: MCPContainerConfig):
        """Initialize the MCP Server."""
        self.config = config
        self._exit_stack: Optional[Any] = None
        self.session: Optional[ClientSession] = None

        # Internally manage the stdio context managers
        self._stdio_cm: Optional[Any] = None
        self._session_cm: Optional[Any] = None

    async def __aenter__(self) -> "MCPServer":
        """Start the MCP server and initialize the session."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the MCP server and clean up resources."""
        await self.stop()

    async def start(self) -> None:
        """Start the server process and initialize the MCP session."""
        if self.session is not None:
            return  # Already started

        # Offload image management (synchronous docker SDK calls) to a thread to avoid blocking asyncio
        resolved_image = await asyncio.to_thread(ensure_image_available, self.config)
        server_params = build_docker_mcp_params(self.config, resolved_image)

        self._stdio_cm = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read_stream, write_stream)
        self.session = await self._session_cm.__aenter__()

        await self.session.initialize()

    async def stop(self) -> None:
        """Stop the server and cleanup."""
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self.session = None

        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None

    async def list_tools(self) -> Any:
        """List available tools from the MCP server."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        return await self.session.list_tools()

    async def list_prompts(self) -> Any:
        """List available prompts from the MCP server."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        return await self.session.list_prompts()

    async def list_resources(self) -> Any:
        """List available resources from the MCP server."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        return await self.session.list_resources()

    async def call_tool(self, name: str, arguments: Optional[dict[str, Any]] = None) -> Any:
        """Call a specific tool on the MCP server."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        return await self.session.call_tool(name, arguments or {})
