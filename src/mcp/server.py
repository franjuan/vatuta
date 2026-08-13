"""MCP Server lifecycle and execution management."""

import asyncio
import logging
from typing import Any, Optional

import docker
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import ClientCapabilities
from pydantic import AnyUrl

from src.mcp.config import MCPContainerConfig, is_item_allowed
from src.utils.logging_config import LoggerWriter

logger = logging.getLogger(__name__)


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
        logger.info("Pulling image %s...", image)
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


def cleanup_existing_container(container_name: str) -> None:
    """Remove any existing container with the given name if it exists (stopped or running)."""
    try:
        client = docker.from_env()
        container = client.containers.get(container_name)
        container.remove(force=True)
    except docker.errors.NotFound:
        pass
    except docker.errors.APIError:
        pass


def ensure_image_available(config: MCPContainerConfig) -> str:
    """Ensure image is available locally and return its resolved RepoDigest."""
    # Cleanup any leftover container from a previous run with the same instance name
    cleanup_existing_container(f"vatuta-mcp-{config.name}")

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
    if config.args:
        args.extend(config.args)

    return StdioServerParameters(command="docker", args=args, env=None)


class MCPServer:
    """Main class for managing an MCP server lifecycle and client session."""

    def __init__(
        self,
        config: MCPContainerConfig,
        capabilities: Optional[ClientCapabilities] = None,
    ):
        """Initialize the MCP Server."""
        self.config = config
        # Explicit client capabilities (defaults to empty ClientCapabilities with no capabilities enabled)
        self.capabilities = capabilities or ClientCapabilities()
        self._exit_stack: Optional[Any] = None
        self.session: Optional[ClientSession] = None

        # Internally manage the stdio context managers
        self._stdio_cm: Optional[Any] = None
        self._session_cm: Optional[Any] = None
        self._errlog: Optional[Any] = None

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

        # Redirect container stderr stream to a dedicated logger for this MCP server instance
        mcp_logger = logging.getLogger(f"external.mcp.{self.config.name}")
        self._errlog = LoggerWriter(mcp_logger, level=logging.INFO)

        self._stdio_cm = stdio_client(server_params, errlog=self._errlog)
        read_stream, write_stream = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read_stream, write_stream)
        self.session = await self._session_cm.__aenter__()

        await self.session.initialize()

    async def stop(self) -> None:
        """Stop the server and cleanup."""
        if self._session_cm is not None:
            try:
                await asyncio.wait_for(self._session_cm.__aexit__(None, None, None), timeout=2.0)
            except Exception:
                pass
            self._session_cm = None
            self.session = None

        # Ensure leftover container is removed FIRST so stdio_client doesn't hang waiting for it
        await asyncio.to_thread(cleanup_existing_container, f"vatuta-mcp-{self.config.name}")

        if self._stdio_cm is not None:
            try:
                await asyncio.wait_for(self._stdio_cm.__aexit__(None, None, None), timeout=2.0)
            except Exception:
                pass
            self._stdio_cm = None

        if hasattr(self, "_errlog") and self._errlog is not None:
            self._errlog.close()
            self._errlog = None

    async def list_tools(self) -> Any:
        """List available tools from the MCP server, filtered by whitelist configuration."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        result = await self.session.list_tools()
        if self.config.allowed_tools is not None:
            result.tools = [t for t in result.tools if is_item_allowed(t.name, self.config.allowed_tools)]
        return result

    async def list_prompts(self) -> Any:
        """List available prompts from the MCP server, filtered by whitelist configuration."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        result = await self.session.list_prompts()
        if self.config.allowed_prompts is not None:
            result.prompts = [p for p in result.prompts if is_item_allowed(p.name, self.config.allowed_prompts)]
        return result

    async def list_resources(self) -> Any:
        """List available resources from the MCP server, filtered by whitelist configuration."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        result = await self.session.list_resources()
        if self.config.allowed_resources is not None:
            result.resources = [
                r
                for r in result.resources
                if is_item_allowed(str(r.uri), self.config.allowed_resources)
                or is_item_allowed(r.name, self.config.allowed_resources)
            ]
        return result

    async def call_tool(self, name: str, arguments: Optional[dict[str, Any]] = None) -> Any:
        """Call a specific tool on the MCP server if authorized by whitelist configuration."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        if not is_item_allowed(name, self.config.allowed_tools):
            raise ValueError(f"Tool '{name}' is not allowed by security whitelist configuration.")
        return await self.session.call_tool(name, arguments or {})

    async def get_prompt(self, name: str, arguments: Optional[dict[str, str]] = None) -> Any:
        """Get a specific prompt from the MCP server if authorized by whitelist configuration."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        if not is_item_allowed(name, self.config.allowed_prompts):
            raise ValueError(f"Prompt '{name}' is not allowed by security whitelist configuration.")
        return await self.session.get_prompt(name, arguments or {})

    async def read_resource(self, uri: str) -> Any:
        """Read a specific resource from the MCP server by its URI if authorized by whitelist configuration."""
        if not self.session:
            raise RuntimeError("MCP session not started.")
        if not is_item_allowed(uri, self.config.allowed_resources):
            raise ValueError(f"Resource URI '{uri}' is not allowed by security whitelist configuration.")
        return await self.session.read_resource(AnyUrl(uri))
