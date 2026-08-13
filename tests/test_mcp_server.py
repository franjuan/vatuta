"""Unit tests for the MCP Server configuration and lifecycle."""

import asyncio

import pytest
from pydantic import ValidationError

from src.mcp.config import MCPContainerConfig, is_forbidden_host_path
from src.mcp.server import build_docker_mcp_params


def test_is_forbidden_host_path_defaults() -> None:
    """Test path validation with default settings."""
    forbidden_paths = [
        "/",
        "/etc",
        "/etc/passwd",
        "/etc/shadow",
        "/var",
        "/var/run/docker.sock",
        "/var/lib/docker",
        "/var/log",
        "/home",
        "/root",
        "/root/.ssh",
    ]
    allowed_paths = [
        "/tmp/vatuta-mcp-test",
        "/home/user/workspace/vatuta/data",
        "/var/lib/vatuta/mcp_data",
    ]

    for path in forbidden_paths:
        assert is_forbidden_host_path(path), f"Path should be forbidden: {path}"

    for path in allowed_paths:
        assert not is_forbidden_host_path(path), f"Path should be allowed: {path}"


def test_is_forbidden_host_path_custom() -> None:
    """Test path validation with custom settings."""
    # Allow /var but block /tmp
    assert is_forbidden_host_path("/tmp/custom", forbidden_exact=set(), forbidden_prefixes=("/tmp/",))
    assert not is_forbidden_host_path("/var/lib", forbidden_exact=set(), forbidden_prefixes=())


def test_mcp_config_from_dict() -> None:
    """Test initializing config from dictionary."""
    data = {
        "name": "test-mcp",
        "image": "mcp/test:latest",
        "auto_pull": False,
        "mounts": [("/var/lib/vatuta/data", "/workspace", "ro")],
    }

    config = MCPContainerConfig.from_dict(data)
    assert config.name == "test-mcp"
    assert config.image == "mcp/test:latest"
    assert config.auto_pull is False
    assert len(config.mounts) == 1


def test_mcp_config_mount_validation() -> None:
    """Test validation of mounts in configuration."""
    data = {"name": "test-mcp", "image": "mcp/test:latest", "mounts": [("/etc", "/workspace", "ro")]}

    with pytest.raises(ValidationError, match="forbidden by security rules"):
        MCPContainerConfig.from_dict(data)

    data2 = {"name": "test-mcp", "image": "mcp/test:latest", "mounts": [("/tmp/safe", "/workspace", "rw")]}

    with pytest.raises(ValidationError, match="must be read-only"):
        MCPContainerConfig.from_dict(data2)


def test_build_docker_mcp_params() -> None:
    """Test that docker parameters are built securely."""
    config = MCPContainerConfig(
        name="test-server",
        image="mcp/test:latest",
        allow_network=False,
        mounts=[("/tmp/vatuta-mcp-test", "/workspace", "ro")],
    )

    params = build_docker_mcp_params(config, "mcp/test@sha256:12345")
    args = params.args

    # Assert critical security flags are present
    assert "-i" in args
    assert "--rm" in args
    assert "--pull=never" in args
    assert "--network=none" in args
    assert "--read-only" in args
    assert "--user=1000:1000" in args
    assert "--cap-drop=ALL" in args
    assert "--name=vatuta-mcp-test-server" in args

    # Assert mount is present
    mount_arg = "--mount=type=bind,src=/tmp/vatuta-mcp-test,dst=/workspace,readonly=true"
    assert mount_arg in args

    # Image must be at the end
    assert args[-1] == "mcp/test@sha256:12345"


def test_mcp_server_capabilities() -> None:
    """Test that MCPServer initializes with default ClientCapabilities."""
    from src.mcp.server import ClientCapabilities, MCPServer

    config = MCPContainerConfig(name="test-cap", image="mcp/test:latest")
    server = MCPServer(config)
    assert isinstance(server.capabilities, ClientCapabilities)
    assert server.capabilities.experimental is None or server.capabilities.experimental == {}


def test_mcp_server_unstarted_methods() -> None:
    """Test that MCPServer methods raise RuntimeError if session is not started."""
    from src.mcp.server import MCPServer

    config = MCPContainerConfig(name="test-unstarted", image="mcp/test:latest")
    server = MCPServer(config)

    with pytest.raises(RuntimeError, match="MCP session not started"):
        asyncio.run(server.get_prompt("test"))

    with pytest.raises(RuntimeError, match="MCP session not started"):
        asyncio.run(server.read_resource("test://uri"))


def test_cleanup_existing_container_nonexistent() -> None:
    """Test that cleanup_existing_container handles non-existent containers safely."""
    from src.mcp.server import cleanup_existing_container

    # Should not raise any exception if container doesn't exist
    cleanup_existing_container("vatuta-mcp-nonexistent-test-container")


def test_is_item_allowed() -> None:
    """Test regex item whitelist helper function."""
    from src.mcp.config import is_item_allowed

    # Allowed when whitelist is None
    assert is_item_allowed("echo", None) is True
    assert is_item_allowed("anything", None) is True

    # Regex matching
    whitelist = ["^echo$", "^jira_.*", "file://.*"]
    assert is_item_allowed("echo", whitelist) is True
    assert is_item_allowed("jira_search_issues", whitelist) is True
    assert is_item_allowed("file://workspace/doc.txt", whitelist) is True

    # Rejected items
    assert is_item_allowed("echos", whitelist) is False
    assert is_item_allowed("confluence_get_page", whitelist) is False
    assert is_item_allowed("http://example.com", whitelist) is False


def test_mcp_config_whitelist_from_dict() -> None:
    """Test parsing whitelist configuration from dictionary."""
    data = {
        "name": "test-whitelist",
        "image": "mcp/test:latest",
        "allowed_tools": ["^echo$"],
        "allowed_prompts": ["simple_.*"],
        "allowed_resources": ["test://.*"],
    }
    config = MCPContainerConfig.from_dict(data)
    assert config.allowed_tools == ["^echo$"]
    assert config.allowed_prompts == ["simple_.*"]
    assert config.allowed_resources == ["test://.*"]


def test_mcp_server_whitelist_enforcement() -> None:
    """Test that MCPServer blocks execution of unwhitelisted items."""
    from unittest.mock import MagicMock

    from src.mcp.server import MCPServer

    config = MCPContainerConfig(
        name="test-whitelist-enforcement",
        image="mcp/test:latest",
        allowed_tools=["^echo$"],
        allowed_prompts=["^simple_prompt$"],
        allowed_resources=["test://static/.*"],
    )
    server = MCPServer(config)
    server.session = MagicMock()

    with pytest.raises(ValueError, match="Tool 'printEnv' is not allowed"):
        asyncio.run(server.call_tool("printEnv"))

    with pytest.raises(ValueError, match="Prompt 'complex_prompt' is not allowed"):
        asyncio.run(server.get_prompt("complex_prompt"))

    with pytest.raises(ValueError, match="Resource URI 'http://forbidden' is not allowed"):
        asyncio.run(server.read_resource("http://forbidden"))
