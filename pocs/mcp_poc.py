"""Proof of Concept for Model Context Protocol (MCP) server integration in Vatuta.

Demonstrates secure container invocation for MCP stdio servers applying minimum
privilege security controls and explicit pre-run image inspection.
"""

import asyncio

from src.mcp import MCPContainerConfig, MCPServer


async def main() -> None:
    """Run the MCP Proof of Concept."""
    # Define configuration as a dictionary, simulating YAML loading
    config_dict = {
        "name": "everything-poc",
        "image": "mcp/everything",
        "auto_pull": True,
        "allow_network": False,
        "read_only": True,
        # Whitelist filtering (regex patterns)
        "allowed_tools": ["^echo$", "^add$"],
        "allowed_prompts": [".*_prompt"],
        "allowed_resources": ["test://static/.*"],
    }

    print(f"Loading configuration: {config_dict}")
    config = MCPContainerConfig.from_dict(config_dict)

    print("Initializing MCP Server...")
    async with MCPServer(config) as server:
        print("MCP Server started successfully.")
        
        # List available tools
        print("\nListing available tools:")
        try:
            tools_result = await server.list_tools()
            for tool in tools_result.tools:
                print(f" - {tool.name}: {tool.description}")
        except Exception as e:
            print(f"Error listing tools: {e}")

        # List available prompts
        print("\nListing available prompts:")
        try:
            prompts_result = await server.list_prompts()
            for prompt in prompts_result.prompts:
                print(f" - {prompt.name}: {prompt.description}")
        except Exception as e:
            print(f"Error listing prompts: {e}")

        # Execute sample MCP tool calls
        print("\nExecuting sample MCP tool calls:")
        try:
            echo_result = await server.call_tool("echo", {"message": "Hello from Vatuta MCPServer!"})
            print(f" -> 'echo' result: {echo_result.content}")
        except Exception as e:
            print(f" -> Error calling 'echo' tool: {e}")

        try:
            add_result = await server.call_tool("add", {"a": 15, "b": 27})
            print(f" -> 'add' result: {add_result.content}")
        except Exception as e:
            print(f" -> Error calling 'add' tool: {e}")

        # Attempt to call a blocked tool not in allowed_tools
        try:
            await server.call_tool("printEnv", {})
            print(" -> Unexpected success calling 'printEnv'")
        except ValueError as e:
            print(f" -> Blocked call to 'printEnv' correctly caught: {e}")

        # Fetch sample prompt
        print("\nFetching sample prompt:")
        try:
            prompt_data = await server.get_prompt("simple_prompt")
            print(f" -> 'simple_prompt' result: {prompt_data}")
        except Exception as e:
            print(f" -> Error fetching prompt: {e}")

        # List and read sample resource
        print("\nReading sample resources:")
        try:
            resources_result = await server.list_resources()
            if resources_result.resources:
                sample_uri = str(resources_result.resources[0].uri)
                resource_content = await server.read_resource(sample_uri)
                print(f" -> Resource '{sample_uri}' content: {resource_content}")
            else:
                print(" -> No resources available to read.")
        except Exception as e:
            print(f" -> Error reading resource: {e}")


if __name__ == "__main__":
    asyncio.run(main())
