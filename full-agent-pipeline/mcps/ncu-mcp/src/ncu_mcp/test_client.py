"""CLI test client for ncu-mcp server."""

import asyncio
import json
import shutil
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  ncu-mcp-test --list                          # List all tools")
        print("  ncu-mcp-test <tool_name>                     # Call tool with no args")
        print("  ncu-mcp-test <tool_name> '{\"arg\": \"val\"}'    # Call tool with JSON args")
        sys.exit(1)

    # Find the ncu-mcp-server command (installed by the same package)
    server_cmd = shutil.which("ncu-mcp-server")
    if server_cmd:
        server_params = StdioServerParameters(command=server_cmd)
    else:
        # Fallback: run via python -m
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ncu_mcp.server"],
        )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if sys.argv[1] == "--list":
                tools = await session.list_tools()
                print(f"\n{'='*60}")
                print(f"ncu-mcp: {len(tools.tools)} tools available")
                print(f"{'='*60}\n")
                for tool in tools.tools:
                    desc_first_line = (tool.description or "").split("\n")[0]
                    print(f"  {tool.name}")
                    print(f"    {desc_first_line}")
                    if tool.inputSchema and tool.inputSchema.get("properties"):
                        params = list(tool.inputSchema["properties"].keys())
                        print(f"    params: {', '.join(params)}")
                    print()
                return

            tool_name = sys.argv[1]
            args = {}
            if len(sys.argv) > 2:
                args = json.loads(sys.argv[2])

            print(f"\n{'='*60}")
            print(f"Calling: {tool_name}({json.dumps(args, indent=2) if args else ''})")
            print(f"{'='*60}\n")

            result = await session.call_tool(tool_name, args)

            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(content)

            print(f"\n{'='*60}")
            print(f"Done: {tool_name}")
            print(f"{'='*60}")


def run():
    """Entry point for the test client."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
