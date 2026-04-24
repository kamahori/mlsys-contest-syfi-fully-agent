"""Allow running as `python -m ncu_mcp`."""

from ncu_mcp.server import server

server.run(transport="stdio")
