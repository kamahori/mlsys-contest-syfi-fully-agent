# ncu-mcp

MCP server wrapping NVIDIA Nsight Compute for GPU kernel profiling.

## Setup

```bash
uv sync                  # Install dependencies
uv run ncu-mcp-server    # Run MCP server (stdio transport)
uv run ncu-mcp-test --list                         # List all tools
uv run ncu-mcp-test <tool_name> '{"arg": "val"}'   # Call a tool
```
## Key Files

- `src/ncu_mcp/server.py` - MCP tool definitions
- `src/ncu_mcp/ncu_runner.py` - async ncu subprocess wrapper
- `src/ncu_mcp/parsers.py` - CSV output parsing
- `src/ncu_mcp/formatters.py` - LLM-friendly output formatting

## Formatter Behavior

`read_report_details` output order: **Summary** (5 key metrics) → **Optimization Opportunities** (rules sorted by speedup) → **Section metrics** (filtered).

Filtering applied automatically in section metrics:
- Array-valued metrics (occupancy sweeps with `;`) are hidden with a `read_report_raw` hint
- Zero-value `%` metrics (idle pipe utilization) are hidden with a `read_report_raw` hint
- Empty sections (identifier-only duplicates) are skipped
- Rules are shown once in the summary table, not repeated per-section

`read_report_source` output: deduped by `(line_number, sass)`, user code sorted before library code (`/usr/`, `/opt/`, `/include/cuda`), each line shows top 3 stall reasons as percentages.

`read_report_raw` summary: excludes `device__` and `profiler__` prefixes from the category table (with access hint).

## Kernel Optimization

Use `/optimize-kernel` skill for the guided profiling workflow.
