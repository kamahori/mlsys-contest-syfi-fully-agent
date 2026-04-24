"""NCU MCP Server - wraps NVIDIA Nsight Compute CLI for GPU kernel profiling."""

import os
import time
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from ncu_mcp.ncu_runner import run_ncu, report_path, REPORTS_DIR
from ncu_mcp.parsers import (
    parse_details_csv,
    parse_raw_csv,
    parse_source_csv,
    parse_session_csv,
    parse_list_output,
    parse_query_metrics,
)
from ncu_mcp.formatters import (
    format_details,
    format_raw_summary,
    format_raw_filtered,
    format_source,
    format_session,
    format_comparison,
    format_sections_and_sets,
    format_rules,
    format_metrics_summary,
    format_reports,
)

server = FastMCP("ncu-mcp")


# ─── Discovery Tools ─────────────────────────────────────────────────────────


@server.tool()
async def list_sections_and_sets() -> str:
    """List available ncu profiling sections and metric sets.

    Use this FIRST to understand what data you can collect before calling profile().

    Returns two lists:
    - SECTIONS: Individual profiling areas (e.g. SpeedOfLight, Occupancy, MemoryWorkloadAnalysis).
      Each section collects a group of related metrics. Use section identifiers with profile(sections=[...]).
    - SETS: Predefined bundles of sections at different granularities:
      - "basic" (~190 metrics, fast) - SpeedOfLight + Occupancy + LaunchStats + WorkloadDistribution
      - "detailed" (~557 metrics) - Adds ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, SourceCounters
      - "full" (~5895 metrics, slow, ~40 passes) - Everything including roofline, instruction stats, scheduler
      - "roofline" (~5260 metrics) - SpeedOfLight + all roofline charts
      - "nvlink" (~52 metrics) - NVLink topology and throughput
      - "pmsampling" (~186 metrics) - PM sampling warp states

    Strategy: Start with set="basic" for a quick overview. If you need deeper analysis of a
    specific area (e.g. memory), re-profile with just those sections rather than using "full".
    """
    sections_result = await run_ncu(["--list-sections"])
    sets_result = await run_ncu(["--list-sets"])

    if not sections_result.ok:
        return f"Error listing sections: {sections_result.error_message}"
    if not sets_result.ok:
        return f"Error listing sets: {sets_result.error_message}"

    sections = parse_list_output(sections_result.stdout)
    sets = parse_list_output(sets_result.stdout)

    return format_sections_and_sets(sections, sets)


@server.tool()
async def list_rules() -> str:
    """List available ncu analysis rules that provide optimization advice.

    Rules are automated checks that analyze collected metrics and produce actionable
    recommendations with estimated speedup percentages. Rules are applied during profiling
    when apply_rules=True (the default).

    Common rules and what they detect:
    - SOLBottleneck: High-level bottleneck (compute vs memory bound)
    - CPIStall: Top warp stall reasons and their causes
    - LaunchConfiguration: Tail effect from partial waves
    - AchievedOccupancy: Gap between theoretical and achieved occupancy
    - HighPipeUtilization: Under-utilized compute pipelines
    - MemoryL2Compression: Inefficient L2 compression usage
    - UncoalescedGlobalAccess: Non-coalesced memory access patterns
    - SharedMemoryConflicts: Bank conflict detection
    - ThreadDivergence: Warp/thread control flow divergence
    - FPInstructions: Opportunities to use fused FP instructions

    Rules only fire if their required sections were collected during profiling.
    Use read_report_details() with include_rules=True to see rule results.
    """
    result = await run_ncu(["--list-rules"])
    if not result.ok:
        return f"Error listing rules: {result.error_message}"

    rules = parse_list_output(result.stdout)
    return format_rules(rules)


@server.tool()
async def query_metrics(substring: str = "", collection: str = "") -> str:
    """Search available ncu metrics by name substring.

    Use this when you need specific low-level metrics beyond what sections provide.
    The ncu metric namespace has thousands of metrics organized by hardware unit prefix:
      sm__    = Streaming Multiprocessor metrics (instruction, pipe utilization)
      dram__  = DRAM/HBM metrics (throughput, cycles, bytes)
      l1tex__ = L1/TEX cache metrics (hit rates, sectors, throughput)
      lts__   = L2 cache metrics (throughput, hit rates, sectors)
      gpu__   = GPU-wide metrics (time duration, throughput)
      gpc__   = GPC metrics (cycles, frequency)
      launch__= Launch configuration metrics (waves, occupancy)

    Args:
      substring: Filter metrics containing this string (case-insensitive).
        Examples: "dram__bytes" for DRAM byte counters, "sm__inst" for SM instructions,
        "throughput" for all throughput metrics. If empty, returns category summary.
      collection: Filter by collection type. One of: "profiling" (default perf counters),
        "launch" (launch attributes), "device" (device properties), "source" (source metrics).

    Returns matching metric names (max 100). Use these names with profile(metrics=[...]).
    """
    args = ["--query-metrics"]

    result = await run_ncu(args, timeout=60)
    if not result.ok:
        return f"Error querying metrics: {result.error_message}"

    metrics = parse_query_metrics(result.stdout)

    # Filter by substring in Python
    if substring:
        sub_lower = substring.lower()
        metrics = [m for m in metrics if sub_lower in m.lower()]

    if not substring and not collection:
        # Return category summary
        return format_metrics_summary(metrics)

    # Return filtered list (cap at 100)
    if len(metrics) > 100:
        metrics = metrics[:100]
        return format_metrics_summary(metrics) + f"\n\n*Showing first 100 of many matches. Narrow your substring filter.*"

    return format_metrics_summary(metrics)


# ─── Profiling Tool ──────────────────────────────────────────────────────────


@server.tool()
async def profile(
    binary: str,
    report_name: str,
    set: str = "",
    sections: list[str] | None = None,
    metrics: list[str] | None = None,
    kernel_filter: str = "",
    launch_count: int = 0,
    launch_skip: int = 0,
    device: int = 7,
    apply_rules: bool = True,
) -> str:
    """Profile a CUDA binary with ncu and save the report.

    This runs NVIDIA Nsight Compute on your binary to collect GPU performance metrics.
    You control what data is collected via set/sections/metrics parameters.

    Args:
      binary: Path to compiled CUDA executable (must exist).
      report_name: Name for the output report (no path or extension).
        Saved as reports/<report_name>.ncu-rep. Overwrites if exists.
      set: Metric set to collect ("basic", "detailed", "full", "roofline", "nvlink", "pmsampling").
        Use list_sections_and_sets() to see what each set includes.
        Cannot be used together with sections parameter.
      sections: Specific section identifiers to collect (e.g. ["SpeedOfLight", "Occupancy"]).
        Use list_sections_and_sets() to see available sections.
        Cannot be used together with set parameter.
      metrics: Additional individual metrics to collect on top of set/sections.
        Use query_metrics() to find metric names.
      kernel_filter: Only profile kernels matching this name. Supports "regex:<pattern>".
      launch_count: Max kernel launches to profile (0 = all). Useful for multi-launch apps.
      launch_skip: Skip this many matching launches before profiling.
      device: GPU device index (default 7).
      apply_rules: Apply analysis rules for optimization advice (default True).

    If neither set, sections, nor metrics is provided, defaults to set="basic".

    Profiling strategy:
    1. Start with set="basic" for a quick ~4-pass profile (~5 seconds)
    2. Read results with read_report_details() to identify the bottleneck area
    3. Re-profile with targeted sections for that area (e.g. sections=["MemoryWorkloadAnalysis"])
    4. Use set="full" only when you need comprehensive data (~40 passes, ~60 seconds)

    Returns: Summary with report path, kernels profiled count, and wall clock duration.
    """
    # Validate
    if not os.path.isfile(binary):
        return f"Error: Binary not found: {binary}"
    if set and sections:
        return "Error: Cannot use both 'set' and 'sections'. Choose one."

    # Build args - use CUDA_VISIBLE_DEVICES to select GPU
    env = {"CUDA_VISIBLE_DEVICES": str(device)}
    args = []

    if set:
        args.extend(["--set", set])
    elif sections:
        for s in sections:
            args.extend(["--section", s])
    elif not metrics:
        # Default to basic
        args.extend(["--set", "basic"])

    if metrics:
        args.extend(["--metrics", ",".join(metrics)])

    if kernel_filter:
        args.extend(["-k", kernel_filter])
    if launch_count > 0:
        args.extend(["-c", str(launch_count)])
    if launch_skip > 0:
        args.extend(["-s", str(launch_skip)])

    if apply_rules:
        args.append("--apply-rules")
        args.append("yes")
    else:
        args.append("--apply-rules")
        args.append("no")

    args.extend(["--import-source", "yes"])

    out_path = report_path(report_name)
    args.extend(["-o", out_path])
    args.extend(["-f"])  # Force overwrite

    args.append(binary)

    start = time.time()
    result = await run_ncu(args, timeout=600, env=env)
    elapsed = time.time() - start

    if not result.ok:
        return f"Profiling failed ({elapsed:.1f}s): {result.error_message}"

    # Count kernels from output
    combined = result.stdout + result.stderr
    kernel_count = combined.count("Kernel:")
    if kernel_count == 0:
        # Count from PROF result lines
        kernel_count = combined.count("==PROF== Profiling")
    if "No kernels were profiled" in combined:
        return f"Profiling completed but no kernels were profiled. Check kernel_filter and binary."

    # Check report was created
    rep_file = out_path
    if not rep_file.endswith(".ncu-rep"):
        rep_file += ".ncu-rep"
    if os.path.isfile(rep_file):
        size_mb = os.path.getsize(rep_file) / (1024 * 1024)
        return (
            f"Profiling complete.\n"
            f"- Report: reports/{report_name}.ncu-rep ({size_mb:.1f} MB)\n"
            f"- Duration: {elapsed:.1f}s\n"
            f"- Kernels profiled: {kernel_count}\n"
        )
    elif os.path.isfile(out_path):
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        return (
            f"Profiling complete.\n"
            f"- Report: reports/{report_name} ({size_mb:.1f} MB)\n"
            f"- Duration: {elapsed:.1f}s\n"
            f"- Kernels profiled: {kernel_count}\n"
        )
    else:
        return (
            f"Profiling ran ({elapsed:.1f}s) but report file not found.\n"
            f"ncu stdout: {result.stdout[:500]}\n"
            f"ncu stderr: {result.stderr[:500]}"
        )


# ─── Reading Tools ───────────────────────────────────────────────────────────


@server.tool()
async def read_report_details(
    report_name: str,
    sections: list[str] | None = None,
    kernel_filter: str = "",
    include_rules: bool = True,
    max_chars: int = 8000,
) -> str:
    """Read profiling results organized by section with metrics and optimization advice.

    This is the PRIMARY tool for understanding kernel performance. It shows metrics
    grouped by profiling section, with human-readable labels and internal metric names.

    Args:
      report_name: Report name (as given to profile()).
      sections: Filter to only these sections (e.g. ["SpeedOfLight", "Occupancy"]).
        If empty, returns all sections in the report.
      kernel_filter: Filter to kernels matching this name.
      include_rules: Include optimization advice from analysis rules (default True).
      max_chars: Maximum output length in characters (default 8000). If output exceeds
        this limit, lower-priority sections are omitted and a truncation notice is shown.
        Increase this (e.g. 16000 or 32000) to see more sections, or use sections=
        filter to request specific sections (which bypasses the limit entirely).

    Output format: Markdown tables grouped by kernel, then by section.
    Each metric shows: Label (internal_metric_name) | Value | Unit
    Rules show: Rule name, advice text, estimated speedup percentage.

    Reading strategy:
    1. First call with no section filter to get the overview
    2. If output says OUTPUT TRUNCATED, either use sections= filter for specific sections,
       or increase max_chars to see more data
    3. Pay attention to rule results - they identify the most impactful optimizations
    4. Use the internal metric names (in parentheses) to drill deeper with read_report_raw()

    Example output:
      ## Kernel: matmul (Block: (16,16,1), Grid: (64,64,1))
      ### GPU Speed Of Light Throughput
      | Metric | Value | Unit |
      |--------|-------|------|
      | Memory Throughput (gpu__compute_memory_throughput...) | 45.2 | % |
      | Compute (SM) Throughput (sm__throughput...) | 12.1 | % |
      **SOLBottleneck** (OPT, est. speedup: 55%): This kernel is memory-bound...
    """
    rep_path = _find_report(report_name)
    if not rep_path:
        return f"Report not found: {report_name}. Use list_reports() to see available reports."

    args = ["-i", rep_path, "--csv", "--page", "details", "--print-details", "all",
            "--print-metric-name", "label-name"]
    if kernel_filter:
        args.extend(["-k", kernel_filter])

    result = await run_ncu(args, timeout=120)
    if not result.ok:
        return f"Error reading report: {result.error_message}"

    kernels = parse_details_csv(result.stdout)
    if not kernels:
        return "No data found. The report may be empty or the kernel filter matched nothing."

    return format_details(kernels, sections, include_rules, max_chars=max_chars)


@server.tool()
async def read_report_raw(
    report_name: str,
    metrics: list[str] | None = None,
    kernel_filter: str = "",
    kernel_index: int = 0,
) -> str:
    """Read raw metric values from a report, with optional filtering.

    Use this for precise metric values when read_report_details() doesn't show enough detail,
    or when you need specific metrics identified via query_metrics().

    Args:
      report_name: Report name (as given to profile()).
      metrics: Metric name substrings to filter by (e.g. ["dram__bytes", "gpu__time"]).
        If empty, returns a SUMMARY of available metric categories with counts
        (e.g. "sm__: 342 metrics, dram__: 56 metrics") instead of raw values.
        This is useful to understand what's available before filtering.
      kernel_filter: Filter to kernels matching this name.
      kernel_index: Which kernel launch to show (0-indexed) when multiple launches exist.

    Returns: When metrics is empty, a category summary. When filtered, a table of matching
    metrics with values and units (max 200 rows).

    Usage pattern:
    1. Call with no metrics arg to see what categories are available
    2. Call with metrics=["dram__"] to see all DRAM-related metrics
    3. Call with metrics=["gpu__time_duration"] for specific metrics
    """
    rep_path = _find_report(report_name)
    if not rep_path:
        return f"Report not found: {report_name}. Use list_reports() to see available reports."

    args = ["-i", rep_path, "--csv", "--page", "raw"]
    if kernel_filter:
        args.extend(["-k", kernel_filter])

    result = await run_ncu(args, timeout=120)
    if not result.ok:
        return f"Error reading report: {result.error_message}"

    kernels = parse_raw_csv(result.stdout)
    if not kernels:
        return "No data found."

    # Select kernel by index
    idx = min(kernel_index, len(kernels) - 1)
    kernel = kernels[idx]

    if not metrics:
        return format_raw_summary(kernel)
    else:
        return format_raw_filtered(kernel, metrics)


@server.tool()
async def read_report_source(
    report_name: str,
    kernel_filter: str = "",
    max_lines: int = 50,
) -> str:
    """Read source code with per-line performance metrics and hotspot annotations.

    Shows CUDA source interleaved with SASS assembly, annotated with warp stall
    metrics per line. Lines are sorted by stall impact (hottest lines first).

    Requires the binary to have been compiled with -lineinfo for source correlation.

    Args:
      report_name: Report name (as given to profile()).
      kernel_filter: Filter to kernels matching this name.
      max_lines: Maximum source lines to return (default 50).

    This is most useful AFTER you've identified the bottleneck type via
    read_report_details(). The source view shows you exactly which lines
    are causing stalls and what type of stall (memory, compute, sync, etc.).

    The SourceCounters section must have been collected during profiling
    (included in "detailed" and "full" sets, or add section "SourceCounters").
    """
    rep_path = _find_report(report_name)
    if not rep_path:
        return f"Report not found: {report_name}. Use list_reports() to see available reports."

    args = ["-i", rep_path, "--csv", "--page", "source", "--print-source", "cuda,sass"]
    if kernel_filter:
        args.extend(["-k", kernel_filter])

    result = await run_ncu(args, timeout=120)
    if not result.ok:
        return f"Error reading report: {result.error_message}"

    kernels = parse_source_csv(result.stdout)
    if not kernels:
        return "No source data. Ensure profiling included SourceCounters section and binary was compiled with -lineinfo."

    # Return first kernel's source
    return format_source(kernels[0], max_lines)


@server.tool()
async def read_report_session(report_name: str) -> str:
    """Read device and session metadata from a profiling report.

    Returns GPU hardware info: device name, compute capability, SM count,
    memory size, clock speeds, CUDA version, driver version.

    Args:
      report_name: Report name (as given to profile()).

    Use this to understand hardware capabilities and verify the profiling target.
    """
    rep_path = _find_report(report_name)
    if not rep_path:
        return f"Report not found: {report_name}. Use list_reports() to see available reports."

    args = ["-i", rep_path, "--csv", "--page", "session"]

    result = await run_ncu(args, timeout=60)
    if not result.ok:
        return f"Error reading report: {result.error_message}"

    data = parse_session_csv(result.stdout)
    return format_session(data)


# ─── Utility Tools ───────────────────────────────────────────────────────────


@server.tool()
async def list_reports() -> str:
    """List available profiling report files.

    Returns names, file sizes, and timestamps of all .ncu-rep files in the reports/ directory.
    Use these names with read_report_* tools.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    reports = []

    for f in sorted(os.listdir(REPORTS_DIR)):
        if f.endswith(".ncu-rep"):
            fpath = os.path.join(REPORTS_DIR, f)
            stat = os.stat(fpath)
            size = stat.st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"

            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            name = f.replace(".ncu-rep", "")
            reports.append({"name": name, "size": size_str, "modified": modified})

    return format_reports(reports)


@server.tool()
async def compare_kernels(
    report_a: str,
    report_b: str,
    metrics: list[str] | None = None,
    kernel_filter: str = "",
) -> str:
    """Compare specific metrics between two profiling reports.

    Shows a side-by-side table with values from both reports, absolute delta, and percentage change.
    Essential for measuring whether an optimization actually improved performance.

    Args:
      report_a: First report name (typically the "before" or baseline).
      report_b: Second report name (typically the "after" or optimized).
      metrics: Metric name substrings to compare (e.g. ["gpu__time_duration", "dram__throughput"]).
        Must specify at least one. Use query_metrics() to find metric names.
      kernel_filter: Filter to kernels matching this name (applied to both reports).

    Key metrics to compare for optimization validation:
    - "gpu__time_duration" - kernel execution time (the bottom line)
    - "dram__throughput" - memory bandwidth utilization
    - "sm__throughput" - compute utilization
    - "l1tex__throughput" - L1 cache throughput
    - "lts__throughput" - L2 cache throughput
    - "launch__waves_per_multiprocessor" - occupancy indicator
    """
    if not metrics:
        return "Error: Must specify at least one metric substring to compare."

    path_a = _find_report(report_a)
    path_b = _find_report(report_b)
    if not path_a:
        return f"Report not found: {report_a}"
    if not path_b:
        return f"Report not found: {report_b}"

    # Read raw data from both reports
    args_base = ["--csv", "--page", "raw"]
    if kernel_filter:
        args_base.extend(["-k", kernel_filter])

    result_a = await run_ncu(["-i", path_a] + args_base, timeout=120)
    result_b = await run_ncu(["-i", path_b] + args_base, timeout=120)

    if not result_a.ok:
        return f"Error reading report A: {result_a.error_message}"
    if not result_b.ok:
        return f"Error reading report B: {result_b.error_message}"

    kernels_a = parse_raw_csv(result_a.stdout)
    kernels_b = parse_raw_csv(result_b.stdout)

    if not kernels_a:
        return f"No data in report A: {report_a}"
    if not kernels_b:
        return f"No data in report B: {report_b}"

    return format_comparison(kernels_a[0], kernels_b[0], metrics)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _find_report(name: str) -> str | None:
    """Find a report file by name, checking with and without extension."""
    candidates = [
        report_path(name),
        os.path.join(REPORTS_DIR, name),
        os.path.join(REPORTS_DIR, f"{name}.ncu-rep"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


# ─── Entry Point ─────────────────────────────────────────────────────────────


def main():
    """Entry point for the console script."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
