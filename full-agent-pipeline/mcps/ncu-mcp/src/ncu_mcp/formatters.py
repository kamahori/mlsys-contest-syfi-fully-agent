"""LLM-friendly output formatting for ncu profiling data."""

from ncu_mcp.parsers import (
    KernelDetails,
    KernelRaw,
    KernelSource,
    RuleRow,
    SectionData,
)

# Priority order for section display when truncating
SECTION_PRIORITY = [
    "SpeedOfLight",
    "Speed Of Light",
    "GPU Speed Of Light Throughput",
    "MemoryWorkloadAnalysis",
    "Memory Workload Analysis",
    "ComputeWorkloadAnalysis",
    "Compute Workload Analysis",
    "Occupancy",
    "LaunchStats",
    "Launch Statistics",
    "WorkloadDistribution",
    "WarpStateStats",
    "Warp State Statistics",
    "SchedulerStats",
    "Scheduler Statistics",
    "InstructionStats",
    "Instruction Statistics",
    "SourceCounters",
    "Source Counters",
]


def _section_sort_key(name: str) -> int:
    """Sort sections by priority (lower = higher priority)."""
    name_lower = name.lower()
    for i, p in enumerate(SECTION_PRIORITY):
        if p.lower() in name_lower or name_lower in p.lower():
            return i
    return len(SECTION_PRIORITY)


# Metric substrings for bottleneck summary extraction
_SUMMARY_METRICS = {
    "Memory Throughput": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "Compute (SM) Throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "Duration": "gpu__time_duration.sum",
    "Waves/SM": "launch__waves_per_multiprocessor",
    "Achieved Occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
}

# Display order for summary
_SUMMARY_ORDER = ["Duration", "Compute (SM) Throughput", "Memory Throughput", "Waves/SM", "Achieved Occupancy"]


def _format_bottleneck_summary(kernel: KernelDetails) -> str:
    """Extract key performance metrics into a summary block."""
    found: dict[str, tuple[str, str]] = {}  # label -> (value, unit)

    for section in kernel.sections:
        for m in section.metrics:
            for label, substring in _SUMMARY_METRICS.items():
                if substring in m.name and label not in found:
                    found[label] = (m.value, m.unit)

    if not found:
        return ""

    lines = ["### Summary"]
    for label in _SUMMARY_ORDER:
        if label in found:
            value, unit = found[label]
            if unit and unit != "n/a":
                lines.append(f"- {label}: {value} {unit}")
            else:
                lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def _format_rules_summary(kernel: KernelDetails) -> str:
    """Collect all rules across sections, deduplicate, sort by speedup."""
    all_rules: list[tuple[RuleRow, str]] = []  # (rule, section_name)
    seen: set[str] = set()

    for section in kernel.sections:
        for r in section.rules:
            dedup_key = r.name + r.description[:80]
            if dedup_key not in seen:
                seen.add(dedup_key)
                all_rules.append((r, section.name))

    if not all_rules:
        return ""

    def speedup_sort_key(item: tuple[RuleRow, str]) -> float:
        try:
            return -float(item[0].estimated_speedup)
        except (ValueError, TypeError):
            return 0.0

    all_rules.sort(key=speedup_sort_key)

    lines = ["### Optimization Opportunities"]
    lines.append("| # | Rule | Est. Speedup | Advice |")
    lines.append("|---|------|-------------|--------|")

    for i, (r, _section_name) in enumerate(all_rules, 1):
        speedup = f"{float(r.estimated_speedup):.1f}%" if r.estimated_speedup else ""
        desc = r.description.replace("\n", " ").replace("|", "\\|").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
        lines.append(f"| {i} | {r.name} | {speedup} | {desc} |")

    return "\n".join(lines)


def format_details(
    kernels: list[KernelDetails],
    section_filter: list[str] | None = None,
    include_rules: bool = True,
    max_chars: int = 8000,
) -> str:
    """Format details page output: summary → rules → filtered section metrics."""
    if not kernels:
        return "No kernel data found in report."

    parts = []
    for kernel in kernels:
        header = f"## Kernel: {kernel.name}"
        if kernel.block or kernel.grid:
            header += f" (Block: {kernel.block}, Grid: {kernel.grid})"
        parts.append(header)

        # Bottleneck summary (always shown)
        summary = _format_bottleneck_summary(kernel)
        if summary:
            parts.append(summary)

        # Rules summary (before sections, if requested)
        if include_rules:
            rules = _format_rules_summary(kernel)
            if rules:
                parts.append(rules)

        sections = kernel.sections
        if section_filter:
            filter_lower = [f.lower() for f in section_filter]
            sections = [
                s for s in sections
                if any(f in s.name.lower() for f in filter_lower)
            ]

        # Sort by priority
        sections.sort(key=lambda s: _section_sort_key(s.name))

        for section in sections:
            section_text = _format_section(section)
            if section_text:
                parts.append(section_text)

    result = "\n\n".join(parts)

    # Only truncate when no explicit section filter is applied
    if not section_filter and len(result) > max_chars:
        result = _truncate_details(kernels, section_filter, include_rules, max_chars)

    return result


def _format_section(section: SectionData) -> str:
    """Format a single section with filtered metrics (no rules — shown in summary)."""
    # Skip sections with no metrics (e.g. identifier-only duplicate sections)
    if not section.metrics:
        return ""

    lines = [f"### {section.name}"]

    array_hidden = 0
    zero_pct_hidden = 0

    if section.metrics:
        lines.append("| Metric | Value | Unit |")
        lines.append("|--------|-------|------|")
        for m in section.metrics:
            # Filter A: skip array-valued metrics (semicolons = occupancy sweep arrays)
            if ";" in m.value:
                array_hidden += 1
                continue
            # Filter B: skip zero-value percentage metrics
            if m.unit == "%" and m.value.strip() == "0":
                zero_pct_hidden += 1
                continue
            lines.append(f"| {m.name} | {m.value} | {m.unit} |")

    if array_hidden > 0:
        lines.append(f"\n*{array_hidden} occupancy sweep metrics hidden (array-valued). Use read_report_raw(metrics=[\"derived__pct_occupancy\"]) to access.*")
    if zero_pct_hidden > 0:
        lines.append(f"\n*{zero_pct_hidden} metrics at 0% hidden. Use read_report_raw(metrics=[\"sm__pipe\"]) for full pipe utilization breakdown.*")

    # Add occupancy analysis if this is the Occupancy section
    if "occupancy" in section.name.lower():
        analysis = _format_occupancy_analysis(section)
        if analysis:
            lines.append(analysis)

    return "\n".join(lines)


def _format_occupancy_analysis(section: SectionData) -> str:
    """Extract occupancy limiters into a compact summary."""
    limit_keys = {
        "occupancy_limit_registers": "Registers",
        "occupancy_limit_warps": "Warps",
        "occupancy_limit_shared_mem": "Shared Mem",
        "occupancy_limit_blocks": "SM Limit",
        "occupancy_limit_barriers": "Barriers",
    }
    info_keys = {
        "registers_per_thread": "Registers/Thread",
        "block_size": "Block Size",
    }

    limits: dict[str, int] = {}
    info: dict[str, str] = {}

    for m in section.metrics:
        # Skip array-valued metrics (occupancy sweeps)
        if ";" in m.value:
            continue
        name_lower = m.name.lower()
        for key, label in limit_keys.items():
            if key in name_lower:
                try:
                    limits[label] = int(float(m.value.replace(",", "")))
                except (ValueError, TypeError):
                    pass
                break
        for key, label in info_keys.items():
            if key in name_lower and label not in info:
                info[label] = m.value.strip()

    if not limits:
        return ""

    bottleneck_label = min(limits, key=lambda k: limits[k])
    bottleneck_val = limits[bottleneck_label]

    # Format: "Tightest limit: Warps (4 blocks/SM). All limits: Warps=4, Registers=5, Shared Mem=12, SM Limit=32."
    all_limits = ", ".join(f"{k}={v}" for k, v in sorted(limits.items(), key=lambda x: x[1]))
    info_str = ", ".join(f"{k}: {v}" for k, v in info.items())

    parts = [f"\n**Occupancy Limits**: Tightest: **{bottleneck_label}** ({bottleneck_val} blocks/SM). All: {all_limits}."]
    if info_str:
        parts.append(info_str + ".")

    return " ".join(parts)


def _truncate_details(
    kernels: list[KernelDetails],
    section_filter: list[str] | None,
    include_rules: bool,
    max_chars: int,
) -> str:
    """Truncate output keeping summary + rules + highest priority sections."""
    parts = []
    remaining = max_chars - 200  # Reserve for truncation notice

    for kernel in kernels:
        header = f"## Kernel: {kernel.name}"
        if kernel.block or kernel.grid:
            header += f" (Block: {kernel.block}, Grid: {kernel.grid})"
        parts.append(header)
        remaining -= len(header) + 2

        # Always include summary and rules (they're compact)
        summary = _format_bottleneck_summary(kernel)
        if summary:
            parts.append(summary)
            remaining -= len(summary) + 2

        if include_rules:
            rules = _format_rules_summary(kernel)
            if rules:
                parts.append(rules)
                remaining -= len(rules) + 2

        sections = kernel.sections
        if section_filter:
            filter_lower = [f.lower() for f in section_filter]
            sections = [
                s for s in sections
                if any(f in s.name.lower() for f in filter_lower)
            ]
        sections.sort(key=lambda s: _section_sort_key(s.name))

        omitted = []
        for section in sections:
            section_text = _format_section(section)
            if not section_text:
                continue
            if len(section_text) <= remaining:
                parts.append(section_text)
                remaining -= len(section_text) + 2
            else:
                omitted.append(section.name)

        if omitted:
            parts.append(
                f"\n**OUTPUT TRUNCATED** (max_chars={max_chars}). "
                f"Omitted sections: {', '.join(omitted)}. "
                f"Use sections= filter to request specific sections, "
                f"or increase max_chars (e.g. max_chars=16000)."
            )

    return "\n\n".join(parts)


def format_raw_summary(kernel: KernelRaw) -> str:
    """Format metric count by prefix category, excluding device__/profiler__ noise."""
    if not kernel.metrics:
        return "No metrics found."

    _EXCLUDED_PREFIXES = ("device__", "profiler__")

    categories: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    for name in kernel.metrics:
        prefix = name.split("__")[0] + "__" if "__" in name else name.split(".")[0]
        if any(name.startswith(ep) for ep in _EXCLUDED_PREFIXES):
            excluded_counts[prefix] = excluded_counts.get(prefix, 0) + 1
        else:
            categories[prefix] = categories.get(prefix, 0) + 1

    total = sum(categories.values())
    lines = [f"## Kernel: {kernel.name}", f"**Total metrics: {total}**", ""]
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} |")

    lines.append(f"\nUse metrics= parameter with a prefix (e.g. \"dram__\") to see values.")

    # Exclusion notice
    total_excluded = sum(excluded_counts.values())
    if total_excluded > 0:
        parts = [f"{count} {prefix} {'properties' if 'device' in prefix else 'internals'}"
                 for prefix, count in sorted(excluded_counts.items(), key=lambda x: -x[1])]
        lines.append(f"\n*Excluded: {', '.join(parts)}. Use read_report_raw(metrics=[\"device__\"]) to access.*")

    return "\n".join(lines)


def format_raw_filtered(kernel: KernelRaw, substrings: list[str], max_rows: int = 200) -> str:
    """Format filtered metric table."""
    if not kernel.metrics:
        return "No metrics found."

    # Filter metrics
    filtered = {}
    for name, (value, unit) in kernel.metrics.items():
        if any(s.lower() in name.lower() for s in substrings):
            filtered[name] = (value, unit)

    if not filtered:
        total = len(kernel.metrics)
        return (
            f"No metrics matching {substrings} found in this report "
            f"(report contains {total} metrics). "
            f"These metrics may exist but weren't collected in this profiling set. "
            f"Use query_metrics(substring=...) to verify metric names exist, "
            f"then re-profile with metrics=[...] to include them."
        )

    lines = [f"## Kernel: {kernel.name} ({len(filtered)} metrics)"]
    lines.append("| Metric | Value | Unit |")
    lines.append("|--------|-------|------|")

    count = 0
    for name in sorted(filtered):
        if count >= max_rows:
            lines.append(f"\n*Truncated at {max_rows} rows. Narrow your filter.*")
            break
        value, unit = filtered[name]
        lines.append(f"| {name} | {value} | {unit} |")
        count += 1

    return "\n".join(lines)


def _is_library_code(file_path: str) -> bool:
    """Check if a file path is library/system code."""
    if not file_path:
        return True
    return (file_path.startswith("/usr/")
            or file_path.startswith("/opt/")
            or "/include/cuda" in file_path
            or "/include/crt/" in file_path)


def _format_stall_breakdown(breakdown: dict[str, int]) -> str:
    """Format top 3 stall reasons as percentage of total samples."""
    if not breakdown:
        return ""
    total = sum(breakdown.values())
    if total == 0:
        return ""
    # Sort by count descending, take top 3
    top = sorted(breakdown.items(), key=lambda x: -x[1])[:3]
    parts = []
    for name, count in top:
        pct = round(count / total * 100)
        # Strip "stall_" prefix for brevity
        short = name.removeprefix("stall_")
        parts.append(f"{short}: {pct}%")
    return ", ".join(parts)


def _format_sass_instruction_mix(source_lines: list) -> str:
    """Group SASS instructions by opcode type and show counts."""
    from collections import Counter
    opcode_counts: Counter[str] = Counter()
    for sl in source_lines:
        if not sl.sass:
            continue
        sass_text = sl.sass.strip()
        if not sass_text or sass_text.startswith("."):
            continue
        tokens = sass_text.split()
        if not tokens:
            continue
        opcode_full = tokens[0]
        if not opcode_full or opcode_full in ("...", "."):
            continue
        # Group by first 2 dot-components (type + datatype)
        parts = opcode_full.split(".")
        key = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else parts[0]
        opcode_counts[key] += 1

    if not opcode_counts:
        return ""

    lines = ["### SASS Instruction Mix"]
    lines.append("| Instruction | Count |")
    lines.append("|-------------|-------|")
    for opcode, count in opcode_counts.most_common(15):
        lines.append(f"| {opcode} | {count} |")
    return "\n".join(lines)


def format_source(kernel: KernelSource, max_lines: int = 50) -> str:
    """Format source hotspots: deduped, stall breakdown, user code first."""
    if not kernel.lines:
        return "No source data. Ensure profiling included SourceCounters section and binary was compiled with -lineinfo."

    # Deduplicate by (line_number, sass) — keep highest stall_value
    dedup: dict[tuple[str, str], "SourceLine"] = {}
    for sl in kernel.lines:
        key = (sl.line_number, sl.sass)
        if key not in dedup:
            dedup[key] = sl
        else:
            try:
                existing = float(dedup[key].stall_value.replace(",", "").replace("%", ""))
            except (ValueError, AttributeError):
                existing = 0
            try:
                new = float(sl.stall_value.replace(",", "").replace("%", ""))
            except (ValueError, AttributeError):
                new = 0
            if new > existing:
                dedup[key] = sl

    unique_lines = list(dedup.values())

    # Check for zero total samples (kernel too fast for sampling)
    total_samples = 0
    for sl in unique_lines:
        try:
            total_samples += int(sl.stall_value.replace(",", "").replace("%", ""))
        except (ValueError, AttributeError):
            pass

    # Sort by stall value descending
    def stall_sort_key(line):
        try:
            return -float(line.stall_value.replace(",", "").replace("%", ""))
        except (ValueError, AttributeError):
            return 0

    unique_lines.sort(key=stall_sort_key)

    # Stable partition: user code first, library code second
    user_lines = [l for l in unique_lines if not _is_library_code(l.file)]
    lib_lines = [l for l in unique_lines if _is_library_code(l.file)]
    ordered = user_lines + lib_lines

    lines = [f"## Kernel: {kernel.name} ({len(user_lines)} user code lines, {len(lib_lines)} library lines)"]

    # Warning: zero stall samples (kernel too fast for sampling)
    if total_samples == 0 and unique_lines:
        lines.append("")
        lines.append("*Warning: 0 stall samples collected — kernel likely too fast or too few launches for sampling. Profile with launch_count=10 or a larger workload to collect sampling data.*")

    # When no user source lines, show SASS instruction mix as fallback
    if len(user_lines) == 0 and len(unique_lines) > 0:
        lines.append("")
        lines.append("*No user source lines — compiler inlined into library headers. SASS instruction analysis follows.*")
        sass_summary = _format_sass_instruction_mix(unique_lines)
        if sass_summary:
            lines.append("")
            lines.append(sass_summary)

    lines.append("| Location | Source | Samples | Top Stalls |")
    lines.append("|----------|--------|---------|------------|")

    for sl in ordered[:max_lines]:
        loc = f"{sl.file}:{sl.line_number}" if sl.file else sl.line_number
        source = (sl.source or sl.sass).replace("|", "\\|").strip()[:80]
        stalls = _format_stall_breakdown(sl.stall_breakdown)
        lines.append(f"| {loc} | {source} | {sl.stall_value} | {stalls} |")

    return "\n".join(lines)


def format_session(data: dict[str, str]) -> str:
    """Format session/device info as clean key-value pairs."""
    if not data:
        return "No session data found."

    lines = ["## Device / Session Info"]
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    for k, v in data.items():
        lines.append(f"| {k} | {v} |")

    return "\n".join(lines)


def format_comparison(
    kernel_a: KernelRaw,
    kernel_b: KernelRaw,
    substrings: list[str],
) -> str:
    """Format side-by-side comparison with deltas."""
    # Find common metrics matching substrings
    matching_a = {
        n: v for n, v in kernel_a.metrics.items()
        if any(s.lower() in n.lower() for s in substrings)
    }
    matching_b = {
        n: v for n, v in kernel_b.metrics.items()
        if any(s.lower() in n.lower() for s in substrings)
    }

    common = set(matching_a.keys()) & set(matching_b.keys())
    if not common:
        return f"No common metrics matching {substrings} found in both reports."

    header = f"## Comparison: {kernel_a.name}"
    if kernel_a.block or kernel_a.grid:
        header += f" (Block: {kernel_a.block}, Grid: {kernel_a.grid})"
    header += f" vs {kernel_b.name}"
    if kernel_b.block or kernel_b.grid:
        header += f" (Block: {kernel_b.block}, Grid: {kernel_b.grid})"
    lines = [header]

    # Warn if grid sizes differ (likely comparing different workloads)
    if kernel_a.grid and kernel_b.grid and kernel_a.grid != kernel_b.grid:
        lines.append("")
        lines.append(f"*Warning: Grid sizes differ ({kernel_a.grid} vs {kernel_b.grid}) — ensure you're comparing the same workload.*")
        lines.append("")

    lines.append("| Metric | Report A | Report B | Delta | Change % | Unit |")
    lines.append("|--------|----------|----------|-------|----------|------|")

    for name in sorted(common):
        val_a_str, unit = matching_a[name]
        val_b_str, _ = matching_b[name]

        try:
            val_a = float(val_a_str.replace(",", ""))
            val_b = float(val_b_str.replace(",", ""))
            delta = val_b - val_a
            pct = (delta / val_a * 100) if val_a != 0 else 0
            delta_str = f"{delta:+.4g}"
            pct_str = f"{pct:+.1f}%"
        except (ValueError, TypeError):
            delta_str = "N/A"
            pct_str = "N/A"

        lines.append(f"| {name} | {val_a_str} | {val_b_str} | {delta_str} | {pct_str} | {unit} |")

    return "\n".join(lines)


def format_sections_and_sets(
    sections: list[tuple[str, str]],
    sets: list[tuple[str, str]],
) -> str:
    """Format list of sections and sets."""
    lines = ["## Profiling Sections"]
    lines.append("| Identifier | Description |")
    lines.append("|------------|-------------|")
    for ident, desc in sections:
        lines.append(f"| {ident} | {desc} |")

    lines.append("\n## Metric Sets")
    lines.append("| Set Name | Description |")
    lines.append("|----------|-------------|")
    for ident, desc in sets:
        lines.append(f"| {ident} | {desc} |")

    return "\n".join(lines)


def format_rules(rules: list[tuple[str, str]]) -> str:
    """Format list of rules."""
    lines = ["## Analysis Rules"]
    lines.append("| Rule | Description |")
    lines.append("|------|-------------|")
    for ident, desc in rules:
        lines.append(f"| {ident} | {desc} |")
    return "\n".join(lines)


def format_metrics_summary(metrics: list[str]) -> str:
    """Format metric query results as category summary or list."""
    if not metrics:
        return "No metrics found."

    # If few enough, list them all
    if len(metrics) <= 100:
        lines = [f"**{len(metrics)} metrics found:**"]
        for m in metrics:
            lines.append(f"- {m}")
        return "\n".join(lines)

    # Otherwise, show category summary
    categories: dict[str, int] = {}
    for name in metrics:
        prefix = name.split("__")[0] + "__" if "__" in name else name
        categories[prefix] = categories.get(prefix, 0) + 1

    lines = [f"**{len(metrics)} metrics total. Categories:**"]
    lines.append("| Prefix | Count |")
    lines.append("|--------|-------|")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} |")
    lines.append(f"\nUse substring filter to narrow results (e.g. \"dram__bytes\").")
    return "\n".join(lines)


def format_reports(reports: list[dict]) -> str:
    """Format list of report files."""
    if not reports:
        return "No reports found in reports/ directory."

    lines = ["## Available Reports"]
    lines.append("| Name | Size | Modified |")
    lines.append("|------|------|----------|")
    for r in reports:
        lines.append(f"| {r['name']} | {r['size']} | {r['modified']} |")
    return "\n".join(lines)
