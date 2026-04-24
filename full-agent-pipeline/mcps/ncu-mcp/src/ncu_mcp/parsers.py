"""CSV parsing for all ncu page types."""

import csv
import io
from dataclasses import dataclass, field


@dataclass
class MetricRow:
    name: str
    label: str
    value: str
    unit: str


@dataclass
class RuleRow:
    name: str
    description: str
    category: str
    estimated_speedup: str = ""  # e.g. "87.81" or ""
    estimated_speedup_type: str = ""  # "global", "local", or ""


@dataclass
class SectionData:
    name: str
    metrics: list[MetricRow] = field(default_factory=list)
    rules: list[RuleRow] = field(default_factory=list)


@dataclass
class KernelDetails:
    name: str
    launch_id: str
    block: str
    grid: str
    sections: list[SectionData] = field(default_factory=list)


@dataclass
class KernelRaw:
    name: str
    launch_id: str
    block: str = ""
    grid: str = ""
    metrics: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> (value, unit)


@dataclass
class SourceLine:
    file: str
    line_number: str
    source: str
    sass: str
    stall_metric: str
    stall_value: str
    stall_breakdown: dict[str, int] = field(default_factory=dict)  # e.g. {"stall_long_sb": 604}


@dataclass
class KernelSource:
    name: str
    launch_id: str
    lines: list[SourceLine] = field(default_factory=list)


def _skip_to_csv(text: str) -> str:
    """Skip any non-CSV preamble lines (ncu sometimes outputs warnings before CSV)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith('"') or "," in line:
            # Check if this looks like a CSV header
            if any(h in line for h in ["Kernel Name", "ID", "Metric Name", "Section Name"]):
                return "\n".join(lines[i:])
    return text


def parse_details_csv(text: str) -> list[KernelDetails]:
    """Parse --page details CSV output.

    Groups rows by kernel (ID column), then by section.
    Rows with 'Rule Name' populated are rule results; others are metrics.
    """
    text = _skip_to_csv(text)
    reader = csv.DictReader(io.StringIO(text))
    kernels: dict[str, KernelDetails] = {}

    for row in reader:
        try:
            kid = row.get("ID", "0")
            kname = row.get("Kernel Name", "unknown")

            if kid not in kernels:
                kernels[kid] = KernelDetails(
                    name=kname,
                    launch_id=kid,
                    block=row.get("Block Size", ""),
                    grid=row.get("Grid Size", ""),
                )
            kernel = kernels[kid]

            section_name = row.get("Section Name", "")
            rule_name = row.get("Rule Name", "")

            # Find or create section
            section = None
            for s in kernel.sections:
                if s.name == section_name:
                    section = s
                    break
            if section is None:
                section = SectionData(name=section_name)
                kernel.sections.append(section)

            if rule_name:
                section.rules.append(RuleRow(
                    name=rule_name,
                    description=row.get("Rule Description", ""),
                    category=row.get("Rule Category", ""),
                    estimated_speedup=row.get("Estimated Speedup", "").strip(),
                    estimated_speedup_type=row.get("Estimated Speedup Type", "").strip(),
                ))
            else:
                metric_name = row.get("Metric Name", "")
                if not metric_name:
                    continue
                # With --print-metric-name label-name, Metric Name has "Label (internal_name)"
                section.metrics.append(MetricRow(
                    name=metric_name,
                    label=row.get("Metric Name", ""),
                    value=row.get("Metric Value", ""),
                    unit=row.get("Metric Unit", ""),
                ))
        except Exception:
            continue

    return list(kernels.values())


def parse_raw_csv(text: str) -> list[KernelRaw]:
    """Parse --page raw CSV output.

    Raw page uses wide format: one row per kernel launch, metrics as columns.
    Line 1: header (metric names), Line 2: units, Line 3+: data.
    """
    text = _skip_to_csv(text)
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return []

    headers = rows[0]
    units = rows[1]

    # Non-metric columns to skip
    skip_cols = {
        "ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
        "Context", "Stream", "Block Size", "Grid Size", "Device", "CC",
    }

    # Build unit map from headers + units row
    unit_map = {}
    for i, h in enumerate(headers):
        if i < len(units) and units[i]:
            unit_map[h] = units[i]

    kernels = []
    for data_row in rows[2:]:
        try:
            # Build dict from header + data
            row = dict(zip(headers, data_row))
            kid = row.get("ID", "0")
            kname = row.get("Kernel Name", "unknown")
            kernel = KernelRaw(
                name=kname,
                launch_id=kid,
                block=row.get("Block Size", ""),
                grid=row.get("Grid Size", ""),
            )

            for i, h in enumerate(headers):
                if h and h not in skip_cols and i < len(data_row):
                    val = data_row[i]
                    if val is not None and val.strip():
                        kernel.metrics[h] = (val, unit_map.get(h, ""))

            kernels.append(kernel)
        except Exception:
            continue

    return kernels


def parse_source_csv(text: str) -> list[KernelSource]:
    """Parse --page source --print-source cuda,sass CSV output.

    Format:
    - Row 1: "File Path","<path>"
    - Row 2: "Function Name","<kernel name>"
    - Row 3: Header row with "Line No","Source","Address","Source",...stall columns
    - Data rows: CUDA source lines have Line No + Source, SASS lines have Address + Source
    - Stall columns include "Warp Stall Sampling (All Samples)", "# Samples", stall_* columns
    """
    lines = text.strip().split("\n")
    kernels = []
    current_kernel = None
    headers = None
    file_path = ""

    for raw_line in lines:
        parsed = list(csv.reader(io.StringIO(raw_line)))
        if not parsed or not parsed[0]:
            continue
        row = parsed[0]

        # Detect file path line
        if len(row) >= 2 and row[0] == "File Path":
            file_path = row[1]
            continue

        # Detect function/kernel name line
        if len(row) >= 2 and row[0] == "Function Name":
            current_kernel = KernelSource(name=row[1], launch_id=str(len(kernels)))
            kernels.append(current_kernel)
            headers = None
            continue

        # Detect header row
        if len(row) > 2 and row[0] in ("Line No", "Address"):
            headers = row
            continue

        if not current_kernel or not headers:
            continue

        try:
            # Build a dict from headers
            row_dict = dict(zip(headers, row))
            line_no = row_dict.get("Line No", "")
            address = row_dict.get("Address", "")

            # Source columns - there are two: one for CUDA, one for SASS
            # Find all "Source" columns
            source_indices = [i for i, h in enumerate(headers) if h == "Source"]
            cuda_source = row[source_indices[0]] if len(source_indices) > 0 and source_indices[0] < len(row) else ""
            sass_source = row[source_indices[1]] if len(source_indices) > 1 and source_indices[1] < len(row) else ""

            stall_samples = row_dict.get("Warp Stall Sampling (All Samples)", "0")
            num_samples = row_dict.get("# Samples", "0")

            # Collect stall breakdown from all stall_* columns (non-zero, exclude "Not Issued")
            stall_breakdown = {}
            for h_name in headers:
                if h_name.startswith("stall_") and "(Not Issued)" not in h_name:
                    h_idx = headers.index(h_name)
                    if h_idx < len(row):
                        try:
                            val = int(row[h_idx].replace(",", ""))
                            if val > 0:
                                stall_breakdown[h_name] = val
                        except (ValueError, IndexError):
                            pass

            # Determine if this is a CUDA source line or SASS line
            if line_no and line_no != "-":
                # CUDA source line
                sl = SourceLine(
                    file=file_path,
                    line_number=line_no,
                    source=cuda_source,
                    sass="",
                    stall_metric="Stall Samples",
                    stall_value=stall_samples,
                    stall_breakdown=stall_breakdown,
                )
                current_kernel.lines.append(sl)
            elif address and address != "-":
                # SASS line
                sl = SourceLine(
                    file="",
                    line_number=address,
                    source="",
                    sass=sass_source,
                    stall_metric="Stall Samples",
                    stall_value=stall_samples,
                    stall_breakdown=stall_breakdown,
                )
                current_kernel.lines.append(sl)
        except Exception:
            continue

    return kernels


def parse_session_csv(text: str) -> dict[str, str]:
    """Parse --page session CSV output into key-value pairs.

    Session page has multiple sections with different headers:
    - "Launch Attribute","Value"
    - "Session Attribute","Value"
    - "Device Attribute","Device 0"
    Each section has a plain-text section header line.
    """
    result = {}
    current_section = ""

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Section headers are plain text lines (no quotes, no commas in them usually)
        if not line.startswith('"') and "," not in line:
            current_section = line
            continue

        # Parse CSV rows
        try:
            parsed = list(csv.reader(io.StringIO(line)))
            if not parsed or len(parsed[0]) < 2:
                continue
            row = parsed[0]
            key = row[0]
            value = row[1]

            # Skip header rows
            if key in ("Launch Attribute", "Session Attribute", "Device Attribute",
                       "Process Id", "Attribute"):
                continue

            # Prefix with section for clarity if needed
            result[key] = value
        except Exception:
            continue

    return result


def parse_list_output(text: str) -> list[tuple[str, str]]:
    """Parse --list-sections or --list-sets or --list-rules output.

    Handles ncu's fixed-width column format with continuation lines.
    Returns list of (identifier, description) tuples where description
    is the second column content.
    """
    import re
    lines = text.split("\n")

    # Find column positions from the dashes separator line
    col_starts = []
    for line in lines:
        if re.match(r"^-{3,}", line):
            # Find start positions of each dash group
            for m in re.finditer(r"-{2,}", line):
                col_starts.append(m.start())
            break

    if len(col_starts) < 2:
        # Fallback: simple 2+ space split
        results = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("-") or line.startswith("="):
                continue
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) == 2:
                results.append((parts[0].strip(), parts[1].strip()))
        return results

    # Parse using column positions - extract col 0 (Identifier) and col 1 (Display Name/Sections/Description)
    results = []
    current_id = None
    current_desc = None

    for line in lines:
        if not line or re.match(r"^-{3,}", line):
            continue

        # Extract identifier (column 0)
        col0_end = col_starts[1] if len(col_starts) > 1 else len(line)
        identifier = line[:col0_end].strip() if len(line) > 0 else ""

        # Extract description (column 1)
        if len(col_starts) > 1 and len(line) > col_starts[1]:
            col1_end = col_starts[2] if len(col_starts) > 2 else len(line)
            description = line[col_starts[1]:col1_end].strip()
        else:
            description = ""

        # Skip the header line
        if identifier in ("Identifier",):
            continue

        if identifier:
            # New entry - save previous if exists
            if current_id is not None:
                results.append((current_id, current_desc))
            current_id = identifier
            current_desc = description
        elif current_id is not None and description:
            # Continuation line
            current_desc += " " + description

    # Save last entry
    if current_id is not None:
        results.append((current_id, current_desc))

    return results


def parse_query_metrics(text: str) -> list[str]:
    """Parse --query-metrics output to extract metric names.

    Format: fixed-width columns with "Metric Name", "Metric Type", "Metric Unit", "Metric Description".
    Skips "Device ..." header line and dashes.
    """
    metrics = []
    for line in text.strip().split("\n"):
        if not line.strip() or line.strip().startswith("-") or line.strip().startswith("="):
            continue
        if line.strip().startswith("Device ") or line.strip().startswith("Metric Name"):
            continue
        # Metric names are the first column, no spaces in metric names
        name = line.split()[0] if line.split() else ""
        if name and ("__" in name or "." in name):
            metrics.append(name)
    return metrics
