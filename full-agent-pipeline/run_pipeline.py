"""Fully autonomous agent pipeline for the FlashInfer-Bench MLSys26 contest.

Two phases, each with N refinement rounds gated by a real
flashinfer-bench eval (correctness + speedup on all workloads of the
chosen definition):

    Phase 1: PyTorch reference  ->  Triton kernel
    Phase 2: Triton best        ->  C++/CUDA kernel

The agent is Claude Code in headless mode (`claude -p`), invoked from
this repo root so it automatically picks up:
    - .claude/skills/*                (GPU skills: cutlass-triton, nsight-profiler, etc.)
    - .mcp.json                       (ncu-mcp server wrapping Nsight Compute)

Requires:
    - FIB_DATASET_PATH pointing at the flashinfer-trace dataset
    - `claude` CLI on PATH
    - uv-managed venv synced from the sibling pyproject.toml
    - NVIDIA GPU + ncu CLI for profiling

Usage:
    uv run python run_pipeline.py \\
        --definition gdn_decode_qk4_v8_d128_k_last \\
        --workdir runs/gdn_decode_v1 \\
        --triton-iters 4 --cuda-iters 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
MCP_CONFIG = REPO_ROOT / ".mcp.json"
REFERENCES_DIR = REPO_ROOT / "references"
BUNDLED_DATASET = REPO_ROOT / "mlsys26-contest"

# Definition-name prefix -> local reference .py (copied verbatim from the
# starter kit under references/). Longest prefix wins.
REFERENCE_BY_PREFIX: dict[str, str] = {
    "gdn_decode": "gdn_decode.py",
    "gdn_prefill": "gdn_prefill.py",
    "dsa_topk_indexer": "dsa_topk_indexer.py",
    "dsa_sparse_attention": "dsa_sparse_attention.py",
    "moe": "moe.py",
}


def local_reference_for(definition_name: str) -> Path | None:
    """Return the starter-kit reference .py for this track, or None."""
    best: tuple[int, str] | None = None
    for prefix, fname in REFERENCE_BY_PREFIX.items():
        if definition_name.startswith(prefix):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), fname)
    if best is None:
        return None
    p = REFERENCES_DIR / best[1]
    return p if p.is_file() else None

# --- skill hints -------------------------------------------------------------
TRITON_SKILLS = [
    "cutlass-triton",
    "parallel-patterns",
    "warp-primitives",
    "gpu-memory-analysis",
    "gpu-benchmarking",
    "nsight-profiler",
]
CUDA_SKILLS = [
    "cuda-toolkit",
    "cuda-debugging",
    "cutlass-triton",
    "warp-primitives",
    "parallel-patterns",
    "gpu-memory-analysis",
    "nsight-profiler",
    "gpu-benchmarking",
    "cublas-cudnn",
]

# MCP tool prefix we want the agent to reach for when diagnosing perf.
NCU_TOOL_PREFIX = "mcp__ncu-mcp"


# --- claude invocation -------------------------------------------------------
def run_claude(
    prompt: str,
    *,
    cwd: Path,
    session_id: str | None = None,
    timeout: int = 5400,
    model: str = "claude-opus-4-7",
    extra_env: dict | None = None,
) -> dict:
    """Spawn `claude -p` headlessly and return parsed JSON envelope."""
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        model,
        "--mcp-config",
        str(MCP_CONFIG),
        "--strict-mcp-config",
    ]
    if session_id:
        cmd += ["--resume", session_id]

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    print(f"[claude] invoking (cwd={cwd}, resume={session_id})", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude timed out after {timeout}s")
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"[claude] exit={proc.returncode} stderr:\n{proc.stderr[:2000]}",
              flush=True)
        raise RuntimeError(f"claude failed rc={proc.returncode}")

    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"[claude] non-JSON stdout (first 1000):\n{proc.stdout[:1000]}")
        raise

    print(
        f"[claude] done in {dt:.1f}s "
        f"session={envelope.get('session_id')} "
        f"turns={envelope.get('num_turns')}",
        flush=True,
    )
    return envelope


# --- flashinfer-bench wrappers ----------------------------------------------
def dataset_path() -> str:
    p = os.environ.get("FIB_DATASET_PATH")
    if p:
        if not Path(p).is_dir():
            sys.exit(f"FIB_DATASET_PATH does not exist: {p}")
        return p
    if BUNDLED_DATASET.is_dir():
        return str(BUNDLED_DATASET)
    sys.exit("FIB_DATASET_PATH not set and no bundled dataset at "
             f"{BUNDLED_DATASET}. Either export FIB_DATASET_PATH or "
             "clone the dataset into the repo (see README).")


def load_definition(definition_name: str):
    from flashinfer_bench import TraceSet
    ts = TraceSet.from_path(dataset_path())
    if definition_name not in ts.definitions:
        avail = sorted(ts.definitions.keys())
        sys.exit(f"definition '{definition_name}' not in TraceSet. "
                 f"available: {avail}")
    return ts, ts.definitions[definition_name]


def dump_definition_context(
    definition,
    out_dir: Path,
    reference_override: Path | None = None,
) -> Path:
    """Write the op definition (IO schema + pytorch reference) for the agent."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # full JSON dump of the Definition for machine consumption
    (out_dir / "definition.json").write_text(
        definition.model_dump_json(indent=2)
    )

    # Resolve the pytorch reference. Precedence:
    #   1. explicit --reference-file override
    #   2. starter-kit reference under references/<track>.py
    #   3. whatever Definition.reference carries (embedded in TraceSet)
    ref_path = out_dir / "reference.py"
    content: str | None = None
    src_note = ""

    if reference_override and reference_override.is_file():
        content = reference_override.read_text()
        src_note = f"explicit --reference-file: {reference_override}"
    else:
        local = local_reference_for(definition.name)
        if local:
            content = local.read_text()
            src_note = f"starter-kit reference: {local.relative_to(REPO_ROOT)}"

    if content is None:
        ref_obj = getattr(definition, "reference", None)
        if ref_obj is not None:
            src_list = getattr(ref_obj, "sources", None) or []
            if src_list:
                parts = []
                for sf in src_list:
                    parts.append(f"# === {getattr(sf, 'path', '?')} ===\n"
                                 f"{getattr(sf, 'content', '')}")
                content = "\n\n".join(parts)
            else:
                content = (
                    getattr(ref_obj, "code", None)
                    or getattr(ref_obj, "content", None)
                    or getattr(ref_obj, "source", None)
                )
            if content is not None:
                src_note = "extracted from Definition.reference in TraceSet"

    if content is None:
        content = "# reference source not available\n"
        src_note = "NOT FOUND"

    header = (f"# Source: {src_note}\n"
              f"# Definition: {definition.name}\n\n")
    ref_path.write_text(header + content)
    print(f"[ctx] reference -> {ref_path} ({src_note})")

    # human readable summary
    lines = [f"# Definition: {definition.name}",
             f"- op_type: {getattr(definition, 'op_type', '?')}",
             f"- axes: {list(getattr(definition, 'axes', {}).keys())}",
             "",
             "## inputs"]
    for nm, t in (definition.inputs or {}).items():
        lines.append(f"- {nm}: shape={t.shape} dtype={t.dtype}")
    lines.append("")
    lines.append("## outputs")
    for nm, t in (definition.outputs or {}).items():
        lines.append(f"- {nm}: shape={t.shape} dtype={t.dtype}")
    if getattr(definition, "description", None):
        lines += ["", "## description", definition.description]
    (out_dir / "DEFINITION.md").write_text("\n".join(lines))

    return ref_path


@dataclass
class EvalResult:
    correct: bool
    num_workloads: int
    num_passed: int
    mean_speedup: float | None
    median_speedup: float | None
    min_speedup: float | None
    max_abs_err: float | None
    max_rel_err: float | None
    per_workload: list[dict] = field(default_factory=list)
    error: str | None = None


# Substrings in EvalResult.error that indicate the host environment is broken
# (driver mismatch, no GPU, etc.) — not anything the agent can fix by editing
# the kernel. When detected, the phase aborts immediately to stop wasting
# Claude API calls on unfixable failures.
ENV_FATAL_MARKERS = (
    "NVIDIA driver on your system is too old",
    "CUDA driver version is insufficient",
    "no CUDA-capable device is detected",
    "Worker startup failed",
    "RunnerFatalError",
    "Failed to start worker",
    "FIB_DATASET_PATH",
)


def is_env_fatal(err: str | None) -> bool:
    if not err:
        return False
    return any(m in err for m in ENV_FATAL_MARKERS)


DEFAULT_BENCH_CONFIG_KWARGS = dict(
    warmup_runs=3,
    iterations=50,
    num_trials=3,
    rtol=1e-2,
    atol=1e-2,
    timeout_seconds=600,
)

REF_CACHE_DIR = REPO_ROOT / "runs" / "_ref_cache"


def _ref_cache_path(definition_name: str) -> Path:
    REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return REF_CACHE_DIR / f"{definition_name}.json"


def _load_ref_cache(definition_name: str) -> dict[str, float]:
    """Return {workload_uuid: ref_latency_ms} from disk."""
    p = _ref_cache_path(definition_name)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    return {k: v["ref_latency_ms"] for k, v in raw.items()
            if isinstance(v, dict) and v.get("ref_latency_ms")}


def _update_ref_cache(definition_name: str, per_workload: list[dict]) -> None:
    """Merge any newly-observed reference latencies into the on-disk cache."""
    p = _ref_cache_path(definition_name)
    existing = {}
    if p.is_file():
        try:
            existing = json.loads(p.read_text())
        except json.JSONDecodeError:
            existing = {}
    updated = False
    for w in per_workload:
        uuid = w.get("workload_uuid") or w.get("workload")
        ref_ms = w.get("ref_latency_ms")
        if uuid and ref_ms and ref_ms > 0:
            existing[uuid] = {
                "ref_latency_ms": ref_ms,
                "updated": time.time(),
            }
            updated = True
    if updated:
        p.write_text(json.dumps(existing, indent=2, sort_keys=True))


def _all_uuids_cached(definition_name: str, uuids: list[str]) -> bool:
    cache = _load_ref_cache(definition_name)
    return bool(uuids) and all(u in cache for u in uuids)


def _fill_ref_from_cache(definition_name: str, per_workload: list[dict]) -> None:
    """Inject cached ref_latency_ms / speedup into entries that lack them."""
    cache = _load_ref_cache(definition_name)
    if not cache:
        return
    for w in per_workload:
        uuid = w.get("workload_uuid") or w.get("workload")
        if not uuid:
            continue
        if not w.get("ref_latency_ms") and uuid in cache:
            w["ref_latency_ms"] = cache[uuid]
        sol_ms = w.get("latency_ms")
        ref_ms = w.get("ref_latency_ms")
        if sol_ms and ref_ms and not w.get("speedup"):
            w["speedup"] = ref_ms / sol_ms


def _aggregate(per: list[dict], error: str | None = None) -> EvalResult:
    """Build an EvalResult from a per-workload list (local or remote)."""
    speedups = [p["speedup"] for p in per if p.get("speedup")]
    abs_errs = [p["max_abs_err"] for p in per if p.get("max_abs_err") is not None]
    rel_errs = [p["max_rel_err"] for p in per if p.get("max_rel_err") is not None]
    passed = sum(1 for p in per if p.get("status") == "PASSED")

    def _med(xs):
        xs = sorted(xs)
        if not xs:
            return None
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    return EvalResult(
        correct=(passed == len(per) and len(per) > 0),
        num_workloads=len(per),
        num_passed=passed,
        mean_speedup=(sum(speedups) / len(speedups)) if speedups else None,
        median_speedup=_med(speedups),
        min_speedup=min(speedups) if speedups else None,
        max_abs_err=max(abs_errs) if abs_errs else None,
        max_rel_err=max(rel_errs) if rel_errs else None,
        per_workload=per,
        error=error,
    )


def _benchmark_local(definition_name, solution, log_dir, config_kwargs) -> tuple[list[dict], str | None]:
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet

    ts = TraceSet.from_path(dataset_path())
    definition = ts.definitions[definition_name]
    workloads = ts.workloads.get(definition_name, [])
    if not workloads:
        return [], f"no workloads for {definition_name}"

    bench_ts = TraceSet(
        root=ts.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )
    config = BenchmarkConfig(
        log_dir=str(log_dir / "bench_logs"),
        **config_kwargs,
    )
    bench = Benchmark(bench_ts, config)
    try:
        result_ts = bench.run_all(dump_traces=True)
        per = []
        for tr in result_ts.traces.get(definition.name, []):
            ev = tr.evaluation
            entry = {"workload": tr.workload.uuid[:8],
                     "workload_uuid": tr.workload.uuid,
                     "status": ev.status.value if ev else None}
            if ev and ev.performance:
                entry["latency_ms"] = ev.performance.latency_ms
                entry["ref_latency_ms"] = ev.performance.reference_latency_ms
                entry["speedup"] = ev.performance.speedup_factor
            if ev and ev.correctness:
                entry["max_abs_err"] = ev.correctness.max_absolute_error
                entry["max_rel_err"] = ev.correctness.max_relative_error
            entry["log"] = (ev.log[:400] if ev and ev.log else None)
            per.append(entry)
        return per, None
    finally:
        try:
            bench.close()
        except Exception as e:
            print(f"[eval] bench.close() failed: {e}")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _modal_function(name: str):
    """Look up a deployed Modal function by name.

    Requires `modal deploy modal_bench.py` to have been run once. We use
    lookup-by-name (not direct import) because direct .remote() on an
    imported function only works inside a `modal run` context — the
    pipeline is a plain Python process making many calls over hours.
    """
    import modal
    from modal_bench import APP_NAME
    try:
        return modal.Function.from_name(APP_NAME, name)
    except Exception as e:
        raise RuntimeError(
            f"could not look up Modal function '{APP_NAME}.{name}': {e}\n"
            "Did you deploy the app? Run:  modal deploy modal_bench.py"
        ) from e


def _benchmark_modal(
    definition_name,
    solution,
    config_kwargs,
    shards: int = 1,
) -> tuple[list[dict], str | None]:
    """Ship the packed solution to Modal and run the benchmark there.

    With shards > 1, the workload list is split into roughly-equal chunks,
    each chunk runs in its own Modal container in parallel via starmap,
    and per-workload results are merged client-side.
    """
    from modal_bench import list_workload_uuids

    sol_json = solution.model_dump_json()

    if shards <= 1:
        fn = _modal_function("remote_evaluate")
        out = fn.remote(
            solution_json=sol_json,
            definition_name=definition_name,
            config_kwargs=config_kwargs,
        )
        return out.get("per_workload", []), out.get("error")

    uuids = list_workload_uuids(definition_name)
    if not uuids:
        return [], f"no workloads for {definition_name}"

    n = max(1, min(shards, len(uuids)))
    chunks: list[list[str]] = [[] for _ in range(n)]
    for i, u in enumerate(uuids):
        chunks[i % n].append(u)
    chunks = [c for c in chunks if c]

    print(f"[eval] modal fan-out: {len(uuids)} workloads -> {len(chunks)} shards")
    args = [
        (sol_json, definition_name, config_kwargs, chunk)
        for chunk in chunks
    ]

    fn = _modal_function("remote_evaluate_shard")
    merged_per: list[dict] = []
    errors: list[str] = []
    for shard_out in fn.starmap(args):
        merged_per.extend(shard_out.get("per_workload", []))
        if shard_out.get("error"):
            errors.append(shard_out["error"])

    err = " | ".join(errors) if errors else None
    return merged_per, err


def evaluate_solution(
    definition_name: str,
    solution_dir: Path,
    spec_kwargs: dict,
    name: str,
    author: str,
    log_dir: Path,
    use_modal: bool = False,
    modal_shards: int = 1,
    use_ref_cache: bool = True,
) -> EvalResult:
    """Pack a solution from `solution_dir` and evaluate on all workloads.

    The per-workload list shape is identical for local and modal paths so
    the rest of the driver doesn't need to know which one ran.

    If `use_ref_cache` is on (default), the reference-baseline timing loop
    is skipped whenever the cache covers every workload — saving the dominant
    wall-time cost when the reference is a slow Python loop (e.g. gdn_prefill).
    Cached values come from prior runs of the same definition; misses fall
    back to a normal full benchmark which then refreshes the cache.
    """
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    try:
        spec = BuildSpec(**spec_kwargs)
        solution = pack_solution_from_files(
            path=str(solution_dir),
            spec=spec,
            name=name,
            definition=definition_name,
            author=author,
        )
        (log_dir / "solution.json").write_text(solution.model_dump_json(indent=2))

        config_kwargs = dict(DEFAULT_BENCH_CONFIG_KWARGS)

        skip_ref = False
        if use_ref_cache:
            try:
                from modal_bench import list_workload_uuids
                uuids = list_workload_uuids(definition_name)
            except Exception:
                uuids = []
            if uuids and _all_uuids_cached(definition_name, uuids):
                skip_ref = True
                config_kwargs["profile_baseline"] = False
                print(f"[eval] reference cache HIT ({len(uuids)} workloads) -> "
                      "skipping baseline profiling")
            else:
                cache = _load_ref_cache(definition_name)
                hit = sum(1 for u in uuids if u in cache)
                print(f"[eval] reference cache: {hit}/{len(uuids)} hits -> "
                      "running full baseline this round to backfill")

        if use_modal:
            print(f"[eval] dispatching to Modal B200 "
                  f"({len(solution.sources)} source files, shards={modal_shards})")
            per, err = _benchmark_modal(
                definition_name, solution, config_kwargs, shards=modal_shards,
            )
        else:
            per, err = _benchmark_local(definition_name, solution, log_dir, config_kwargs)

        if use_ref_cache:
            if not skip_ref:
                _update_ref_cache(definition_name, per)
            _fill_ref_from_cache(definition_name, per)
        return _aggregate(per, error=err)
    except Exception as e:
        import traceback
        return EvalResult(
            correct=False, num_workloads=0, num_passed=0,
            mean_speedup=None, median_speedup=None, min_speedup=None,
            max_abs_err=None, max_rel_err=None,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[:3000]}",
        )


# --- prompts -----------------------------------------------------------------
def _skill_block(skills: list[str]) -> str:
    lines = ["Available skills — invoke via the Skill tool when useful:"]
    for s in skills:
        lines.append(f"  - {s}")
    lines.append("")
    lines.append(
        "Profiling tools available via MCP: tools prefixed with "
        f"`{NCU_TOOL_PREFIX}__` wrap Nsight Compute. Typical flow: "
        "list_sections_and_sets -> profile(binary=..., set='basic') -> "
        "read_report_details. Use the `optimize-kernel` skill for the "
        "full workflow."
    )
    return "\n".join(lines)


def _contract_block(lang: str, entry_point: str, dps: bool, binding: str) -> str:
    if lang == "triton":
        body = textwrap.dedent(f"""
            Language: Triton (Python). Source files go in
            `solution/triton/`. Destination-passing style: {dps}.
            Entry point: `{entry_point}` — i.e. file `{entry_point.split('::')[0]}`
            must expose top-level function `{entry_point.split('::')[1]}` with
            the exact parameter names and order from the definition.
            Do NOT use variadic args. Use `torch.empty_like(...)` only if you
            also set destination_passing_style=false (we did not).
            Import only: torch, triton, triton.language, math, typing.
        """).strip()
    else:
        dps_explain = (
            "DPS = inputs THEN outputs as parameters. The harness pre-allocates "
            "the output tensors and passes them in; your kernel writes into "
            "them and returns void. Function arity = (#inputs + #outputs)."
            if dps else
            "Value-returning style: the kernel allocates and returns output(s)."
        )
        body = textwrap.dedent(f"""
            Language: C++/CUDA. Source files go in `solution/cuda/`.
            Destination-passing style: {dps}. Binding: {binding}.
            Entry point: `{entry_point}` — the .cu file must export a
            function `{entry_point.split('::')[1]}` with the definition's
            parameter names/order via {binding} bindings.

            {dps_explain}

            For tvm-ffi: parameters are `tvm::ffi::Tensor`; register the
            symbol via `TVM_FFI_DLL_EXPORT_TYPED_FUNC(...)` so the harness
            can `getattr(mod, "{entry_point.split('::')[1]}")` it.
            For torch binding: parameters are `torch::Tensor` and you bind
            with `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){{ m.def("...", ...) }}`.

            Read DEFINITION.md and definition.json carefully to get the
            exact parameter names, order, dtypes and axis sizes. Mismatched
            arity will be rejected by the builder's signature validator.

            Prefer tensor-core MMA (wmma / wgmma / tcgen05), TMA or cp.async
            for shared-memory staging, mbarrier producer/consumer, and launch
            bounds for occupancy. Target sm_90 (Hopper) and sm_100 (Blackwell).
        """).strip()
    return body


def initial_prompt(
    lang: str,
    phase_dir: Path,
    entry_point: str,
    dps: bool,
    binding: str,
    skills: list[str],
    definition_name: str,
    prior_best_hint: str | None,
) -> str:
    hint = ""
    if prior_best_hint:
        hint = "\n\nPrior-phase best kernel you may want to study:\n" + prior_best_hint

    return textwrap.dedent(f"""
        You are a GPU kernel engineer competing in the FlashInfer-Bench
        MLSys26 contest. Target definition: `{definition_name}`.

        The op's IO schema, axes and pytorch reference are already staged
        in this directory:
          - ./DEFINITION.md      (human summary)
          - ./definition.json    (full schema)
          - ./reference.py       (pytorch reference implementation)

        {_contract_block(lang, entry_point, dps, binding)}

        Write the implementation files under `./solution/{lang}/`. The
        driver will pack them via flashinfer-bench's pack_solution_from_files
        and evaluate against ALL workloads of `{definition_name}` — both
        correctness (rtol=atol=1e-2) and speedup vs pytorch reference.
        Correctness AND speed are graded.

        Start by reading reference.py and DEFINITION.md. Then think about
        tile sizes, memory access pattern, and kernel structure before
        writing. Do NOT run the benchmark yourself — the driver does that
        and feeds results back next round.

        {_skill_block(skills)}{hint}
    """).strip()


def refine_prompt(
    lang: str,
    round_idx: int,
    total_rounds: int,
    last: EvalResult,
    best: EvalResult | None,
    skills: list[str],
    definition_name: str,
) -> str:
    status = "CORRECT" if last.correct else "INCORRECT"
    summary = (
        f"workloads: {last.num_passed}/{last.num_workloads} passed. "
        f"mean speedup: "
        f"{(f'{last.mean_speedup:.3f}x' if last.mean_speedup else 'n/a')}, "
        f"median: "
        f"{(f'{last.median_speedup:.3f}x' if last.median_speedup else 'n/a')}, "
        f"min: "
        f"{(f'{last.min_speedup:.3f}x' if last.min_speedup else 'n/a')}. "
        f"max_abs_err={last.max_abs_err}, max_rel_err={last.max_rel_err}."
    )
    per_lines = []
    for w in last.per_workload[:8]:
        per_lines.append(
            f"  - {w.get('workload')}: status={w.get('status')} "
            f"speedup={w.get('speedup')} "
            f"abs_err={w.get('max_abs_err')} log={(w.get('log') or '')[:180]}"
        )
    per_block = "\n".join(per_lines)

    err_block = ""
    if last.error:
        err_block = f"\nHarness-level error:\n```\n{last.error[:1500]}\n```"

    best_block = ""
    if best and best.correct:
        best_block = (
            f"\nBest correct version so far: mean_speedup="
            f"{best.mean_speedup:.3f}x, min_speedup={best.min_speedup:.3f}x. "
            "Make targeted edits rather than a full rewrite unless you have a "
            "concrete reason to expect a big gain."
        )

    focus = {
        "triton": (
            "Tuning levers to consider: BLOCK_M/N/K, num_warps, num_stages, "
            "autotune configs per workload shape, accumulator dtype for "
            "tl.dot, masking on tails, grouping M for L2 reuse, "
            "avoiding redundant loads, vectorized pointer arithmetic, "
            "persistent kernels when launch overhead dominates."
        ),
        "cuda": (
            "Tuning levers to consider: warp-level MMA (wgmma / tcgen05 on "
            "Hopper/Blackwell), TMA descriptors for bulk load, cp.async for "
            "overlap, smem swizzling to avoid bank conflicts, mbarrier "
            "producer/consumer pipelines, register-tiling + launch bounds, "
            "persistent blocks with a scheduler. Profile with the "
            f"{NCU_TOOL_PREFIX}__ tools: compile your binding first, then "
            "profile the kernel to see SoL, roofline and stall reasons."
        ),
    }[lang]

    header = (f"Refinement round {round_idx}/{total_rounds} on "
              f"`{definition_name}`. Status: {status}.")

    if not last.correct:
        body = textwrap.dedent(f"""
            Your previous {lang} solution is incorrect or failed to build on
            some workloads.

            Summary: {summary}
            Per-workload:
            {per_block}{err_block}

            Diagnose the root cause, edit `./solution/{lang}/` in place, and
            keep the entry-point contract. Favor correctness this round;
            speed is secondary when not all workloads pass.
        """).strip()
    else:
        body = textwrap.dedent(f"""
            Your previous {lang} solution is correct on all {last.num_workloads}
            workloads.

            Summary: {summary}
            Per-workload:
            {per_block}

            Goal: increase the minimum speedup (not just mean) while keeping
            every workload passing. {focus}{best_block}

            Edit `./solution/{lang}/` in place. Do NOT run the benchmark
            yourself.
        """).strip()

    return f"{header}\n\n{body}\n\n{_skill_block(skills)}"


# --- phase driver ------------------------------------------------------------
def run_phase(
    *,
    lang: str,
    phase_dir: Path,
    num_iters: int,
    skills: list[str],
    definition,
    entry_point: str,
    dps: bool,
    binding: str,
    solution_name: str,
    author: str,
    prior_best_hint: str | None,
    reference_override: Path | None,
    use_modal: bool = False,
    modal_shards: int = 1,
    use_ref_cache: bool = True,
    resume: bool = False,
) -> tuple[Path | None, EvalResult | None, list[dict]]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    dump_definition_context(definition, phase_dir, reference_override)

    solution_src_dir = phase_dir / "solution" / lang
    solution_src_dir.mkdir(parents=True, exist_ok=True)

    spec_kwargs = {
        "language": lang,
        "target_hardware": ["cuda"],
        "entry_point": entry_point,
        "destination_passing_style": dps,
    }
    if lang == "cuda":
        spec_kwargs["binding"] = binding

    history: list[dict] = []
    session_id: str | None = None
    best_snapshot_dir: Path | None = None
    best_result: EvalResult | None = None

    def _record_and_update_best(rnd: int, result: EvalResult, archive: Path,
                                session_id: str | None):
        """Append history entry + promote best_src if this round is the new winner."""
        nonlocal best_snapshot_dir, best_result
        history.append({
            "round": rnd,
            "eval": asdict(result),
            "session_id": session_id,
            "archive": str(archive.relative_to(phase_dir)),
        })
        better = (
            result.correct
            and (best_result is None
                 or not best_result.correct
                 or (result.mean_speedup or 0) > (best_result.mean_speedup or 0))
        )
        if better:
            best_result = result
            best_snapshot_dir = phase_dir / "best_src"
            if best_snapshot_dir.exists():
                shutil.rmtree(best_snapshot_dir)
            shutil.copytree(archive, best_snapshot_dir)

    for rnd in range(num_iters):
        archive = phase_dir / f"round_{rnd:02d}_src"

        # Resume path: if this round's source archive already exists,
        # re-evaluate it instead of re-running the agent. Lets us pick up
        # where a crashed/rebooted phase left off without losing the
        # already-generated kernels.
        if resume and archive.exists() and any(archive.iterdir()):
            print(f"\n=== [{lang}] round {rnd}/{num_iters - 1} (RESUMED from archive) ===",
                  flush=True)
            if solution_src_dir.exists():
                shutil.rmtree(solution_src_dir)
            shutil.copytree(archive, solution_src_dir)

            log_dir = phase_dir / f"logs_round_{rnd:02d}"
            log_dir.mkdir(exist_ok=True)

            result = evaluate_solution(
                definition_name=definition.name,
                solution_dir=solution_src_dir,
                spec_kwargs=spec_kwargs,
                name=f"{solution_name}-{lang}-r{rnd}",
                author=author,
                log_dir=log_dir,
                use_modal=use_modal,
                modal_shards=modal_shards,
                use_ref_cache=use_ref_cache,
            )
            print(f"[{lang}] resumed eval: correct={result.correct} "
                  f"passed={result.num_passed}/{result.num_workloads} "
                  f"mean_speedup={result.mean_speedup}")
            if result.error:
                print(f"[{lang}] error: {result.error[:400]}")

            _record_and_update_best(rnd, result, archive, session_id)

            if is_env_fatal(result.error):
                print(f"[{lang}] ENVIRONMENTAL failure detected — aborting phase.",
                      flush=True)
                (phase_dir / "history.json").write_text(json.dumps(history, indent=2))
                return best_snapshot_dir, best_result, history
            continue

        if rnd == 0:
            prompt = initial_prompt(
                lang, phase_dir, entry_point, dps, binding,
                skills, definition.name, prior_best_hint,
            )
        else:
            prompt = refine_prompt(
                lang, rnd, num_iters - 1,
                EvalResult(**history[-1]["eval"]),
                best_result, skills, definition.name,
            )

        print(f"\n=== [{lang}] round {rnd}/{num_iters - 1} ===", flush=True)
        envelope = run_claude(prompt, cwd=phase_dir, session_id=session_id)
        session_id = envelope.get("session_id", session_id)

        if not any(solution_src_dir.iterdir()):
            print(f"[{lang}] no source files produced; skipping eval")
            history.append({
                "round": rnd,
                "eval": asdict(EvalResult(
                    False, 0, 0, None, None, None, None, None,
                    error="no source files produced",
                )),
                "session_id": session_id,
            })
            continue

        log_dir = phase_dir / f"logs_round_{rnd:02d}"
        log_dir.mkdir(exist_ok=True)

        result = evaluate_solution(
            definition_name=definition.name,
            solution_dir=solution_src_dir,
            spec_kwargs=spec_kwargs,
            name=f"{solution_name}-{lang}-r{rnd}",
            author=author,
            log_dir=log_dir,
            use_modal=use_modal,
            modal_shards=modal_shards,
            use_ref_cache=use_ref_cache,
        )
        print(f"[{lang}] eval: correct={result.correct} "
              f"passed={result.num_passed}/{result.num_workloads} "
              f"mean_speedup={result.mean_speedup}")
        if result.error:
            print(f"[{lang}] error: {result.error[:400]}")

        # archive source
        if archive.exists():
            shutil.rmtree(archive)
        shutil.copytree(solution_src_dir, archive)

        _record_and_update_best(rnd, result, archive, session_id)

        if is_env_fatal(result.error):
            print(f"[{lang}] ENVIRONMENTAL failure detected — aborting phase. "
                  "No further claude rounds will be invoked. Fix the host "
                  "environment (driver / CUDA / torch / dataset path) and "
                  "re-run.", flush=True)
            (phase_dir / "history.json").write_text(json.dumps(history, indent=2))
            return best_snapshot_dir, best_result, history

    (phase_dir / "history.json").write_text(json.dumps(history, indent=2))
    return best_snapshot_dir, best_result, history


def dir_to_prompt_hint(d: Path | None) -> str | None:
    if not d or not d.is_dir():
        return None
    parts = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}:
            try:
                txt = p.read_text()
            except Exception:
                continue
            if len(txt) > 20_000:
                txt = txt[:20_000] + "\n... (truncated)"
            parts.append(f"--- {p.name} ---\n{txt}")
    if not parts:
        return None
    return "\n\n".join(parts)


# --- entrypoint --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--definition", required=True,
                    help="Name of the kernel definition in the flashinfer-trace dataset")
    ap.add_argument("--workdir", required=True, help="Directory for this run")
    ap.add_argument("--solution-name", default=None,
                    help="Solution name prefix (default: <definition>-auto)")
    ap.add_argument("--author", default="auto-agent")
    ap.add_argument("--triton-iters", type=int, default=4)
    ap.add_argument("--cuda-iters", type=int, default=4)
    ap.add_argument("--triton-entry", default="kernel.py::kernel",
                    help="Triton entry point, format file::fn")
    ap.add_argument("--cuda-entry", default="kernel.cu::run_kernel",
                    help="CUDA entry point, format file::fn")
    ap.add_argument("--binding", default="tvm-ffi", choices=["tvm-ffi", "torch"])
    ap.add_argument("--no-dps", action="store_true",
                    help="Disable destination-passing style (kernel returns outputs)")
    ap.add_argument("--skip-triton", action="store_true")
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument("--no-seed-cuda-from-triton", action="store_true",
                    help="Don't show the best triton kernel as a hint to the CUDA phase")
    ap.add_argument("--reference-file", default=None,
                    help="Override the pytorch reference shown to the agent. "
                         "Defaults to references/<track>.py matched by definition prefix, "
                         "falling back to the reference embedded in the TraceSet.")
    ap.add_argument("--gpus", default=None,
                    help="Restrict visible GPUs for this run. Sets "
                         "CUDA_VISIBLE_DEVICES for the driver and every child "
                         "process (claude subprocess, flashinfer-bench runner, "
                         "ncu-mcp). Example: --gpus 0,1")
    ap.add_argument("--use-modal", action="store_true",
                    help="Run the framework benchmark step on Modal B200 "
                         "instead of locally. The agent (claude) still runs "
                         "locally. Requires one-time setup: `modal setup`, "
                         "`modal volume create flashinfer-trace`, "
                         "`modal volume put flashinfer-trace ./mlsys26-contest /`. "
                         "Pre-warm with `modal run modal_bench.py::warm`.")
    ap.add_argument("--modal-shards", type=int, default=1,
                    help="When --use-modal is on, split workloads into N "
                         "shards and fan out to N Modal containers in parallel. "
                         "Each shard re-runs the reference baseline once, so "
                         "speedup is sub-linear; values of 2-8 are typical. "
                         "Default 1 = single container.")
    ap.add_argument("--no-ref-cache", action="store_true",
                    help="Disable the on-disk reference-latency cache. By "
                         "default the pipeline records ref_latency_ms per "
                         "(definition, workload) at runs/_ref_cache/<def>.json "
                         "and skips the slow reference timing loop on rounds "
                         "where every workload is already cached. Use this "
                         "flag to force a fresh baseline (e.g. after changing "
                         "GPU model).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from a partially-completed run. For each "
                         "round whose `round_NN_src/` archive already exists, "
                         "skip the agent and re-evaluate the archive instead. "
                         "Lets a crashed/rebooted phase pick up without losing "
                         "the kernels already generated.")
    args = ap.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"[main] CUDA_VISIBLE_DEVICES={args.gpus}")

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if not SKILLS_DIR.is_dir():
        print(f"[warn] skills dir missing at {SKILLS_DIR}", file=sys.stderr)
    if not MCP_CONFIG.is_file():
        print(f"[warn] .mcp.json missing at {MCP_CONFIG}", file=sys.stderr)

    _ts, definition = load_definition(args.definition)
    print(f"[main] loaded definition {definition.name} "
          f"({len(_ts.workloads.get(definition.name, []))} workloads)")

    reference_override = Path(args.reference_file).resolve() if args.reference_file else None
    if reference_override and not reference_override.is_file():
        sys.exit(f"--reference-file not found: {reference_override}")
    if reference_override is None:
        auto_ref = local_reference_for(definition.name)
        if auto_ref:
            print(f"[main] will use local reference {auto_ref.relative_to(REPO_ROOT)}")
        else:
            print("[main] no local reference matched; will try Definition.reference from TraceSet")

    solution_name = args.solution_name or f"{definition.name}-auto"
    summary = {
        "definition": definition.name,
        "workdir": str(workdir),
        "solution_name": solution_name,
        "author": args.author,
        "phases": {},
    }

    triton_best_dir: Path | None = None

    if not args.skip_triton:
        tri_best, tri_res, tri_hist = run_phase(
            lang="triton",
            phase_dir=workdir / "triton",
            num_iters=args.triton_iters,
            skills=TRITON_SKILLS,
            definition=definition,
            entry_point=args.triton_entry,
            dps=not args.no_dps,
            binding=args.binding,
            solution_name=solution_name,
            author=args.author,
            prior_best_hint=None,
            reference_override=reference_override,
            use_modal=args.use_modal,
            modal_shards=args.modal_shards,
            use_ref_cache=not args.no_ref_cache,
            resume=args.resume,
        )
        summary["phases"]["triton"] = {
            "best_dir": str(tri_best) if tri_best else None,
            "result": asdict(tri_res) if tri_res else None,
            "rounds": len(tri_hist),
        }
        triton_best_dir = tri_best

    if not args.skip_cuda:
        hint = (None if args.no_seed_cuda_from_triton
                else dir_to_prompt_hint(triton_best_dir))
        cuda_best, cuda_res, cuda_hist = run_phase(
            lang="cuda",
            phase_dir=workdir / "cuda",
            num_iters=args.cuda_iters,
            skills=CUDA_SKILLS,
            definition=definition,
            entry_point=args.cuda_entry,
            dps=not args.no_dps,
            binding=args.binding,
            solution_name=solution_name,
            author=args.author,
            prior_best_hint=hint,
            reference_override=reference_override,
            use_modal=args.use_modal,
            modal_shards=args.modal_shards,
            use_ref_cache=not args.no_ref_cache,
            resume=args.resume,
        )
        summary["phases"]["cuda"] = {
            "best_dir": str(cuda_best) if cuda_best else None,
            "result": asdict(cuda_res) if cuda_res else None,
            "rounds": len(cuda_hist),
        }

    (workdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
