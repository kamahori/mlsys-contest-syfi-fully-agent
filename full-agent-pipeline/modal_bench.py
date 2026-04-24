"""Modal-hosted benchmark runner for the autonomous pipeline.

The pipeline normally runs `Benchmark.run_all()` on the local GPU. With
`--use-modal` it ships the packed solution to a Modal B200 instead, runs
the same benchmark there, and returns the per-workload dicts the driver
expects. The agent (claude) still runs locally; only the framework
benchmark step is offloaded.

One-time setup (per machine):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace ./mlsys26-contest /
    modal deploy modal_bench.py        # makes remote_evaluate(_shard) callable from any process

The driver looks the deployed functions up by name (modal.Function.from_name),
so deploy is what matters — `modal run` alone (used by the warm helper
below) is ephemeral and would not persist the function for subsequent
.remote() calls.
"""

from __future__ import annotations

import modal

APP_NAME = "full-agent-pipeline-bench"
TRACE_VOLUME_NAME = "flashinfer-trace"
TRACE_MOUNT = "/data"

app = modal.App(APP_NAME)
trace_volume = modal.Volume.from_name(TRACE_VOLUME_NAME, create_if_missing=True)

image = (
    # Use the NVIDIA CUDA devel image so tvm-ffi's CUDA path finds nvcc
    # and /usr/local/cuda with headers/libs. debian_slim has no toolkit,
    # so TVMFFIBuilder raised "Could not find CUDA installation" for every
    # .cu workload. Version matches the torch cu128 wheel.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install(
        # Match the local pipeline's pinned versions / index.
        "torch>=2.7,<2.11",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "flashinfer-bench",
        "triton",
        "numpy",
        "tomli",
    )
)


def _run_bench_on_volume(
    solution_json: str,
    definition_name: str,
    config_kwargs: dict,
    workload_uuids: list[str] | None,
) -> dict:
    """Shared body for both single-shot and sharded eval paths."""
    import traceback

    from flashinfer_bench import (
        Benchmark,
        BenchmarkConfig,
        Solution,
        TraceSet,
    )

    bench = None
    try:
        solution = Solution.model_validate_json(solution_json)
        config = BenchmarkConfig(**config_kwargs)

        ts = TraceSet.from_path(TRACE_MOUNT)
        if definition_name not in ts.definitions:
            return {
                "error": (f"definition '{definition_name}' not in volume; "
                          "did you `modal volume put` the dataset?"),
                "per_workload": [],
            }
        definition = ts.definitions[definition_name]
        workloads = ts.workloads.get(definition_name, [])
        if workload_uuids is not None:
            wanted = set(workload_uuids)
            workloads = [w for w in workloads if w.workload.uuid in wanted]
        if not workloads:
            return {
                "error": f"no workloads for {definition_name} (uuids={workload_uuids})",
                "per_workload": [],
            }

        bench_ts = TraceSet(
            root=ts.root,
            definitions={definition.name: definition},
            solutions={definition.name: [solution]},
            workloads={definition.name: workloads},
            traces={definition.name: []},
        )
        bench = Benchmark(bench_ts, config)
        result_ts = bench.run_all(dump_traces=True)

        per = []
        for tr in result_ts.traces.get(definition.name, []):
            ev = tr.evaluation
            entry = {
                "workload": tr.workload.uuid[:8],
                "workload_uuid": tr.workload.uuid,
                "status": ev.status.value if ev else None,
            }
            if ev and ev.performance:
                entry["latency_ms"] = ev.performance.latency_ms
                entry["ref_latency_ms"] = ev.performance.reference_latency_ms
                entry["speedup"] = ev.performance.speedup_factor
            if ev and ev.correctness:
                entry["max_abs_err"] = ev.correctness.max_absolute_error
                entry["max_rel_err"] = ev.correctness.max_relative_error
            entry["log"] = ev.log[:400] if ev and ev.log else None
            per.append(entry)
        return {"per_workload": per, "error": None}
    except Exception as e:
        return {
            "per_workload": [],
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:3000]}",
        }
    finally:
        if bench is not None:
            try:
                bench.close()
            except Exception:
                pass


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_MOUNT: trace_volume},
)
def remote_evaluate(
    solution_json: str,
    definition_name: str,
    config_kwargs: dict,
) -> dict:
    """Run all workloads of `definition_name` on one Modal B200."""
    return _run_bench_on_volume(solution_json, definition_name, config_kwargs, None)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_MOUNT: trace_volume},
)
def remote_evaluate_shard(
    solution_json: str,
    definition_name: str,
    config_kwargs: dict,
    workload_uuids: list[str],
) -> dict:
    """Run a SUBSET of workloads (selected by UUID) on one Modal B200.

    Used by the driver when fan-out is enabled: split the workload list
    into N shards, dispatch each shard to its own container via .map(),
    merge the per-workload results client-side. Each shard re-builds the
    solution and runs the reference baseline once, so amortize that
    overhead by keeping the shard count modest (typically 2–8).
    """
    return _run_bench_on_volume(
        solution_json, definition_name, config_kwargs, workload_uuids
    )


def list_workload_uuids(definition_name: str) -> list[str]:
    """Local helper — read the bundled dataset to enumerate workload UUIDs.

    The driver uses this to decide how to shard before dispatching.
    """
    from pathlib import Path
    import os

    from flashinfer_bench import TraceSet

    # Match run_pipeline.py's resolution order: env var, then ./mlsys26-contest.
    p = os.environ.get("FIB_DATASET_PATH")
    if not p:
        candidate = Path(__file__).resolve().parent / "mlsys26-contest"
        if candidate.is_dir():
            p = str(candidate)
        else:
            raise RuntimeError(
                "FIB_DATASET_PATH not set and no bundled dataset at "
                f"{candidate}. Set FIB_DATASET_PATH or add the dataset."
            )
    ts = TraceSet.from_path(p)
    return [w.workload.uuid for w in ts.workloads.get(definition_name, [])]


@app.local_entrypoint()
def warm():
    """Build/cache the image and warm one B200 container (ephemeral).

    Run after `modal deploy modal_bench.py` if you want to amortize the
    image-build cost before the first real eval. This is `modal run`-style
    (ephemeral), so it doesn't replace `modal deploy` — both are useful.
    """
    print("warming modal image + container...")
    out = remote_evaluate.remote(
        solution_json="{}",  # invalid; we just want the container up
        definition_name="__warmup__",
        config_kwargs={},
    )
    print("warm result:", out)
