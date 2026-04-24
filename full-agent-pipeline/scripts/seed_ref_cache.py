"""Populate runs/_ref_cache/<def>.json with cheap reference-latency
measurements so subsequent benchmark rounds can skip the expensive
in-framework baseline timing loop.

Per workload we run the reference `iterations` times after `warmup`
warmups, capture mean ms, and write to the cache. Then `evaluate_solution`
sees a full cache hit, sets `profile_baseline=False` on `BenchmarkConfig`,
and only times the solution + does correctness check (a few ref calls).

Usage:
    uv run python scripts/seed_ref_cache.py \\
        --definition gdn_prefill_qk4_v8_d128_k_last \\
        --warmup 1 --iters 5 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from run_pipeline import _ref_cache_path, _load_ref_cache, dataset_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--definition", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap number of workloads (0 = all)")
    ap.add_argument("--force", action="store_true",
                    help="Re-time workloads even if already in cache")
    args = ap.parse_args()

    from flashinfer_bench import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.compile import BuilderRegistry

    ts = TraceSet.from_path(dataset_path())
    if args.definition not in ts.definitions:
        sys.exit(f"definition not in dataset: {args.definition}")
    definition = ts.definitions[args.definition]
    workloads = ts.workloads.get(args.definition, [])
    print(f"[seed] {args.definition}: {len(workloads)} workloads")
    if args.limit:
        workloads = workloads[: args.limit]

    cache_path = _ref_cache_path(args.definition)
    existing = {}
    if cache_path.is_file():
        try:
            existing = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            existing = {}

    ref_runnable = BuilderRegistry.get_instance().build_reference(definition)

    written = 0
    skipped = 0
    failed = 0
    for i, wl_trace in enumerate(workloads):
        wl = wl_trace.workload
        uuid = wl.uuid
        if not args.force and uuid in existing and existing[uuid].get("ref_latency_ms"):
            skipped += 1
            continue

        try:
            safe = (load_safetensors(definition, wl, ts.root)
                    if any(d.type == "safetensors" for d in wl.inputs.values())
                    else {})
            inp = gen_inputs(definition, wl, device=args.device, safe_tensors=safe)

            with torch.no_grad():
                for _ in range(args.warmup):
                    ref_runnable(*inp)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(args.device)

                t0 = time.perf_counter()
                for _ in range(args.iters):
                    ref_runnable(*inp)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(args.device)
                ms = (time.perf_counter() - t0) / args.iters * 1e3

            existing[uuid] = {
                "ref_latency_ms": ms,
                "warmup": args.warmup,
                "iters": args.iters,
                "device": args.device,
                "updated": time.time(),
            }
            cache_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
            written += 1
            print(f"  [{i+1}/{len(workloads)}] {uuid[:8]}: {ms:.3f} ms  axes={wl.axes}",
                  flush=True)
        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{len(workloads)}] {uuid[:8]}: FAILED {type(e).__name__}: {e}",
                  flush=True)

    print(f"\n[seed] done. written={written} skipped={skipped} failed={failed} "
          f"cached_total={len(existing)}")
    print(f"[seed] cache file: {cache_path}")


if __name__ == "__main__":
    main()
