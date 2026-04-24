"""One-shot local benchmark for an agent-produced source dir.

Real script (not a heredoc) so flashinfer-bench's spawn-based persistent
runner can re-import __main__ in worker subprocesses.

Usage:
    uv run python scripts/manual_eval.py \\
        --definition gdn_prefill_qk4_v8_d128_k_last \\
        --lang triton --entry gdn_prefill.py::run_prefill \\
        --src runs/gdn_prefill/triton/round_03_src \\
        --lang cuda --entry gdn_prefill.cu::run_prefill \\
        --src runs/gdn_prefill/cuda/round_05_src
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from run_pipeline import evaluate_solution  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--definition", required=True)
    ap.add_argument("--lang", action="append", required=True,
                    choices=["triton", "cuda"])
    ap.add_argument("--entry", action="append", required=True)
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--binding", default="tvm-ffi")
    ap.add_argument("--no-dps", action="store_true")
    ap.add_argument("--log-root", default="/tmp/manual_eval")
    ap.add_argument("--no-ref-cache", action="store_true",
                    help="Don't read/write runs/_ref_cache/<def>.json")
    args = ap.parse_args()

    if not (len(args.lang) == len(args.entry) == len(args.src)):
        sys.exit("--lang / --entry / --src must be repeated the same number of times")

    for lang, entry, src in zip(args.lang, args.entry, args.src):
        src_dir = Path(src).resolve()
        if not src_dir.is_dir():
            sys.exit(f"src not found: {src_dir}")

        print(f"\n=== {lang} :: {src_dir} ===", flush=True)
        log_dir = Path(args.log_root) / f"{args.definition}_{lang}"
        log_dir.mkdir(parents=True, exist_ok=True)

        spec_kwargs = {
            "language": lang,
            "target_hardware": ["cuda"],
            "entry_point": entry,
            "destination_passing_style": not args.no_dps,
        }
        if lang == "cuda":
            spec_kwargs["binding"] = args.binding

        res = evaluate_solution(
            definition_name=args.definition,
            solution_dir=src_dir,
            spec_kwargs=spec_kwargs,
            name=f"manual-{args.definition}-{lang}",
            author="manual-check",
            log_dir=log_dir,
            use_modal=False,
            use_ref_cache=not args.no_ref_cache,
        )

        print(f"correct = {res.correct}")
        print(f"passed  = {res.num_passed}/{res.num_workloads}")
        print(f"speedup mean   = {res.mean_speedup}")
        print(f"speedup median = {res.median_speedup}")
        print(f"speedup min    = {res.min_speedup}")
        print(f"max_abs_err    = {res.max_abs_err}")
        print(f"max_rel_err    = {res.max_rel_err}")
        if res.error:
            print(f"ERROR: {res.error[:600]}")
        statuses: dict[str, int] = {}
        for w in res.per_workload:
            s = w.get("status") or "NONE"
            statuses[s] = statuses.get(s, 0) + 1
        print(f"status counts  = {statuses}")


if __name__ == "__main__":
    main()
