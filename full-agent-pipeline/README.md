# full-agent-pipeline

Fully autonomous Triton + C++/CUDA kernel pipeline for the **FlashInfer-Bench
MLSys26 Contest**. Drives `claude` CLI in headless mode through two phases per
track, each gated by real `flashinfer-bench` evaluation:

```
phase 1: PyTorch reference  ->  Triton  (N refinement rounds)
phase 2: Triton best         ->  C++/CUDA (N refinement rounds, seeded with phase-1 source)
```

The agent has access to:
- `.claude/skills/*` — GPU optimization skills (cutlass-triton, nsight-profiler, …)
- `.mcp.json` — `ncu-mcp` server wrapping Nsight Compute for in-loop profiling
- `references/<track>.py` — verbatim PyTorch reference for each contest track,
  copied from the starter kit. The driver auto-selects one by definition-name
  prefix (`gdn_decode*` → `references/gdn_decode.py`, etc.) and writes it into
  the per-phase workdir as `reference.py` for the agent to read. Override with
  `--reference-file path/to/your_ref.py`.
- `mlsys26-contest/` — the contest dataset (definitions, workloads, traces,
  solutions). **Not tracked by git** (see `.gitignore`); you download it
  yourself — see "Dataset" below. The pipeline auto-detects it as the
  default `FIB_DATASET_PATH`, so once it's on disk no env var is needed.
  Override by exporting `FIB_DATASET_PATH` to point elsewhere.

## One-time setup

```bash
# from this directory
uv sync                       # installs flashinfer-bench, torch, triton, modal
(cd mcps/ncu-mcp && uv sync)  # installs the ncu MCP server
```

Requires: NVIDIA GPU + `ncu` CLI on PATH; `claude` CLI authenticated; `git-lfs`
installed (for the dataset download below).

## Dataset

The contest dataset (`mlsys26-contest/`) is ~1.8 GB and is **not checked in**
— `.gitignore` excludes it. Download it once into the repo root:

```bash
# from this directory (run_pipeline.py's folder)
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
# optional: drop the .git directory to save ~1.8 GB of LFS store
rm -rf mlsys26-contest/.git
```

That's it — no `export FIB_DATASET_PATH` needed; the pipeline auto-detects
`./mlsys26-contest`. If you already have the dataset somewhere else, either
symlink it here (`ln -s /path/to/flashinfer-trace ./mlsys26-contest`) or
export `FIB_DATASET_PATH=/path/to/flashinfer-trace`.

Verify the dataset loaded:

```bash
uv run python -c "
from flashinfer_bench import TraceSet
ts = TraceSet.from_path('./mlsys26-contest')
for name in sorted(ts.definitions.keys()):
    print(f'{name}: {len(ts.workloads.get(name, []))} workloads')
"
# Expected: 5 definitions, 324 workloads total.
```

## Run per track

All five contest tracks share the same driver. Pick the track-specific
`--definition` and entry-point names below. Tune `--triton-iters` /
`--cuda-iters` to your time budget (each round = 1 claude session +
1 full benchmark sweep across all workloads).

### 1. `gdn_decode`
```bash
uv run python run_pipeline.py \
    --definition gdn_decode_qk4_v8_d128_k_last \
    --workdir runs/gdn_decode \
    --triton-entry gdn_decode.py::run_decode \
    --cuda-entry   gdn_decode.cu::run_decode \
    --binding tvm-ffi \
    --triton-iters 4 --cuda-iters 6
```

### 2. `gdn_prefill`
```bash
uv run python run_pipeline.py \
    --definition gdn_prefill_qk4_v8_d128_k_last \
    --workdir runs/gdn_prefill \
    --triton-entry gdn_prefill.py::run_prefill \
    --cuda-entry   gdn_prefill.cu::run_prefill \
    --binding tvm-ffi \
    --triton-iters 4 --cuda-iters 6
```

### 3. `dsa_topk_indexer`
```bash
uv run python run_pipeline.py \
    --definition dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 \
    --workdir runs/dsa_topk_indexer \
    --triton-entry dsa_topk_indexer.py::run \
    --cuda-entry   dsa_topk_indexer.cu::run \
    --binding tvm-ffi \
    --triton-iters 4 --cuda-iters 6
```

### 4. `dsa_sparse_attention`
```bash
uv run python run_pipeline.py \
    --definition dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 \
    --workdir runs/dsa_sparse_attention \
    --triton-entry dsa_sparse_attention.py::run \
    --cuda-entry   dsa_sparse_attention.cu::run \
    --binding tvm-ffi \
    --triton-iters 4 --cuda-iters 6
```

### 5. `moe`
The starter kit's `moe` baseline is `language = "python"` (free-form Python
that may call Triton / CUTLASS / cuBLAS internally). Run the Triton phase
first and skip CUDA, or run both for a hand-written kernel attempt:
```bash
# triton-only (matches the starter kit shape)
uv run python run_pipeline.py \
    --definition moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
    --workdir runs/moe \
    --triton-entry moe.py::run \
    --triton-iters 6 \
    --skip-cuda

# or push for a CUDA implementation as well
uv run python run_pipeline.py \
    --definition moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
    --workdir runs/moe_cuda \
    --triton-entry moe.py::run \
    --cuda-entry   moe.cu::run \
    --binding tvm-ffi \
    --triton-iters 4 --cuda-iters 6
```

## Run all five sequentially

```bash
declare -A TRACKS=(
    [gdn_decode]="gdn_decode_qk4_v8_d128_k_last:gdn_decode.py::run_decode:gdn_decode.cu::run_decode"
    [gdn_prefill]="gdn_prefill_qk4_v8_d128_k_last:gdn_prefill.py::run_prefill:gdn_prefill.cu::run_prefill"
    [dsa_topk_indexer]="dsa_topk_indexer_fp8_h64_d128_topk2048_ps64:dsa_topk_indexer.py::run:dsa_topk_indexer.cu::run"
    [dsa_sparse_attention]="dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64:dsa_sparse_attention.py::run:dsa_sparse_attention.cu::run"
    [moe]="moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:moe.py::run:moe.cu::run"
)
for track in "${!TRACKS[@]}"; do
    IFS=":" read -r DEF TRI_ENTRY CU_ENTRY <<< "${TRACKS[$track]}"
    uv run python run_pipeline.py \
        --definition "$DEF" \
        --workdir "runs/$track" \
        --triton-entry "$TRI_ENTRY" \
        --cuda-entry "$CU_ENTRY" \
        --triton-iters 4 --cuda-iters 6 \
        2>&1 | tee "runs/${track}.log"
done
```

## Outputs per track

```
runs/<track>/
├── triton/
│   ├── DEFINITION.md            # IO schema for the agent
│   ├── definition.json
│   ├── reference.py             # extracted pytorch reference
│   ├── solution/triton/         # current source (live, last round)
│   ├── round_NN_src/            # archive per round
│   ├── best_src/                # best correct source by mean speedup
│   ├── logs_round_NN/           # bench logs + packed solution.json
│   └── history.json             # per-round eval results
├── cuda/                        # same shape
└── summary.json                 # top-level result summary
```

## Modal acceleration (optional)

The framework benchmark step can be offloaded to Modal B200s while the
agent (claude) keeps running locally. Useful when local GPUs are
contended or slower than B200.

**One-time setup**:

```bash
# 1. authenticate with Modal
modal setup

# 2. create the trace volume + upload the dataset
modal volume create flashinfer-trace
modal volume put flashinfer-trace ./mlsys26-contest /

# 3. deploy the benchmark functions so the driver can look them up
#    (this also builds + caches the image)
modal deploy modal_bench.py

# 4. (optional) pre-warm a container so the first round skips cold start
modal run modal_bench.py::warm
```

If you skip step 3 you'll get
`ExecutionError: Function has not been hydrated ...` on the first
`--use-modal` round — the driver looks the functions up by name and that
requires a deployed app. Re-run `modal deploy modal_bench.py` whenever you
edit the dependency pins or remote function code.

**Use it**:

```bash
# single B200 per round, all workloads sequential on it
uv run python run_pipeline.py --use-modal \
    --definition gdn_decode_qk4_v8_d128_k_last \
    --workdir runs/gdn_decode \
    --triton-entry gdn_decode.py::run_decode \
    --cuda-entry   gdn_decode.cu::run_decode

# fan out across N B200 containers in parallel (one per shard)
uv run python run_pipeline.py --use-modal --modal-shards 4 \
    --definition gdn_prefill_qk4_v8_d128_k_last \
    --workdir runs/gdn_prefill \
    --triton-entry gdn_prefill.py::run_prefill \
    --cuda-entry   gdn_prefill.cu::run_prefill
```

Each round:
1. Agent (`claude`) runs locally — uses your local GPU for any quick tests it does.
2. The packed `solution.json` is shipped to one or more Modal B200s.
3. `Benchmark.run_all()` runs there; per-workload statuses + speedups come back.
4. Driver feeds them to the next refinement round, identical to the local path.

**Why shards?** flashinfer-bench's `Benchmark` parallelizes across
*solutions*, and we always have one. So a single container with
`gpu="B200:N"` doesn't help — the extra GPUs sit idle. The right way to
scale is fan-out: split the workload list into N chunks, run each chunk
on its own B200 container in parallel via `.starmap()`, merge per-workload
results client-side.

Trade-off: each shard re-runs the reference baseline once (one trial per
workload). For a definition with W workloads + reference cost R per
workload, single-shard wall time is roughly W·(R+S) where S = solution
benchmark cost. With N shards it becomes ~(W/N)·(R+S) + container start.
Sub-linear because R is paid per shard, but for W=100, N=4 you typically
see 3-3.5× wall-clock speedup. Diminishing returns beyond N=8.

Per-round Modal cost: container start (~5–10s, cached after first call,
× N for shards) + benchmark wall time. No Modal GPU is held between rounds.

## Common flags

| flag | default | meaning |
|---|---|---|
| `--triton-iters` / `--cuda-iters` | `4` | refinement rounds per phase |
| `--binding` | `tvm-ffi` | C++ binding for CUDA solutions; alternative `torch` |
| `--no-dps` | off | disable destination-passing style (use if your kernel returns outputs) |
| `--skip-triton` / `--skip-cuda` | off | run only one phase |
| `--no-seed-cuda-from-triton` | off | don't pre-load the best Triton kernel into the CUDA initial prompt |
| `--reference-file` | auto | override the pytorch reference shown to the agent (default: `references/<track>.py` matched by definition prefix, then `Definition.reference` from TraceSet) |
| `--gpus` | all visible | set `CUDA_VISIBLE_DEVICES` for the whole pipeline (agent subprocess, flashinfer-bench runner, ncu-mcp). Example: `--gpus 0,1` |
| `--use-modal` | off | run the framework benchmark on Modal B200 (see Modal section). Agent stays local. |
| `--modal-shards` | `1` | with `--use-modal`, split workloads into N parallel Modal containers. 1 = no fan-out. |
| `--solution-name` | `<def>-auto` | name written into the packed solution.json |
| `--author` | `auto-agent` | author field in solution.json |

## Resuming / packing for submission

Each round's source tree is archived under `runs/<track>/<lang>/round_NN_src/`,
and the best correct one is at `runs/<track>/<lang>/best_src/`. To build a
submission `solution.json` from any of those, point the starter-kit
`pack_solution.py` at it (or call `flashinfer_bench.agents.pack_solution_from_files`
directly with the appropriate `BuildSpec`).
