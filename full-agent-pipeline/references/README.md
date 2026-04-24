# Baseline (reference) implementations

One file per track. Each is the **exact Python code** embedded in the
`reference` field of the corresponding `mlsys26-contest/definitions/*.json`
— extracted verbatim so you can diff against our optimized implementation
without opening the JSON.

The flashinfer-bench harness (`BuilderRegistry.build_reference(definition)`
in `default.py::eval_performance`) builds a runnable from this code and
times it under `warmup_runs=3, iterations=100, num_trials=5`. Reported
speedup = `reference_latency_ms / solution_latency_ms`.

Per the contest FAQ: *"The reference is intentionally kept simple to define
correctness."* It is **not** an optimized baseline (no flashinfer, no
Triton, no grouped GEMM — just per-expert Python loops with
`.to(fp32).matmul(…)`).

## Mapping

| Reference                               | Optimized                             |
|-----------------------------------------|---------------------------------------|
| `references/moe.py`                     | `solution/python/moe.py`              |
| `references/gdn_decode.py`              | `solution/python/gdn_decode.py`       |
| `references/gdn_prefill.py`             | `solution/python/gdn_prefill.py`      |
| `references/dsa_sparse_attention.py`    | `solution/python/dsa_sparse_attention.py` |
| `references/dsa_topk_indexer.py`        | `solution/python/dsa_topk_indexer.py` |

## Diff

```bash
diff -u references/moe.py solution/python/moe.py | less
# or for a side-by-side:
diff -y --width=220 references/moe.py solution/python/moe.py | less
```

## Signature caveat

The reference functions **return** their output; our solutions use
destination-passing style (`run(..., output)` writes into the caller-
allocated output tensor, returns `None`). This is why the harness's
`is_dps` flag exists — it adapts between the two conventions. The
computation is identical otherwise.

## Regenerate

If the dataset definitions change upstream, re-extract with:
```bash
source .venv/bin/activate
python -c "
import json
from pathlib import Path
tracks = {
    'moe.py': 'mlsys26-contest/definitions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json',
    'gdn_decode.py': 'mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json',
    'gdn_prefill.py': 'mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json',
    'dsa_sparse_attention.py': 'mlsys26-contest/definitions/dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json',
    'dsa_topk_indexer.py': 'mlsys26-contest/definitions/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json',
}
for name, path in tracks.items():
    Path(f'references/{name}').write_text(json.load(open(path)).get('reference', ''))
"
```
