# syfi — Full-Agent Track

Submission repo for team **syfi** in the [MLSys 2026 FlashInfer Kernel Generation Contest](http://mlsys26.flashinfer.ai/), **full-agent** approach.

Per contest rules, the full-agent track requires that an agent fully reproduce each kernel end-to-end. This repo contains (1) the generated kernels and (2) the agent workflow / reproduction material (TODO).

Each subdirectory is a self-contained submission for one kernel definition, following the multi-kernel layout from the contest FAQ. There is no root `config.toml` — the evaluation pipeline scans immediate subdirs for configs.

## Layout

```
syfi-full-agent/
├── moe/
│   ├── config.toml           # definition = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"
│   └── solution/cuda/moe.cu
├── gdn_prefill/
│   ├── config.toml           # definition = "gdn_prefill_qk4_v8_d128_k_last"
│   └── solution/cuda/gdn_prefill.cu
├── gdn_decode/
│   ├── config.toml           # definition = "gdn_decode_qk4_v8_d128_k_last"
│   └── solution/cuda/gdn_decode.cu
├── dsa_indexer/
│   ├── config.toml           # definition = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
│   └── solution/cuda/dsa_topk_indexer.cu
└── dsa_sparse_attention/
    ├── config.toml           # definition = "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"
    └── solution/cuda/dsa_sparse_attention.cu
```

## Local testing

From any kernel subdir:

```bash
python scripts/pack_solution.py   # regenerates solution.json
python scripts/run_local.py       # requires FIB_DATASET_PATH
```

## Agent reproduction

TODO — describe the agent workflow (prompts, tools, orchestration) used to generate each kernel. Full-agent submissions placing in the top 3 require a technical writeup and reproduction guide.

## Submission

Tag a commit (e.g. `git tag submission-v1`) and push to the submission remote. The evaluation pipeline scans tags and groups by definition (one tag per definition wins; latest tag per definition is evaluated).
