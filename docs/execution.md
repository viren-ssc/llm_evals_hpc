# Execution status

Recorded 2026-11-06T04:25:13.958479+00:00. Stage 1 is in progress. All three checkpoints are found and structurally verified, all 756 questions validated (404 English, 352 French), and all 14 tests pass. The authoritative input checksum remains unchanged. No judge has been run.

## Scheduler jobs

CLUSTER: gpsc7
JOBID STATE TIME NODELIST(REASON)
11816065 PENDING 0:00 (Resources)
11816066 PENDING 0:00 (Priority)
11816015 RUNNING 10:16 ib14gpu-006


Each worker requests four A100-SXM4-40GB GPUs, 16 CPUs and 192 GiB RAM on gpsc7, with vLLM 0.17.0, BF16 and four-way tensor parallelism. Each runs an English/French smoke test before the full benchmark and resumes saved successful answers. At this snapshot Llama is loading; GC and Apertus are queued. Saved successes: GC 0/756, Apertus 0/756, Llama 0/756.

## Recovered model locations

GC: `/gpfs/fs7/ssc/ssc_dsai/abm002/.cache/huggingface/hub/models--ssc-dsai--gc-llm-apertus-70b-instruct-2509/snapshots/58277e94d16ccf486c1702c0aac8115c69448585`

Apertus: `/gpfs/fs7/ssc/ssc_dsai/abm002/.cache/huggingface/hub/models--swiss-ai--Apertus-70B-Instruct-2509/snapshots/0f2767662a013c0fb631079cea883155252bc3b6`

Each has 30 readable shards, 70,599,864,640 parameters and 141,199,854,344 shard bytes. The official [GC model card](https://huggingface.co/ssc-dsai/gc-llm-apertus-70b-instruct-2509/blob/main/README.md) identifies Apertus Instruct 2509 as the parent. GC is distributed as full weights following LoRA training, with no adapter-only load required.

The alternate cache was located by following `scripts/launch_canbench_responses.slurm` in the accessible abm002 data-curation project. Initial group-root caches are indeed incomplete, but they are not the only available copies. Both complete tokenizers store native templates in standalone chat_template.jinja files; discovery now handles this format. No shared model or environment was modified.

Only the original question enters each request, wrapped in checkpoint-compatible role tokens. Native system/developer/date/cutoff boilerplate is omitted explicitly; both native and effective templates are preserved in configuration/manifests.

## History and monitoring

Initial Llama job 11815905 failed before generation due to excessive IPC path length; the worker now uses short node-local temporary paths. Job 11816009 was cancelled before generation to enforce strict question-only formatting. Active replacement is 11816015.

Run `PYTHONPATH=src python -m llm_evals status` and `squeue -M gpsc7 -j 11816015,11816065,11816066` for live state. This document is a recorded snapshot. Complete runs are not yet claimed. The PARTIAL workbook currently has empty answer columns. Final export remains gated on all 2,268 successful responses.

## Latest batching configuration — 2026-11-06T05:33:09.006934+00:00

User authorized engine restarts to enable adaptive continuous batching. Configuration: sequence ceiling 64, scheduled-token budget 8192, admission within 95% of measured KV tokens with whole-response reservations. Actual batch concurrency depends on question lengths and the newly profiled cache; no fixed 64-request admission is claimed.

Current jobs: GC LLM 11817121, Apertus 11817122, Llama 11816015. The initial GC/Apertus batch startups failed on callable RPC serialization before producing new answers; the query now reads the coordinator’s own cache profile without disabling serialization guards. Llama received that repair while loading. All three corrected engines are loading. Saved counts: GC 69, Apertus 42, Llama 161. Existing answers and generation fingerprints are preserved. All 24 tests passed, including the complete batched-wrapper path, independent prompts, cache parsing, out-of-order completions and immediate saves.

For live state, run `PYTHONPATH=src python -m llm_evals status` and `squeue -M gpsc7 -u "$USER" -n canadianness`.
