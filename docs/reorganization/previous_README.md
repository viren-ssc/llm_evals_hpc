# Canadian-ness benchmark generation: Stage 1

Question-only inference for GC LLM, Apertus and Llama. No DeepEval, judge service or external inference endpoint is called.

The authoritative input is `Canadian-ness Benchmark - 756 CORE with Context.xlsx`, sheet `CORE Questions`. Validation observed 756 unique questions: 404 English and 352 French. SHA-256: `ef8a729f58c0dbcc63c3d3b31d572d15291aaac18daea1795afb51050fe38cb2`. No input cells are changed.

From this directory, with the system Python (openpyxl, PyYAML and pytest already installed):

```bash
export PYTHONPATH="$PWD/src"
python -m llm_evals discover
python -m llm_evals validate
python -m pytest -q
python scripts/submit.py
python -m llm_evals status
python -m llm_evals export --partial
# Only after all three models have 756 successes:
python -m llm_evals export
```

`discover` records real candidates in `outputs/runs/discovery.json` and writes `configs/models.yaml`. It checks metadata hashes, safetensors headers, indexed tensors, payload bounds and readability without reading 141 GB of payload or modifying shared files. An incomplete download is never runnable. Rediscovery changes experiment identity if metadata changes; existing results are never silently reused across identities.

Llama's complete flat directory has 70,553,706,496 parameters and 141,107,497,872 bytes of shards. Its exact upstream commit is not established; the local config, tokenizer, index and shard header hashes pin the observed copy. The other Llama HF snapshot has incomplete links. Expanded research found complete GC LLM and Apertus snapshots in `/gpfs/fs7/ssc/ssc_dsai/abm002/.cache/huggingface/hub`. The official GC model card confirms Apertus-70B-Instruct-2509 as its parent. Initial group-root caches contain configs only and are rejected. Both Apertus-family tokenizers use standalone `chat_template.jinja` files; discovery now recognizes and hashes them. See `docs/model_discovery.md`.

`submit.py` submits only ready checkpoints and records acknowledgements before any later submission. It requests four A100-SXM4-40GB GPUs, 16 CPUs, 192 GiB RAM and 24 hours on gpsc7, account `ssc_dsai__gpu_a100`, qos `low`. It does not touch unrelated jobs. The worker runs all 756 questions directly with resume, loading a model once. Separate smoke inference and the smoke prerequisite are disabled by the user’s instruction in `configs/execution.yaml`. Empty output, exact prompt echo, truncation and inference errors are checked during normal generation.

```bash
squeue -M gpsc7 -j 11816015
# Cancel only this project job if desired:
scancel -M gpsc7 11816015
# Explicit resubmission after cancellation/failure, without losing answers:
PROJECT_ROOT="$PWD" MODEL_ID=llama sbatch \
  --output="$PWD/outputs/runs/%x-%j.out" \
  --error="$PWD/outputs/runs/%x-%j.err" scripts/slurm/worker.sh
```

Inspect `outputs/runs/submissions.json` and scheduler state before resubmitting. `submission_pending` is an ambiguous acknowledgement and requires reconciliation, not blind retry. The helper deliberately refuses duplicate submissions, including after completed/failed jobs; use the explicit command above after inspection. Requeued jobs and explicit reruns resume saved successes.

The discovered inference runtime is `/gpfs/fs7/ssc/ssc_dsai/.cache/envs/vib000/canai-chandra-ocr2/bin/python`: vLLM 0.17.0, Transformers 4.57.6, PyTorch 2.10.0. The existing runtime lacks openpyxl. Local `.runtime/openpyxl` and `.runtime/et_xmlfile` are copies of the installed pure Python system libraries, used through PYTHONPATH without modifying shared environments. On another installation, install this package and its inference extra in a dedicated environment instead.

Manual GPU operations, **inside a Slurm GPU allocation only**:

```bash
python -m llm_evals generate --model llama --resume
python -m llm_evals generate --all --resume
```

The `--all` command tries each model, reports blocked models and exits nonzero if any fail. Slurm submission is the preferred workflow for individual model allocations. Ready state and current files are checked again before loading. GPU monitoring is written every 30 seconds. SIGTERM requests an orderly stop after the current answer; a hard scheduler kill can lose only the in-flight unsaved answer.

Raw checkpoints are atomic per-question JSON files in `outputs/raw/<experiment-fingerprint>/<model-id>/`. A nonblocking per-model lock prevents duplicate workers. Run manifests preserve configs, source hashes, package versions, prompt template, job ID, timestamps, exact local model metadata and shard headers. Records retain answer text, input/output tokens, elapsed time, finish reason, retries and truncation. The baseline remains 1,024 output tokens. Historical truncation retries used 2,048, but the user has now disabled those retries; error-only retries may still occur. Empty outputs remain failures. Under the user’s retention override, token-limit outputs are kept as saved generation successes with `truncated=true` and `accepted_truncation=true`; they are not retried. Historical attempts remain in the record. Explicit resume retries unresolved failures; successful records are never regenerated.

`export` refuses a final workbook until all 2,268 successful records exist and match IDs, questions, configuration and workbook checksum. It produces:

- `outputs/exports/canadianness_756_three_model_answers.xlsx`: 756 rows, original fields plus three answer columns.
- Same-stem `.jsonl`: one wide record per question, suitable for laptop DeepEval ingestion later.
- Same-stem `_raw.jsonl`: normalized model/question records with generation metadata.
- Same-stem `_failures.jsonl`: unresolved failures and missing answers.
- `outputs/exports/integrity.json`: current verified counts and input hash.

Before completion, `export --partial` writes the same files with `_PARTIAL` in the stem and empty answer cells for missing/failed results. A partial workbook is not a finished deliverable. Excel strings are preserved literally, including leading `=`; overlong cells fail explicitly rather than silently truncate. Raw JSON preserves full text.

For local Stage 2, transfer the final workbook, wide/raw JSONL, integrity report, `configs/` and `outputs/runs/` manifests. Keep the model files on HPC. Context and reference answers are retained only in the dataset and export for later judging.

Execution history: initial job 11815905 failed before generation because its IPC socket path exceeded the Linux limit. The corrected worker uses `/tmp/caneval-$SLURM_JOB_ID` for short node-local IPC paths; replacement job 11816009 runs the same model and decoding configuration.

Llama formatting clarification: the upstream template automatically inserts a system block containing a fixed date and knowledge-cutoff text. To enforce question-only natural-language content, the pinned Llama template omits that block while retaining the checkpoint's BOS, user header, end-of-turn and assistant header tokens. It preserves the question string exactly, including whitespace. Both the upstream and effective templates are recorded in manifests; the effective template is also in `configs/models.yaml`. Replacement job 11816009 was stopped before any answers when this behavior was identified.

The active replacement submission is job 11816015. The exact rendered-template test confirms that only the question and Llama role tokens are present.

Expanded research corrected the initial blocked status: all three checkpoints are available and verified structurally. Current jobs are GC LLM **11816065**, Apertus **11816066**, and Llama **11816015**, on gpsc7. Each requests four A100 40 GB GPUs. GC and Apertus queue normally until resources/priority permit. Their question-only templates retain BOS/user/end-user/assistant markers and omit native system/developer boilerplate. The new discovery and rendered-prompt regressions pass; the complete suite has 14 passing tests. See `docs/execution.md` for the latest recorded snapshot and `outputs/runs/expanded-research.json` for checkpoint/source evidence.

Current execution override: GC job 11816065 and Apertus job 11816066 were promoted while their first model loads were still in progress. A one-time CPython interpreter-safe callback expanded the existing work lists from two to all 756 questions without restarting model processes or changing prompts/decoding. Both workers acknowledged the update in `outputs/runs/runtime-promotion-*.json`. Their legacy Slurm command labels may still say smoke-test, but their actual work lists cover the full benchmark. The old copied shell script’s subsequent resume invocation finds completed answers and does not construct another engine. New worker submissions invoke `generate` directly. Llama job 11816015 was already in its full-generation phase and was left untouched.

Output-retention policy: `configs/execution.yaml` enables `retain_generated_outputs`. Every token emitted is preserved without post-processing cuts. Token-limit responses are included in the workbook and raw JSONL, with flags preserved in raw records. The status `success` means a saved nonempty generation result and may include accepted token-limit outputs; the `truncated` counter also reports those flags and therefore overlaps with success. No truncation retry is performed. The policy was applied to all three existing model processes without a model reload; audit files are `outputs/runs/output-retention-<job>.json`.

Continuous batching is now enabled for every model in `configs/execution.yaml`: `max_num_seqs: 64`, `max_num_batched_tokens: 8192`, and `kv_cache_reserve_fraction: 0.95`. The ceiling is deliberately above what the available KV cache normally permits. After loading, the worker reads the current engine coordinator’s measured KV-cache token capacity, rounds each question-plus-output reservation to cache blocks, and admits as many independent requests as fit within 95% of capacity. Long questions reduce the active count automatically; no question is shortened. Completed requests free their reservations and new requests fill those slots. Every completion is atomically saved immediately, even when completions arrive out of input order. Logs print `BATCH_RUNTIME` and `BATCH_ADMISSION` with real capacity and active counts. Raw records preserve batch timing/concurrency metadata.

The obsolete `batch_size: 1` field in generation.yaml remains only to preserve the decoding/input fingerprints of already captured answers; it is not the active batching control. Execution topology is recorded separately in run manifests. Batching can change floating-point execution order, so the manifests distinguish pre-batch and batched runs; previously saved outputs are preserved as requested. No model, question, chat template, precision, seed or output limit changes.

A one-time controlled handoff drains each current question, releases the old engine, and execs the batched generation process inside the same Slurm allocation. This necessary weight reload is authorized by the latest batching request. No scheduler requeue and no smoke inference occur. Existing model paths/results remain in place. Handoff audit files: `outputs/runs/batching-restart-<job>.json`.

Batching startup correction: vLLM 0.17 rejects callable RPC serialization by default. No serialization guard was disabled. Capacity is now read from the engine’s own `GPU KV cache size` profile, and the configured block size is used to round reservations. Replacement jobs: GC LLM 11817121; Apertus 11817122; Llama remains 11816015. Prior answers remain in their original checkpoint namespace and are skipped.
