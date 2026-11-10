# LLM Evals · HPC inference (Part 1)

Generate answers to the Canadian-ness benchmark with **GC LLM, Apertus, and Llama**, then transfer the answered benchmark to the **separate icelab LLM judge repository (Part 2)**. This repository runs inference on HPC; judging happens in icelab.

756 questions · 404 English / 352 French · 7 metric labels · 3 models

```mermaid
flowchart LR
    A[Original benchmark] --> B[Validate IDs, text and checksum]
    B --> C[Slurm GPU allocation]
    C --> D[One independent question per request]
    D --> E[Saved raw answers and provenance]
    E --> F[Filled XLSX and JSONL]
    F --> G[Transfer to Part 2 on icelab]
    G --> H[LLM judge and evaluation scores]
```

**Start here:** [existing results](experiments/README.md) · [new inference run](#run-from-start-to-export) · [handoff to icelab](#send-answers-to-part-2) · [technical details](docs/architecture.md)

## Find your files

```text
llm_evals_hpc/
├── data/benchmarks/       Original, unchanged CORE workbook
├── configs/              Default dataset, checkpoint and decoding settings
├── src/llm_evals/         Generation, checkpointing, validation and export code
├── scripts/              Slurm submission and handoff preparation
├── tests/                Synthetic CPU tests; no real model calls
├── experiments/          Four experiments + the earlier baseline
│   └── <experiment>/
│       ├── configs/      Exact run settings (including continuation overrides)
│       ├── src/          Historical implementation needed to interpret/resume run
│       ├── scripts/      Run-specific submit/export tools
│       └── outputs/
│           ├── raw/      Full text per question, flags and provenance
│           ├── exports/ Filled benchmarks, JSONL and integrity reports
│           └── runs/    Manifests, Slurm logs and GPU measurements
├── outputs/              Created by NEW runs; never mix with archived experiments
└── docs/                 Design, handoff and reorganization records
```

Historical source copies are intentional: the experiments used different output limits, continuation policies and export rules. Do not replace them with the default implementation. Absolute paths inside old result records document the original execution location; current navigation uses the folders above.

## Run from start to export

### 1 · Set up Python

Use Python 3.11+ and a **dedicated environment**, without changing shared environments. Run these commands from the repository root:

```bash
cd /home/vib000/canai_projects/llm_evals_hpc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export PYTHONPATH="$PWD/src"
python -m pytest -q
python -m llm_evals validate
```

For GPU inference, install the pinned inference packages into a dedicated environment compatible with the cluster's CUDA/driver setup:

```bash
python -m pip install -e '.[inference]'
export INFERENCE_PYTHON="$PWD/.venv/bin/python"
```

The worker also supports the previously used shared runtime through its default `INFERENCE_PYTHON`. `.runtime/` contains the local workbook-library compatibility copies required by that runtime; keep it when using that setup. Model weights remain in the shared HPC cache and are not shipped in this repo.

### 2 · Check inputs and model settings

| File                        | What to check                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------ |
| `configs/dataset.yaml`    | Workbook location, CORE worksheet, 756 rows, SHA-256                                 |
| `configs/models.yaml`     | Local checkpoint/tokenizer paths, identity hashes, role-only templates, GPU settings |
| `configs/generation.yaml` | Output limit, temperature, seed, context length                                      |
| `configs/execution.yaml`  | Dynamic batching, KV reservation and output retention                                |

The default is temperature 0 with a 1,024-token initial cap. Nonempty token-capped answers are retained and flagged. **Saved output does not mean a naturally finished response.** Inspect raw records before interpreting results.

For a new deployment, discover available checkpoints before submitting:

```bash
python -m llm_evals discover
python -m llm_evals validate
```

Discovery writes `configs/models.yaml` and `outputs/runs/discovery.json`. Review its readiness and paths. It validates checkpoint metadata and shard structure; it does not download weights. Do not rediscover or edit settings inside an archived experiment: model/decoding changes alter its fingerprint.

Only the original `question` enters inference, with checkpoint-compatible user/assistant role tokens. Context, references and metric labels stay in the dataset for Part 2. Questions are never concatenated.

### 3 · Submit on Slurm

Review `scripts/slurm/worker.sh` for your cluster. The recorded setup uses `gpsc7`, `gpu_a100`, account `ssc_dsai__gpu_a100`, qos `low`, four A100 GPUs, 16 CPUs, 192 GiB RAM and 24 hours **per model job**.

```bash
mkdir -p outputs/runs
# Inspect existing submissions and queue before submitting.
if [ -f outputs/runs/submissions.json ]; then cat outputs/runs/submissions.json; fi
squeue -M gpsc7 -u "$USER"
python scripts/submit.py
squeue -M gpsc7 -u "$USER"
python -m llm_evals status
```

`submit.py` submits each ready model once. Keep the exported `INFERENCE_PYTHON` from setup in the submitting shell. No smoke inference is required; each worker loads its model once and processes the benchmark with dynamic batching. Never run GPU inference on login nodes.

If a job fails, inspect `outputs/runs/` and reconcile its scheduler acknowledgement first. Only after confirming the old worker has stopped, resume a specific model:

```bash
PROJECT_ROOT="$PWD" MODEL_ID=gc_llm sbatch \
  --output="$PWD/outputs/runs/%x-%j.out" \
  --error="$PWD/outputs/runs/%x-%j.err" scripts/slurm/worker.sh
```

Replace `gc_llm` with `apertus` or `llama` as needed. Resume skips saved successes. A `submission_pending` record needs manual scheduler reconciliation before retrying.

### 4 · Export and verify

```bash
python -m llm_evals status
# While incomplete (also produced by the worker):
python -m llm_evals export --partial
# Final export: requires 756 saved successes for EACH configured model.
python -m llm_evals export
cat outputs/exports/integrity.json
```

Final outputs in `outputs/exports/`:

| File ending         | Purpose                                                       |
| ------------------- | ------------------------------------------------------------- |
| `.xlsx`           | Original benchmark fields + one answer column per model       |
| `.jsonl`          | Same wide dataset, one question per line                      |
| `_raw.jsonl`      | Model/question records, complete text and generation metadata |
| `_failures.jsonl` | Missing or failed records (empty for a complete export)       |
| `integrity.json`  | Counts, completion flag and input checksum                    |

The stem is `canadianness_756_three_model_answers`. `_PARTIAL` means incomplete export; retain the label. Excel cannot hold more than 32,767 characters in one cell; full raw JSON is the authority for long answers. The historical experiments have their own exporters and completion policies: use the [results guide](experiments/README.md).

## Send answers to Part 2

Pick the experiment and model you intend to evaluate. For example, validate the completed sampling Apertus dataset and create a portable JSONL package:

```bash
python scripts/prepare_judge_dataset.py \
  --input experiments/sampling_t07_p095_1024_20261107T013517Z/outputs/exports/apertus_756_evaluation_t07_p095_1024.jsonl \
  --answer-column apertus_answer --expected-rows 756 \
  --output outputs/handoff/apertus_sampling.jsonl
```

The helper rejects empty answers, duplicate IDs and missing questions/references. It preserves all columns, answer text and truncation metadata, and writes a checksum manifest. It calls no judge. For another **already answered** benchmark, pass its `.xlsx` or `.jsonl`, answer-column name, expected row count, and optional `--sheet`. See [the handoff guide](docs/icelab_handoff.md) for column mapping and transfer details.

Copy the package, original benchmark, raw generation metadata, run configs and integrity reports to the machine that can access icelab. Configure and launch judging using **Part 2's README**; its judge cannot be accessed from HPC. The Part 2 URL and CLI are not recorded here yet. Judge scores belong to that repo; the results here are generated answers and generation diagnostics.

## Use another benchmark for inference

The present loader deliberately enforces the Canadian-ness CORE schema and language/metric counts. Merely changing a filename will not make an arbitrary benchmark compatible. [Dataset extension steps](docs/new_benchmarks.md) explain the required changes and checks. Already answered benchmarks can use the handoff helper directly.

## Reproducibility

The original workbook lives in `data/benchmarks/` and must retain SHA-256:

```text
ef8a729f58c0dbcc63c3d3b31d572d15291aaac18daea1795afb51050fe38cb2
```

Keep raw records, model identity hashes, configs, manifests and truncation flags with every export. The sampling GC dataset includes one explicit cross-run replacement; use its **MIXED_TEMPERATURE_COMPLETE** files and report that fact. [Reorganization audit](docs/reorganization/report.json) records cleanup and verifies historical result bytes were preserved.

## Git: push this repo and clone on HPC

Secrets belong in ignored local files such as `.env` or `configs/credentials.json`. Never put real credentials in tracked configs, examples, logs or results. Ignore rules protect filenames; review staged content before every push. Shared examples must contain placeholders only. Model weights and Python environments stay outside Git; benchmark inputs, filled datasets and experiment results are included.

After creating an empty remote repository, replace `YOUR_REMOTE_URL` below:

```bash
git remote add origin YOUR_REMOTE_URL
git push -u origin main
```

On HPC, replace `YOUR_REMOTE_URL` and choose a suitable working directory:

```bash
git clone YOUR_REMOTE_URL llm_evals_hpc
cd llm_evals_hpc
# Follow Python/environment setup above; environments and credentials are local.
# For later updates, after saving any local changes:
git pull --ff-only
```

Before later commits:

```bash
git status --short
git add -- .
git diff --cached --stat
git diff --cached
# Commit and push only after reviewing the staged content.
```

If a credential is ever committed, rotate/revoke it immediately and remove it from Git history before publishing. Adding its filename to `.gitignore` does not remove an already tracked secret.
