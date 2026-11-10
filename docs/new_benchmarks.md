# Add a benchmark

## Already answered

Use `scripts/prepare_judge_dataset.py` with your input, output, answer column and expected row count. Rename/map your source columns to `question_id`, `question`, `reference_answer` and a model answer column first; preserve context and provenance where available. The helper accepts arbitrary row counts and does not enforce Canadian-ness language/metric labels. Part 2 must define the new benchmark's rubrics and dataset adapter before judging.

## Needs inference

1. Keep the original input immutable in a separately named `data/benchmarks/<benchmark>/` folder. Record its SHA-256.
2. Start an isolated working copy with its own configs and empty `outputs/`. Do not edit archived experiments or reuse their answers.
3. Add an explicit dataset profile/loader in `src/llm_evals/dataset.py`. The current implementation hard-codes `CORE Questions`, required fields, seven metric labels, the 756-row count and 404/352 language split. For a new profile, replace those checks with the benchmark's actual invariants; do not disable validation globally.
4. Update `configs/dataset.yaml` for the input path and checksum. Currently the sheet/expected_rows entries document intent; the loader does not read them dynamically. Wire profile-specific values through `load`, CLI validation, generation, export and status. Update the export stem/messages, which currently contain `756`.
5. Retain unique stable IDs, exact questions, original order, context and references. Keep inference's interface question-only. Reference/context fields must never enter model prompts for this protocol.
6. Add synthetic tests for duplicate IDs, empty questions, wrong checksums, correct row counts and export alignment. Run the CPU test suite and validation.
7. Configure/check local model identities, prompt templates and decoding; then follow the README's Slurm submission, monitoring and export commands in the new working copy.
8. Validate the answered dataset and hand it to Part 2. Add the benchmark's actual rubric mapping there; M1–M7 labels are specific to the current dataset.

For a new model, the current CLI and discovery module also assume `gc_llm`, `apertus`, `llama`. Extend the CLI choices, discovery configuration and checkpoint-compatible template tests before submission. Do not rename a model key to impersonate an existing checkpoint.
