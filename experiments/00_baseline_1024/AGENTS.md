# Project instructions

Stage 1 only. Never call a judge or use gpt-oss to generate answers.
The root CORE workbook is immutable. Its recorded SHA-256 must match every result.
Only the unmodified `question` string may enter a single-user checkpoint-compatible chat template.
Do not include context, reference answers, metadata, other questions, or conversation history.
Never substitute checkpoints, invent answers, hide truncation flags, or run GPU inference on login nodes.
Use `PYTHONPATH=src python -m llm_evals` from this directory.
Use Slurm for GPU work; inspect existing submission records before submitting again.
Keep tests synthetic and in temporary directories. Never mix test fixtures into real outputs.
Final export requires 756 successes per model. Incomplete exports must say PARTIAL.
Changes to shared caches or shared Python environments are outside this project's scope.

User override: do not run smoke inference or require a smoke gate. Run all 756 questions directly, keeping each model loaded once. Preserve active model processes when changing orchestration.

User output-retention override: keep all nonempty generated text, including token-limit outputs, and move to the next question without truncation retries. Retain truncated=true and accepted_truncation=true. Here success means a saved generation output, not proof that the response ended naturally. Include accepted text in the answer export.

Latest user authorization: configure maximum safe batching for all three models, including the necessary one-time engine restarts. Preserve Slurm GPU allocations and successful records. Actual concurrency is dynamically bounded by measured KV-cache tokens and full per-request output reservations; configured sequence ceiling is 64. Never concatenate questions.
