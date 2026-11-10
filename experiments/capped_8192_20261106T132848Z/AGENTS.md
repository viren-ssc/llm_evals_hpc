Fresh full-dataset run of GC LLM and Apertus only. Generate all 756 answers per model, no reuse of previous answers. Question-only prompts. 8192 output tokens, temperature 0, batch concurrency 3 with full KV reservation, context 16384, CPU offload 0. No smoke inference or judging. Preserve capped text and flag incomplete, no automatic retry of capped answers. Leave Llama and earlier runs untouched.

User requested higher concurrency: execution ceiling 16, scheduler-managed KV growth, initial reservation 2048 tokens per request, actual output cap remains 8192. Scheduler may preempt/recompute as long answers grow.

Latest user condition: increase concurrency only if safe. Keep ceiling 16 but require full 8192-token KV reservations for every admitted request. Do not use scheduler-managed speculative reservations; actual concurrency is limited by measured cache, likely 3.

Latest user override: 1024 output tokens for both GC/Apertus immediately. Preserve all already captured answers, including capped answers, and continue unanswered questions. Use continuation_overrides to retain original checkpoint lineage; record effective decoding in each new answer. Batch ceiling 16, full cache reservation. No smoke inference.
