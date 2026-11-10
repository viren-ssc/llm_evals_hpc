# Full-answer rerun instructions

This is a fresh rerun of every question for all three models. No reuse of the earlier 1,024-token outputs.
Only the unmodified question field enters an independent request. Preserve checkpoint-compatible role-only templates. No context, references, metric labels, rubrics, history, judge calls or smoke inference.
Use max_tokens=None; never replace it with vLLM's default 16 or another fixed response cap. The only token boundary is the checkpoint context window. Preserve all emitted text and flag context-boundary outputs as incomplete, not accepted full answers. Do not retry a context-boundary result automatically.
Use batched generation and Slurm GPU allocation. CPU weight offload is 8 GiB per GPU; no quantization or checkpoint substitution. The scheduler manages KV growth and may preempt/recompute to fit memory.
Keep the first run in the parent project untouched. Output all text to raw JSON and JSONL; Excel has a 32,767-character cell limit and cannot be the sole authoritative full-text store.

User override 2026-11-06: continue GC/Apertus with 8192 generated tokens, preserve all capped text as incomplete. Runtime context 16384, no CPU offload, three concurrent full-KV-reserved requests. Existing natural completions remain reusable in original lineage; execution policy and per-answer decoding record the change. Llama remains untouched.
