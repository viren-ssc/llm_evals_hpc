# Experiments and saved results

Four reruns are preserved below, alongside the earlier baseline. Counts mean nonempty captured answers, including capped text; they are not judge scores. Historical exports remain unchanged, including intermediate PARTIAL files.

| Folder | GC LLM | Apertus | Llama | Policy / recommended deliverable |
|---|---:|---:|---:|---|
| [00_baseline_1024](00_baseline_1024/) | 756 | 756 | 756 | Original 1,024-token run; retained capped outputs. Use `canadianness_756_three_model_answers.xlsx` / `.jsonl`; 756 answers per model. Earlier PARTIAL exports are historical snapshots. |
| [capped_8192_20261106T132848Z](capped_8192_20261106T132848Z/) | 130 | 0 | 0 | GC/Apertus planned; only GC records present. 8,192 cap then 1,024 continuation. Review files are partial diagnostics. |
| [consistent_1024_20261106T144122Z](consistent_1024_20261106T144122Z/) | 756 | 756 | 0 | Fresh temperature-zero 1,024 cap. Use `gc_llm_756_evaluation_1024_tokens.*` and `apertus_756_evaluation_1024_tokens.*`. |
| [full_answers_20261106T061020Z](full_answers_20261106T061020Z/) | 118 | 73 | 756 | Initially checkpoint-window output; GC/Apertus later continued with 8,192 cap. Llama: `llama_756_COMPLETE.xlsx` / `.jsonl`. |
| [sampling_t07_p095_1024_20261107T013517Z](sampling_t07_p095_1024_20261107T013517Z/) | 755 | 756 | 756 | T=0.7, top_p=0.95, 1,024 cap. Use combined `MIXED_TEMPERATURE_COMPLETE.*`; GC has 755 sampling answers + 1 authorized T=0 replacement. |

## Choose files for judging

- **Consistent temperature zero:** use the two standalone model datasets from `consistent_1024...`; each includes all 756 answers, including flagged token-capped outputs. Llama was not part of that experiment.
- **Sampling:** Apertus and Llama each have a standalone 756-row dataset. Use the GC `MIXED_TEMPERATURE_COMPLETE` dataset to include its explicitly authorized replacement; its strict sampling-only dataset remains 755-row PARTIAL.
- **Combined coverage:** the sampling combined `MIXED_TEMPERATURE_COMPLETE` workbook/JSONL has 756 questions × 3 models. Preserve its matching raw JSONL and `mixed_temperature_integrity.json`.
- **Long-answer experiment:** Llama has a complete standalone export. GC/Apertus and the capped rerun are incomplete; keep those labels.

No LLM judge scores are stored here. Final score files belong to Part 2 on icelab. No incomplete experiment was promoted to a completed result during cleanup.

## Inspect or reproduce

Each folder keeps configs, its historical source implementation and run-specific exporters. Source variants are required for continuation and output-retention semantics. Run the `scripts/export_*.py` tools only with that folder’s `src` on `PYTHONPATH`, and only when intentionally regenerating exports. Back up historical exports first.

For new inference, follow the root README in a fresh output directory. Do not blindly resubmit archived jobs or rediscover models in archived configs.
