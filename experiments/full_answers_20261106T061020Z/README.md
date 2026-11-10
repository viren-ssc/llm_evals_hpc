# full_answers_20261106T061020Z

Initially checkpoint-window output; GC/Apertus later continued with 8,192 cap. Llama: `llama_756_COMPLETE.xlsx` / `.jsonl`.

- [Filled benchmarks and integrity reports](outputs/exports/)
- [Raw per-question records](outputs/raw/)
- [Run manifests and scheduler logs](outputs/runs/)
- [Exact configurations](configs/)

Captured nonempty raw answers: GC LLM **118**, Apertus **73**, Llama **756** (756 questions/model). These counts include token-capped responses. Review per-answer flags; historical integrity files can reflect earlier snapshots or stricter natural-completion policies.

This folder retains its historical source version. From this directory, CPU inspection uses:

```bash
export PYTHONPATH="$PWD/src"
python -m llm_evals validate
python -m llm_evals status
```

Do not submit jobs or overwrite exports just to inspect results. For a new run use the [main README](../../README.md). Historical absolute paths inside JSON records are original provenance and may reference the previous repository name.
