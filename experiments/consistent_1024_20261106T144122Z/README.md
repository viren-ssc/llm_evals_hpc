# consistent_1024_20261106T144122Z

Fresh temperature-zero 1,024 cap. Use `gc_llm_756_evaluation_1024_tokens.*` and `apertus_756_evaluation_1024_tokens.*`.

- [Filled benchmarks and integrity reports](outputs/exports/)
- [Raw per-question records](outputs/raw/)
- [Run manifests and scheduler logs](outputs/runs/)
- [Exact configurations](configs/)

Captured nonempty raw answers: GC LLM **756**, Apertus **756**, Llama **0** (756 questions/model). These counts include token-capped responses. Review per-answer flags; historical integrity files can reflect earlier snapshots or stricter natural-completion policies.

This folder retains its historical source version. From this directory, CPU inspection uses:

```bash
export PYTHONPATH="$PWD/src"
python -m llm_evals validate
python -m llm_evals status
```

Do not submit jobs or overwrite exports just to inspect results. For a new run use the [main README](../../README.md). Historical absolute paths inside JSON records are original provenance and may reference the previous repository name.
