# sampling_t07_p095_1024_20261107T013517Z

T=0.7, top_p=0.95, 1,024 cap. Use combined `MIXED_TEMPERATURE_COMPLETE.*`; GC has 755 sampling answers + 1 authorized T=0 replacement.

- [Filled benchmarks and integrity reports](outputs/exports/)
- [Raw per-question records](outputs/raw/)
- [Run manifests and scheduler logs](outputs/runs/)
- [Exact configurations](configs/)

Captured nonempty raw answers: GC LLM **755**, Apertus **756**, Llama **756** (756 questions/model). These counts include token-capped responses. Review per-answer flags; historical integrity files can reflect earlier snapshots or stricter natural-completion policies.

This folder retains its historical source version. From this directory, CPU inspection uses:

```bash
export PYTHONPATH="$PWD/src"
python -m llm_evals validate
python -m llm_evals status
```

Do not submit jobs or overwrite exports just to inspect results. For a new run use the [main README](../../README.md). Historical absolute paths inside JSON records are original provenance and may reference the previous repository name.
