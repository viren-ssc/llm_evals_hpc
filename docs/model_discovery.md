# Local model discovery

Read-only discovery; no downloads. Header/index validation checks readability and payload bounds, not full weight cryptographic integrity.

## gc_llm: ssc-dsai/gc-llm-apertus-70b-instruct-2509
Path: `/gpfs/fs7/ssc/ssc_dsai/abm002/.cache/huggingface/hub/models--ssc-dsai--gc-llm-apertus-70b-instruct-2509/snapshots/58277e94d16ccf486c1702c0aac8115c69448585`
Ready: True; revision: `58277e94d16ccf486c1702c0aac8115c69448585`.
Published GC model card identifies swiss-ai/Apertus-70B-Instruct-2509 as its parent; complete full checkpoints found
Problems: 
Architecture/precision: ['ApertusForCausalLM'] / bfloat16
Parameter count: 70599864640

## apertus: swiss-ai/Apertus-70B-Instruct-2509
Path: `/gpfs/fs7/ssc/ssc_dsai/abm002/.cache/huggingface/hub/models--swiss-ai--Apertus-70B-Instruct-2509/snapshots/0f2767662a013c0fb631079cea883155252bc3b6`
Ready: True; revision: `0f2767662a013c0fb631079cea883155252bc3b6`.
Published GC model card identifies swiss-ai/Apertus-70B-Instruct-2509 as its parent; complete full checkpoints found
Problems: 
Architecture/precision: ['ApertusForCausalLM'] / bfloat16
Parameter count: 70599864640

## llama: meta-llama/Llama-3.3-70B-Instruct
Path: `/gpfs/fs7/ssc/ssc_dsai/.cache/Meta-Llama-3.3-70B-Instruct`
Ready: True; revision: `Meta-Llama-3.3-70B-Instruct`.
Local model card and config identify Llama 3.3 70B Instruct
Problems: 
Architecture/precision: ['LlamaForCausalLM'] / bfloat16
Parameter count: 70553706496

The initial shared root caches are config-only. Complete readable GC and Apertus snapshots were found in /gpfs/fs7/ssc/ssc_dsai/abm002/.cache/huggingface/hub by following the existing CANBENCH_RESPONSES launcher. Each has 30 shards and 70,599,864,640 parameters. GC uses full weights, not an adapter-only load. The official GC model card verifies the Apertus Instruct 2509 parent: https://huggingface.co/ssc-dsai/gc-llm-apertus-70b-instruct-2509/blob/main/README.md. Both native templates live in separate chat_template.jinja files. The effective question-only templates retain their native BOS/user/end/assistant tokens and omit automatic system/developer boilerplate.
The shared full_finetune_conservative and lora_optimized models have no established connection to the requested GC checkpoint and are not substituted.
Llama alternative flat copy: /gpfs/fs7/ssc/ssc_dsai/.cache/Meta-Llama-3.3-70B-Instruct. The alternate HF snapshot contains incomplete download links; the complete flat copy is selected and pinned by metadata and shard header hashes.
BF16 70B weights require approximately 132 GiB aggregate plus runtime/KV overhead. Use four 40 GB A100 GPUs with eager execution, one sequence and an 8192-token context. Memory feasibility must pass smoke testing. V100 4×16 GB is insufficient without changing precision.
