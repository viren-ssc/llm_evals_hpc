"""Question-only, independent requests; imports GPU libraries only inside allocations."""
import json
from pathlib import Path
import os
import time


def messages(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Empty question')
    return [{'role': 'user', 'content': question}]


class VLLM:
    def __init__(self, model, settings, batching=None):
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('GPU inference requires a Slurm GPU allocation')
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model['tokenizer_path'], local_files_only=True,
                                                       trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError('No verified native chat template')
        config = json.loads((Path(model['path']) / 'config.json').read_text())
        eos = config.get('eos_token_id')
        self.stops = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
        self.native_template = self.tokenizer.chat_template
        self.template = model.get('prompt_template') or self.native_template
        self.settings = settings
        self.batching = batching or {}
        self.max_sequences = self.batching.get('max_num_seqs', 1)
        self.engine = LLM(model=model['path'], tokenizer=model['tokenizer_path'],
                          dtype=model['dtype'], tensor_parallel_size=model['tensor_parallel_size'],
                          gpu_memory_utilization=model['gpu_memory_utilization'],
                          max_model_len=settings['max_model_len'], max_num_seqs=self.max_sequences,
                          max_num_batched_tokens=self.batching.get('max_num_batched_tokens', 8192),
                          enforce_eager=True, seed=settings['seed'], trust_remote_code=False)

    def answer(self, question, max_tokens):
        from vllm import SamplingParams
        ids = self.tokenizer.apply_chat_template(messages(question), tokenize=True, add_generation_prompt=True, chat_template=self.template)
        if len(ids) + max_tokens > self.settings['max_model_len']:
            raise ValueError('Question plus output budget exceeds context; question will not be truncated')
        started = time.monotonic()
        out = self.engine.generate([{'prompt_token_ids': ids}], SamplingParams(
            temperature=0.0, max_tokens=max_tokens, seed=self.settings['seed'],
            stop_token_ids=self.stops), use_tqdm=False)[0].outputs[0]
        if out.finish_reason not in ('stop', 'length'):
            raise RuntimeError(f'Unexpected finish reason: {out.finish_reason}')
        return {'generated_answer': out.text, 'input_tokens': len(ids), 'output_tokens': len(out.token_ids),
                'generation_seconds': time.monotonic() - started,
                'finish_reason': out.finish_reason, 'stop_reason': out.stop_reason,
                'truncated': out.finish_reason == 'length',
                'prompt_token_ids_sha256': __import__('hashlib').sha256(str(ids).encode()).hexdigest()}


    def answers(self, jobs, stopped=lambda: False):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from .batching import stream_requests, profiled_cache_tokens
        def prepared():
            for job in jobs:
                ids = self.tokenizer.apply_chat_template(messages(job['question']), tokenize=True,
                    add_generation_prompt=True, chat_template=self.template)
                if len(ids) + job['max_tokens'] > self.settings['max_model_len']:
                    raise ValueError(f"Question {job['question_id']} exceeds context; no truncation applied")
                yield dict(job, ids=ids, params=SamplingParams(temperature=0.0,
                    max_tokens=job['max_tokens'], seed=self.settings['seed'],
                    stop_token_ids=self.stops, output_kind=RequestOutputKind.FINAL_ONLY))
        log = Path(__file__).resolve().parents[2] / 'outputs/runs' / f"canadianness-{os.environ['SLURM_JOB_ID']}.out"
        capacity = profiled_cache_tokens(log)
        block_size = self.engine.llm_engine.vllm_config.cache_config.block_size or 256
        budget = int(capacity * self.batching.get('kv_cache_reserve_fraction', 0.95))
        budget = budget // block_size * block_size
        self.batch_runtime = dict(max_num_seqs=self.max_sequences, kv_cache_tokens=capacity,
                                  reserved_token_budget=budget, block_size=block_size)
        print('BATCH_RUNTIME ' + json.dumps(self.batch_runtime), flush=True)
        for job, result, timing in stream_requests(self.engine.llm_engine, prepared(),
                self.max_sequences, budget, block_size, stopped):
            out = result.outputs[0]
            if out.finish_reason not in ('stop', 'length'):
                raise RuntimeError(f'Unexpected finish reason: {out.finish_reason}')
            answer = {'generated_answer': out.text, 'input_tokens': len(job['ids']),
                      'output_tokens': len(out.token_ids), 'finish_reason': out.finish_reason,
                      'stop_reason': out.stop_reason, 'truncated': out.finish_reason == 'length',
                      'prompt_token_ids_sha256': __import__('hashlib').sha256(str(job['ids']).encode()).hexdigest(),
                      'batch_runtime': self.batch_runtime, **timing}
            yield job['question_id'], answer
