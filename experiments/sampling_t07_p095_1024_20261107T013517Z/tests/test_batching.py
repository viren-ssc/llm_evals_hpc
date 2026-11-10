from types import SimpleNamespace
import pytest
from llm_evals.batching import reservation, stream_requests
from llm_evals.generation import consume_batched
from llm_evals.checkpointing import Store


class Scheduler:
    def __init__(self):
        self.active = {}
        self.peak = 0
        self.admissions = []
        self.aborted = []
    def add_request(self, qid, prompt, params):
        assert set(prompt) == {'prompt_token_ids'}
        self.active[qid] = prompt
        self.peak = max(self.peak, len(self.active))
        self.admissions.append((qid, prompt))
    def step(self):
        qid = sorted(self.active)[-1]  # intentionally finish out of submission order
        del self.active[qid]
        return [SimpleNamespace(request_id=qid, finished=True)]
    def abort_request(self, ids):
        self.aborted.extend(ids)


def requests(n=7):
    return [{'question_id': f'q{i}', 'ids': [i] * 5, 'max_tokens': 20, 'params': object()} for i in range(n)]


def test_stream_bounds_memory_and_refills_out_of_order():
    engine = Scheduler()
    stream = stream_requests(engine, requests(), max_sequences=64, token_budget=96, block_size=16)
    first = next(stream)
    assert first[0]['question_id'] == 'q2'
    assert len(engine.admissions) == 3  # 3 x rounded 32-token reservations
    assert engine.peak == 3
    rest = list(stream)
    assert {job['question_id'] for job, _, _ in [first] + rest} == {f'q{i}' for i in range(7)}
    assert engine.peak == 3
    assert not engine.aborted


def test_stream_stop_drains_without_new_submissions():
    engine = Scheduler()
    stop = [False]
    stream = stream_requests(engine, requests(), 3, 96, 16, lambda: stop[0])
    first = next(stream)
    stop[0] = True
    rest = list(stream)
    assert len([first] + rest) == 3
    assert len(engine.admissions) == 3


def test_reservation_never_shortens_oversized_question():
    assert reservation(5, 20, 16) == 32
    with pytest.raises(ValueError, match='exceeds'):
        list(stream_requests(Scheduler(), requests(1), 64, 16, 16))


def test_batched_answers_saved_immediately_with_correct_question(tmp_path):
    store = Store(tmp_path, 'apertus')
    store.retain_truncated = True
    rows = [{'question_id': 'q1', 'question': 'English question'},
            {'question_id': 'q2', 'question': 'Question française'}]
    class Engine:
        def answers(self, jobs, stopped):
            assert [r['question'] for r in jobs] == ['English question', 'Question française']
            yield 'q2', {'generated_answer': 'Réponse', 'truncated': True}
            # First completion must already be durably saved before continuing.
            assert store.get('q2', 'fp')['generated_answer'] == 'Réponse'
            yield 'q1', {'generated_answer': 'Answer', 'truncated': False}
    consume_batched(Engine(), rows, store, 'fp', 'apertus', {'path': 'fixture', 'revision': 'fixture'},
                    {'max_attempts': 3, 'max_new_tokens': 1024}, 'run', 'sha', lambda: False)
    assert store.get('q1', 'fp')['question'] == 'English question'
    assert store.get('q2', 'fp')['question'] == 'Question française'
    assert store.get('q2', 'fp')['accepted_truncation']
    assert len(store.get('q2', 'fp')['attempts']) == 1


def test_missing_batched_completion_is_not_silent(tmp_path):
    class Missing:
        def answers(self, jobs, stopped):
            return iter(())
    with pytest.raises(RuntimeError, match='omitted'):
        consume_batched(Missing(), [{'question_id': 'q', 'question': 'Question'}], Store(tmp_path, 'llama'),
                        'fp', 'llama', {'path': 'fixture', 'revision': 'fixture'},
                        {'max_attempts': 1, 'max_new_tokens': 1024}, 'run', 'sha', lambda: False)


def test_cache_capacity_uses_latest_measured_profile(tmp_path):
    from llm_evals.batching import profiled_cache_tokens
    path = tmp_path / 'engine.log'
    path.write_text('old engine GPU KV cache size: 39,344 tokens\n'
                    'new engine GPU KV cache size: 32,176 tokens\n')
    assert profiled_cache_tokens(path) == 32176
    path.write_text('Loading weights; no measured profile\n')
    with pytest.raises(RuntimeError, match='unavailable'):
        profiled_cache_tokens(path)


@pytest.mark.parametrize('requested_cap', [20, None])
def test_vllm_batch_wrapper_uses_profile_without_callable_rpc(tmp_path, monkeypatch, requested_cap):
    import sys
    import types
    import llm_evals.inference as inference
    root = tmp_path / 'project'
    log = root / 'outputs/runs/canadianness-test.out'
    log.parent.mkdir(parents=True)
    log.write_text('GPU KV cache size: 96 tokens\n')
    monkeypatch.setattr(inference, '__file__', str(root / 'src/llm_evals/inference.py'))
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    fake_vllm = types.ModuleType('vllm')
    fake_vllm.SamplingParams = Params
    fake_sampling = types.ModuleType('vllm.sampling_params')
    fake_sampling.RequestOutputKind = SimpleNamespace(FINAL_ONLY='final_only')
    monkeypatch.setitem(sys.modules, 'vllm', fake_vllm)
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', fake_sampling)
    prompts = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 1 and messages[0]['role'] == 'user'
            prompts.append(messages[0]['content'])
            return [len(prompts)] * 3
    class Framework(Scheduler):
        def add_request(self, qid, prompt, params):
            assert params.max_tokens is requested_cap or params.max_tokens == requested_cap
            assert params.temperature == 0.7 and params.top_p == 0.95
            super().add_request(qid, prompt, params)
        vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
        def step(self):
            result = super().step()[0]
            result.outputs = [SimpleNamespace(text='Answer ' + result.request_id,
                              token_ids=[1, 2], finish_reason='stop', stop_reason=68)]
            return [result]
    framework = Framework()
    obj = inference.VLLM.__new__(inference.VLLM)
    obj.tokenizer = Tokenizer()
    obj.engine = SimpleNamespace(llm_engine=framework)  # deliberately no callable RPC API
    obj.template = 'fixture'
    obj.settings = {'max_model_len': 8192, 'seed': 0, 'temperature': 0.7, 'top_p': 0.95}
    obj.batching = {'kv_cache_reserve_fraction': 0.95, 'admission_mode': 'scheduler_managed',
                    'initial_output_reservation_tokens': 20}
    obj.max_sequences = 64
    obj.stops = [68]
    results = list(obj.answers([{'question_id': 'q1', 'question': 'Question one', 'max_tokens': requested_cap},
                               {'question_id': 'q2', 'question': 'Question deux', 'max_tokens': requested_cap}]))
    assert prompts == ['Question one', 'Question deux']
    assert [qid for qid, answer in results] == ['q2', 'q1']
    assert framework.peak == 2
    assert all(answer['requested_max_new_tokens'] == requested_cap for _, answer in results)
    if requested_cap is None:
        assert all(answer['effective_max_new_tokens'] == 8189 for _, answer in results)
    assert obj.batch_runtime['kv_cache_tokens'] == 96
    assert all(answer['generated_answer'] == 'Answer ' + qid for qid, answer in results)


def test_no_response_cap_uses_remaining_checkpoint_context():
    from llm_evals.inference import output_allowance
    assert output_allowance(65536, 100, None) == 65436
    assert output_allowance(131072, 120, None) == 130952
    with pytest.raises(ValueError, match='fills'):
        output_allowance(65536, 65536, None)


def test_scheduler_managed_reservation_does_not_cap_generation():
    jobs = requests(2)
    for job in jobs:
        job['max_tokens'] = 65436
        job['reservation_output_tokens'] = 20
    engine = Scheduler()
    result = list(stream_requests(engine, jobs, 64, 96, 16))
    assert engine.peak == 2
    assert all(job['max_tokens'] == 65436 for job, _, _ in result)
