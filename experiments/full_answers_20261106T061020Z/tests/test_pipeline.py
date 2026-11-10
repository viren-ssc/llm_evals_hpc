import copy
import hashlib
import json
from pathlib import Path
import openpyxl
import pytest
import yaml
from llm_evals.dataset import validate_rows, load, FIELDS
from llm_evals.checkpointing import Store, atomic, locked
from llm_evals.inference import messages
from llm_evals.generation import experiment


def row(qid='q1', lang='en'):
    return dict(zip(FIELDS, [qid, lang, 'M1', 'M1, M7', 'SECRET_CONTEXT',
                            'A question ' + qid, 'SECRET_REFERENCE']))


def test_validation_rejects_duplicates_and_bad_fields():
    a, b = row(), row('q2', 'fr')
    assert validate_rows([a, b], 2) == {'en': 1, 'fr': 1}
    for mutation in ({'question_id': 'q1'}, {'question': a['question']}, {'language': 'xx'},
                     {'primary_metric': 'M8'}, {'applicable_metrics': 'M7'},
                     {'question': ''}, {'reference_answer': ''}, {'scope': 'held'}):
        with pytest.raises(ValueError):
            validate_rows([a, dict(b, **mutation)], 2)
    with pytest.raises(ValueError):
        validate_rows([a], 2)


def test_question_only_prompt():
    r = row()
    assert messages(r['question']) == [{'role': 'user', 'content': r['question']}]
    assert 'SECRET' not in json.dumps(messages(r['question']))


def record(fp='fingerprint', qid='q1', model='llama'):
    return dict(question_id=qid, model_id=model, fingerprint=fp, status='success',
                generated_answer='Actual fixture answer', truncated=False)


def test_resume_and_association(tmp_path):
    s = Store(tmp_path, 'llama')
    assert s.get('q1', 'fingerprint') is None
    s.put(record())
    assert s.get('q1', 'fingerprint')['status'] == 'success'
    with pytest.raises(ValueError):
        s.get('q1', 'different-settings')
    with pytest.raises(ValueError):
        s.put(record(model='apertus'))
    for changes in ({'generated_answer': ''}, {'truncated': True}):
        with pytest.raises(ValueError):
            s.put(dict(record(), **changes))
    (s.root / '.pending-interrupted').write_text('{')
    assert s.get('q1', 'fingerprint')['generated_answer']


def test_lock_rejects_second_worker(tmp_path):
    with locked(tmp_path / 'lock'):
        with pytest.raises(BlockingIOError):
            with locked(tmp_path / 'lock'):
                pass


def setup_export(tmp_path):
    (tmp_path / 'configs').mkdir()
    models = {key: {'path': 'fixture-' + key} for key in ['gc_llm', 'apertus', 'llama']}
    (tmp_path / 'configs/models.yaml').write_text(yaml.safe_dump(models))
    (tmp_path / 'configs/generation.yaml').write_text(yaml.safe_dump({'max_new_tokens': 1024}))
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = 'CORE Questions'
    sheet.append(FIELDS)
    for i in range(756):
        sheet.append(list(row(f'q{i}', 'en' if i < 404 else 'fr').values()))
    path = tmp_path / 'benchmark.xlsx'
    book.save(path)
    return models, path


def test_export_missing_and_complete(tmp_path):
    from llm_evals.export import export
    models, path = setup_export(tmp_path)
    initial = path.read_bytes()
    with pytest.raises(ValueError, match='requires 756'):
        export(tmp_path)
    result = export(tmp_path, partial=True)
    book = openpyxl.load_workbook(result['workbook'])
    assert book.active.max_row == 757
    assert book.active.cell(2, 8).value is None
    book.close()
    rows, _, sha = load(path)
    for key, model in models.items():
        fp, _ = experiment(model, {'max_new_tokens': 1024}, sha)
        s = Store(tmp_path / 'outputs/raw' / fp, key)
        for r in rows:
            rec = record(fp, r['question_id'], key)
            rec.update(question=r['question'], benchmark_sha256=sha, generated_answer='=literal ' + key)
            s.put(rec)
    result = export(tmp_path)
    assert result['successful'] == {key: 756 for key in models}
    book = openpyxl.load_workbook(result['workbook'])
    sheet = book.active
    assert sheet.max_row == 757
    assert sheet.max_column == 10
    assert sheet.cell(2, 7).value == 'SECRET_REFERENCE'
    assert sheet.cell(2, 8).value == '=literal apertus'  # yaml sorts fixture model keys
    assert sheet.cell(2, 8).data_type == 's'
    assert len(Path(result['workbook']).with_suffix('.jsonl').read_text().splitlines()) == 756
    assert path.read_bytes() == initial
    book.close()


def test_fingerprint_tracks_model_settings_and_dataset():
    fp, _ = experiment({'path': 'a'}, {'seed': 0}, 'abc')
    for model, settings, sha in [({'path': 'b'}, {'seed': 0}, 'abc'),
                                 ({'path': 'a'}, {'seed': 1}, 'abc'),
                                 ({'path': 'a'}, {'seed': 0}, 'def')]:
        assert experiment(model, settings, sha)[0] != fp


def test_generation_retries_and_resume(tmp_path, monkeypatch):
    import llm_evals.generation as gen
    models, path = setup_export(tmp_path)
    models = {'llama': {'path': 'fixture', 'ready': True, 'problems': [], 'revision': 'fixture',
                        'metadata_sha256': {}, 'shard_manifest': []}}
    (tmp_path / 'configs/models.yaml').write_text(yaml.safe_dump(models))
    settings = {'max_new_tokens': 1024, 'retry_max_new_tokens': 2048, 'max_attempts': 3}
    (tmp_path / 'configs/generation.yaml').write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(gen, 'inspect', lambda p: {'ready': True, 'metadata_sha256': {}, 'shards': []})
    monkeypatch.setattr(gen.time, 'sleep', lambda seconds: None)
    calls = []
    class Fake:
        template = 'fixture template'
        def __init__(self, *args):
            pass
        def answer(self, question, max_tokens):
            calls.append((question, max_tokens))
            return {'generated_answer': 'fixture answer', 'truncated': max_tokens == 1024}
    result = gen.run(tmp_path, 'llama', smoke=True, resume=True, engine_factory=Fake)
    assert result['successful'] == 2
    assert [c[1] for c in calls] == [1024, 2048, 1024, 2048]
    gen.run(tmp_path, 'llama', smoke=True, resume=True, engine_factory=Fake)
    assert len(calls) == 4  # successful checkpoints never regenerated
    assert all('SECRET' not in q for q, _ in calls)


def test_input_checksum_pin(tmp_path):
    from llm_evals.dataset import benchmark_path
    models, path = setup_export(tmp_path)
    from llm_evals.dataset import checksum
    (tmp_path / 'configs/dataset.yaml').write_text(yaml.safe_dump({'path': path.name, 'sha256': checksum(path)}))
    assert benchmark_path(tmp_path) == path
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum changed'):
        benchmark_path(tmp_path)


def test_incomplete_checkpoint_not_runnable(tmp_path):
    from llm_evals.discovery import inspect
    atomic(tmp_path / 'config.json', {'architectures': ['ApertusForCausalLM'], 'model_type': 'apertus'})
    result = inspect(tmp_path)
    assert not result['ready']
    assert 'No indexed weight shards' in result['problems']


def test_effective_llama_template_contains_only_question_and_role_tokens():
    import jinja2
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/models.yaml').read_text())
    template = config['llama']['prompt_template']
    question = '  Pourquoi?\nKeep whitespace.  '
    rendered = jinja2.Environment().from_string(template).render(
        bos_token='<|begin_of_text|>', messages=messages(question), add_generation_prompt=True)
    assert rendered == ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' +
                        question + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    assert 'system' not in rendered
    assert 'Today Date' not in rendered
    assert 'Knowledge' not in rendered


def test_empty_generations_record_failure_and_never_pass_smoke(tmp_path, monkeypatch):
    import llm_evals.generation as gen
    models, path = setup_export(tmp_path)
    model = {'path': 'fixture', 'ready': True, 'problems': [], 'revision': 'fixture',
             'metadata_sha256': {}, 'shard_manifest': []}
    (tmp_path / 'configs/models.yaml').write_text(yaml.safe_dump({'llama': model}))
    settings = {'max_new_tokens': 1024, 'retry_max_new_tokens': 2048, 'max_attempts': 3}
    (tmp_path / 'configs/generation.yaml').write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(gen, 'inspect', lambda p: {'ready': True, 'metadata_sha256': {}, 'shards': []})
    monkeypatch.setattr(gen.time, 'sleep', lambda seconds: None)
    class Empty:
        template = 'fixture'
        def __init__(self, *args):
            pass
        def answer(self, *args):
            return {'generated_answer': '', 'truncated': False}
    with pytest.raises(RuntimeError, match='Incomplete run'):
        gen.run(tmp_path, 'llama', smoke=True, resume=True, engine_factory=Empty)
    rows, _, sha = load(path)
    fp, _ = experiment(model, settings, sha)
    store = Store(tmp_path / 'outputs/raw' / fp, 'llama')
    first = store.get('q0', fp)
    assert first['status'] == 'error'
    assert len(first['attempts']) == 3
    assert first['generated_answer'] == ''
    assert not json.loads((tmp_path / f'outputs/runs/smoke-llama-{fp}.json').read_text())['passed']


def test_separate_native_template_is_recognized_and_pinned(tmp_path):
    import struct
    from llm_evals.discovery import inspect
    atomic(tmp_path / 'config.json', {'architectures': ['ApertusForCausalLM'], 'model_type': 'apertus'})
    atomic(tmp_path / 'tokenizer_config.json', {'bos_token': '<s>'})
    atomic(tmp_path / 'tokenizer.json', {})
    (tmp_path / 'chat_template.jinja').write_text('{{ bos_token }}{{ messages[0].content }}')
    header = json.dumps({'weight': {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
    (tmp_path / 'model-00001-of-00001.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0\0')
    atomic(tmp_path / 'model.safetensors.index.json', {'weight_map': {'weight': 'model-00001-of-00001.safetensors'}})
    first = inspect(tmp_path)
    assert first['ready']
    assert first['parameter_count'] == 1
    assert first['chat_template'].startswith('{{ bos_token }}')
    assert 'chat_template.jinja' in first['metadata_sha256']
    (tmp_path / 'chat_template.jinja').write_text('changed')
    assert inspect(tmp_path)['metadata_sha256'] != first['metadata_sha256']


@pytest.mark.parametrize('key', ['gc_llm', 'apertus'])
def test_apertus_family_rendered_prompts_are_question_only(key):
    import jinja2
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/models.yaml').read_text())
    question = '  Pourquoi le Canada?\nExact whitespace.  '
    rendered = jinja2.Environment().from_string(config[key]['prompt_template']).render(
        bos_token='<s>', messages=messages(question))
    assert rendered == '<s><|user_start|>' + question + '<|user_end|><|assistant_start|>'
    assert '<|system_start|>' not in rendered
    assert '<|developer_start|>' not in rendered


def test_disabled_smoke_runs_full_dataset_once_without_gate(tmp_path, monkeypatch):
    import llm_evals.generation as gen
    setup_export(tmp_path)
    model = {'path': 'fixture', 'ready': True, 'problems': [], 'revision': 'fixture',
             'metadata_sha256': {}, 'shard_manifest': []}
    (tmp_path / 'configs/models.yaml').write_text(yaml.safe_dump({'llama': model}))
    (tmp_path / 'configs/execution.yaml').write_text('run_smoke_tests: false\n')
    settings = {'max_new_tokens': 1024, 'retry_max_new_tokens': 2048, 'max_attempts': 1}
    (tmp_path / 'configs/generation.yaml').write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(gen, 'inspect', lambda p: {'ready': True, 'metadata_sha256': {}, 'shards': []})
    calls = {'loads': 0, 'questions': 0}
    class Once:
        template = 'fixture'
        def __init__(self, *args):
            calls['loads'] += 1
        def answer(self, question, max_tokens):
            calls['questions'] += 1
            return {'generated_answer': 'fixture response', 'truncated': False}
    result = gen.run(tmp_path, 'llama', smoke=True, resume=True, engine_factory=Once)
    assert result['smoke'] is False
    assert result['successful'] == 756
    assert calls == {'loads': 1, 'questions': 756}
    gen.run(tmp_path, 'llama', resume=True, engine_factory=Once)
    assert calls == {'loads': 1, 'questions': 756}


def test_retained_limit_output_is_saved_with_truthful_flag(tmp_path):
    store = Store(tmp_path, 'apertus')
    store.retain_truncated = True
    rec = record(model='apertus')
    rec.update(status='truncated', truncated=True, generated_answer='Full emitted text ' * 200)
    original = rec['generated_answer']
    store.put(rec)
    saved = store.get('q1', 'fingerprint')
    assert saved['status'] == 'success'
    assert saved['accepted_truncation'] is True
    assert saved['truncated'] is True
    assert saved['generated_answer'] == original


def test_export_retains_limit_output_in_answer_column(tmp_path):
    from llm_evals.export import export
    models, path = setup_export(tmp_path)
    rows, _, sha = load(path)
    key = 'apertus'
    fp, _ = experiment(models[key], {'max_new_tokens': 1024}, sha)
    store = Store(tmp_path / 'outputs/raw' / fp, key)
    store.retain_truncated = True
    rec = record(fp, 'q0', key)
    text = 'Entire emitted answer ' * 200
    rec.update(question=rows[0]['question'], benchmark_sha256=sha,
               status='truncated', truncated=True, generated_answer=text)
    store.put(rec)
    result = export(tmp_path, partial=True)
    book = openpyxl.load_workbook(result['workbook'])
    assert book.active.cell(2, 8).value == text
    assert result['successful']['apertus'] == 1
    records = Path(result['workbook']).with_name(Path(result['workbook']).stem + '_raw.jsonl')
    saved = json.loads(records.read_text().splitlines()[0])
    assert saved['truncated'] and saved['accepted_truncation']
    book.close()


def test_full_context_record_is_not_retried_or_called_complete(tmp_path):
    from llm_evals.generation import consume_batched
    class AtContext:
        calls = 0
        def answers(self, jobs, stopped):
            self.calls += 1
            assert jobs[0]['max_tokens'] is None
            yield 'q1', {'generated_answer': 'All text up to the native context boundary',
                          'truncated': True, 'response_complete': False, 'completion_boundary': 'model_context'}
    engine = AtContext()
    store = Store(tmp_path, 'apertus')
    consume_batched(engine, [{'question_id': 'q1', 'question': 'Question'}], store, 'fp', 'apertus',
                    {'path': 'fixture', 'revision': 'fixture'}, {'max_attempts': 3, 'max_new_tokens': None},
                    'run', 'sha', lambda: False)
    saved = store.get('q1', 'fp')
    assert engine.calls == 1
    assert saved['status'] == 'truncated'
    assert not saved['response_complete']


def test_large_answer_is_preserved_in_jsonl_and_excel_chunks(tmp_path):
    from llm_evals.export import export
    models, path = setup_export(tmp_path)
    rows, _, sha = load(path)
    fp, _ = experiment(models['apertus'], {'max_new_tokens': 1024}, sha)
    store = Store(tmp_path / 'outputs/raw' / fp, 'apertus')
    text = 'Réponse complète 🎉 ' * 5000
    rec = record(fp, 'q0', 'apertus')
    rec.update(question=rows[0]['question'], benchmark_sha256=sha, generated_answer=text, response_complete=True)
    store.put(rec)
    result = export(tmp_path, partial=True)
    book = openpyxl.load_workbook(result['workbook'])
    assert book.active.cell(2, 8).value.startswith('FULL GENERATED TEXT:')
    chunks = list(book['Long Answers'].iter_rows(min_row=2, values_only=True))
    assert ''.join(r[4] for r in chunks) == text
    wide = json.loads(Path(result['workbook']).with_suffix('.jsonl').read_text().splitlines()[0])
    assert wide['apertus_answer'] == text
    book.close()
