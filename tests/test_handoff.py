import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('handoff', Path(__file__).resolve().parents[1] / 'scripts/prepare_judge_dataset.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def test_handoff_preserves_text_flags_and_source(tmp_path):
    row = dict(question_id='q1', question='  exact question\n', reference_answer='reference',
               model_answer='=literal answer\n', truncated=True, temperature=0.7)
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps(row) + '\n')
    original = source.read_bytes()
    destination = tmp_path / 'handoff.jsonl'
    report = handoff.prepare(source, destination, 'model_answer', 1)
    assert json.loads(destination.read_text()) == row
    assert source.read_bytes() == original
    assert report['rows'] == 1 and not report['judge_called']
    with pytest.raises(ValueError, match='exists'):
        handoff.prepare(source, destination, 'model_answer', 1)


@pytest.mark.parametrize('rows,expected', [
    ([dict(question_id='q', question='Q', reference_answer='R', answer='')], 1),
    ([dict(question_id='q', question='Q', reference_answer='R', answer='A')]*2, 2),
    ([dict(question_id='q', question='Q', reference_answer='R', answer='A')], 756),
])
def test_invalid_handoff_does_not_create_output(tmp_path, rows, expected):
    source = tmp_path / 'input.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    output = tmp_path / 'output.jsonl'
    with pytest.raises(ValueError):
        handoff.prepare(source, output, 'answer', expected)
    assert not output.exists()
