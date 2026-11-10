"""Authoritative workbook ingestion; generation consumes only question strings."""
import hashlib
import re
from collections import Counter
from pathlib import Path

FIELDS = ['question_id', 'language', 'primary_metric', 'applicable_metrics',
          'context_text', 'question', 'reference_answer']


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(rows, expected=756, counts=None):
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} CORE rows, found {len(rows)}')
    seen, questions = set(), set()
    for row in rows:
        qid = row.get('question_id')
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError(f'Empty or duplicate question_id: {qid!r}')
        seen.add(qid)
        q = row.get('question')
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f'Empty question: {qid}')
        normalized = ' '.join(q.split()).casefold()
        if normalized in questions:
            raise ValueError(f'Duplicate question text: {qid}')
        questions.add(normalized)
        if row.get('language') not in ('en', 'fr'):
            raise ValueError(f'Invalid language: {qid}')
        if not re.fullmatch(r'M[1-7]', str(row.get('primary_metric'))):
            raise ValueError(f'Invalid primary metric: {qid}')
        metrics = {v.strip() for v in str(row.get('applicable_metrics', '')).split(',')}
        if not metrics or any(not re.fullmatch(r'M[1-7]', m) for m in metrics):
            raise ValueError(f'Invalid applicable metrics: {qid}')
        if row['primary_metric'] not in metrics:
            raise ValueError(f'Primary metric absent from applicable metrics: {qid}')
        for field in ('context_text', 'reference_answer'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Missing {field}: {qid}')
        for field in ('status', 'scope', 'inclusion', 'tier'):
            if field in row and str(row[field]).strip().lower() not in ('core', 'included', 'include'):
                raise ValueError(f'Non-CORE row: {qid}')
    observed = dict(Counter(r['language'] for r in rows))
    if counts and observed != counts:
        raise ValueError(f'Language count mismatch: {observed}')
    return observed


def load(path, expected=756):
    import openpyxl
    book = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        if 'CORE Questions' not in book.sheetnames:
            raise ValueError('Missing explicit CORE Questions worksheet')
        sheet = book['CORE Questions']
        cells = sheet.iter_rows(values_only=True)
        headers = list(next(cells))
        if len(set(headers)) != len(headers) or not set(FIELDS) <= set(headers):
            raise ValueError(f'Invalid workbook headers: {headers}')
        rows = []
        for values in cells:
            if all(v is None for v in values):
                continue
            row = dict(zip(headers, values))
            if any(isinstance(v, str) and v.startswith('=') for v in values):
                raise ValueError('Formula cells are not authoritative literal question data')
            rows.append(row)
        validate_rows(rows, expected, {'en': 404, 'fr': 352} if expected == 756 else None)
        return rows, headers, checksum(path)
    finally:
        book.close()


def benchmark_path(root):
    import yaml
    config_path = Path(root) / 'configs/dataset.yaml'
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
        path = Path(root) / config['path']
        if checksum(path) != config['sha256']:
            raise ValueError('Authoritative benchmark checksum changed; review dataset configuration explicitly')
        return path
    candidates = list(Path(root).glob('*.xlsx'))
    if len(candidates) != 1:
        raise ValueError('Configure the unique authoritative workbook in configs/dataset.yaml')
    return candidates[0]
