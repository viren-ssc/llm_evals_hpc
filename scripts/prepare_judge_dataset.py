"""Validate an answered benchmark and package it for the separate Part 2 repository.

No inference, judge calls, filtering, or answer rewriting occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, destination, answer_column, expected_rows=None, sheet=None):
    if source.resolve() == destination.resolve():
        raise ValueError('Input and output must differ')
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise ValueError('Output already exists; choose a new destination')
    if source.suffix == '.jsonl':
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    elif source.suffix == '.xlsx':
        import openpyxl
        book = openpyxl.load_workbook(source, read_only=True, data_only=False)
        try:
            worksheet = book[sheet] if sheet else book.worksheets[0]
            cells = worksheet.iter_rows(values_only=True)
            headers = list(next(cells))
            if len(headers) != len(set(headers)):
                raise ValueError('Duplicate workbook columns')
            rows = [dict(zip(headers, values)) for values in cells if any(v is not None for v in values)]
        finally:
            book.close()
    else:
        raise ValueError('Expected .jsonl or .xlsx')
    if not rows or (expected_rows is not None and len(rows) != expected_rows):
        raise ValueError(f'Unexpected row count: {len(rows)}')
    seen = set()
    for row in rows:
        for field in ('question_id', 'question', 'reference_answer', answer_column):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Missing/empty {field} for {row.get("question_id")}')
        if row['question_id'] in seen:
            raise ValueError(f'Duplicate question_id: {row["question_id"]}')
        seen.add(row['question_id'])
    # Keep the original column names and all metadata; Part 2 owns its adapter schema.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'rows': len(rows), 'answer_column': answer_column,
              'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
              'judge_called': False, 'schema': 'original columns preserved'}
    destination.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--answer-column', required=True)
    parser.add_argument('--expected-rows', type=int)
    parser.add_argument('--sheet')
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.answer_column, args.expected_rows, args.sheet), indent=2))


if __name__ == '__main__':
    main()
