"""User-authorized one-answer temperature-zero replacement; preserve source runs."""
import hashlib
import json
from pathlib import Path
import openpyxl

root = Path(__file__).resolve().parents[1]
project = root.parents[1]
folder = root / 'outputs/exports'
qid = 'CA-EXP2-152'
source = project / 'experiments/consistent_1024_20261106T144122Z/outputs/exports/gc_llm_756_evaluation_1024_tokens_raw.jsonl'
def read(path):
    return [json.loads(line) for line in path.open()]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
replacement, = [r for r in read(source) if r['question_id'] == qid and r['model_id'] == 'gc_llm']
assert replacement['status'] == 'success' and replacement['decoding']['temperature'] == 0
assert replacement['generated_answer'].strip()
sha = digest(project / 'data/benchmarks/Canadian-ness Benchmark - 756 CORE with Context.xlsx')
assert sha == replacement['benchmark_sha256']
raw_source = folder / 'canadianness_756_three_model_answers_PARTIAL_raw.jsonl'
raw = read(raw_source)
old, = [r for r in raw if r['question_id'] == qid and r['model_id'] == 'gc_llm']
assert not old['generated_answer'].strip()
for key in ('question', 'model_revision', 'model_path', 'benchmark_sha256'):
    assert old[key] == replacement[key], key
provenance = {'type': 'user_authorized_cross_run_replacement', 'source_file': str(source),
              'source_file_sha256': digest(source), 'source_run_id': replacement['run_id'],
              'source_fingerprint': replacement['fingerprint'], 'source_temperature': 0.0,
              'target_run_id': old['run_id'], 'target_temperature': 0.7,
              'reason': 'User authorized filling the single empty answer with the earlier temperature-zero GC-LLM answer.'}
new = dict(replacement, replacement_provenance=provenance, replaced_empty_record=old)
raw = [new if r is old else r for r in raw]
assert len(raw) == 2268
for model in ('gc_llm', 'apertus', 'llama'):
    records = [r for r in raw if r['model_id'] == model]
    assert len(records) == 756 and len({r['question_id'] for r in records}) == 756
    assert all(r['generated_answer'].strip() and r['benchmark_sha256'] == sha for r in records)
    for r in records:
        if r['truncated']:
            r['accepted_truncation'] = True
            r['retention_policy'] = 'User authorized retaining every nonempty token-capped output'
        r['status'] = 'success'
wide = read(folder / 'canadianness_756_three_model_answers_PARTIAL.jsonl')
for row in wide:
    row['gc_llm_temperature'] = 0.0 if row['question_id'] == qid else 0.7
    row['gc_llm_answer_reused'] = row['question_id'] == qid
    row['gc_llm_source_run_id'] = replacement['run_id'] if row['question_id'] == qid else old['run_id']
    if row['question_id'] == qid:
        assert row['question'] == replacement['question']
        row['gc_llm_answer'] = replacement['generated_answer']
assert len(wide) == 756
assert all(row[model+'_answer'].strip() for row in wide for model in ('gc_llm','apertus','llama'))
gc_rows = read(folder / 'gc_llm_756_evaluation_t07_p095_1024_PARTIAL.jsonl')
template = next(row for row in wide if row['question_id'] == qid)
base_keys = [k for k in gc_rows[0] if k not in ('gc_llm_answer','truncated','response_complete','output_tokens','finish_reason','temperature','top_p','max_new_tokens','model_id','model_revision','run_id')]
entry = {k: template[k] for k in base_keys}
entry.update(gc_llm_answer=replacement['generated_answer'], truncated=False, response_complete=True,
             output_tokens=replacement['output_tokens'], finish_reason=replacement['finish_reason'],
             temperature=0.0, top_p=replacement['decoding'].get('top_p'), max_new_tokens=1024,
             model_id='gc_llm', model_revision=replacement['model_revision'], run_id=replacement['run_id'])
gc_rows.append(entry)
order = {row['question_id']: i for i,row in enumerate(wide)}
gc_rows.sort(key=lambda row: order[row['question_id']])
for row in gc_rows:
    row['answer_reused'] = row['question_id'] == qid
    row['source_file'] = str(source) if row['answer_reused'] else str(raw_source)
    row['accepted_truncation'] = bool(row['truncated'])

def write(stem, rows, workbook=True):
    path = folder / (stem + '.jsonl')
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    if workbook:
        book = openpyxl.Workbook(); sheet = book.active; sheet.title='Evaluation Dataset'
        columns = list(rows[0]); sheet.append(columns)
        for row in rows:
            values = [row.get(k) for k in columns]
            assert all(not isinstance(v,str) or len(v)<=32767 for v in values)
            sheet.append(values)
            for cell in sheet[sheet.max_row]:
                if isinstance(cell.value,str): cell.data_type='s'
        sheet.freeze_panes='A2'; sheet.auto_filter.ref=sheet.dimensions
        meta=book.create_sheet('Replacement Provenance'); meta.append(['field','value'])
        for key,value in provenance.items(): meta.append([key,str(value)])
        book.save(folder / (stem+'.xlsx'))
    return str(path)
gc_path = write('gc_llm_756_evaluation_MIXED_TEMPERATURE_COMPLETE', gc_rows)
combined_path = write('canadianness_756_three_model_answers_MIXED_TEMPERATURE_COMPLETE', wide)
raw_path = write('canadianness_756_three_model_answers_MIXED_TEMPERATURE_COMPLETE_raw', raw, False)
report = {'complete': True, 'successful': {'gc_llm':756,'apertus':756,'llama':756},
          'benchmark_sha256':sha, 'mixed_temperature':True, 'replacement_question_id':qid,
          'replacement_provenance':provenance, 'gc_llm_temperature_counts':{'0.7':755,'0.0':1},
          'original_empty_record_preserved_in_raw_export':True,
          'files':{'gc_llm':gc_path,'combined':combined_path,'raw':raw_path}}
(folder/'mixed_temperature_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
# Read back persisted artifacts and verify the replacement verbatim.
for path,key in ((gc_path,'gc_llm_answer'),(combined_path,'gc_llm_answer')):
    rows=read(Path(path)); assert len(rows)==756
    item, = [r for r in rows if r['question_id']==qid]
    assert item[key] == replacement['generated_answer']
    book=openpyxl.load_workbook(Path(path).with_suffix('.xlsx'),read_only=True)
    assert book['Evaluation Dataset'].max_row==757
    book.close()
assert digest(project/'Canadian-ness Benchmark - 756 CORE with Context.xlsx')==sha
print(json.dumps(report,indent=2))
