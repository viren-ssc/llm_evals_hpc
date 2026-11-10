"""Export and verify the completed, consistent-cap GC LLM dataset."""
from pathlib import Path
import json,collections,hashlib
import openpyxl,yaml
from llm_evals.dataset import load,benchmark_path
from llm_evals.generation import experiment
from llm_evals.checkpointing import Store
root=Path(__file__).resolve().parents[1]
rows,headers,sha=load(benchmark_path(root))
model=yaml.safe_load((root/'configs/models.yaml').read_text())['gc_llm']
settings=yaml.safe_load((root/'configs/generation.yaml').read_text());assert settings['max_new_tokens']==1024
fp,_=experiment(model,settings,sha);store=Store(root/'outputs/raw'/fp,'gc_llm')
columns=headers+['gc_llm_answer','truncated','response_complete','output_tokens','finish_reason','max_new_tokens','model_id','model_revision','run_id']
values=[];raw=[]
for row in rows:
 r=store.get(row['question_id'],fp);assert r and r['question']==row['question'] and r['benchmark_sha256']==sha
 assert r['status'] in ('success','truncated') and r['generated_answer'].strip()
 assert r['requested_max_new_tokens']==r['effective_max_new_tokens']==1024
 v=dict(row,gc_llm_answer=r['generated_answer'],truncated=r['truncated'],response_complete=r['response_complete'],output_tokens=r['output_tokens'],finish_reason=r['finish_reason'],max_new_tokens=1024,model_id='gc_llm',model_revision=r['model_revision'],run_id=r['run_id'])
 assert all(not isinstance(x,str) or len(x)<=32767 for x in v.values())
 values.append(v);raw.append(r)
folder=root/'outputs/exports';stem=folder/'gc_llm_756_evaluation_1024_tokens'
for path,data in [(stem.with_suffix('.jsonl'),values),(folder/(stem.name+'_raw.jsonl'),raw)]:
 with path.open('w') as f:
  for v in data:f.write(json.dumps(v,ensure_ascii=False)+'\n')
book=openpyxl.Workbook();sheet=book.active;sheet.title='GC LLM Evaluation';sheet.append(columns)
for v in values:
 sheet.append([v[c] for c in columns])
 for cell in sheet[sheet.max_row]:
  if isinstance(cell.value,str):cell.data_type='s'
sheet.freeze_panes='H2';sheet.auto_filter.ref=sheet.dimensions;book.save(stem.with_suffix('.xlsx'))
check=openpyxl.load_workbook(stem.with_suffix('.xlsx'),read_only=True,data_only=False)
back=list(check.active.values);assert len(back)==757 and list(back[0])==columns
for expected,actual in zip(values,back[1:]):assert actual[columns.index('gc_llm_answer')]==expected['gc_llm_answer'] and actual[0]==expected['question_id']
check.close()
counts=dict(collections.Counter(r['status'] for r in raw));info={'rows':len(values),'model':'gc_llm','max_new_tokens':1024,'counts':counts,'benchmark_sha256':sha,'fingerprint':fp,'question_only_generation':True,'evaluation_scores_included':False,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [stem.with_suffix('.xlsx'),stem.with_suffix('.jsonl'),folder/(stem.name+'_raw.jsonl')]}}
(folder/(stem.name+'_integrity.json')).write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
