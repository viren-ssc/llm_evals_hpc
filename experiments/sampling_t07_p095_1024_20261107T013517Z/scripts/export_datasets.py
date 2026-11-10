"""Export verified per-model evaluation datasets and a combined workbook."""
import json,collections
from pathlib import Path
import openpyxl,yaml
from llm_evals.checkpointing import Store,locked,atomic
from llm_evals.dataset import load,benchmark_path
from llm_evals.generation import experiment
from llm_evals.export import export
root=Path(__file__).resolve().parents[1];folder=root/'outputs/exports';folder.mkdir(parents=True,exist_ok=True)
with locked(folder/'dataset-export.lock'):
 rows,headers,sha=load(benchmark_path(root));settings=yaml.safe_load((root/'configs/generation.yaml').read_text());models=yaml.safe_load((root/'configs/models.yaml').read_text());summary={}
 for model_id,model in models.items():
  fp,_=experiment(model,settings,sha);store=Store(root/'outputs/raw'/fp,model_id);values=[]
  for row in rows:
   r=store.get(row['question_id'],fp)
   if not r or r['status'] not in ('success','truncated'):continue
   assert r['question']==row['question'] and r['benchmark_sha256']==sha and r['requested_max_new_tokens']==1024
   assert r['decoding']['temperature']==0.7 and r['decoding']['top_p']==0.95
   values.append(dict(row,**{model_id+'_answer':r['generated_answer']},truncated=r['truncated'],response_complete=r['response_complete'],output_tokens=r['output_tokens'],finish_reason=r['finish_reason'],temperature=0.7,top_p=0.95,max_new_tokens=1024,model_id=model_id,model_revision=r['model_revision'],run_id=r['run_id']))
  complete=len(values)==756;stem=folder/(model_id+'_756_evaluation_t07_p095_1024'+('' if complete else '_PARTIAL'));columns=headers+[model_id+'_answer','truncated','response_complete','output_tokens','finish_reason','temperature','top_p','max_new_tokens','model_id','model_revision','run_id']
  with stem.with_suffix('.jsonl').open('w') as f:
   for v in values:f.write(json.dumps(v,ensure_ascii=False)+'\n')
  book=openpyxl.Workbook();sheet=book.active;sheet.title='Evaluation Dataset';sheet.append(columns)
  for v in values:
   assert all(not isinstance(x,str) or len(x)<=32767 for x in v.values());sheet.append([v[c] for c in columns])
   for c in sheet[sheet.max_row]:
    if isinstance(c.value,str):c.data_type='s'
  sheet.freeze_panes='H2';sheet.auto_filter.ref=sheet.dimensions;book.save(stem.with_suffix('.xlsx'))
  summary[model_id]={'captured':len(values),'expected':756,'coverage_complete':complete,'naturally_finished':sum(v['response_complete'] for v in values),'capped':sum(v['truncated'] for v in values),'workbook':str(stem.with_suffix('.xlsx')),'jsonl':str(stem.with_suffix('.jsonl'))}
 export(root,partial=True)
 atomic(folder/'evaluation_datasets_integrity.json',{'benchmark_sha256':sha,'sampling':settings,'models':summary})
 print(json.dumps(summary,indent=2))
