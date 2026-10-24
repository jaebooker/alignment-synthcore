"""Fresh equal-budget tuning and post-tuning mechanism audit.
python -m experiments.tuning_audit.run initialize|select|freeze|audit|confirm|report
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import scipy
from scipy import stats
import torch
from harness.runner import get_testbed, load_hparams, run_method, _gate

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/post-tuning-mechanism-audit-v1';PFILE=HERE/'protocol.json'

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def specs(p):return {i:load(ROOT/'ideas'/i/'idea.json') for i in p['ideas']}

def signature(p):
 files=set(HERE.glob('*.py'))|{PFILE,ROOT/'harness/runner.py',ROOT/'harness/sandbox_exec.py'}
 for i,s in specs(p).items():
  tb=get_testbed(s['testbed']);files|={ROOT/'ideas'/i/'method.py',ROOT/'ideas'/i/'idea.json',Path(tb.baseline_method),Path(tb.public_module)}
  files|={Path(f) for f in tb.evaluator_files}
 return {str(f.relative_to(ROOT)):sha(f) for f in sorted(files)}

def manifest_job(tbname,method,split,config,seed,hp,p):
 sig=signature(p);tb=get_testbed(tbname)
 return {'testbed':tbname,'evaluator_hash':tb.evaluator_hash(),'method':str(Path(method).relative_to(ROOT)),'method_sha256':sha(Path(method)),
         'split':split,'config':config,'seed':seed,'hparams':hp,'experiment_source_hashes':sig}

def job_id(job):return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:32]

def execute(job):
 key=job_id(job);path=OUT/'arms'/f'{key}.json'
 if path.exists():
  row=load(path)
  if row['job']!=job:raise RuntimeError('Cached job mismatch')
  return row
 tb=get_testbed(job['testbed']);t=time.monotonic()
 print('START',key[:8],job['method'],job['split'],job['config'],job['seed'],json.dumps(job['hparams']),flush=True)
 inst=tb.instance(job['split'],job['config'],job['seed'])
 outputs=run_method(tb,ROOT/job['method'],inst,job['hparams'],job['seed'],timeout=600)
 score=tb.evaluate(outputs,inst,job['seed'])
 # Store fitted models without pickle, enabling evaluator-side independent re-scoring.
 arrays={};meta=[]
 for index,o in enumerate(outputs):
  item={k:v for k,v in o.items() if k!='state_dict'};item['state_keys']=[]
  for number,(name,value) in enumerate(o.get('state_dict',{}).items()):
   label=f'task{index}_parameter{number}';arrays[label]=np.asarray(value);item['state_keys'].append([name,label])
  meta.append(item)
 artifact=OUT/'models'/f'{key}.npz';artifact.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(artifact,**arrays)
 row={'id':key,'job':job,'score':score,'runtime_s':time.monotonic()-t,'completed_utc':now(),'model_metadata':meta,'model_sha256':sha(artifact)}
 dump(path,row)
 print('DONE',key[:8],f'{row["runtime_s"]:.1f}s',f'safety={score["safety"]:.4f}',flush=True)
 return row

def run_jobs(jobs,workers):
 unique={job_id(j):j for j in jobs};results={}
 with ThreadPoolExecutor(max_workers=workers) as pool:
  tasks={pool.submit(execute,j):key for key,j in unique.items()}
  for future in as_completed(tasks):results[tasks[future]]=future.result()
 return results

def candidate_sets(p):
 answer={}
 for i,s in specs(p).items():
  tb=get_testbed(s['testbed'])
  for method in [ROOT/'ideas'/i/'method.py',Path(tb.baseline_method)]:
   rel=str(method.relative_to(ROOT));grid=load_hparams(method)
   if len(grid)<p['tuning']['candidate_count']:raise RuntimeError('Insufficient equal-budget candidates')
   answer[(tb.name,rel)]=grid[:p['tuning']['candidate_count']]
 return answer

def verify_source(p):
 f=load(OUT/'initial-freeze.json')
 if f['source_hashes']!=signature(p):raise RuntimeError('Source changed after initial freeze')

def selection(p):
 verify_source(p);jobs=[]
 for (tb,rel),grid in candidate_sets(p).items():
  for hp in grid:
   for seed in p['selection_seeds']:jobs.append(manifest_job(tb,ROOT/rel,'dev','full',seed,hp,p))
 results=run_jobs(jobs,p['parallel_jobs']);chosen={}
 for (tb,rel),grid in candidate_sets(p).items():
  table=[]
  for hp in grid:
   rr=[results[job_id(manifest_job(tb,ROOT/rel,'dev','full',seed,hp,p))] for seed in p['selection_seeds']]
   table.append({'hparams':hp,'safety':float(np.mean([r['score']['safety'] for r in rr])),'arm_ids':[r['id'] for r in rr]})
  selected=max(range(len(table)),key=lambda n:table[n]['safety'])
  chosen[tb+'|'+rel]={'hparams':table[selected]['hparams'],'selected_index':selected,'table':table}
 dump(OUT/'selection.json',{'selected':chosen,'created_utc':now(),'source_hashes':signature(p)})

def selected_hp(chosen,tb,method):return chosen['selected'][tb+'|'+str(Path(method).relative_to(ROOT))]['hparams']

def evaluation(stage,p):
 verify_source(p);freeze=load(OUT/'selection-freeze.json');chosen=load(OUT/'selection.json')
 if freeze['selection_sha256']!=sha(OUT/'selection.json'):raise RuntimeError('Selection differs from frozen settings')
 split,config,seeds,conditions=('dev','toy',p['gate_audit_seeds'],['default','selected']) if stage=='audit' else ('test','full',p['confirmation_seeds'],['selected'])
 jobs=[];mapping={}
 for i,s in specs(p).items():
  tb=get_testbed(s['testbed'])
  for condition in conditions:
   for seed in seeds:
    pair={}
    for label,method in [('method',ROOT/'ideas'/i/'method.py'),('baseline',Path(tb.baseline_method))]:
     hp=load_hparams(method)[0] if condition=='default' else selected_hp(chosen,tb.name,method)
     job=manifest_job(tb.name,method,split,config,seed,hp,p);jobs.append(job);pair[label]=job_id(job)
    mapping[f'{i}|{condition}|{seed}']={'idea':i,'condition':condition,'seed':seed,'pair':pair}
 results=run_jobs(jobs,p['parallel_jobs'])
 rows=[{**row,'method':results[row['pair']['method']]['score'],'baseline':results[row['pair']['baseline']]['score']} for row in mapping.values()]
 dump(OUT/f'{stage}.json',{'stage':stage,'created_utc':now(),'rows':rows,'source_hashes':signature(p),'selection_sha256':sha(OUT/'selection.json')})

def contrast(a,b,alpha=.05):
 d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
 if len(d)<2 or not np.isfinite(d).all():raise ValueError('Need finite paired observations')
 mean=float(d.mean());half=float(stats.t.ppf(1-alpha/2,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
 return {'n':len(d),'method_mean':float(np.mean(a)),'baseline_mean':float(np.mean(b)),'mean_diff':mean,'ci':[mean-half,mean+half],'alpha':alpha,'paired_differences':d.tolist()}

def practical_status(bounds,minimum):
 if bounds[0]>=minimum:return 'declared_minimum_supported'
 if bounds[1]<minimum:return 'declared_minimum_ruled_out_in_this_setup'
 return 'inconclusive_at_declared_minimum'

def report(p):
 verify_source(p)
 summaries=[]
 for stage in ['audit','confirm']:
  if not (OUT/f'{stage}.json').exists():continue
  data=load(OUT/f'{stage}.json')
  for i,s in specs(p).items():
   for condition in (['default','selected'] if stage=='audit' else ['selected']):
    rr=[r for r in data['rows'] if r['idea']==i and r['condition']==condition]
    ms,bs=[r['method'] for r in rr],[r['baseline'] for r in rr]
    gate=_gate(get_testbed(s['testbed']),s['prediction'],ms,bs);key=s['prediction']['mechanism']['diagnostic'];higher=s['prediction']['mechanism']['direction']=='higher'
    ma=[r['diagnostics'][key] for r in ms];ba=[r['diagnostics'][key] for r in bs]
    oriented=contrast(ma,ba) if higher else contrast(ba,ma)
    simultaneous=contrast(ma,ba,.05/len(p['ideas'])) if higher else contrast(ba,ma,.05/len(p['ideas']))
    result={'idea':i,'stage':stage,'condition':condition,'gate':gate,'diagnostic':key,'method_diagnostic_mean':float(np.mean(ma)),
            'baseline_diagnostic_mean':float(np.mean(ba)),'oriented_effect':oriented,'simultaneous_oriented_effect':simultaneous,
            'practical_status':practical_status(simultaneous['ci'],s['prediction']['mechanism']['min_effect']),
            'safety':contrast([r['safety'] for r in ms],[r['safety'] for r in bs]),
            'capability':contrast([r['capability'] for r in ms],[r['capability'] for r in bs])}
    summaries.append(result)
 dump(OUT/'summary.json',{'results':summaries,'source_hashes':signature(p),'note':'Historical records unchanged. Primary decisions use simultaneous practical-effect intervals; point gates are reported separately.'})

def main():
 torch.set_num_threads(1)
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['initialize','select','freeze','audit','confirm','report']);args=ap.parse_args();p=load(PFILE);OUT.mkdir(parents=True,exist_ok=True)
 if args.action=='initialize':
  if (OUT/'initial-freeze.json').exists():raise RuntimeError('Already initialized')
  hist={}
  for i,s in specs(p).items():
   old=load(ROOT/'results'/i/'record.json');hist[i]={'record_sha256':sha(ROOT/'results'/i/'record.json'),'record':old,'current_evaluator_hash':get_testbed(s['testbed']).evaluator_hash()}
  dump(OUT/'historical-context.json',hist)
  dump(OUT/'initial-freeze.json',{'created_utc':now(),'source_hashes':signature(p),'protocol_sha256':sha(PFILE),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
 elif args.action=='select':selection(p)
 elif args.action=='freeze':
  verify_source(p)
  if (OUT/'selection-freeze.json').exists():raise RuntimeError('Selection already frozen')
  dump(OUT/'selection-freeze.json',{'created_utc':now(),'source_hashes':signature(p),'selection_sha256':sha(OUT/'selection.json'),'selection_seeds':p['selection_seeds'],'gate_audit_seeds':p['gate_audit_seeds'],'confirmation_seeds':p['confirmation_seeds']})
 elif args.action in ['audit','confirm']:evaluation(args.action,p);report(p)
 else:report(p)

if __name__=='__main__':main()
