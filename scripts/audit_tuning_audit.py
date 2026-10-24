"""Check saved Test 2 provenance and independently reconstruct selected model scores."""
from pathlib import Path
import argparse,hashlib,json,math,sys,subprocess
import numpy as np
import torch
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from harness.runner import get_testbed,load_hparams

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/post-tuning-mechanism-audit-v1'
load=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rescore',action='store_true');args=ap.parse_args();torch.set_num_threads(1)
 initial=load(OUT/'initial-freeze.json');freeze=load(OUT/'selection-freeze.json');p=load(ROOT/'experiments/tuning_audit/protocol.json');chosen=load(OUT/'selection.json')
 assert all(sha(ROOT/k)==v for k,v in initial['source_hashes'].items());assert freeze['source_hashes']==initial['source_hashes'];assert sha(OUT/'selection.json')==freeze['selection_sha256']
 supporting={}
 for folder in ['harness','testbeds/confounded_imitation','testbeds/ued_gridworld']:
  for f in sorted((ROOT/folder).rglob('*.py')):
   rel=str(f.relative_to(ROOT));original=subprocess.check_output(['git','show',initial['git_base']+':'+rel],cwd=ROOT)
   assert hashlib.sha256(original).hexdigest()==sha(f),'Dependency changed: '+rel
   supporting[rel]=sha(f)
 arms={f.stem:load(f) for f in (OUT/'arms').glob('*.json')};checks=[]
 stage_seeds=[set(p[k]) for k in ['selection_seeds','gate_audit_seeds','confirmation_seeds']]
 assert all(not a&b for i,a in enumerate(stage_seeds) for b in stage_seeds[i+1:])
 assert all(p['engineering_smoke_seed'] not in a for a in stage_seeds)
 expected_ids=set()
 for key,row in arms.items():
  assert hashlib.sha256(json.dumps(row['job'],sort_keys=True).encode()).hexdigest()[:32]==key
  assert row['job']['experiment_source_hashes']==initial['source_hashes'];assert sha(OUT/'models'/f'{key}.npz')==row['model_sha256'];assert row['completed_utc']>initial['created_utc']
 for key,c in chosen['selected'].items():
  assert len(c['table'])==4
  tbname,method=key.split('|');assert [e['hparams'] for e in c['table']]==load_hparams(ROOT/method)[:4]
  assert c['hparams']==c['table'][c['selected_index']]['hparams']
  for entry in c['table']:
   expected_ids.update(entry['arm_ids']);rr=[arms[k] for k in entry['arm_ids']];assert {r['job']['seed'] for r in rr}==set(p['selection_seeds']);assert all(r['job']['split']=='dev' and r['job']['config']=='full' for r in rr)
   assert all(r['job']['method']==method and r['job']['testbed']==tbname and r['job']['hparams']==entry['hparams'] for r in rr)
   assert np.isclose(np.mean([r['score']['safety'] for r in rr]),entry['safety'])
  assert c['selected_index']==max(range(4),key=lambda i:c['table'][i]['safety'])
 stages={s:load(OUT/f'{s}.json') for s in ['audit','confirm']}
 for stage,data in stages.items():
  assert data['source_hashes']==initial['source_hashes'];assert data['selection_sha256']==freeze['selection_sha256']
  for idea in p['ideas']:
   conditions=['default','selected'] if stage=='audit' else ['selected']
   for condition in conditions:
    rr=[r for r in data['rows'] if r['idea']==idea and r['condition']==condition]
    assert {r['seed'] for r in rr}==set(p['gate_audit_seeds'] if stage=='audit' else p['confirmation_seeds'])
    for pair in rr:
     for role in ['method','baseline']:
      expected_ids.add(pair['pair'][role]);a=arms[pair['pair'][role]];assert a['completed_utc']>freeze['created_utc'];assert pair[role]==a['score']
      assert a['job']['split']==('dev' if stage=='audit' else 'test')
      assert a['job']['config']==('toy' if stage=='audit' else 'full')
 assert expected_ids==set(arms),'Missing or unregistered fit records'
 summary=load(OUT/'summary.json')
 for item in summary['results']:
  spec=load(ROOT/'ideas'/item['idea']/'idea.json');key=spec['prediction']['mechanism']['diagnostic'];rr=[r for r in stages[item['stage']]['rows'] if r['idea']==item['idea'] and r['condition']==item['condition']]
  d=np.array([r['method']['diagnostics'][key]-r['baseline']['diagnostics'][key] for r in rr]);d=d if spec['prediction']['mechanism']['direction']=='higher' else -d
  for name,alpha in [('oriented_effect',.05),('simultaneous_oriented_effect',.05/3)]:
   half=t.ppf(1-alpha/2,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));assert np.allclose([d.mean()-half,d.mean()+half],item[name]['ci'])
 if args.rescore:
  selected=[]
  for tbname in ['confounded_imitation','ued_gridworld']:
   rr=[r for r in arms.values() if r['job']['testbed']==tbname and r['job']['split']=='test' and r['job']['seed']==p['confirmation_seeds'][0]]
   selected.extend(sorted(rr,key=lambda r:r['job']['method']))
  for row in selected:
   job=row['job'];tb=get_testbed(job['testbed']);inst=tb.instance(job['split'],job['config'],job['seed']);outputs=[]
   with np.load(OUT/'models'/f'{row["id"]}.npz',allow_pickle=False) as model:
    for meta in row['model_metadata']:
     out={k:v for k,v in meta.items() if k!='state_keys'};out['state_dict']={name:model[label] for name,label in meta['state_keys']};outputs.append(out)
   score=tb.evaluate(outputs,inst,job['seed'])
   def compare(a,b):
    if isinstance(a,dict):assert set(a)==set(b);[compare(a[k],b[k]) for k in a]
    elif a is None:assert b is None
    else:assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
   compare(score,row['score']);checks.append({'id':row['id'],'testbed':tb.name,'method':job['method'],'rescored':True})
 result={'status':'passed','verified_model_archives':len(arms),'fresh_tuning_candidates_per_arm':4,'selection_seeds_per_candidate':2,'gate_audit_seeds':4,'confirmation_seeds_per_idea':8,'source_and_selection_hashes_match':True,'all_audit_and_confirmation_fits_after_selection_freeze':True,'paired_intervals_independently_recalculated':True,'selected_saved_models_rescored':checks,'supporting_source_matches_initial_git_commit':supporting,'scope':'Engineering integrity check; same evaluator re-scoring is not independent scientific reproduction.'}
 (OUT/'integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
