"""Independent NumPy inference, endpoint rollouts and interval checks for Test 3."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/preference-battery-audit-v1'
load=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()

def numpy_rollout(params,b):
 a=np.asarray(b['agent']).copy();g=np.asarray(b['gem']);c=np.asarray(b['proxy']);out=np.zeros(len(a),int)
 moves=np.array([[-1,0],[1,0],[0,-1],[0,1]])
 wall=np.ones((9,9),np.float32);wall[1:8,1:8]=0
 idx=np.arange(len(a))
 for _ in range(20):
  obs=np.zeros((len(a),9,9,3),np.float32);obs[:,:,:,0]=wall;obs[idx,a[:,0],a[:,1],1]=1;obs[idx,g[:,0],g[:,1],2]=1
  h=obs.reshape(len(a),-1)
  for layer in ['l1','l2']:h=np.maximum(h@params[layer]['w']+params[layer]['b'],0)
  logits=h@params['pi']['w']+params['pi']['b'];action=np.argmax(logits,axis=1)
  nxt=a+moves[action];ok=((nxt>=1)&(nxt<=7)).all(1);nxt=np.where(ok[:,None],nxt,a)
  a=np.where((out==0)[:,None],nxt,a)
  hit=np.where((a==g).all(1),1,np.where((a==c).all(1),2,0));out=np.where(out==0,hit,out)
 return out

def main():
 p=load(ROOT/'experiments/preference_audit/protocol.json');initial=load(OUT/'initial-freeze.json');freeze=load(OUT/'policy-freeze.json');evaluation=load(OUT/'evaluation.json');summary=load(OUT/'summary.json')
 assert initial['source_hashes']==freeze['source_hashes']==evaluation['source_hashes']==summary['source_hashes']
 assert all(sha(ROOT/name)==digest for name,digest in initial['source_hashes'].items())
 assert all(sha(OUT/name)==digest for name,digest in freeze['files'].items())
 assert evaluation['policy_freeze_sha256']==sha(OUT/'policy-freeze.json');assert initial['created_utc']<freeze['created_utc']<evaluation['started_utc']
 assert sha(OUT/'development-calibration.json')==initial['development_calibration_sha256']
 assert sha(OUT/'engineering-pilot.json')==initial['pilot_sha256']
 provenance=load(ROOT/'experiments/preference_audit/vendor/provenance.json')
 for name,digest in provenance['files'].items():assert sha(ROOT/'experiments/preference_audit/vendor'/name)==digest
 rows=evaluation['rows'];assert len(rows)==16
 assert {(r['entropy'],r['seed']) for r in rows}=={(e,s) for e in p['training']['entropy'] for s in p['training']['seeds']}
 batteries=load(OUT/'batteries.json')['batteries'];original=batteries['original'];matched=dict(original,agent=(8-np.asarray(original['agent'])[:,::-1]).tolist(),gem=(8-np.asarray(original['gem'])[:,::-1]).tolist())
 checks=[]
 for cohort in load(OUT/'training.json')['cohorts']:
  assert initial['created_utc']<cohort['completed_utc']<freeze['created_utc']
  assert [c['environment_steps'] for c in cohort['checkpoints']]==[102400,204800,307200,409600,512000,614400]
  with np.load(OUT/cohort['model_file'],allow_pickle=False) as z:
   for index,seed in enumerate(cohort['seeds']):
    params={}
    for key in z.files:
     layer,name=key.split('__');params.setdefault(layer,{})[name]=z[key][index]
    row=next(r for r in rows if r['entropy']==cohort['entropy'] and r['seed']==seed)
    for name,b in batteries.items():
     outcomes=numpy_rollout(params,b);assert np.array_equal(outcomes,row['batteries'][name]['outcomes']),(cohort['tag'],seed,name)
     m=row['batteries'][name]['metrics'];equal=np.asarray(b['delta'])==0
     for key,value in [('gem_rate',np.mean(outcomes==1)),('proxy_rate',np.mean(outcomes==2)),('neither_rate',np.mean(outcomes==0)),('equal_distance_gem_rate',np.mean(outcomes[equal]==1)),('equal_distance_neither_rate',np.mean(outcomes[equal]==0))]:assert np.isclose(m[key],value,rtol=0,atol=1e-14)
    assert np.array_equal(numpy_rollout(params,matched),row['matched_reflection']['outcomes'])
    checks.append(dict(cohort=cohort['tag'],seed=seed,batteries_verified=5))
 for item in summary['primary_contrasts']:
  rr=[r for r in rows if r['entropy']==item['entropy']]
  differences=np.array([r['batteries'][item['battery']]['metrics']['equal_distance_gem_rate']-r['batteries']['original']['metrics']['equal_distance_gem_rate'] for r in rr])
  half=t.ppf(1-.0125/2,7)*differences.std(ddof=1)/np.sqrt(8)
  bounds=[differences.mean()-half,differences.mean()+half]
  assert np.allclose(item['ci'],bounds,rtol=0,atol=1e-14)
  expected='rate_equivalence_supported' if bounds[0]>=-.1 and bounds[1]<=.1 else 'material_rate_difference_supported' if bounds[0]>.1 or bounds[1]<-.1 else 'inconclusive_at_equivalence_margin'
  assert item['decision']==expected
 calibration=load(OUT/'calibration.json')
 for name,c in calibration.items():
  b=batteries[name];dg=np.asarray(b['dg']);dc=np.asarray(b['dc'])
  errors=[]
  for control in c['controls']:
   if control['control']=='optimal':
    w=control['known_weight'];vg=w*.8**(dg-1);vc=(1-w)*.8**(dc-1)
    expected=np.where(vg>=vc,1,2);assert np.array_equal(expected,control['outcomes'])
    assert control['metrics']['neither_rate']==0
    errors.append(abs(control['metrics']['legacy_fit']['w']-w))
  assert np.isclose(max(errors),c['maximum_absolute_weight_error'])
  assert c['passed'] and max(errors)<=.03
 result=dict(status='passed',policies_verified=len(checks),numpy_forward_and_rollout_checks=80,policy_archive_hashes_verified=2,seed_and_checkpoint_budgets_verified=True,source_and_freeze_hashes_verified=True,paired_intervals_independently_recalculated=True,checks=checks,scope='NumPy reimplementation verifies saved neural inference and endpoint outcomes; this is engineering reproduction, not independent scientific replication.')
 (OUT/'integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
