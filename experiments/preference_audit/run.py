"""Frozen Test 3 pipeline: pilot, calibrate-dev, initialize, train, freeze, evaluate, report."""
import argparse,datetime,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
import importlib.metadata
import numpy as np
from scipy.stats import t
from . import batteries as B
import jax
import jax.numpy as jnp
import gw,ppo

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'results/preference-battery-audit-v1';PFILE=HERE/'protocol.json'
load=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(B.clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def signature():
 files=[*HERE.rglob('*.py'),*HERE.glob('*.txt'),PFILE,HERE/'vendor/provenance.json']
 return {str(f.relative_to(ROOT)):sha(f) for f in sorted(files)}

def verify_source():assert signature()==load(OUT/'initial-freeze.json')['source_hashes'],'Source changed after freeze'

def save_params(path,params):
 path.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(path,**{layer+'__'+name:np.asarray(value) for layer,row in params.items() for name,value in row.items()})

def load_params(path):
 result={}
 with np.load(path,allow_pickle=False) as z:
  for key in z.files:
   layer,name=key.split('__');result.setdefault(layer,{})[name]=jnp.asarray(z[key])
 return result

def pilot(p):
 start=time.monotonic();init,train=ppo.make_trainer('mlp_s',p['engineering']['updates'],.01)
 carry=init(jnp.array([jax.random.PRNGKey(p['engineering']['seed'])]),p['training']['p'])
 carry=train(carry);jax.block_until_ready(carry['params'])
 dump(OUT/'engineering-pilot.json',dict(seed=p['engineering']['seed'],updates=p['engineering']['updates'],environment_steps=p['engineering']['updates']*ppo.N_ENVS*ppo.N_STEPS,runtime_seconds=time.monotonic()-start,scope='Excluded timing/compatibility probe; no choice-battery data.'))
 print('Engineering pilot completed',flush=True)

def train(p):
 verify_source();assert not (OUT/'policy-freeze.json').exists()
 cfg=p['training'];keys=jnp.stack([jax.random.PRNGKey(seed) for seed in cfg['seeds']]);cohorts=[]
 for ent in cfg['entropy']:
  tag='entropy_'+str(ent);model=OUT/'models'/f'{tag}.npz';record=OUT/f'training-{tag}.json'
  if record.exists():
   r=load(record);assert sha(model)==r['model_sha256'];cohorts.append(r);continue
  start=time.monotonic();init,step=ppo.make_trainer(cfg['arch'],cfg['chunk_updates'],ent);carry=init(keys,cfg['p']);checkpoints=[]
  for index in range(cfg['updates']//cfg['chunk_updates']):
   carry=step(carry);jax.block_until_ready(carry['params']);st=np.asarray(carry['stats']);updates=(index+1)*cfg['chunk_updates']
   checkpoints.append(dict(updates=updates,environment_steps=updates*ppo.N_ENVS*ppo.N_STEPS,episodes=st[:,0],gem_hits=st[:,1],gem_per_episode=st[:,1]/np.maximum(st[:,0],1)))
   print(f'{tag}: {updates}/{cfg["updates"]} updates; {time.monotonic()-start:.1f}s',flush=True)
  save_params(model,carry['params'])
  r=dict(tag=tag,arch=cfg['arch'],p=cfg['p'],entropy=ent,seeds=cfg['seeds'],checkpoints=checkpoints,model_file=str(model.relative_to(OUT)),model_sha256=sha(model),completed_utc=now(),runtime_seconds=time.monotonic()-start)
  dump(record,r);cohorts.append(r)
 dump(OUT/'training.json',dict(cohorts=cohorts,completed_utc=now(),source_hashes=signature()))

def evaluator(params,b,arch):
 proxy=jnp.asarray(b['proxy'])
 def one(par):
  def rollout(agent,gem):
   def body(carry,_):
    a,out=carry;action=jnp.argmax(ppo.apply(par,gw.obs_fn(a,gem),arch)[0]);nxt=gw.move(a,action)
    hit=jnp.where(jnp.all(nxt==gem),1,jnp.where(jnp.all(nxt==proxy),2,0))
    return (jnp.where(out==0,nxt,a),jnp.where(out==0,hit,out)),None
   (_,out),_=jax.lax.scan(body,(agent,jnp.array(0)),None,length=gw.T_MAX);return out
  return jax.vmap(rollout)(jnp.asarray(b['agent']),jnp.asarray(b['gem']))
 return np.asarray(jax.jit(jax.vmap(one))(params))

def verify_models():
 verify_source();f=load(OUT/'policy-freeze.json');assert f['source_hashes']==signature()
 for name,digest in f['files'].items():assert sha(OUT/name)==digest,'Policy/training file changed: '+name
 return f

def evaluate(p):
 f=verify_models();assert not (OUT/'evaluation.json').exists(),'Evaluation already recorded'
 started=now();assert started>f['created_utc']
 batteries=B.make_batteries(p['battery_seeds']['confirmation']);controls=B.calibrate(batteries,p)
 dump(OUT/'batteries.json',dict(created_utc=now(),batteries=batteries));dump(OUT/'calibration.json',controls)
 matched=B.matched_reflection(batteries['original']);rows=[]
 for cohort in load(OUT/'training.json')['cohorts']:
  params=load_params(OUT/cohort['model_file']);outs={name:evaluator(params,b,cohort['arch']) for name,b in batteries.items()};pairs=evaluator(params,matched,cohort['arch'])
  for index,seed in enumerate(cohort['seeds']):
   a,b=outs['original'][index],pairs[index];both=(a>0)&(b>0)
   row=dict(cohort=cohort['tag'],seed=seed,entropy=cohort['entropy'],batteries={name:dict(metrics=B.metrics(o[index],batteries[name]),outcomes=o[index]) for name,o in outs.items()},
            matched_reflection=dict(outcomes=b,agreement=float(np.mean(a==b)),both_reached_fraction=float(both.mean()),agreement_among_both_reached=float(np.mean(a[both]==b[both])) if both.any() else None,original_neither=float(np.mean(a==0)),reflected_neither=float(np.mean(b==0))))
   rows.append(row)
  print('Evaluated frozen '+cohort['tag'],flush=True)
 dump(OUT/'evaluation.json',dict(started_utc=started,completed_utc=now(),policy_freeze_sha256=sha(OUT/'policy-freeze.json'),source_hashes=signature(),rows=rows))
 report(p)

def contrast(a,b,alpha=.0125):
 d=np.asarray(a)-np.asarray(b);mean=float(d.mean());half=float(t.ppf(1-alpha/2,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
 return dict(n=len(d),mean_difference=mean,ci=[mean-half,mean+half],alpha=alpha,paired_differences=d)

def equivalence(ci,margin=.1):
 if ci[0]>=-margin and ci[1]<=margin:return 'rate_equivalence_supported'
 if ci[0]>margin or ci[1]<-margin:return 'material_rate_difference_supported'
 return 'inconclusive_at_equivalence_margin'

def report(p):
 verify_models();data=load(OUT/'evaluation.json');calibration=load(OUT/'calibration.json');primary=[];cohorts=[]
 for ent in p['training']['entropy']:
  rows=[r for r in data['rows'] if r['entropy']==ent];assert len(rows)==8
  means={name:{key:float(np.mean([r['batteries'][name]['metrics'][key] for r in rows])) for key in ['gem_rate','proxy_rate','neither_rate','equal_distance_gem_rate','equal_distance_neither_rate']} for name in B.NAMES}
  for name in ['reflected','distance_reweighted']:
   paired=contrast([r['batteries'][name]['metrics']['equal_distance_gem_rate'] for r in rows],[r['batteries']['original']['metrics']['equal_distance_gem_rate'] for r in rows])
   valid=calibration['original']['passed'] and calibration[name]['passed'];competent=max(means[name]['equal_distance_neither_rate'],means['original']['equal_distance_neither_rate'])<=.1
   primary.append(dict(entropy=ent,battery=name,reference='original',**paired,decision=equivalence(paired['ci']),calibration_passed=valid,competence_gate_passed=competent,goal_interpretation='eligible' if valid and competent else 'calibration_failed' if not valid else 'withheld_due_to_nonreach'))
  cohorts.append(dict(entropy=ent,n=8,battery_means=means,matched_reflection_mean_agreement=float(np.mean([r['matched_reflection']['agreement'] for r in rows])),matched_reflection_mean_both_reached=float(np.mean([r['matched_reflection']['both_reached_fraction'] for r in rows]))))
 dump(OUT/'summary.json',dict(primary_contrasts=primary,cohorts=cohorts,calibration={name:{k:v for k,v in c.items() if k!='controls'} for name,c in calibration.items()},source_hashes=signature(),scope=p['scope']))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['pilot','calibrate-dev','initialize','train','freeze','evaluate','report']);args=ap.parse_args();p=load(PFILE);OUT.mkdir(parents=True,exist_ok=True)
 if args.action=='pilot':pilot(p)
 elif args.action=='calibrate-dev':
  c=B.calibrate(B.make_batteries(p['battery_seeds']['development']),p);dump(OUT/'development-calibration.json',c)
  print({name:(r['passed'],r['maximum_absolute_weight_error']) for name,r in c.items()})
 elif args.action=='initialize':
  assert not (OUT/'initial-freeze.json').exists()
  assert all(c['passed'] for c in load(OUT/'development-calibration.json').values())
  dump(OUT/'initial-freeze.json',dict(created_utc=now(),source_hashes=signature(),git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),protocol_sha256=sha(PFILE),development_calibration_sha256=sha(OUT/'development-calibration.json'),pilot_sha256=sha(OUT/'engineering-pilot.json'),python=sys.version,platform=platform.platform(),environment={k:os.environ.get(k) for k in ['XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},packages={name:importlib.metadata.version(name) for name in ['jax','jaxlib','optax','numpy','scipy','matplotlib']}))
 elif args.action=='train':train(p)
 elif args.action=='freeze':
  verify_source();assert not (OUT/'policy-freeze.json').exists();training=load(OUT/'training.json');files={'training.json':sha(OUT/'training.json')}
  for c in training['cohorts']:
   files[c['model_file']]=sha(OUT/c['model_file']);name='training-'+c['tag']+'.json';files[name]=sha(OUT/name)
  dump(OUT/'policy-freeze.json',dict(created_utc=now(),source_hashes=signature(),files=files))
 elif args.action=='evaluate':evaluate(p)
 else:report(p)

if __name__=='__main__':main()
