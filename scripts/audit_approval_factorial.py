import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t
root=Path(__file__).resolve().parents[1];out=root/'results/approval-factorial-v1'
frozen=json.loads((out/'freeze.json').read_text());p=json.loads((root/'experiments/approval_factorial/protocol.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert all(sha(root/k)==v for k,v in frozen['source_hashes'].items())
assert sha(root/'experiments/approval_factorial/protocol.json')==frozen['protocol_sha256']
rows=[];verified=0
for stage,seeds,families in [('pilot',p['development_seeds'],p['development_families']),('confirmation',p['confirmation_seeds'],p['confirmation_families'])]:
 files=list((out/stage).glob('seed-*.json'));assert len(files)==len(seeds)
 assert {json.loads(f.read_text())['seed'] for f in files}==set(seeds)
 for f in files:
  r=json.loads(f.read_text());assert r['source_hashes']==frozen['source_hashes'];assert len(r['rows'])==len(families)*len(p['sizes'])
  if stage=='confirmation':assert f.stat().st_mtime>=(out/'freeze.json').stat().st_mtime
  for row in r['rows']:
   trace=out/stage/'traces'/f"{r['seed']}-{row['family']}-{row['size']}.npz";assert sha(trace)==row['trace_sha256'];verified+=1
   for g,arms in row['arms'].items():
    for arm,a in arms.items():
     assert a['training']['approval_samples']==p['training_steps'];assert a['training']['environment_steps']==p['training_steps']
     assert all(math.isfinite(v) for v in a['metrics'].values())
    if row['family']=='feedback_lever':
     for channel in ['coupled','decoupled']:assert arms[channel+'_reward0']['metrics']==arms[channel+'_reward1']['metrics']
   if stage=='confirmation':rows.append((r['seed'],row))
summary=json.loads((out/'confirmation-summary.json').read_text());byseed={}
for seed,row in rows:
 if row['family']=='feedback_lever':
  a=row['arms']['0.95'];byseed.setdefault(seed,[]).append(a['decoupled_reward0']['metrics']['lever_use']-a['coupled_reward0']['metrics']['lever_use'])
d=np.array([np.mean(v) for v in byseed.values()]);mean=d.mean();h=t.ppf(.975,31)*d.std(ddof=1)/math.sqrt(32)
assert np.allclose([mean-h,mean+h],summary['primary']['lever_difference']['ci95'])
assert summary['primary_support_criteria_met'] is False
secondaries=[]
for fam in p['confirmation_families']:
 for channel in ['coupled','decoupled']:
  samples={}
  for seed,row in rows:
   if row['family']==fam:
    arms=row['arms']['0.95'];samples.setdefault(seed,[]).append(arms[channel+'_reward1']['metrics']['button_use']-arms[channel+'_reward0']['metrics']['button_use'])
  ds=np.array([np.mean(v) for v in samples.values()]);half=t.ppf(.975,31)*ds.std(ddof=1)/np.sqrt(32)
  secondaries.append({'family':fam,'contrast':channel+' reward1-minus-reward0 button probability','mean':float(ds.mean()),'ci95':[float(ds.mean()-half),float(ds.mean()+half)],'status':'secondary descriptive contrast'})
unknown=[{'seed':seed,'family':row['family'],'size':row['size'],'gamma':g,'arm':arm,'probability':a['metrics']['unknown_state_probability']} for seed,row in rows for g,arms in row['arms'].items() for arm,a in arms.items() if a['metrics']['unknown_state_probability']>1e-10]
audit={'status':'passed','source_hashes_match_freeze':True,'trace_archives_hash_verified':verified,'confirmation_seeds':32,'confirmation_environment_instances':len(rows),'conditions_per_environment':8,'primary_intervals_independently_recalculated':True,'confirmation_created_after_freeze':True,'identical_budgets':True,'reward_toggle_identity_on_zero_reward_family':True,'secondary_descriptive_contrasts':secondaries,'unknown_state_fallback_cases':unknown,'scope':'Engineering/integrity audit of these saved runs, not independent scientific reproduction.'}
(out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k not in ['unknown_state_fallback_cases','secondary_descriptive_contrasts']},indent=2));print(json.dumps(secondaries,indent=2))
