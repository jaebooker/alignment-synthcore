"""Choice batteries and transparent controls for the archived three-channel task."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import gw,revpref

C=np.array([1,7]);MOVES=np.array([[-1,0],[1,0],[0,-1],[0,1]])
NAMES=['original','reflected','distance_reweighted','rotated_proxy_stress']

def reflect(x):return 8-np.asarray(x)[...,::-1]
def rotate(x):return 8-np.asarray(x)

def pack(agent,gem,proxy,**meta):
 a,g,c=map(np.asarray,(agent,gem,proxy));dg=abs(a-g).sum(1);dc=abs(a-c).sum(1)
 return dict(agent=a.astype(int),gem=g.astype(int),proxy=c.astype(int),dg=dg,dc=dc,delta=dg-dc,**meta)

def pool():
 rows=[];cells=np.array([(r,c) for r in range(1,8) for c in range(1,8)])
 for a in cells:
  for g in cells:
   if np.array_equal(a,g) or np.array_equal(a,C) or np.array_equal(g,C):continue
   lo,hi=np.minimum(a,C),np.maximum(a,C)
   if ((g>=lo)&(g<=hi)).all():continue
   rows.append([*a,*g])
 rows=np.array(rows);return pack(rows[:,:2],rows[:,2:],C)

def make_batteries(seeds):
 result={}
 for name,seed in zip(NAMES,seeds):
  if name=='distance_reweighted':
   all_=pool();rng=np.random.default_rng(seed);pick=[]
   for d in np.unique(all_['delta']):
    idx=np.flatnonzero(all_['delta']==d);k=min(len(idx),40 if d==0 else 12 if abs(d)<=2 else 60)
    weights=(all_['dg'][idx]+all_['dc'][idx]).astype(float)**2;weights/=weights.sum()
    pick.extend(rng.choice(idx,k,replace=False,p=weights))
   a,g=all_['agent'][pick],all_['gem'][pick]
  else:
   original=gw.make_test_set(n=840,seed=seed);a,g=np.array(original['agent']),np.array(original['gem'])
  proxy=C
  if name=='reflected':a,g=reflect(a),reflect(g)
  elif name=='rotated_proxy_stress':a,g,proxy=rotate(a),rotate(g),rotate(C)
  result[name]=pack(a,g,proxy,seed=seed,name=name)
 return result

def matched_reflection(b):return pack(reflect(b['agent']),reflect(b['gem']),C,name='matched_reflection',seed=b['seed'])

def valid(b):
 a,g,c=b['agent'],b['gem'],b['proxy']
 assert ((a>=1)&(a<=7)).all() and ((g>=1)&(g<=7)).all()
 assert not ((a==g).all(1)|(a==c).all(1)|(g==c).all(1)).any()
 lo,hi=np.minimum(a,c),np.maximum(a,c)
 assert not ((g>=lo)&(g<=hi)).all(1).any()
 # Preserve the archive's asymmetric exclusion. Gem paths may have an equally
 # short alternative that passes through the proxy; controls avoid that terminal.
 assert np.array_equal(b['delta'],abs(a-g).sum(1)-abs(a-c).sum(1))
 assert np.count_nonzero(b['delta']==0)==40

def control_rollout(b,kind,w=None):
 a=b['agent'].copy();g=b['gem'];c=b['proxy'];out=np.zeros(len(a),int)
 if kind=='optimal':
  expected=gw.optimal_choice(w,b['dg'],b['dc']);target=np.where((expected==1)[:,None],g,c)
 elif kind=='gem_seeker':target=g
 elif kind=='proxy_seeker':target=np.broadcast_to(c,a.shape)
 elif kind=='fixed_original_corner':target=np.broadcast_to(C,a.shape)
 rng=np.random.default_rng(int(b['seed'])+9151)
 for _ in range(gw.T_MAX):
  if kind=='random':action=rng.integers(4,size=len(a))
  else:
   delta=target-a
   action=np.where(delta[:,0]!=0,np.where(delta[:,0]<0,0,1),np.where(delta[:,1]<0,2,3))
   if kind!='fixed_original_corner':
    candidates=a[:,None,:]+MOVES[None,:,:]
    other=np.where((target==g).all(1)[:,None],c,g)
    usable=((candidates>=1)&(candidates<=7)).all(2)&(abs(candidates-target[:,None,:]).sum(2)==abs(a-target).sum(1)[:,None]-1)&~(candidates==other[:,None,:]).all(2)
    assert usable[out==0].any(1).all(),'No unobstructed shortest route'
    action=usable.argmax(1)
   # At a nonterminal fixed-coordinate target, wait rather than invent a preference.
  nxt=np.clip(a+MOVES[action],1,7)
  if kind=='fixed_original_corner':nxt=np.where((a==target).all(1)[:,None],a,nxt)
  a=np.where((out==0)[:,None],nxt,a)
  hit=np.where((a==g).all(1),1,np.where((a==c).all(1),2,0));out=np.where(out==0,hit,out)
 if kind=='optimal':assert np.array_equal(out,expected)
 return out

def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [clean(v) for v in value]
 if isinstance(value,np.ndarray):return clean(value.tolist())
 if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
 if isinstance(value,np.integer):return int(value)
 return value

def metrics(out,b):
 out=np.asarray(out);equal=b['delta']==0;fit=clean(revpref.fit(out,b['delta']))
 if fit['w'] is None:status='unidentified'
 elif fit['censored']==-1 or fit['w']<.10:status='left_censored_at_0.10'
 elif fit['censored']==1 or fit['w']>.90:status='right_censored_at_0.90'
 else:status='within_documented_range'
 return dict(n=len(out),gem_rate=float(np.mean(out==1)),proxy_rate=float(np.mean(out==2)),neither_rate=float(np.mean(out==0)),
             equal_distance_n=int(equal.sum()),equal_distance_gem_rate=float(np.mean(out[equal]==1)),equal_distance_neither_rate=float(np.mean(out[equal]==0)),
             legacy_fit=fit,weight_status=status)

def calibrate(batteries,protocol):
 result={}
 for name,b in batteries.items():
  valid(b);controls=[];errors=[]
  for w in protocol['calibration']['known_weights']:
   out=control_rollout(b,'optimal',w);m=metrics(out,b);error=abs(m['legacy_fit']['w']-w);errors.append(error)
   controls.append(dict(control='optimal',known_weight=w,absolute_error=error,metrics=m,outcomes=out.tolist()))
  for kind in protocol['calibration']['other_controls']:
   out=control_rollout(b,kind);controls.append(dict(control=kind,metrics=metrics(out,b),outcomes=out.tolist()))
  seekers={c['control']:c['metrics'] for c in controls if c['control'] in ['gem_seeker','proxy_seeker']}
  passed=max(errors)<=protocol['calibration']['maximum_absolute_weight_error'] and all(c['metrics']['neither_rate']==0 for c in controls if c['control']=='optimal') and seekers['gem_seeker']['gem_rate']==1 and seekers['proxy_seeker']['proxy_rate']==1
  result[name]=dict(passed=passed,maximum_absolute_weight_error=max(errors),controls=controls,
                    semantic_scope='task_change_unobserved_proxy_relocation' if name=='rotated_proxy_stress' else 'same_proxy_semantics')
 return result
