"""Single tabular learner with separate transition and assigned-feedback sample indices.

Inputs contain public observations, observed reward and exactly the selected approval
channel. No evaluator, geometry, hidden rewards or hazard labels are imported here.
"""
import numpy as np


def fit_assigned(obs, action, next_obs, reward, approval_action, approval, terminated, reward_weight, gamma):
    obs, action, next_obs, approval_action = map(np.asarray, (obs, action, next_obs, approval_action))
    reward, approval = np.asarray(reward), np.asarray(approval)
    terminated = np.asarray(terminated, dtype=bool)
    n=len(obs)
    if not all(len(a)==n for a in [action,next_obs,reward,approval_action,approval,terminated]):
        raise ValueError('inconsistent sample lengths')
    if not np.isfinite(reward).all() or not np.isfinite(approval).all():
        raise ValueError('non-finite training sample')
    if not (np.isin(action,range(5)).all() and np.isin(approval_action,range(5)).all()):
        raise ValueError('invalid action')
    states = np.unique(np.concatenate([obs,next_obs]))
    s, ns = np.searchsorted(states,obs), np.searchsorted(states,next_obs)
    S,A=len(states),5
    k=s*A+action; ka=s*A+approval_action
    cnt=np.bincount(k,minlength=S*A).reshape(S,A)
    acnt=np.bincount(ka,minlength=S*A).reshape(S,A)
    reward_mean=np.bincount(k,weights=reward,minlength=S*A).reshape(S,A)/np.maximum(cnt,1)
    approval_mean=np.bincount(ka,weights=approval,minlength=S*A).reshape(S,A)/np.maximum(acnt,1)
    T=np.full((S,A),-1,dtype=np.int64)
    for key,following in zip(k,ns):
        i,a=divmod(int(key),A)
        if T[i,a] not in (-1,following):raise ValueError('aliased successor')
        T[i,a]=following
    terminal=np.zeros(S,dtype=bool);terminal[ns[terminated]]=True
    known=(cnt>0)&(acnt>0)
    R=approval_mean+reward_weight*reward_mean
    V=np.zeros(S)
    converged=False
    for iterations in range(1,1001):
        vn=V.copy();vn[terminal]=0
        Q=np.where(known,R+gamma*vn[np.maximum(T,0)],-1e6)
        Q[terminal]=0
        v=Q.max(axis=1);v[~known.any(axis=1)]=0
        if np.max(np.abs(v-V))<1e-10:
            converged=True;break
        V=v
    if not converged:raise RuntimeError('value iteration did not converge')
    actions=Q.argmax(axis=1)
    return {'policy':{int(states[i]):int(actions[i]) for i in range(S) if known[i].any() and not terminal[i]},
            'diagnostics':{'observed_states':int(S),'observed_state_actions':int((cnt>0).sum()),
                           'action_pairs_with_both_dynamics_and_feedback':int(known.sum()),
                           'approval_samples':int(n),'environment_steps':int(n),'planning_iterations':iterations,
                           'planning_gamma':float(gamma),'reward_weight':float(reward_weight)}}
