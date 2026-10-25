"""Additional independent verification; not used to choose protocol or fits."""
import numpy as np
import pytest
import torch
from experiments.regret_budget.core import certify, Environment, OriginalACCEL, train
from testbeds.ued_gridworld import public as P
from testbeds.ued_gridworld import generate as G
from testbeds.ued_gridworld.testbed import Session


def test_real_horizon_long_connected_path():
    # A connected serpentine corridor whose far end exceeds the actual deadline.
    walls=np.ones((9,9),bool)
    for row in [1,3,5,7]:walls[row,1:8]=False
    walls[2,7]=walls[4,1]=walls[6,7]=False
    level=dict(walls=walls,cheese=(7,1),lamp=(1,2),zone=[],n=7)
    values,_=certify(level,H=19)
    assert G._bfs(walls,level['cheese'])[(1,1)]==30
    assert values[(1,1)]==0
    assert values[(7,2)]==pytest.approx(1-.5/19)


@pytest.mark.parametrize('arm',['dr','practical'])
def test_copied_ppo_matches_original_step_checkpoint(monkeypatch,arm):
    # Includes genuine parameter updates, level choice, resets and replay masks.
    # Original shared trainer and new instrumentation must yield identical weights.
    torch.set_num_threads(2)
    seed=9753;steps=4096;hp=dict(lr=.001,ent=.01,p_replay=.7,edits=3,beta=.3,rho=.3,capacity=1000)
    class Direct:
        def __init__(self):self.session=Session([dict(family='corner',perm=np.arange(4))],dict(steps=steps,levels=60000),seed)
        def call(self,op,**args):
            try:return self.session.handle(dict(op=op,task=0,**args))
            except Exception as error:
                if type(error).__name__=='BudgetExceeded':raise P.BudgetExceeded() from None
                raise
        def new_levels(self,k=1):return self.call('new_levels',n=k)['levels']
        def edit_levels(self,levels,k=1):return self.call('edit_levels',levels=levels,k=k)['levels']
        def reset(self,slots,levels):return Environment.decode(self.call('reset',slots=slots,levels=levels)['obs'])
        def step(self,slots,actions):
            r=self.call('step',slots=slots,actions=actions)
            return Environment.decode(r['obs']),np.asarray(r['reward'],np.float32),np.asarray(r['done'],bool)
    v=Direct();monkeypatch.setattr(P,'venv',lambda i:v)
    curriculum=P.Uniform() if arm=='dr' else OriginalACCEL(hp,np.random.default_rng(seed),v)
    original=P.train_ppo(0,dict(n_channels=4,view=7),hp,curriculum,seed)
    saved=[]
    def save(key,policy,ledger,certificates):
        saved.append(policy);return ledger
    train(arm,hp,seed,'corner',np.arange(4),[steps],None,save)
    assert original['arch']==saved[0]['arch']
    for name,value in original['state_dict'].items():
        np.testing.assert_array_equal(value,saved[0]['state_dict'][name])
