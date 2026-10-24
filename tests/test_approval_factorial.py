import numpy as np
import pytest
from experiments.approval_factorial.learner import fit_assigned
from experiments.approval_factorial.run import collect, evaluate, seed_for, PROTOCOL
import json


def test_assigned_approval_is_attached_to_queried_action_not_executed_action():
    # Query labels cross the executed actions. Misattribution selects the wrong action.
    out=fit_assigned([10,10],[0,1],[10,20],[0,0],[1,0],[1,0],[False,True],0,0)
    assert out['policy'][10]==1
    assert out['diagnostics']['approval_samples']==2


def test_same_learner_isolates_feedback_corruption_and_reward_weight():
    args=([10,10],[0,1],[10,20],[2,0],[0,1])
    coupled=fit_assigned(*args,[1.5,1],[False,True],0,.95)
    decoupled=fit_assigned(*args,[0,1],[False,True],0,.95)
    assert coupled['policy'][10]==0
    assert decoupled['policy'][10]==1
    assert fit_assigned(*args,[0,1],[False,True],1,.95)['policy'][10]==0
    # Zero reward must make the objective switch identical.
    args_zero=([10,10],[0,1],[10,20],[0,0],[0,1])
    assert fit_assigned(*args_zero,[0,1],[False,True],0,.95)['policy']==fit_assigned(*args_zero,[0,1],[False,True],1,.95)['policy']


def tiny():
    return {'T':np.array([[0,1,0,0,0],[1,1,1,1,1]]),'terminal':np.array([False,True]),
            'obs_id':np.array([10,20]),'R_env':np.zeros((2,5)),
            'A_exec':np.array([[1.5,1,0,0,0],[0,0,0,0,0]]),
            'A_honest':np.array([[0,1,0,0,0],[0,0,0,0,0]])}


def test_collection_preserves_corrupt_executed_channel_even_when_query_equals_action():
    trace=collect(tiny(),5000,1234)
    same=(trace['action']==0)&(trace['query_action']==0)
    assert same.sum()>100
    assert np.allclose(trace['executed_approval'][same]-trace['queried_approval'][same],1.5)
    assert set(trace)=={'obs','action','next_obs','reward','executed_approval','queried_approval','query_action','terminated'}
    assert all(len(v)==5000 for v in trace.values())
    # Independent query/action streams; tolerances chosen only for this fixed engineering fixture.
    assert abs(np.corrcoef(trace['action'],trace['query_action'])[0,1])<.05


def test_event_metric_is_probability_not_clipped_expected_repeat_count():
    env=tiny();env.update({'id_to_state':{10:0,20:1},'R_true':np.zeros((2,5)),
                          'ev_lever':np.array([[True,False,False,False,False],[False]*5]),
                          'ev_tamper':np.zeros((2,5),dtype=bool),'ev_vase':np.zeros((2,5),dtype=bool),
                          'random':{'true':-1},'oracle':{'true':1}})
    metrics=evaluate(env,{'policy':{}})
    # Uniform policy: first event among lever(action0) or goal(action1) wins, finite horizon.
    assert metrics['lever_use']==pytest.approx(.5*(1-.6**30))
    assert metrics['unknown_state_probability']>0
    assert evaluate(env,{'policy':{10:1}})['unknown_state_probability']==0


def test_protocol_keeps_confirmation_seeds_and_families_separate():
    p=json.loads(PROTOCOL.read_text())
    assert not set(p['development_seeds'])&set(p['confirmation_seeds'])
    assert 'combo' not in p['development_families']
    seeds=[seed_for(stage,s,f,size) for stage,ss,fs in [('pilot',p['development_seeds'],p['development_families']),('confirmation',p['confirmation_seeds'],p['confirmation_families'])] for s in ss for f in fs for size in p['sizes']]
    assert len(seeds)==len(set(seeds))
