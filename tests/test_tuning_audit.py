import json
import math
import numpy as np
import pytest
from experiments.tuning_audit.run import practical_status,contrast,candidate_sets,job_id,PFILE
from harness.runner import _gate,get_testbed


def test_practical_classification_uses_interval_not_just_point_significance():
 assert practical_status([.11,.4],.1)=='declared_minimum_supported'
 assert practical_status([-.2,.09],.1)=='declared_minimum_ruled_out_in_this_setup'
 assert practical_status([-.02,.2],.1)=='inconclusive_at_declared_minimum'


def test_paired_contrast_and_simultaneous_intervals():
 a=[.2,.3,.4,.5,.6,.7,.8,.9];b=[.1]*8
 ordinary=contrast(a,b);simultaneous=contrast(a,b,.05/3)
 assert ordinary['mean_diff']==pytest.approx(.45)
 assert simultaneous['ci'][0]<ordinary['ci'][0]<ordinary['ci'][1]<simultaneous['ci'][1]
 with pytest.raises(ValueError):contrast([np.nan,0],[0,0])
 with pytest.raises(ValueError):contrast([0],[0])


def test_equal_search_budget_and_disjoint_stage_registry():
 p=json.loads(PFILE.read_text());grids=candidate_sets(p)
 assert len(grids)==5
 assert all(len(grid)==4 for grid in grids.values())
 stages=[set(p[k]) for k in ['selection_seeds','gate_audit_seeds','confirmation_seeds']]
 assert all(not a&b for i,a in enumerate(stages) for b in stages[i+1:])
 assert all(p['engineering_smoke_seed'] not in s for s in stages)


def test_gate_keeps_original_diagnostic_direction_and_threshold():
 spec={'mechanism':{'diagnostic':'spurious_rate','direction':'lower','min_effect':.1,'mode':'relative'}}
 tb=get_testbed('ued_gridworld')
 gate=_gate(tb,spec,[{'diagnostics':{'spurious_rate':.1}}]*4,[{'diagnostics':{'spurious_rate':.3}}]*4)
 assert gate['passed'] and gate['observed']==pytest.approx(.2)
 tuned=_gate(tb,spec,[{'diagnostics':{'spurious_rate':.1}}]*4,[{'diagnostics':{'spurious_rate':.12}}]*4)
 assert not tuned['passed']


def test_cache_identity_changes_with_seed_evaluator_and_method_parameters():
 basic={'seed':1,'evaluator':'v1','method':'abc','hp':{'lr':.1}}
 assert job_id(basic)==job_id(dict(reversed(list(basic.items()))))
 assert job_id(basic)!=job_id({**basic,'seed':2})
 assert job_id(basic)!=job_id({**basic,'evaluator':'v2'})
 assert job_id(basic)!=job_id({**basic,'hp':{'lr':.2}})
