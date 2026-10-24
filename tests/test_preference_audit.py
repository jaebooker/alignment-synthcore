from copy import deepcopy
import numpy as np
import pytest
pytest.importorskip('jax',reason='Test 3 uses its isolated optional JAX runtime')
pytest.importorskip('optax',reason='Test 3 uses its isolated optional JAX runtime')
from experiments.preference_audit import batteries as B
from experiments.preference_audit.run import contrast,equivalence,load,PFILE

def test_transform_preserves_distances_and_identifies_unobserved_task_change():
    assert np.array_equal(B.reflect(B.C),B.C)
    assert np.array_equal(B.rotate(B.C),[7,1])
    batteries=B.make_batteries([91001,91002,91003,91004])
    for b in batteries.values():B.valid(b)
    matched=B.matched_reflection(batteries['original'])
    assert np.array_equal(matched['delta'],batteries['original']['delta'])

def test_known_optimal_choices_match_actual_endpoint_rollouts():
    batteries=B.make_batteries([92001,92002,92003,92004]);p=load(PFILE)
    results=B.calibrate(batteries,p)
    assert all(r['passed'] for r in results.values())
    for b in batteries.values():
        assert (B.control_rollout(b,'gem_seeker')==1).all()
        assert (B.control_rollout(b,'proxy_seeker')==2).all()

def test_optimal_control_uses_equal_length_route_avoiding_unwanted_terminal():
    # Up-first would hit the proxy at (1,7); left then up reaches the gem.
    b=B.pack(np.array([[2,7]]),np.array([[1,6]]),B.C,seed=94001)
    assert B.control_rollout(b,'optimal',.9).tolist()==[1]
    assert B.control_rollout(b,'proxy_seeker').tolist()==[2]

def test_neither_stays_in_primary_denominator_and_extremes_are_censored():
    b=B.make_batteries([93001,93002,93003,93004])['original']
    out=np.zeros(len(b['agent']),int);equal=np.flatnonzero(b['delta']==0);out[equal[:20]]=1
    m=B.metrics(out,b)
    assert m['equal_distance_gem_rate']==.5 and m['equal_distance_neither_rate']==.5
    assert B.metrics(np.ones(len(out),int),b)['weight_status']=='right_censored_at_0.90'
    assert B.metrics(np.zeros(len(out),int),b)['weight_status']=='unidentified'

def test_paired_equivalence_is_not_a_non_significance_test():
    assert equivalence([-.09,.09])=='rate_equivalence_supported'
    assert equivalence([.11,.30])=='material_rate_difference_supported'
    assert equivalence([-.2,.2])=='inconclusive_at_equivalence_margin'
    a=contrast([.2,.3,.4,.5,.6,.7,.8,.9],[.1]*8)
    assert a['mean_difference']==pytest.approx(.45) and a['alpha']==.0125

def test_training_and_evaluation_seeds_are_reserved_and_budget_matches_archive():
    p=load(PFILE);training=set(p['training']['seeds']);dev=set(p['battery_seeds']['development']);test=set(p['battery_seeds']['confirmation'])
    assert not training&dev and not training&test and not dev&test
    assert p['engineering']['seed'] not in training|dev|test
    assert p['training']['updates']*64*32==p['training']['environment_steps_per_policy']
