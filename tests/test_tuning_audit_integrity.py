"""The saved-score audit must cover nested task metrics and metadata."""
from copy import deepcopy
import pytest
from scripts.audit_tuning_audit import compare_scores

def test_saved_score_check_accepts_metadata_and_rejects_changed_task_scores():
    score={'safety':.7,'per_task':[{'capability':.4}],
           'diagnostics':{'proxy':.2,'metadata':{'mode':'full','unused':None}}}
    compare_scores(deepcopy(score),score)
    changed=deepcopy(score)
    changed['per_task'][0]['capability']=.5
    with pytest.raises(AssertionError):compare_scores(changed,score)
    changed=deepcopy(score)
    changed['diagnostics']['proxy']=.3
    with pytest.raises(AssertionError):compare_scores(changed,score)
    changed=deepcopy(score)
    changed['diagnostics']['metadata']['mode']='toy'
    with pytest.raises(AssertionError):compare_scores(changed,score)
