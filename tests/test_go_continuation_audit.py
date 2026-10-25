"""Cross-check independent audit transitions against sgfmill on new fixtures."""
import importlib.util
from pathlib import Path
import numpy as np
from experiments.go_continuations.native import _LineRulesState
spec=importlib.util.spec_from_file_location('go_audit',Path(__file__).resolve().parents[1]/'scripts/audit_go_continuations.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def test_independent_random_replay_agrees_with_sgfmill():
    rng=np.random.default_rng(77019)
    for seed in range(3):
        state=_LineRulesState(5);a=np.zeros((5,5),np.int8);ko=None;seen={a.tobytes()}
        for turn in range(150):
            p=None if rng.random()<.1 else tuple(int(x) for x in rng.integers(0,5,2));color=1 if turn%2==0 else -1
            vertex='pass' if p is None else audit.COLUMNS[p[1]]+str(5-p[0])
            result=audit.move(a,p,color,ko,seen)
            try:proposal=state.propose(vertex,'b' if color==1 else 'w',positional_superko=True)
            except ValueError:
                assert result is None;continue
            assert result is not None;np.testing.assert_array_equal(proposal[2],result[0]);state.commit(proposal);a,ko=result;seen.add(a.tobytes())


def test_native_board_first_column_last_move_marker():
    # Leela can print the last-move parenthesis adjacent to the row number.
    text='3(X) . . 3\n2 . O . 2\n1 . . . 1\n'
    b=audit.native_board(text,n=3)
    assert b[0,0]==1 and b[1,1]==-1 and b.shape==(3,3)
