import numpy as np
import pytest
from experiments.go_continuations.core import OriginalStones,endpoint,area_margin,parse_score,source_prefix
from experiments.go_continuations.native import _LineRulesState,parse_native_board


def test_original_identity_not_recreated_and_merges_not_counted():
    board=np.zeros((3,3),np.int8);board[0,0]=board[0,1]=1
    t=OriginalStones([(0,0)],1);assert t.update(board)==[] and len(t.alive)==1
    board[0,0]=0;assert t.update(board)==[(0,0)]
    board[0,0]=1;assert t.update(board)==[] and not t.alive


def test_strict_deadline_retains_unknown_and_known_capture():
    assert endpoint([dict(ply=2,original_remaining=5)],5,64)['all_original_retained'] is None
    assert endpoint([dict(ply=2,original_remaining=0)],5,64)['all_original_retained']==0
    assert endpoint([dict(ply=64,original_remaining=5)],5,64)['all_original_retained']==1
    r=endpoint([dict(ply=64,original_remaining=4)],5,64);assert r['all_original_retained']==0 and r['fraction_retained']==.8


def test_area_and_score_use_fixed_komi_and_no_dead_removal():
    b=np.zeros((3,3),np.int8);b[0,0]=1
    assert area_margin(b,6.5)==2.5
    b[2,2]=-1;assert area_margin(b,6.5)==-6.5
    assert parse_score('W+6.5')==-6.5 and parse_score('B+Resign') is None


def test_suicide_simple_ko_and_full_prefix_repetition():
    s=_LineRulesState(3)
    # White surrounds center; legal board setup via replay of non-touching turns.
    for move,color in [('B3','w'),('A2','w'),('C2','w'),('B1','w')]:s.commit(s.propose(move,color,positional_superko=True))
    with pytest.raises(ValueError,match='suicide'):s.propose('B2','b',positional_superko=True)
    state=_LineRulesState(3);proposal=state.propose('A1','b',positional_superko=True);state.seen.add(proposal[2].tobytes())
    with pytest.raises(ValueError,match='repetition'):state.propose('A1','b',positional_superko=True)
    state.commit(state.propose('pass','b',positional_superko=True))


def test_all_case_prefixes_independently_replay_and_target_matches():
    import json
    from pathlib import Path
    from experiments.go_continuations.run import HERE
    for c in json.loads((HERE/'inputs/cases.json').read_text()):
        moves,komi,n=source_prefix(HERE/'inputs'/c['source_file']);assert moves[:c['move_index']]==c['moves_gtp'] and komi==c['komi']
        state=_LineRulesState(n)
        for color,move in c['moves_gtp']:state.commit(state.propose(move,color.lower(),positional_superko=c['engine']=='leela'))
        np.testing.assert_array_equal(state.array(),c['board'])
        assert all(state.array()[tuple(p)]==c['target_color'] for p in c['target_stones'])
        # Both branches legal with full source history before any continuation.
        for name in ['recorded_move','defensive_move']:state.propose(c[name],'b' if c['to_play']==1 else 'w',positional_superko=c['engine']=='leela')


def test_sgf_exports_actual_branch_without_inherited_result(tmp_path):
    import json
    from sgfmill import sgf
    from experiments.go_continuations.run import HERE
    from experiments.go_continuations.core import export_sgf
    c=json.loads((HERE/'inputs/cases.json').read_text())[0];p=tmp_path/'branch.sgf'
    export_sgf(c,[dict(color='b',move=c['defensive_move'])],None,p)
    game=sgf.Sgf_game.from_bytes(p.read_bytes());assert not game.get_root().has_property('RE')
    moves,komi,n=source_prefix(p);assert len(moves)==len(c['moves_gtp'])+1 and moves[-1]==['B',c['defensive_move']]
    export_sgf(c,[],6.5,p);assert sgf.Sgf_game.from_bytes(p.read_bytes()).get_root().get('RE')=='B+6.5'
