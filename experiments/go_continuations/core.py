"""Native branch continuations with independent board checks and stone identities."""
import hashlib,json,os,re,time
from pathlib import Path
import numpy as np
from sgfmill import sgf
from .native import GTPClient,search_settings,parse_search_log,parse_native_board,_LineRulesState,_katago_policy_allows,gtp_to_canonical


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(array):return hashlib.sha256(np.asarray(array,dtype=np.int8).tobytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class OriginalStones:
    """Once removed, an original stone stays lost even if its point is reoccupied."""
    def __init__(self,points,color):self.initial=set(map(tuple,points));self.alive=set(self.initial);self.color=color
    def update(self,board):
        removed={p for p in self.alive if board[p]!=self.color};self.alive-=removed
        return sorted(removed)


def endpoint(steps,n_original,horizon):
    """Strict deadline: early termination with retained stones is unresolved.

    Loss of any original stone resolves all-original retention as failure. A
    replacement never restores it. Fraction bounds retain all early-stop cases.
    """
    eligible=[r for r in steps if r['ply']<=horizon]
    remaining=eligible[-1]['original_remaining'] if eligible else n_original
    reached=bool(eligible and eligible[-1]['ply']==horizon)
    binary=0 if remaining<n_original else 1 if reached else None
    fraction=remaining/n_original if reached or remaining==0 else None
    return dict(horizon=horizon,all_original_retained=binary,fraction_retained=fraction,
                fraction_lower=remaining/n_original if reached else 0.,fraction_upper=remaining/n_original,
                observed_plies=eligible[-1]['ply'] if eligible else 0)


def area_margin(board,komi):
    """Physical Chinese-area tally, no dead-stone removal/adjudication."""
    b=np.asarray(board);black=int((b==1).sum());white=int((b==-1).sum());empty=set(map(tuple,np.argwhere(b==0)))
    while empty:
        todo=[min(empty)];region=set();adj=set()
        while todo:
            p=todo.pop()
            if p in region:continue
            region.add(p)
            for q in [(p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)]:
                if min(q)<0 or max(q)>=len(b):continue
                if b[q]==0 and q not in region:todo.append(q)
                elif b[q]!=0:adj.add(int(b[q]))
        empty-=region
        if adj=={1}:black+=len(region)
        elif adj=={-1}:white+=len(region)
    return float(black-white-komi)


def parse_score(text):
    text=text.strip()
    if text in ['0','Jigo','jigo']:return 0.
    m=re.fullmatch(r'([BW])\+([0-9]+(?:\.[0-9]+)?)',text,re.I)
    return (1 if m[1].upper()=='B' else -1)*float(m[2]) if m else None


def source_prefix(path):
    game=sgf.Sgf_game.from_bytes(Path(path).read_bytes());moves=[];n=game.get_size();root=game.get_root()
    for node in game.get_main_sequence():
        if any(node.has_property(p) for p in ['AB','AW','AE','PL']):raise ValueError('Unsupported setup/turn edit')
        color,point=node.get_move()
        if color is not None:
            moves.append([color.upper(),'pass' if point is None else 'ABCDEFGHJKLMNOPQRSTUVWXYZ'[point[1]]+str(point[0]+1)])
    return moves,float(root.get('KM')),n


def export_sgf(case,steps,score,path):
    game=sgf.Sgf_game(size=19);root=game.get_root();root.set('KM',case['komi']);root.set('RU','Chinese')
    root.set('AP',('AlignmentCommons','Test6'));root.set('C','Bounded native continuation. Original target identities retained across mergers/captures. Source result not inherited. Native rule details are in protocol.json.')
    node=root
    for color,vertex in case['moves_gtp']+[[r['color'].upper(),r['move']] for r in steps]:
        node=node.new_child();point=gtp_to_canonical(vertex);node.set_move(color.lower(),None if point is None else (18-point[0],point[1]))
    if score is not None:root.set('RE','0' if score==0 else ('B+' if score>0 else 'W+')+str(abs(score)))
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(game.serialise())


def branch_settings(kind,visits,seed):
    settings=search_settings(kind,visits,seed)
    settings.update(fresh_process_per_query=False,fresh_process_per_branch=True,tree_reuse_within_branch=True)
    return settings

def continuation(case,arm,seed,locations,out,protocol_hash,settings_visits=128,max_plies=128):
    """One fresh process per arm/case/seed; same engine plays both colors.

    Native tree/cache reuse within a branch is retained and accounted for.
    No tree or cache is reused between paired arms. Root visits are checked on
    every completed search; new playout counts and wall time are also retained.
    """
    out=Path(out);key=f'{case["stem"]}-{arm}-s{seed}';dest=out/'branches'/f'{key}.json'
    if dest.exists():
        old=json.loads(dest.read_text());assert old['protocol_sha256']==protocol_hash
        return old  # Includes errors; no outcome-selective retries.
    kind=case['engine'];e=locations[kind];settings=branch_settings(kind,settings_visits,seed)
    logpath=out/'logs'/f'{key}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
    state=_LineRulesState(19);color='b';steps=[];transcript=[];terminal=None;error=None;score=None;first_loss=None;all_loss=None
    tracker=OriginalStones(case['target_stones'],case['target_color']);passes=0;t0=time.monotonic()
    env={k:v for k,v in os.environ.items() if not k.startswith('GO_CONCEPT_DUMP_') and k!='GCLAB_DUMP'}
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cfg=out/'configs'/f'{kind}-s{seed}.cfg';cfg.parent.mkdir(parents=True,exist_ok=True)
    if kind=='leela':command=[e['binary'],*settings['arguments'],'--weights',e['weights']]
    else:
        text=''.join(f'{k} = {v}\n' for k,v in settings['config'].items())
        if cfg.exists():assert cfg.read_text()==text
        else:cfg.write_text(text)
        command=[e['binary'],'gtp','-model',e['weights'],'-config',str(cfg)]
    native_rules=None;version=None
    try:
        with GTPClient(command,log_path=logpath,env=env,timeout=120) as client:
            def req(command):
                result=client.request(command);transcript.append(dict(command=command,response=result));return result
            def snapshot():
                client.log.flush();offset=logpath.stat().st_size;text=req('showboard');req('version');client.log.flush()
                if not text.strip():text=logpath.read_bytes()[offset:].decode(errors='replace')
                return parse_native_board(text),text
            version=req('version');req('boardsize 19');req('clear_board')
            if kind=='katago':req('kata-set-rules chinese');native_rules=req('kata-get-rules'); observed=json.loads(native_rules); assert observed['ko']=='SIMPLE' and observed['scoring']=='AREA' and not observed['suicide']
            req(f'komi {case["komi"]}');req('time_settings 0 86400 1')
            for prior_color,vertex in case['moves_gtp']:
                assert prior_color.lower()==color
                state.commit(state.propose(vertex,color,positional_superko=kind=='leela'))
                req(f'play {color} {vertex}');color='w' if color=='b' else 'b'
            board,text=snapshot();assert np.array_equal(board,state.array()) and np.array_equal(board,case['board'])
            assert (1 if color=='b' else -1)==case['to_play'];assert all(board[p]==tracker.color for p in tracker.initial)
            for ply in range(1,max_plies+1):
                before=state.array();stats=None;search_seconds=0.
                if ply==1:move=case['recorded_move'] if arm=='recorded' else case['defensive_move']
                else:
                    client.log.flush();offset=logpath.stat().st_size;st=time.monotonic();move=req(f'{settings["search_command"]} {color}').strip()
                    search_seconds=time.monotonic()-st;req('version');client.log.flush()
                    parsed=parse_search_log(kind,logpath.read_bytes()[offset:].decode(errors='replace'))
                    stats={k:v for k,v in parsed.items() if k not in ['printed_candidates','candidate_precision','pv']}
                    assert stats['root_visits']==settings_visits,('root visit budget',stats)
                proposal=state.propose(move,color,positional_superko=kind=='leela')
                # Native selected moves must pass independent rules. Forced moves
                # additionally receive KataGo's native legal-mask check before play.
                if kind=='katago':
                    allowed,_=_katago_policy_allows(req(f'kata-raw-nn {color} 0 0'),move,19);assert allowed
                if ply==1 or not settings['search_plays_move']:req(f'play {color} {move}')
                board,boardtext=snapshot();assert np.array_equal(board,proposal[2]),'Native transition mismatch'
                state.commit(proposal);lost=tracker.update(board)
                if lost and first_loss is None:first_loss=ply
                if not tracker.alive and all_loss is None:all_loss=ply
                passes=passes+1 if move.lower()=='pass' else 0
                steps.append(dict(ply=ply,color=color,move=move,forced=ply==1,stats=stats,search_seconds=search_seconds,
                                  board_before_sha256=digest(before),board_after_sha256=digest(board),native_board_text=boardtext,
                                  lost_original=[list(p) for p in lost],original_remaining=len(tracker.alive)))
                # Persist after every legal verified move, so errors retain the prefix.
                dump(out/'progress'/f'{key}.json',dict(protocol_sha256=protocol_hash,case=case['stem'],arm=arm,seed=seed,steps=steps))
                color='w' if color=='b' else 'b'
                if passes==2:
                    terminal=dict(ply=ply,reason='two consecutive passes',native_final_score=req('final_score'),physical_area_margin_black=area_margin(board,case['komi']))
                    native_score=parse_score(terminal['native_final_score']);terminal['score_matches_physical_area']=native_score is not None and abs(native_score-terminal['physical_area_margin_black'])<1e-6
                    terminal['score_scope']='Physical terminal-board area only; no perfect-play or independently adjudicated life/death claim'
                    if terminal['score_matches_physical_area']:score=native_score
                    break
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc));print('ERROR',key,error,flush=True)
    row=dict(id=key,case=case['stem'],engine=kind,arm=arm,seed=seed,protocol_sha256=protocol_hash,status='error' if error else 'complete',
             version=version,native_rules=native_rules,error=error,steps=steps,terminal=terminal,first_original_loss_ply=first_loss,all_original_lost_ply=all_loss,
             endpoints={str(h):endpoint(steps,len(tracker.initial),h) for h in [64,128]},final_original_remaining=len(tracker.alive),
             elapsed_seconds=time.monotonic()-t0,log_sha256=sha(logpath),transcript=transcript)
    export_sgf(case,steps,score,out/'sgfs'/f'{key}.sgf');row['sgf_sha256']=sha(out/'sgfs'/f'{key}.sgf')
    dump(dest,row);print('DONE',key,row['status'],len(steps),'plies',flush=True);return row
