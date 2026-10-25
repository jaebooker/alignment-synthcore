"""Separate NumPy transition/identity/score audit; no production move helpers."""
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
from sgfmill import sgf
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'experiments/go_continuations';OUT=ROOT/'results/go-continuation-audit-v1'
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
COLUMNS='ABCDEFGHJKLMNOPQRSTUVWXYZ'
def point(vertex,n=19):return None if vertex.lower()=='pass' else (n-int(vertex[1:]),COLUMNS.index(vertex[0].upper()))
def adjacent(p,n):return [q for q in [(p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1)] if min(q)>=0 and max(q)<n]
def group(a,p):
    color=a[p];todo=[p];chain=set();libs=set()
    while todo:
        q=todo.pop()
        if q in chain:continue
        chain.add(q)
        for v in adjacent(q,len(a)):
            if a[v]==0:libs.add(v)
            elif a[v]==color and v not in chain:todo.append(v)
    return chain,libs

def move(a,p,color,ko,seen=None):
    if p is None:return a.copy(),None
    if a[p]!=0 or p==ko:return None
    b=a.copy();b[p]=color;removed=set()
    for q in adjacent(p,len(a)):
        if b[q]==-color:
            stones,libs=group(b,q)
            if not libs:
                removed|=stones
                for v in stones:b[v]=0
    stones,libs=group(b,p)
    if not libs or seen is not None and b.tobytes() in seen:return None
    return b,next(iter(removed)) if len(removed)==len(stones)==len(libs)==1 else None

def native_board(text,n=19):
    rows={}
    for line in text.splitlines():
        m=re.match(r'^\s*(\d{1,2})(?!\d)(.*)',line)
        if m and 1<=int(m[1])<=n:
            cells=re.findall(r'[XO.+]',m[2])
            if len(cells)==n:rows[n-int(m[1])]=[1 if c=='X' else -1 if c=='O' else 0 for c in cells]
    assert len(rows)==n
    return np.array([rows[r] for r in range(n)],np.int8)

def physical_area(b,komi):
    owner=b.copy();visited=set()
    for y,x in np.argwhere(b==0):
        start=(int(y),int(x))
        if start in visited:continue
        todo=[start];empty=set();border=set()
        while todo:
            p=todo.pop()
            if p in empty:continue
            empty.add(p)
            for q in adjacent(p,len(b)):
                if b[q]:border.add(int(b[q]))
                elif q not in empty:todo.append(q)
        visited|=empty
        if len(border)==1:
            color=next(iter(border))
            for p in empty:owner[p]=color
    return float(np.count_nonzero(owner==1)-np.count_nonzero(owner==-1)-komi)

def main():
    f=load(OUT/'initial-freeze.json');p=f['protocol'];cases=load(HERE/'inputs/cases.json')
    for rel,digest in f['source_hashes'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(HERE/'protocol.json')==f['protocol_sha256']
    expected={f'{c["stem"]}-{arm}-s{seed}' for c in cases for arm in p['arms'] for seed in p['seeds']}
    files={x.stem for x in (OUT/'branches').glob('*.json')};assert files==expected,('missing/extra',expected^files)
    count=0;captures=0;certs=[];records=[]
    for c in cases:
        game=sgf.Sgf_game.from_bytes((HERE/'inputs'/c['source_file']).read_bytes());assert sha(HERE/'inputs'/c['source_file'])==c['game_id']
        parsed=[]
        for node in game.get_main_sequence():
            color,pos=node.get_move()
            if color is not None:parsed.append([color.upper(),'pass' if pos is None else COLUMNS[pos[1]]+str(pos[0]+1)])
        assert parsed[:c['move_index']]==c['moves_gtp']
        a=np.zeros((19,19),np.int8);ko=None;seen={a.tobytes()}
        for color,vertex in c['moves_gtp']:
            result=move(a,point(vertex),1 if color=='B' else -1,ko,seen if c['engine']=='leela' else None);assert result is not None
            a,ko=result;seen.add(a.tobytes())
        assert np.array_equal(a,c['board']);target=set(map(tuple,c['target_stones']));assert group(a,min(target))[0]==target
        # Reconfirm both archived immediate tactical statements, separately from outcomes.
        tactical={}
        for arm,vertex in [('recorded',c['recorded_move']),('defense',c['defensive_move'])]:
            result=move(a,point(vertex),c['target_color'],ko,seen if c['engine']=='leela' else None);assert result is not None
            b,k=result;bs=seen|{b.tobytes()};worst=0
            for q in [tuple(q) for q in np.argwhere(b==0)]+[None]:
                reply=move(b,q,-c['target_color'],k,bs if c['engine']=='leela' else None)
                if reply is not None:worst=max(worst,sum(reply[0][v]!=c['target_color'] for v in target))
            tactical[arm]=int(worst)
        assert tactical['recorded']>=10 and tactical['defense']==0;certs.append(dict(case=c['stem'],worst_one_reply_original_loss=tactical))
        for seed in p['seeds']:
            for arm in p['arms']:
                key=f'{c["stem"]}-{arm}-s{seed}';path=OUT/'branches'/f'{key}.json';r=load(path)
                assert (r['case'],r['arm'],r['seed'],r['engine'])==(c['stem'],arm,seed,c['engine'])
                assert r['protocol_sha256']==sha(OUT/'initial-freeze.json')
                assert sha(OUT/'logs'/f'{key}.log')==r['log_sha256']
                assert sha(OUT/'sgfs'/f'{key}.sgf')==r['sgf_sha256']
                current=a.copy();current_ko=ko;visited=set(seen);alive=set(target);first_loss=None;all_loss=None;passes=0
                counts=[]
                for index,step in enumerate(r['steps']):
                    ply=index+1;assert step['ply']==ply
                    color=c['target_color']*((-1)**index);assert step['color']==('b' if color==1 else 'w')
                    assert hashlib.sha256(current.tobytes()).hexdigest()==step['board_before_sha256']
                    result=move(current,point(step['move']),color,current_ko,visited if c['engine']=='leela' else None);assert result is not None
                    current,current_ko=result;visited.add(current.tobytes())
                    assert hashlib.sha256(current.tobytes()).hexdigest()==step['board_after_sha256']
                    np.testing.assert_array_equal(native_board(step['native_board_text']),current)
                    lost={v for v in alive if current[v]!=c['target_color']};alive-=lost
                    assert sorted(lost)==sorted(map(tuple,step['lost_original'])) and len(alive)==step['original_remaining']
                    if lost and first_loss is None:first_loss=ply
                    if not alive and all_loss is None:all_loss=ply
                    if index==0:assert step['forced'] and step['move']==(c['recorded_move'] if arm=='recorded' else c['defensive_move']) and step['stats'] is None
                    else:assert not step['forced'] and step['stats']['root_visits']==p['visits']
                    counts.append(len(alive));passes=passes+1 if step['move'].lower()=='pass' else 0
                    count+=1;captures+=len(lost)
                assert r['first_original_loss_ply']==first_loss and r['all_original_lost_ply']==all_loss and r['final_original_remaining']==len(alive)
                assert len(r['steps'])<=p['max_plies']
                searches=[t for t in r['transcript'] if t['command'].startswith(('genmove ','kata-search '))]
                assert len(searches)==max(0,len(r['steps'])-1) or r['status']=='error'
                for step,t in zip(r['steps'][1:],searches):assert t['command'].split()[-1]==step['color'] and t['response'].strip()==step['move']
                log=(OUT/'logs'/f'{key}.log').read_text(errors='replace')
                visits=[int(v) for v in re.findall(r'(?m)^(\d+) visits,',log)] if c['engine']=='leela' else [int(v) for v in re.findall(r'Root visits:\s*(\d+)',log)]
                if r['status']=='complete':assert len(visits)==len(searches) and all(v==p['visits'] for v in visits)
                for h in [64,128]:
                    eligible=counts[:h];left=eligible[-1] if eligible else len(target);reached=len(eligible)==h
                    binary=0 if left<len(target) else 1 if reached else None
                    fraction=left/len(target) if reached or left==0 else None;e=r['endpoints'][str(h)]
                    assert e['all_original_retained']==binary and e['fraction_retained']==fraction
                    assert e['fraction_lower']==(left/len(target) if reached else 0.) and e['fraction_upper']==left/len(target)
                g=sgf.Sgf_game.from_bytes((OUT/'sgfs'/f'{key}.sgf').read_bytes());line=[]
                for node in g.get_main_sequence():
                    col,pos=node.get_move()
                    if col is not None:line.append([col.upper(),'pass' if pos is None else COLUMNS[pos[1]]+str(pos[0]+1)])
                assert line==c['moves_gtp']+[[s['color'].upper(),s['move']] for s in r['steps']]
                if r['terminal']:
                    assert passes==2 and r['terminal']['ply']==len(r['steps'])
                    assert physical_area(current,c['komi'])==r['terminal']['physical_area_margin_black']
                    if r['terminal']['score_matches_physical_area']:assert g.get_root().has_property('RE')
                    else:assert not g.get_root().has_property('RE')
                else:assert not g.get_root().has_property('RE')
                records.append(dict(id=key,status=r['status'],plies=len(counts),record_sha256=sha(path)))
    result=dict(status='passed',branches=len(records),verified_continuation_plies=count,original_stone_identities_lost_across_branches=captures,tactical_checks=certs,records=records,
                limits='Independent implementation audit of saved native observations; not independent scientific replication or perfect-play proof.')
    (OUT/'integrity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','tactical_checks']},indent=2))
if __name__=='__main__':main()
