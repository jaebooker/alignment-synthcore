"""Initialize and execute a prospectively frozen native Go continuation audit."""
import argparse,datetime,json,platform,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from .core import continuation,dump,sha,source_prefix,branch_settings
from .native import search_settings
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/go-continuation-audit-v1'
load=lambda p:json.loads(Path(p).read_text())
def signature():
    files=list(HERE.glob('*.py'))+list((HERE/'inputs').glob('*'))+[HERE/'protocol.json',HERE/'input-provenance.json',ROOT/'tests/test_go_continuations.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
def verify(locations):
    f=load(OUT/'initial-freeze.json');assert f['source_hashes']==signature(),'Frozen source changed'
    assert f['protocol_sha256']==sha(HERE/'protocol.json')
    for kind,paths in locations.items():
        for key,path in paths.items():assert sha(path)==f['engines'][kind][key+'_sha256'],kind+' '+key
    return f

def initialize(locations):
    if (OUT/'initial-freeze.json').exists():raise RuntimeError('Refuse to overwrite freeze')
    p=load(HERE/'protocol.json');cases=load(HERE/'inputs/cases.json');provenance=load(HERE/'input-provenance.json')
    assert len(cases)==4 and len({c['source_file'] for c in cases})==2
    for c in cases:
        path=HERE/'inputs'/c['source_file'];assert sha(path)==c['game_id'];moves,komi,n=source_prefix(path)
        assert moves[:c['move_index']]==c['moves_gtp'] and n==19 and komi==c['komi']
    for kind,paths in locations.items():
        for key,path in paths.items():assert sha(path)==provenance['engines'][kind][key+'_sha256']
    settings={k:{str(s):branch_settings(k,p['visits'],s) for s in p['seeds']} for k in provenance['engines']}
    for kind,entries in settings.items():
        if kind=='katago':
            for seed,item in entries.items():
                cfg=OUT/'configs'/f'{kind}-s{seed}.cfg';cfg.parent.mkdir(parents=True,exist_ok=True);cfg.write_text(''.join(f'{k} = {v}\n' for k,v in item['config'].items()))
    dump(OUT/'initial-freeze.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=p,protocol_sha256=sha(HERE/'protocol.json'),source_hashes=signature(),engines=provenance['engines'],settings=settings,runtime=dict(python=platform.python_version(),numpy=np.__version__)))

def worker(args):
    case,arm,seed,locations=args;f=verify(locations);p=f['protocol']
    return continuation(case,arm,seed,locations,OUT,sha(OUT/'initial-freeze.json'),p['visits'],p['max_plies'])

def execute(locations):
    f=verify(locations);p=f['protocol'];cases=load(HERE/'inputs/cases.json')
    jobs=[(case,arm,seed,locations) for case in cases for seed in p['seeds'] for arm in p['arms']]
    with ProcessPoolExecutor(max_workers=p['parallel_branches']) as pool:
        fs=[pool.submit(worker,j) for j in jobs]
        for x in as_completed(fs):x.result()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initialize','execute']);parser.add_argument('--locations',required=True);a=parser.parse_args()
    globals()[a.command](load(a.locations))
