"""Reproducible Test 5 runner: initialize, select, freeze, confirm, audit.

Each fit owns a process. No training/checkpoint is resumed across source versions.
All checkpoint evaluation happens after the fit ends and outside its CPU ledger.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import scipy
import torch
from .core import train
from testbeds.ued_gridworld import generate as G
from testbeds.ued_gridworld.testbed import UEDGridworld

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/regret-budget-audit-v1'
PFILE=HERE/'protocol.json'

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)
def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def signature():
    paths=list(HERE.glob('*.py'))+[PFILE,ROOT/'ideas/accel-levels/method.py',ROOT/'testbeds/ued_gridworld/generate.py',ROOT/'testbeds/ued_gridworld/testbed.py',ROOT/'testbeds/ued_gridworld/public.py',ROOT/'tests/test_regret_budget.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def verify():
    if load(OUT/'initial-freeze.json')['source_hashes'] != signature():
        raise RuntimeError('Source changed after pre-fit freeze')
def job_id(job): return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()[:24]
def hp(rate): return dict(lr=rate,ent=.01,p_replay=.7,edits=3,beta=.3,rho=.3,capacity=1000)
def make_job(arm, rate, seed, split, keys):
    p=load(PFILE)
    return dict(arm=arm, hp=hp(rate), seed=seed, split=split, keys=sorted(keys),
                step_budgets=p['step_budgets'], cpu_budget_s=p['cpu_budget_s'], source_hashes=signature())

def execute(job):
    verify()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    key=job_id(job);dest=OUT/'fits'/f'{key}.json'
    if dest.exists():
        cached=load(dest)
        assert cached['job']==job
        return cached
    print('START',key,job['split'],job['seed'],job['arm'],job['hp']['lr'],job['keys'],flush=True)
    perm=np.random.default_rng(G.SPLIT_OFFSET[job['split']]+1000*job['seed']).permutation(4)
    checkpoints=[n for n in job['step_budgets'] if f'steps_{n}' in job['keys']]
    models={}
    def save(name, policy, ledger, certs):
        path=OUT/'models'/f'{key}-{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        names=list(policy['state_dict'])
        np.savez_compressed(path,**{f'p{i}':policy['state_dict'][n] for i,n in enumerate(names)})
        info=dict(model=str(path.relative_to(OUT)),model_sha256=sha(path),state_names=names,arch=policy['arch'],ledger=dict(ledger))
        # A certificate digest per encountered training level, with count checked by audit.
        certpath=OUT/'certificates'/f'{key}-{name}.json'
        dump(certpath,certs)
        info['certificates']=str(certpath.relative_to(OUT));info['certificates_sha256']=sha(certpath)
        models[name]=policy
        return info
    saved,curve=train(job['arm'],job['hp'],job['seed'],'corner',perm,checkpoints,
                      job['cpu_budget_s'] if 'cpu' in job['keys'] else None,save)
    # Eval tables did not exist in Environment or learner during training.
    task=UEDGridworld().instance(job['split'],'full',job['seed']).private['tasks'][0]
    assert np.array_equal(perm,task['perm']) and task['family']=='corner'
    for name,info in saved.items():
        info['score']=UEDGridworld().score_one(models[name],task)
    result=dict(id=key,job=job,checkpoints=saved,curve=curve,completed_utc=now())
    dump(dest,result)
    print('DONE',key,[(n,round(i['ledger']['training_cpu_s'],1)) for n,i in saved.items()],flush=True)
    return result

def jobs_run(jobs):
    p=load(PFILE)
    with ProcessPoolExecutor(max_workers=p['parallel_fits'], max_tasks_per_child=1) as pool:
        fs=[pool.submit(execute,j) for j in jobs]
        for f in as_completed(fs): f.result()

def initialize():
    if (OUT/'initial-freeze.json').exists(): raise RuntimeError('Already frozen')
    p=load(PFILE)
    dump(OUT/'initial-freeze.json',dict(created_utc=now(),source_hashes=signature(),git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),protocol=p,hardware=dict(machine=platform.machine(),system=platform.system(),logical_cpus=os.cpu_count()),packages=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__)))

def select():
    verify();p=load(PFILE);keys=[f'steps_{n}' for n in p['step_budgets']]+['cpu']
    jobs=[make_job(a,r,s,'dev',keys) for a in p['arms'] for r in p['rates'] for s in p['selection_seeds']]
    jobs_run(jobs)
    chosen={}
    for arm in p['arms']:
        chosen[arm]={}
        for key in keys:
            table=[]
            for rate in p['rates']:
                rows=[load(OUT/'fits'/f'{job_id(make_job(arm,rate,seed,"dev",keys))}.json') for seed in p['selection_seeds']]
                table.append(dict(rate=rate,mean_safety=float(np.mean([r['checkpoints'][key]['score']['safety'] for r in rows])),fits=[r['id'] for r in rows]))
            best=max(range(len(table)),key=lambda i:table[i]['mean_safety'])
            chosen[arm][key]=dict(rate=table[best]['rate'],table=table)
    dump(OUT/'selection.json',dict(selected=chosen,created_utc=now()))

def freeze():
    verify()
    if (OUT/'selection-freeze.json').exists():raise RuntimeError('Selection already frozen')
    dump(OUT/'selection-freeze.json',dict(created_utc=now(),selection_sha256=sha(OUT/'selection.json'),source_hashes=signature()))

def confirm():
    verify();p=load(PFILE);f=load(OUT/'selection-freeze.json');assert f['selection_sha256']==sha(OUT/'selection.json')
    chosen=load(OUT/'selection.json')['selected'];jobs=[]
    for arm, entries in chosen.items():
        for rate in sorted({i['rate'] for i in entries.values()}):
            keys=[k for k,i in entries.items() if i['rate']==rate]
            jobs.extend(make_job(arm,rate,seed,'test',keys) for seed in p['confirmation_seeds'])
    dump(OUT/'confirmation-jobs.json',dict(jobs=jobs,created_utc=now()))
    jobs_run(jobs)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initialize','select','freeze','confirm']);args=parser.parse_args()
    globals()[args.command]()
