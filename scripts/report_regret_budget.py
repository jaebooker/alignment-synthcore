"""Analysis and static scientific figure; never used for training or selection."""
from pathlib import Path
import json,sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.regret_budget.run import job_id
OUT=ROOT/'results/regret-budget-audit-v1';load=lambda p:json.loads(p.read_text())

def interval(values):
    a=np.asarray(values,float);m=float(a.mean());se=float(stats.sem(a));h=float(stats.t.ppf(.975,len(a)-1)*se)
    return dict(mean=m,lower=m-h,upper=m+h,n=len(a),unit='paired seed-group proportion difference')

def main():
    p=load(ROOT/'experiments/regret_budget/protocol.json');audit=load(OUT/'integrity-audit.json');assert audit['status']=='passed'
    selected=load(OUT/'selection.json')['selected'];jobs=load(OUT/'confirmation-jobs.json')['jobs'];rows=[load(OUT/'fits'/f'{job_id(j)}.json') for j in jobs]
    data={a:{} for a in p['arms']};cost={a:{} for a in p['arms']}
    for a in p['arms']:
        for key in selected[a]:
            rr=sorted([r for r in rows if r['job']['arm']==a and key in r['checkpoints']],key=lambda r:r['job']['seed'])
            assert [r['job']['seed'] for r in rr]==p['confirmation_seeds']
            data[a][key]=[dict(seed=r['job']['seed'],**r['checkpoints'][key]['score']) for r in rr]
            cost[a][key]=[dict(seed=r['job']['seed'],**r['checkpoints'][key]['ledger']) for r in rr]
    contrasts=[]
    for key in selected['dr']:
        for a,b in [('oracle','practical'),('practical','dr'),('random','practical')]:
            for metric in ['safety','capability','spurious']:
                delta=[x[metric]-y[metric] for x,y in zip(data[a][key],data[b][key])]
                r=dict(budget=key,arm=a,reference=b,metric=metric,**interval(delta))
                r['primary']=key=='steps_1000000' and a=='oracle' and b=='practical' and metric=='safety'
                contrasts.append(r)
    primary=next(r for r in contrasts if r['primary']);cap=float(np.mean([r['capability'] for r in data['oracle']['steps_1000000']]))
    supported=primary['lower']>.1 and cap>=.8
    verdict='supported' if supported else ('declared effect ruled out locally' if primary['upper']<.1 else 'inconclusive at declared effect')
    means={a:{k:{m:float(np.mean([r[m] for r in rr])) for m in ['safety','capability','spurious']} for k,rr in data[a].items()} for a in p['arms']}
    summary=dict(study_id=p['study_id'],scope=p['scope'],verdict=verdict,primary=primary,oracle_iid_capability=cap,primary_supported=supported,
                 means=means,contrasts=contrasts,confirmation_seed_metrics=data,training_costs=cost,selected_rates={a:{k:r['rate'] for k,r in v.items()} for a,v in selected.items()},
                 fits_total=audit['fits'],policy_checkpoints=audit['policy_checkpoints_rescored'],limits=['Corner family only; 3M and lamp unrun.','Two development seeds, two rates; tuning is limited and development outcomes noisy.','The 2025 paper includes 200M-step corner experiments; this local 1M-step ceiling is substantially smaller and uses different mechanics.','Eight seed groups yield wide intervals; no episode-level pseudoreplication.','Oracle has privileged geometric information, and corrects both max-ever underestimation and cross-start reference bias.','Only the attainable-return reference is exact; sampled policy returns and recent-window curriculum scores remain noisy.','In-process PPO ablation; no new official benchmark verdict.','180 CPU-second checkpoint is local resource matching, with recorded rollout-boundary overshoot. No FLOP/energy claim.','Secondary comparisons and budget changes are descriptive. One CPU cap cannot establish catch-up across a CPU-budget curve.','Entering the cue zone is not exclusive decoy pursuit; successful trajectories can enter it.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    labels={'dr':'Domain randomization','practical':'Max-latest ACCEL','random':'Random replay/edit','oracle':'Oracle ACCEL'}
    keys=['steps_300000','steps_1000000','cpu'];titles=['301,056 environment steps','1,001,472 environment steps','180 training CPU seconds + boundary']
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    for ax,key,title in zip(axes,keys,titles):
        for idx,a in enumerate(p['arms']):
            values=[r['safety']*100 for r in data[a][key]]
            ax.scatter(np.full(8,idx)+np.linspace(-.13,.13,8),values,s=20,alpha=.65)
            ax.plot([idx-.23,idx+.23],[np.mean(values)]*2,color='black',lw=2)
        ax.set_xticks(range(4),[labels[a] for a in p['arms']],rotation=32,ha='right')
        ax.set_title(title,fontsize=11);ax.set_ylim(-3,103);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Held-out intended-goal success (%)')
    fig.suptitle('Test 5: regret estimator and budget audit — corner task only',fontsize=14)
    fig.text(.01,.01,'Dots: eight reserved seed groups; black bars: means. Rates tuned separately per arm and budget. Oracle is privileged.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(OUT/'regret-budget.png',dpi=160);plt.close(fig)
    lines=['# Test 5 — regret estimator and training budget audit','',f'**Primary verdict: {verdict}.** Oracle minus max-latest at 1M requested steps: {primary["mean"]*100:.1f} percentage points (95% paired-seed t interval {primary["lower"]*100:.1f} to {primary["upper"]*100:.1f}). Oracle IID capability: {cap*100:.1f}%. The prespecified rule requires the interval lower bound above 10 points and IID capability at least 80%.','',
    'Four locally implemented curricula use the same student CNN/PPO: domain randomization, existing max-latest ACCEL, uniform random replay/editing, and privileged exact-start oracle regret. Two learning rates were tuned on two separate development seeds, independently at each budget. Confirmation uses eight reserved seeds. This is a new corner-family ablation of the local implementation, not a replication or a replacement for Test 2.','',
    '| Budget | Arm | Goal success | IID capability | Cue entry |','|---|---|---:|---:|---:|']
    for key,title in zip(keys,titles):
        for a in p['arms']:
            v=means[a][key];lines.append(f'| {title} | {labels[a]} | {v["safety"]*100:.1f}% | {v["capability"]*100:.1f}% | {v["spurious"]*100:.1f}% |')
    metric=lambda a,k,m='safety':means[a][k][m]*100
    early,long,cpu='steps_300000','steps_1000000','cpu'
    lines+=['','## Interpretation','',
    f"The primary oracle-replacement decision is {verdict}. At1M requested steps, random replay/editing reaches {metric('random',long):.1f}% intended-goal success, compared with {metric('practical',long):.1f}% for max-latest and {metric('oracle',long):.1f}% for oracle. At matched CPU, random reaches {metric('random',cpu):.1f}%, oracle {metric('oracle',cpu):.1f}%, and max-latest {metric('practical',cpu):.1f}%. These are descriptive secondary comparisons. The random control is essential when assessing whether privileged information is necessary; random-versus-oracle superiority or equivalence was not a registered inferential contrast.", '',
    f"Selected max-latest performance is {metric('practical',early):.1f}% at 300k and {metric('practical',long):.1f}% at 1M steps, with independently tuned DR at {metric('dr',long):.1f}% at 1M. Selected rates may differ between budgets, so this does not isolate extra training of one fixed configuration. A single CPU budget cannot establish how the methods converge with larger total compute. The local ranking/eviction scheme and remaining budget explanations must both be considered.", '',
    f"Oracle IID capability is {metric('oracle',long,'capability'):.1f}% at the primary checkpoint, compared with the 80% gate. At early/matched-CPU budgets its capability is {metric('oracle',early,'capability'):.1f}%/{metric('oracle',cpu,'capability'):.1f}%, versus {metric('dr',early,'capability'):.1f}%/{metric('dr',cpu,'capability'):.1f}% for DR. Oracle-versus-DR superiority was not the registered primary contrast.", '']
    lines+=['','## Secondary comparisons (descriptive)','','Positive differences favor the first named arm for intended-goal success. These intervals are not multiplicity-adjusted and do not replace the primary test.','','| Budget | Comparison | Difference, percentage points | 95% paired-seed interval |','|---|---|---:|---:|']
    for r in contrasts:
        if r['metric']=='safety':
            lines.append(f"| {r['budget']} | {labels[r['arm']]} minus {labels[r['reference']]} | {r['mean']*100:.1f} | {r['lower']*100:.1f} to {r['upper']*100:.1f} |")
    lines+=['','Learning rates may differ between budget checkpoints because they were selected independently. An early-to-late difference therefore describes selected performance at each budget; it does not isolate extra training of one fixed policy configuration. One matched CPU checkpoint cannot establish recovery across a CPU-budget curve.','','| Arm at CPU budget | Mean actual CPU seconds | Mean env steps | Mean scoring steps | Mean oracle CPU seconds |','|---|---:|---:|---:|---:|']
    for a in p['arms']:
        cc=cost[a]['cpu'];avg=lambda k:float(np.mean([r[k] for r in cc]))
        lines.append(f"| {labels[a]} | {avg('training_cpu_s'):.2f} | {avg('steps'):.0f} | {avg('scoring_steps'):.0f} | {avg('oracle_cpu_s'):.2f} |")
    lines+=['','## Instrumentation correction','','Before development selection or any confirmation run, code inspection found that checkpoint scoring retained NumPy views backed by live network parameters. Saved weights were correct, but early metrics could use later weights. The entire interrupted attempt was excluded and archived separately; every valid study fit was restarted after an immutable on-disk snapshot loader and a mutation regression test were added. Scientific design, seeds, budgets and outcomes were unchanged. The old and corrected source freezes are preserved in instrumentation-correction.json. No invalid metrics were used for selection or interpretation.','']
    lines+=['','## Accounting and oracle certification','','Training CPU seconds include both training threads, environment draws, edits, ranking, score-only rollouts and finite-horizon oracle certification. Initialization is included; imports, snapshot serialization and held-out evaluation are excluded. Each checkpoint is saved after a complete 2,048-step PPO rollout. The 180-second checkpoint overshoots by at most '+f'{audit["maximum_cpu_boundary_overshoot_s"]:.2f} seconds in this run. This matches a local CPU resource, not FLOPs or energy.','',
    'Each oracle label is the exact finite-horizon attainable return from the actual training start minus the observed return. Integer Bellman dynamic programming is checked against independent shortest-path values for every permissible start on each queried level. Distances exceeding the 19-step horizon have value zero. Labels affect only curriculum scoring, eviction and ranking. Evaluation tables are generated only after training. The oracle changes both the max-ever reference and cross-start bias, so it does not isolate one estimator defect.','',
    f'The independent geometry-based audit re-scored all {audit["policy_checkpoints_rescored"]} saved policies from {audit["fits"]} fresh fits, checked source freezes, model hashes, all job counts, tuning winners, seed separation and budget boundaries. Maximum metric discrepancy: {audit["maximum_metric_difference"]}. This is engineering verification, not independent scientific reproduction.','',
    '## Literature scope','','[Abdel Sadek et al., Mitigating Goal Misgeneralization via Minimax Regret](https://arxiv.org/html/2507.03068v1) studies max-latest and graph-based oracle-latest estimators. Its reported corner experiments include 200M training steps (Appendix J), compared with this local 1M-step ceiling. These environments, curriculum details and evaluation protocols differ. This audit examines the local implementation, not the paper’s theoretical guarantees or full empirical claim.','','## Limits','']+['- '+v for v in summary['limits']]+['','## Reproduce','','```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run initialize','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run select','python -m experiments.regret_budget.run freeze','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.regret_budget.run confirm','python scripts/audit_regret_budget.py','python scripts/report_regret_budget.py','```','',
    'Start with an absent results/regret-budget-audit-v1 directory for a fresh independent run. CPU checkpoint weights and realized step counts may vary with hardware and process scheduling. Step checkpoints remain deterministic within the documented numerical runtime. All fitted weights and per-level certificate digest tables are in the complete local complete-evidence.zip. Source/public records archives omit these large artifacts; per-file SHA256 hashes and aggregate certification counts remain in the fit records. Independent re-scoring requires the complete local bundle.','',
    'Related method: [Parker-Holder et al., Evolving Curricula with Regret-Based Environment Design](https://arxiv.org/abs/2203.01302). The published method motivates this audit; its broader claims are not tested here.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(verdict=verdict,primary=primary,means=means),indent=2))
if __name__=='__main__':main()
