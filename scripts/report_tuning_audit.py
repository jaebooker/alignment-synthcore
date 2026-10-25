"""Publish Test 2's saved audit; fitting and interval rules are defined in the frozen runner."""
from pathlib import Path
import json,hashlib
from decimal import Decimal,ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/post-tuning-mechanism-audit-v1'
load=lambda p:json.loads(p.read_text())
def pct(v):return str(Decimal(str(100*v)).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))+'%'
def points(v):return str(Decimal(str(100*v)).quantize(Decimal('.01' if 0<abs(100*v)<.1 else '.1'),rounding=ROUND_HALF_UP))
def main():
 p=load(ROOT/'experiments/tuning_audit/protocol.json');s=load(OUT/'summary.json');selection=load(OUT/'selection.json');hist=load(OUT/'historical-context.json');audit=load(OUT/'integrity-audit.json')
 rows={(r['idea'],r['stage'],r['condition']):r for r in s['results']};names={'causal-confusion-interventions':'Causal confusion','minimax-regret-levels':'Minimax regret','accel-levels':'ACCEL'}
 md=['# Test 2 — does the predicted mechanism survive baseline tuning?','',
 '24 October 2026. Fresh equal-budget selection on current evaluators, followed by disjoint development gate checks and a locally frozen eight-seed paired confirmation per idea. This is an audit of the local implementations/testbeds, not an independent reproduction of the papers. Original result records and verdicts remain unchanged.','',
 '## Finding','',
 'All three fresh toy point gates passed at defaults and failed after safety-based tuning of both arms. In reserved full confirmation, all three simultaneous intervals had upper bounds below the declared 10-percentage-point mechanism advantage. This rules out that minimum in this bounded setup under the stated interval assumptions; it does not establish zero effect or refute the papers. Minimax-regret and ACCEL also had worse safety and capability than the tuned domain-randomization baseline.','',
 '## Design','',
 '- Ideas: causal-confusion interventions, minimax-regret levels and ACCEL.',
 '- Fresh tuning: the existing four-candidate grid per method and per baseline, each evaluated on two fresh development/full seeds (12000–12001). Select highest mean safety; resolve exact ties by original candidate order. No selection on mechanism diagnostics.',
 '- Gate audit: four separate development/toy seeds (22000–22003), comparing defaults with selected settings on the same instances. These comparisons isolate a tuning-setting change within the toy configuration.',
 '- Confirmation: eight reserved test/full seeds (92000–92007). Retain the original diagnostic, orientation and minimum effect of 0.1. Do not attribute all toy-to-full changes to tuning.',
 '- Primary decision: the selected method’s oriented diagnostic improvement relative to its selected baseline must clear the declared minimum. Report Bonferroni 98.333% paired-seed t intervals across three ideas: a lower bound ≥0.1 supports the declared minimum; an upper bound <0.1 rules it out in this setup; otherwise the result is inconclusive at that minimum. Original point-gate decisions are also shown separately.',
 '- Safety/capability comparisons and 95% intervals are secondary/descriptive. Eight seeds can leave broad intervals; a failed significance test is not proof of no effect. The t intervals assume seed-level mean differences are adequately described by this approximation.',
 '- Original per-task budgets are unchanged, including 300,000 UED environment steps per task. Equal candidate counts do not mean equal wall-clock compute, model size, or supervision access.',
 '- Both UED methods share the same baseline fits when the configuration, seed and selected hyperparameters match. This saves compute without creating extra independent observations.',
 '- Protocol/source were frozen before selection; selected hyperparameters were frozen before gate audit or confirmation. The audit always tunes both arms even if the default gate fails.','',
 '## Fresh toy gate: defaults versus selected settings','',
 '| Idea | Default diagnostic advantage | Default point gate | Selected diagnostic advantage | Selected point gate |',
 '| --- | ---: | --- | ---: | --- |']
 for idea in p['ideas']:
  d=rows[(idea,'audit','default')];r=rows[(idea,'audit','selected')]
  md.append(f"| {names[idea]} | {points(d['oriented_effect']['mean_diff'])} pp | {'pass' if d['gate']['passed'] else 'fail'} | {points(r['oriented_effect']['mean_diff'])} pp | {'pass' if r['gate']['passed'] else 'fail'} |")
 md+=['','Positive advantage means less copycat/proxy behavior than the baseline. The declared gate minimum is 10 percentage points. These four-seed development comparisons are descriptive, not a literature-wide gate failure rate.','',
 '## Reserved full confirmation','',
 '| Idea | Baseline diagnostic rate | Method diagnostic rate | Oriented improvement | Simultaneous 98.333% interval | Practical-effect decision |',
 '| --- | ---: | ---: | ---: | --- | --- |']
 for idea in p['ideas']:
  r=rows[(idea,'confirm','selected')];ci=r['simultaneous_oriented_effect']['ci']
  decision={'declared_minimum_supported':'Minimum supported here','declared_minimum_ruled_out_in_this_setup':'10-point minimum ruled out here','inconclusive_at_declared_minimum':'Inconclusive at 10 points'}[r['practical_status']]
  md.append(f"| {names[idea]} | {pct(r['baseline_diagnostic_mean'])} | {pct(r['method_diagnostic_mean'])} | {points(r['oriented_effect']['mean_diff'])} pp | [{points(ci[0])}, {points(ci[1])}] pp | {decision} |")
 md+=['','## Secondary safety and capability','',
 '| Idea | Method safety | Baseline safety | Safety difference (95% interval) | Capability difference (95% interval) |',
 '| --- | ---: | ---: | --- | --- |']
 for idea in p['ideas']:
  r=rows[(idea,'confirm','selected')];a,b=r['safety'],r['capability']
  md.append(f"| {names[idea]} | {pct(a['method_mean'])} | {pct(a['baseline_mean'])} | {points(a['mean_diff'])} pp [{points(a['ci'][0])}, {points(a['ci'][1])}] | {points(b['mean_diff'])} pp [{points(b['ci'][0])}, {points(b['ci'][1])}] |")
 md+=['','The diagnostic is not the safety endpoint. A reduced proxy/copycat rate can coexist with worse task capability or worse intended-goal success. A safety advantage without the specified diagnostic improvement leaves the claimed mechanism unresolved.','',
 '## Selected settings','']
 for key,c in selection['selected'].items():md+=['- '+key+': `'+json.dumps(c['hparams'],sort_keys=True)+'`.']
 md+=['','## Historical version boundary','',
 'The archived minimax-regret record uses an earlier UED evaluator hash. The new minimax and ACCEL fits both use current UED v0.2, with fresh tuning. Historical point gates are context, not same-version replications. The causal-confusion and ACCEL historical evaluator hashes match the current source; their old seeds are still not reused as fresh evidence.','',
 '## Integrity and reproduction','',
 '- '+str(audit['verified_model_archives'])+' unique fitted-model archives passed SHA-256 checks.',
 '- Candidate budgets, safety-based selection, disjoint stage seeds, selection-freeze ordering, source hashes and paired intervals were independently recalculated by the integrity script.',
 '- '+str(len(audit['selected_saved_models_rescored']))+' saved fitted models were re-scored against regenerated evaluation instances using the same evaluator. This tests reproducibility of saved evidence, not independent scientific replication.',
 '- The public records ZIP contains scores, metadata and hashes; the complete local evidence bundle also includes fitted NPZ weights for re-scoring.',
 '- Fourteen new-audit plus existing interactive engineering tests passed before fitting. Fifteen passed after the saved-score checker received a regression test for nested metrics and metadata.','',
 'Run `python -m experiments.tuning_audit.run initialize`, then `select`, `freeze`, `audit`, `confirm`, and `report` in a clean output directory. Keep the exact source/protocol hashes and runtime metadata. Model archives use pickle-free NPZ; no old baseline cache was used for fitting.','',
 '## Interpretation limits','',
 'This audit uses three selected ideas and small toy testbeds, a limited four-setting search, two selection seeds and eight confirmation seeds. It cannot estimate how common baseline sensitivity is across alignment research, prove the methods never work, or identify a unique causal explanation for a performance difference. Causal-confusion interventions have extra interaction/query access by design; the baseline remains the declared behavior-cloning comparator. Practical-regret curricula here do not implement an oracle-regret variant or the papers’ larger training budgets.','',
 'The next study should change one uncertainty at a time—such as a broader prespecified search, more independently reserved seeds, or a different task family—without rewriting this audit’s outcome.','']
 (OUT/'REPORT.md').write_text('\n'.join(md))
 fig,axes=plt.subplots(1,2,figsize=(11,4.8));indices=np.arange(3)
 for delta,condition,label,color in [(-.17,'default','Defaults','#657b92'),(.17,'selected','After tuning','#b77d40')]:
  means=[];lo=[];hi=[]
  for idea in p['ideas']:
   r=rows[(idea,'audit',condition)]['oriented_effect'];m=r['mean_diff'];means.append(100*m);lo.append(100*(m-r['ci'][0]));hi.append(100*(r['ci'][1]-m))
  axes[0].bar(indices+delta,means,.30,label=label,color=color,yerr=[lo,hi],capsize=3)
 means=[];lo=[];hi=[]
 for idea in p['ideas']:
  r=rows[(idea,'confirm','selected')]['simultaneous_oriented_effect'];m=r['mean_diff'];means.append(100*m);lo.append(100*(m-r['ci'][0]));hi.append(100*(r['ci'][1]-m))
 axes[1].errorbar(indices,means,yerr=[lo,hi],fmt='o',markersize=7,capsize=4,color='#27624e')
 for ax,title in zip(axes,['Fresh toy gate audit (4 seeds; descriptive 95% intervals)','Reserved full confirmation (8 seeds; simultaneous intervals)']):
  ax.axhline(10,color='#9e4840',linestyle='--',label='Declared minimum: 10 pp');ax.axhline(0,color='#aaa',linewidth=.7);ax.set_xticks(indices,[names[i].replace(' ','\n',1) for i in p['ideas']]);ax.set_ylabel('Oriented diagnostic improvement (percentage points)');ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
 axes[0].legend(fontsize=8)
 fig.suptitle('Test 2: mechanisms checked after fresh equal-budget tuning',fontsize=13)
 fig.text(.5,.015,'Copycat/proxy metrics and point gates are distinct from safety/capability. Different toy/full outcomes are not a tuning-only causal contrast.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.07,1,.93));fig.savefig(OUT/'mechanism-audit.png',dpi=180);plt.close(fig)
 print('Saved REPORT.md and mechanism-audit.png')

if __name__=='__main__':main()
