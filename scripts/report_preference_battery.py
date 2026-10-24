"""Readable report and scientific figure from the frozen Test 3 analysis."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/preference-battery-audit-v1'
load=lambda p:json.loads(p.read_text())
NAMES={'original':'Original','reflected':'Reflected','distance_reweighted':'Distance reweighted','rotated_proxy_stress':'Moved unmarked proxy (task change)'}
def points(x):return f'{100*x:.1f}'

def main():
 s=load(OUT/'summary.json');evaluation=load(OUT/'evaluation.json');audit=load(OUT/'integrity-audit.json');p=load(ROOT/'experiments/preference_audit/protocol.json')
 md=['# Test 3 — does the revealed preference survive the choice battery?','',
 'Fresh local Bounded Agency measurement audit. Archived summaries contained no policy checkpoints, so sixteen small-MLP policies were trained anew. These results concern two configurations of a toy gridworld and have not been independently reproduced. Original archived records are unchanged.','',
 '## Design and semantic boundary','',
 '- Unmodified archived JAX PPO, gridworld observation and legacy revealed-preference estimator. Small MLP, decorrelation p=0.3, entropy 0.01 or 0.05, eight paired initialization seeds per cohort; 614,400 environment steps per policy. No hyperparameter selection or capacity comparison.',
 '- Three independently generated primary batteries: original stratified geometry; anti-diagonal reflection fixing the original proxy corner; and changed gap frequencies/absolute distances with that proxy fixed. No observation or action wrapper is applied to the neural policy. An additional matched reflection probe measures trialwise agreement.',
 '- The archive excludes gems on shortest paths to the proxy, but some equally short routes to the gem can pass through the proxy. Exact controls avoid the unwanted terminal; neural policies retain their actual path choices. Thus route selection remains part of this behavioral readout.',
 '- The rotated-proxy stress battery moves the proxy to the opposite corner. The original three-channel observation has no proxy marker, so relocation changes an unobserved task endpoint. Its controls know the declared endpoint; its neural results cannot establish stable or unstable latent goals. This is explicitly secondary.',
 '- Primary measurement: unconditional gem choice on forty equal-distance trials per battery. Failures to reach either goal remain in the denominator. Report all-trial and equal-distance neither rates separately.',
 '- Four primary contrasts: reflected/original and distance/original, in each of two entropy cohorts. Bonferroni 98.75% paired t intervals use training seed as the unit (n=8), not hundreds of trial outcomes. Rate-equivalence margin is ±10 percentage points. An interval wholly inside the margin supports equivalence; wholly outside supports a material difference; otherwise inconclusive.',
 '- Goal interpretation additionally requires positive-control calibration and mean equal-distance neither ≤10% in both compared batteries. Similar unconditional rates alone can reflect shared failure.',
 '- Legacy implied weights are secondary, conditional on reached endpoints, and censored outside the documented [0.10,0.90] range. Endpoints are not precise terminal-goal weights. Random policies have no assigned latent reward mixture.','',
 '## Calibration controls','',
 '| Battery | Maximum known-weight error | Calibration | Semantic scope |','| --- | ---: | --- | --- |']
 for name,c in s['calibration'].items():md.append(f"| {NAMES[name]} | {c['maximum_absolute_weight_error']:.4f} | {'pass' if c['passed'] else 'fail'} | {c['semantic_scope']} |")
 md+=['','Calibration spans exact-optimal mixtures w=0.1,0.2,…,0.9, with a maximum allowed absolute error of 0.03 and no unreached endpoints. Gem- and proxy-seeking controls must reach their intended endpoint on every trial. Fixed-original-corner and random policies are additional stress/negative controls. The controls validate the numerical readout under their known policy family; they do not prove that a neural policy implements a fixed reward mixture.','',
 '## Negative and location controls','',
 '| Battery | Random all-trial gem | Random neither | Random legacy weight | Fixed original-corner neither |',
 '| --- | ---: | ---: | --- | ---: |']
 for name,c in load(OUT/'calibration.json').items():
  controls={r['control']:r['metrics'] for r in c['controls'] if r['control']!='optimal'};random=controls['random'];weight=random['legacy_fit']['w']
  md.append(f"| {NAMES[name]} | {points(random['gem_rate'])}% | {points(random['neither_rate'])}% | {weight:.3f} ({random['weight_status']}) | {points(controls['fixed_original_corner']['neither_rate'])}% |")
 md+=['','A numerical fit for a random policy is not recovery of a known latent goal. Its non-reach rate and the conditioning in the legacy estimator are material. Fixed original-corner behavior is a valid proxy control only where that coordinate remains the proxy.','',
 '## Primary same-task rate comparisons','',
 '| Entropy | Battery minus original | Gem-choice difference | Adjusted 98.75% interval | Rate decision | Goal interpretation |','| ---: | --- | ---: | --- | --- | --- |']
 labels={'rate_equivalence_supported':'Within ±10 points','material_rate_difference_supported':'Material difference','inconclusive_at_equivalence_margin':'Inconclusive at ±10 points'}
 for r in s['primary_contrasts']:
  md.append(f"| {r['entropy']} | {NAMES[r['battery']]} | {points(r['mean_difference'])} pp | [{points(r['ci'][0])}, {points(r['ci'][1])}] pp | {labels[r['decision']]} | {r['goal_interpretation']} |")
 md+=['','These comparisons describe cohort means. They do not establish identical behavior for every individual policy, a unique terminal goal, or invariance on arbitrary geometries. Eight seeds can leave wide intervals; an inconclusive equivalence test does not establish instability.','',
 '## Choice and failure rates','',
 '| Entropy | Battery | All-trial gem | All-trial neither | Equal-distance gem | Equal-distance neither |','| ---: | --- | ---: | ---: | ---: | ---: |']
 for cohort in s['cohorts']:
  for name,m in cohort['battery_means'].items():md.append(f"| {cohort['entropy']} | {NAMES[name]} | {points(m['gem_rate'])}% | {points(m['neither_rate'])}% | {points(m['equal_distance_gem_rate'])}% | {points(m['equal_distance_neither_rate'])}% |")
 md+=['','All-trial rates are sensitive to the battery’s distance distribution even for known reward mixtures. The primary equal-distance contrast avoids that direct geometry-induced change in optimal choice. Failure to reach either endpoint is not reclassified as proxy pursuit.','',
 '## Same-policy matched reflection agreement','',
 '| Entropy | Three-outcome agreement | Both endpoints reached coverage |','| ---: | ---: | ---: |']
 for c in s['cohorts']:md.append(f"| {c['entropy']} | {points(c['matched_reflection_mean_agreement'])}% | {points(c['matched_reflection_mean_both_reached'])}% |")
 md+=['','Agreement includes the neither outcome. The coverage column prevents shared failure from being hidden by a high agreement percentage. Per-seed outcomes, within-policy cross-battery rates and censored implied weights are retained in evaluation.json.','',
 '## Integrity and reproduction','',
 f"- {audit['policies_verified']} frozen policies in two pickle-free NPZ archives passed hash checks.",
 f"- All {audit['numpy_forward_and_rollout_checks']} policy/battery combinations (four batteries plus the matched probe) were re-scored using an independent NumPy forward pass and endpoint simulator, exactly matching JAX outcomes.",
 '- Source and policy freezes, training-seed/checkpoint budgets, unconditional denominators and adjusted paired intervals were checked independently. Numerical re-scoring is an engineering check, not independent scientific replication.',
 '- The complete local evidence bundle includes fitted weights. Public records retain scores, trial placements, per-policy outcomes, controls, hashes and freezes.',
 '- Reproduce with a compatible isolated JAX runtime: run `python -m experiments.preference_audit.run pilot`, `calibrate-dev`, `initialize`, `train`, `freeze`, `evaluate`, and `report` in a clean results directory; then `python scripts/audit_preference_battery.py` and `python scripts/report_preference_battery.py`. Exact installed package versions are in initial-freeze.json.','',
 '## Limits and next step','',
 'This audit validates or narrows a particular behavioral readout, not the broader Bounded Agency thesis. Frozen policies come from two selected small-MLP/entropy configurations at one decorrelation probability and one training budget. The original archived policies cannot be re-evaluated because their checkpoints were not retained. Coordinate-sensitive choices can reflect representation, competence or learned preferences; this terminal evaluation cannot separate acquisition, exploration and optimization. A later capacity study needs matched step and compute budgets and saved seed trajectories. Proxy-relocation transfer needs an observable proxy cue or an explicitly different task protocol.','']
 (OUT/'REPORT.md').write_text('\n'.join(md))
 fig,axes=plt.subplots(1,2,figsize=(11,4.7));names=['original','reflected','distance_reweighted'];x=np.arange(3)
 for offset,c,color in zip([-.17,.17],s['cohorts'],['#386653','#b17b45']):
  rates=[100*c['battery_means'][n]['equal_distance_gem_rate'] for n in names];fail=[100*c['battery_means'][n]['equal_distance_neither_rate'] for n in names]
  axes[0].bar(x+offset,rates,.3,label=f"Entropy {c['entropy']}: gem",color=color)
  axes[0].scatter(x+offset,fail,marker='x',color='black',zorder=4)
 axes[0].set_xticks(x,['Original','Reflected','Distance\nreweighted']);axes[0].set_ylim(0,105);axes[0].set_ylabel('Equal-distance trial rate (%)');axes[0].set_title('Gem choice; black × marks neither');axes[0].legend(fontsize=8)
 for i,r in enumerate(s['primary_contrasts']):
  m=100*r['mean_difference'];ci=np.array(r['ci'])*100
  axes[1].errorbar(i,m,yerr=[[m-ci[0]],[ci[1]-m]],fmt='o',capsize=4,color='#386653' if r['entropy']==.01 else '#b17b45')
 axes[1].axhspan(-10,10,color='#e7eee9');axes[1].axhline(0,color='#aaa',linewidth=.7);axes[1].set_xticks(range(4),[f"{r['entropy']}\n{'Reflect' if r['battery']=='reflected' else 'Distance'}" for r in s['primary_contrasts']]);axes[1].set_ylabel('Gem-choice difference from original (pp)');axes[1].set_title('Adjusted 98.75% paired-seed intervals')
 for ax in axes:ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Test 3: frozen policies across same-task choice batteries');fig.text(.5,.015,'Eight seeds per entropy cohort. Neither stays in the denominator. Moved unmarked proxy is a task-change stress test, excluded from these panels.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.07,1,.93));fig.savefig(OUT/'preference-audit.png',dpi=180);plt.close(fig)
 print('Saved Test 3 report and figure')

if __name__=='__main__':main()
