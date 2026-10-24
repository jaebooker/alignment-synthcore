"""Render the saved frozen E1 results; does not change fitting or primary analysis."""
from pathlib import Path
import json
import hashlib
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/approval-factorial-v1'
x=json.loads((OUT/'confirmation-summary.json').read_text());p=json.loads((ROOT/'experiments/approval_factorial/protocol.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
rows={(r['family'],r['gamma'],r['arm']):r for r in x['table']}
def pct(v):return str(Decimal(str(100*v)).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP))+'%'
def value(f,g,a,k):return rows[(f,g,a)]['metrics'][k]['mean']
md=['# Test 1: matched approval feedback × environment reward','',
'24 October 2026. Locally frozen 32-seed confirmation, preceded by eight development seeds. This is a fixed-exploration tabular ablation, not a replication of the original online decoupled-approval algorithm. No independent scientific reproduction or peer review is claimed.','',
'## Main result','',
'Independent approval substantially reduced lever activation, but the full-horizon learner did not complete the task. The combined primary support rule was **not met**. Hack avoidance alone is not successful task behavior.','',
'On feedback-lever environments at reward weight zero and gamma=0.95:',
'- Coupled lever use: '+pct(value('feedback_lever',.95,'coupled_reward0','lever_use'))+'.',
'- Independent-feedback lever use: '+pct(value('feedback_lever',.95,'decoupled_reward0','lever_use'))+'.',
'- Paired difference: −87.5 percentage points; 95% paired-seed t interval [−95.4, −79.6] points.',
'- Task completion: 0% in both arms. The independent-feedback arm failed the required absolute task-completion floor of 80%.','',
'The zero-width t intervals for all-zero goal outcomes describe zero observed seed variation; they do not establish certainty about every possible environment. The observed reduction measures lever activation, not inferred intent: an honest-feedback policy can sometimes activate a lever incidentally.','',
'## Fixed design and budgets','',
'- Four conditions: coupled/independent approval × environment-reward coefficient 0/1. Approval coefficient fixed at 1.',
'- Same public exploration trace across conditions: 20,000 uniform-action environment steps and exactly 20,000 used approval samples per condition/environment. Independent query actions have their own RNG stream; executed feedback is discarded by independent-feedback arms.',
'- Same learner and planning rule, zero hyperparameter tuning in every condition. Primary gamma=0.95; gamma=0 is a predeclared descriptive sensitivity, not a replacement primary analysis.',
'- 32 reserved evaluation seeds × three families × two layouts = 192 new environment instances. Eight policy conditions per instance (four objective/feedback arms at two gammas). Within each seed, average the two layouts before inference.',
'- Combo family was absent from development. Each evaluation task is still explored locally under the fixed budget; this is not zero-shot transfer.',
'- Reward in feedback_lever is always zero, so its reward-coefficient switch is structurally identical. Reward effects are evaluated on button and combo families.',
'- Primary rule required all three: lever-difference CI upper ≤−20 points; goal-difference CI lower ≥−5 points; independent-feedback absolute-goal CI lower ≥80%. Only the last component failed.',
'- Freeze manifest includes source/protocol hashes, eight-seed pilot summary hash and all 32 confirmation seeds. Every confirmation record was created after that freeze. There was no adaptive extension. Pilot lever variance was zero; its plug-in power estimate was consequently unreliable and is not treated as a power guarantee.','',
'## Full-horizon confirmation (primary planning setting)','',
'| Family | Feedback | Environment reward used | Lever activation | Button activation | Goal completion |',
'| --- | --- | --- | ---: | ---: | ---: |']
for f in p['confirmation_families']:
 for a in p['arms']:
  md.append(f"| {f} | {a['feedback']} | {a['reward_weight']} | {pct(value(f,.95,a['id'],'lever_use'))} | {pct(value(f,.95,a['id'],'button_use'))} | {pct(value(f,.95,a['id'],'goal_completion'))} |")
md+=['','## Reward-use contrast: secondary, descriptive','',
'In the button-only family, using environment reward increased button activation:',
'- Coupled feedback: 9.4% → 68.8% (+59.4 points; descriptive 95% paired interval [+48.7,+70.1]).',
'- Independent feedback: 6.3% → 65.6% (+59.4 points; descriptive 95% paired interval [+47.8,+71.0]).','',
'This supports a role for objective construction at this fixed coefficient and learner. It does not establish that every positive environment-reward weight causes hacking. Competing lever and button incentives change the pattern in the combo family. These secondary intervals are not multiplicity-adjusted confirmatory tests.','',
'## Myopic sensitivity (gamma=0): secondary, descriptive','',
'| Family | Feedback | Environment reward used | Lever activation | Button activation | Goal completion |',
'| --- | --- | --- | ---: | ---: | ---: |']
for f in p['confirmation_families']:
 for a in p['arms']:
  md.append(f"| {f} | {a['feedback']} | {a['reward_weight']} | {pct(value(f,0.,a['id'],'lever_use'))} | {pct(value(f,0.,a['id'],'button_use'))} | {pct(value(f,0.,a['id'],'goal_completion'))} |")
md+=['',
'On the feedback-lever family at reward coefficient zero, myopic task completion was 31.3% with coupled feedback and 98.4% with independent feedback. The primary full-horizon learner accumulated approval while failing to reach the goal, consistent with the already documented summed-approval farming problem. The changed planning horizon matters in this controlled setup; this was a predeclared sensitivity, not the primary success criterion.','',
'One of 64 myopic independent-feedback lever layouts used the default uniform policy at an unobserved state; reward0 and reward1 are identical copies there. The resulting small fractional lever-use rate must not be presented as uniformly deliberate corruption exploitation. Primary full-horizon arms had zero such fallback probability.','',
'## Unmodified online reference: development only','',
'The existing decoupled-approval method was also run unchanged on four development seeds with epsilon=0.2 and one query per step. It completed 87.5% of lever tasks and 100% of button tasks, with no lever or button activation. Its adaptive exploration differs from the matched trace experiment, so it is contextual reference evidence and not a factorial causal contrast.','',
'## Engineering and integrity checks','',
'- Fourteen tests passed across the new factorial controls and existing interactive testbed checks.',
'- Independent recomputation verified the primary interval, all frozen source hashes, the 32 reserved seeds, identical sample budgets and the reward-toggle identity on the zero-reward family.',
'- All 224 pilot/confirmation trace archives passed their saved SHA-256 checks.',
'- Traces, per-seed metrics, runtime versions, protocol, source hashes, freeze record and audit are retained. This integrity audit is not independent scientific reproduction.','',
'## What this establishes','',
'Within this fixed-exploration learner and planted environment, independently collected feedback sharply reduces lever activation, and including environment reward increases delayed-button activation. Neither observation alone establishes useful safe behavior: every primary full-horizon condition failed task completion. Planning horizon, feedback collection and objective construction need separate controls.','',
'## Limits and next experiment','',
'The experiment uses procedural toy environments, a selected reward coefficient, shared uniform exploration and a tabular learner. It is not an evaluation of general-purpose or superhuman agents. Honest counterfactual feedback is a privileged property of this environment; the mechanism may fail when that channel can also be corrupted. Hyperparameters were fixed, not equally tuned to optimality. The original published Synthcore verdicts remain unchanged.','',
'The next clean extension is a separately frozen matched online-exploration experiment, followed by a corrupted-counterfactual-channel stress test. It must retain goal-completion gates and must not relabel this primary failure as a safety win.','']
(OUT/'REPORT.md').write_text('\n'.join(md))
colors=['#8e6843','#27624e'];fig,axes=plt.subplots(1,2,figsize=(10,4.9))
for ax,metric,title in zip(axes,['lever_use','goal_completion'],['Lever activation (lower is better)','Task completion (higher is better)']):
 for offset,arm,label,color in zip([-.18,.18],['coupled_reward0','decoupled_reward0'],['Executed feedback','Independent feedback'],colors):
  means=[];lo=[];hi=[]
  for g in [.95,0.]:
   stat=rows[('feedback_lever',g,arm)]['metrics'][metric];m=stat['mean'];means.append(100*m)
   lo.append(100*(m-max(0,stat['ci95'][0])));hi.append(100*(min(1,stat['ci95'][1])-m))
  ax.bar(np.arange(2)+offset,means,.32,label=label,color=color,yerr=[lo,hi],capsize=3)
  for i,m in enumerate(means):ax.text(i+offset,m+3,str(Decimal(str(m)).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP))+'%',ha='center',fontsize=9)
 ax.set_xticks([0,1],['Full horizon\n(primary)','Myopic\n(descriptive sensitivity)']);ax.set_ylim(0,118);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Percent');ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False)
axes[0].legend(loc='upper right',fontsize=8)
fig.suptitle('Independent feedback reduced lever use; primary task-completion gate failed',fontsize=12)
fig.text(.5,.015,'32 paired seeds · 2 layouts per seed · feedback-lever family · environment reward = 0\nSeed-level 95% t intervals; zero observed variation gives degenerate intervals. Myopic includes one unknown-state fallback layout.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.09,1,.93));fig.savefig(OUT/'factorial-results.png',dpi=180);plt.close(fig)
print('Wrote REPORT.md and factorial-results.png')
