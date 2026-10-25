"""Report frozen case-level endpoints; no position-population inference."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'experiments/go_continuations';OUT=ROOT/'results/go-continuation-audit-v1'
load=lambda p:json.loads(Path(p).read_text())
def bounds(value):return (0,1) if value is None else (value,value)
def main():
    p=load(HERE/'protocol.json');cases=load(HERE/'inputs/cases.json');audit=load(OUT/'integrity-audit.json');assert audit['status']=='passed'
    records={f.stem:load(f) for f in (OUT/'branches').glob('*.json')};results=[]
    for case in cases:
        entry=dict(case=case['stem'],engine=case['engine'],source_game=case['source_file'],move=case['next_move_number'],original_stones=len(case['target_stones']),recorded_move=case['recorded_move'],defensive_move=case['defensive_move'],horizons={},terminal_pairs=[])
        entry['unique_trajectories_per_arm']={arm:len({hashlib.sha256(json.dumps([s['move'] for s in records[f'{case["stem"]}-{arm}-s{seed}']['steps']]).encode()).hexdigest() for seed in p['seeds']}) for arm in p['arms']}
        for h in [64,128]:
            arms={};effects=[]
            for arm in p['arms']:
                rr=[records[f'{case["stem"]}-{arm}-s{s}'] for s in p['seeds']];ee=[r['endpoints'][str(h)] for r in rr]
                arms[arm]=dict(retained=sum(e['all_original_retained']==1 for e in ee),lost=sum(e['all_original_retained']==0 for e in ee),unresolved=sum(e['all_original_retained'] is None for e in ee),fraction_lower=float(np.mean([e['fraction_lower'] for e in ee])),fraction_upper=float(np.mean([e['fraction_upper'] for e in ee])),first_loss_plies=[r['first_original_loss_ply'] for r in rr])
            for seed in p['seeds']:
                recorded=records[f'{case["stem"]}-recorded-s{seed}']['endpoints'][str(h)]['all_original_retained']
                defense=records[f'{case["stem"]}-defense-s{seed}']['endpoints'][str(h)]['all_original_retained'];rl,rh=bounds(recorded);dl,dh=bounds(defense)
                effects.append(dict(seed=seed,recorded=recorded,defense=defense,lower=dl-rh,upper=dh-rl))
            low=float(np.mean([e['lower'] for e in effects]));high=float(np.mean([e['upper'] for e in effects]))
            verdict='positive preservation advantage within selected cohort' if low>0 else 'no positive advantage in selected cohort' if high<=0 else 'unresolved'
            entry['horizons'][str(h)]=dict(arms=arms,paired_effect_lower=low,paired_effect_upper=high,verdict=verdict,paired=effects)
        for seed in p['seeds']:
            pair={a:records[f'{case["stem"]}-{a}-s{seed}'] for a in p['arms']};terminal={a:r['terminal'] for a,r in pair.items()}
            valid=all(t and t['score_matches_physical_area'] for t in terminal.values())
            margins={a:(t['physical_area_margin_black']*case['target_color'] if t and t['score_matches_physical_area'] else None) for a,t in terminal.items()}
            entry['terminal_pairs'].append(dict(seed=seed,both_verified_terminal=valid,target_relative_area_margins=margins,defense_worsens_terminal_result=bool(valid and (margins['defense']>0)-(margins['recorded']>0)<0)))
        results.append(entry)
    terminal_count=sum(bool(r['terminal']) for r in records.values());matched_scores=sum(bool(r['terminal'] and r['terminal']['score_matches_physical_area']) for r in records.values())
    supported=sum(r['horizons']['64']['paired_effect_lower']>0 for r in results);unresolved=sum(r['horizons']['64']['verdict']=='unresolved' for r in results)
    summary=dict(study_id=p['study_id'],scope=p['scope'],case_engine_pairs=4,source_games=2,paired_seeds_per_case=8,branches=len(records),primary_horizon=64,secondary_horizon=128,
                 primary_positive_case_comparisons=supported,primary_unresolved_case_comparisons=unresolved,cases=results,
                 branch_errors=[r['id'] for r in records.values() if r['status']=='error'],verified_continuation_plies=audit['verified_continuation_plies'],generated_searches=sum(max(0,len(r['steps'])-1) for r in records.values()),
                 two_pass_terminations=terminal_count,terminal_native_area_score_agreements=matched_scores,verified_terminal_pairs=sum(t['both_verified_terminal'] for r in results for t in r['terminal_pairs']),
                 worsening_terminal_pairs=sum(t['defense_worsens_terminal_result'] for r in results for t in r['terminal_pairs']),
                 limitations=['Four outcome-selected case/engine strata overlap within only two source games; not four independent positions or a failure-rate sample.','Eight search seeds measure sensitivity within a fixed case and policy, not an independent position population. Distinct seeds may produce identical trajectories; unique counts are reported.','Both colors use the same associated historical engine; no optimal-adversary, forced-life or forced-win certificate. Absence of capture in self-play does not establish robustness against an opponent choosing a known legal capture witness.','Only the first archived immediate defensive alternative is tested per case; other legal defenses may produce different later outcomes.','Strict horizon retention is unresolved after early termination with all original stones still present. Captured identities never return when a point is reoccupied.','Terminal-board occupancy/area is not independently adjudicated life and death. Only exact native/physical-area agreement receives a scored terminal record.','Leela positional superko and native symmetries differ from Kata Chinese simple ko and forced identity; do not pool engines.','Historical engineering checkpoints match archived hashes; the original published victim identities remain unresolved.','No source-game result is reused as a continuation outcome. Truncated branches receive no invented winner.','Reserved attack cohort and matched ordinary controls are later stages, not run here. The manuscript study remains separate.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    labels=[('LZ' if c['engine']=='leela' else 'KataGo')+' G'+c['source_file'].split('-')[-1].split('.')[0]+f' m{c["next_move_number"]}' for c in cases]
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,h in zip(axes,[64,128]):
        for i,case in enumerate(results):
            for arm,offset,col in [('recorded',-.19,'#4c78a8'),('defense',.19,'#59a14f')]:
                r=case['horizons'][str(h)]['arms'][arm];known=r['retained']/8*100;unknown=r['unresolved']/8*100
                ax.bar(i+offset,known,.36,color=col)
                if unknown:ax.bar(i+offset,unknown,.36,bottom=known,color='white',edgecolor=col,hatch='///')
                ax.text(i+offset,known+unknown+2,f'{r["retained"]}/8'+(' +?' if unknown else ''),ha='center',fontsize=9)
        ax.set_title(f'Strict original-stone retention at {h} plies');ax.set_xticks(range(4),labels,rotation=20,ha='right');ax.set_ylim(0,115);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('All original target stones retained (%)');fig.legend(handles=[Patch(color='#4c78a8',label='Recorded engine move'),Patch(color='#59a14f',label='Defensive alternative'),Patch(facecolor='white',edgecolor='gray',hatch='///',label='Unresolved possible retention')],loc='upper center',ncol=3,bbox_to_anchor=(.5,.94))
    fig.suptitle('Test 6: native continuation outcomes on selected development cases');fig.text(.01,.01,'Four overlapping case/engine pairs from two games; eight paired search seeds each. No population inference or life/death proof.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.85]);fig.savefig(OUT/'go-continuations.png',dpi=160);plt.close(fig)
    lines=['# Test 6 — native Go continuation audit','',f'**Primary outcome:** {supported}/4 selected case/engine comparisons show a positive 64-ply original-stone preservation advantage under the frozen worst-case rule; {unresolved}/4 remain unresolved. These four comparisons overlap within two source games and are not a population sample.','',
           'Each archived case branches into its recorded 128-visit engine choice and the first independently legal safe alternative in its existing annotation. The historical source-game move can differ from the recorded engine move. A fresh process replays the complete original prefix; the associated engine then plays both colors at 128 root visits per generated move. Eight fixed seeds per branch and a 128-ply ceiling were frozen before outcomes. Ply 1 is the forced move.','',
           f'Completed {len(records)} planned branches and independently checked {audit["verified_continuation_plies"]} continuation plies. Engine errors: {len(summary["branch_errors"])}. Binaries and model hashes match the archived milestone-4 engines.','',
           '| Case / engine | Branch moves (recorded / defense) | Original stones | 64-ply retained / lost / unknown, recorded | 64-ply retained / lost / unknown, defense | Worst/best paired effect |','|---|---|---:|---:|---:|---:|']
    for label,r in zip(labels,results):
        h=r['horizons']['64'];a=h['arms']['recorded'];b=h['arms']['defense'];fmt=lambda x:f'{x["retained"]} / {x["lost"]} / {x["unresolved"]}'
        lines.append(f'| {label} | {r["recorded_move"]} / {r["defensive_move"]} | {r["original_stones"]} | {fmt(a)} | {fmt(b)} | {h["paired_effect_lower"]*100:.1f} to {h["paired_effect_upper"]*100:.1f} pp |')
    lines+=['','Unique move trajectories across the eight seeds (recorded / defense): '+ '; '.join(label+': '+str(r['unique_trajectories_per_arm']['recorded'])+' / '+str(r['unique_trajectories_per_arm']['defense']) for label,r in zip(labels,results))+'.']
    lines+=['','## Interpretation','',
    'A legal immediate capture opportunity and a beneficial longer continuation are separate properties. The NumPy audit reconfirms every archived immediate loss/defense certificate, while the continuation tables report what the frozen native policies actually do. Those policies need not exploit every available capture. Only the first archived defensive alternative is tested; other legal defenses may yield different outcomes.', '',
    'These data support case-specific material-retention conclusions under the tested policies. They do not establish an avoidable whole-game loss: most branches are horizon-truncated, and a terminal-board score remains a bounded self-play result rather than a forced-win certificate.', '',
    '| Case | Recorded first original-stone capture | Defense first original-stone capture |','|---|---|---|']
    for label,r in zip(labels,results):
        def timing(arm):
            values=r['horizons']['128']['arms'][arm]['first_loss_plies'];parts=[]
            for value in sorted(set(v for v in values if v is not None)):parts.append(f'ply{value}: {values.count(value)}/8')
            if None in values:parts.append(f'not observed: {values.count(None)}/8')
            return '; '.join(parts)
        lines.append(f"| {label} | {timing('recorded')} | {timing('defense')} |")
    lines+=['','The effects above are deterministic sensitivity bounds over these eight selected paired seeds, not confidence intervals or population estimates. No pooling over four case/engine strata is used for inference.','',
            '## Strict endpoints and identity accounting','','An original stone is permanently lost at its first capture. Merging with other stones does not add original targets, and later replacement stones do not resurrect captured identities. A loss before the deadline resolves all-original retention as failure. Reaching the deadline with every original stone retained resolves success. An early terminal or engine error with all original stones still present remains unresolved. Fractional retention bounds and first/complete capture times are retained in summary.json.','',
            '## Later horizon','','| Case | 128-ply recorded retained / lost / unknown | 128-ply defense retained / lost / unknown | Worst/best paired effect |','|---|---:|---:|---:|']
    for label,r in zip(labels,results):
        h=r['horizons']['128'];a=h['arms']['recorded'];b=h['arms']['defense'];fmt=lambda x:f'{x["retained"]} / {x["lost"]} / {x["unresolved"]}'
        lines.append(f'| {label} | {fmt(a)} | {fmt(b)} | {h["paired_effect_lower"]*100:.1f} to {h["paired_effect_upper"]*100:.1f} pp |')
    lines+=['','## Game outcomes and remaining limits','',f'There were {terminal_count} two-pass terminations, with {matched_scores} exact native/physical terminal-area score agreements. Both branches had verified terminal scores in {summary["verified_terminal_pairs"]} paired seeds; {summary["worsening_terminal_pairs"]} of those pairs worsened the target-side win/loss result under defense. Horizon-truncated branches receive no winner. Terminal-board area is a bounded protocol outcome, not an independently adjudicated life/death or perfect-play result.','']+['- '+v for v in summary['limitations']]+['',
    '## Verification and reproduction','','A separate NumPy simulator replays every source prefix and continuation without using production sgfmill board transitions. It checks native boards after every move, permanent target removals, the original immediate tactical certificates, root visits, search/GTP response identities, output SGFs, score tallies, all branch counts and frozen file hashes. This is engineering verification, not independent scientific replication.','',
    'The native executables and models are external pinned dependencies; input-provenance.json lists hashes. Supply a private locations JSON mapping leela/katago to binary and weights paths. Run from the source checkout using an environment with NumPy and sgfmill:','',
    '```sh','python -m experiments.go_continuations.run initialize --locations /path/to/locations.json','python -m experiments.go_continuations.run execute --locations /path/to/locations.json','python scripts/audit_go_continuations.py','python scripts/report_go_continuations.py','```','',
    'Use an absent results/go-continuation-audit-v1 directory for an independent run. Existing error records are retained rather than retried. Native raw logs are included in the complete local evidence bundle; public records retain their hashes, move sequences, board snapshots, transcripts and full endpoint data. Absolute machine paths are not published. Source models/binaries are not redistributed.','',
    'The next ordered stage is a separately reserved attack-family cohort, followed by matched ordinary controls. No such cohort has been selected or evaluated here. The Loops Look Alive manuscript/native-engine check remains separate.','']
    lines+=['','## Verified terminal-score details','','| Case | Both terminal scores verified | Distinct target-relative area margins, recorded / defense |','|---|---:|---|']
    for label,r in zip(labels,results):
        vals={a:sorted({t['target_relative_area_margins'][a] for t in r['terminal_pairs'] if t['target_relative_area_margins'][a] is not None}) for a in p['arms']}
        lines.append(f"| {label} | {sum(t['both_verified_terminal'] for t in r['terminal_pairs'])}/8 | {vals['recorded']} / {vals['defense']} |")
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','limitations']},indent=2))
if __name__=='__main__':main()
