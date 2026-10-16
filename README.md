# Alignment SynthCore

Automatically tested alignment research ideas. Each idea is run against a testbed where the right answer is planted and known. It goes through a toy mechanism gate, then a benchmark against a tuned baseline over 8 paired seeds. The verdict is computed by code, and failures are published too.

The aim is direction-finding: ground truth about which research directions deserve more effort. This is the tested-ideas layer of Alignment Commons. Design follows SynthCore (synthcore.org) and the spec in "Alignment Testbed Spec v0".

## Status

Phase 1 of the spec: the shared harness plus Testbeds 1 and 2, sized to run on a CPU.

| Testbed | What is planted | Safety metric | Baseline |
| --- | --- | --- | --- |
| `planted_interp` | Toy superposition models. Feature directions are known, a share of features only fire with a parent feature (the absorption setting), and activations carry noise | Mean max cosine similarity to the planted features | TopK SAE, tuned |
| `backdoor` | Populations of small digit classifiers, half with trigger backdoors at 2% poison. The pixel trigger family is held out of dev entirely | AUROC, clean vs held-out family | Neural Cleanse-style mask search, tuned |

## Run

```
pip install torch numpy scipy scikit-learn
python run.py ideas/<idea>          # one idea
python run.py --all                 # everything in ideas/
python summarize.py                 # results/SUMMARY.md and summary.json
python -m pytest tests              # stats and verdict tests
```

First runs generate and cache models under `results/_cache/`.

## Adding an idea

Make `ideas/<slug>/` with two files.

`idea.json` holds the title, description, source, Alignment Commons direction, testbed, and a prediction written before any run:

```json
"prediction": {
  "safety_min_effect": 0.0,
  "mechanism": {"diagnostic": "child_mmcs", "direction": "higher", "mode": "relative", "min_effect": 0.01}
}
```

`mode: relative` means the diagnostic must beat the baseline by `min_effect` on the toy config. `mode: absolute` means the method's own diagnostic must clear `threshold`. With no mechanism, the gate uses the safety metric against the baseline.

`method.py` defines `HPARAMS` (a list; the first entry is the default, and the whole list is the tuning grid, capped at 8 like the baseline's) and `run(tasks, hparams, seed) -> list of outputs`. See each testbed's `testbed.py` docstring for its task and output format.

## Pipeline

1. **Stage 1, toy gate.** Dev split, toy config, 3 seeds, default hyperparameters for both arms. Fail stops the idea: `mechanism_failed`.
2. **Tuning.** Method and baseline each get up to 8 configs, scored on the dev split. Test data is never used for tuning.
3. **Stage 2.** Test split, full config, 8 paired seeds. Exact sign-flip permutation test on per-seed differences, alpha 0.05, with 95% CI and minimum detectable effect.
4. **Verdict.** `beats_baseline`, `mech_ok_no_win`, `no_effect`, `hurts`, `mechanism_failed`, `implementation_failed`, `untestable`. Flags: `capability_confounded` (capability also rose significantly), `cause_unresolved` (won without the predicted mechanism).

## Integrity

Methods run in a separate process. They receive only public inputs, written to a temporary folder. Ground truth stays in the harness process. Every record carries a hash of the evaluator files, so results from different evaluator versions are never mixed. Tuned baselines are cached per evaluator hash.

Known v0 limits: process isolation is not a filesystem jail, so a method could still read repo files. A container is the next step before running LLM-written methods. On the backdoor testbed the population is fixed, so seeds vary method randomness plus a paired bootstrap over models.

## Not built yet

- LLM extraction from papers. Needs an API key; `idea.json` is the target format.
- Coding agent that writes `method.py`. Example methods here were written by hand.
- Testbeds 3 to 6 (reward hacking, goal misgeneralization, weak-to-strong, control).
- Alignment Commons integration: `results/summary.json` is the hand-off format.

## License

MIT. Baseline and method code may reuse SynthCore's MIT-licensed experiment code with attribution.
