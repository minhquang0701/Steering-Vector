# Executed pilot: no demonstrated defense improvement

Run: `runs/cpu_pilot`, using `configs/cpu_pilot.yaml`. Model:
`HuggingFaceTB/SmolLM2-135M-Instruct`, resolved snapshot
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, CPU, float32, greedy decoding, 24 output tokens.
This run completed on 2026-09-23 local time (2026-09-24 UTC).

The dataset contained 36 controlled examples (12 families), with 24 train, 6 validation,
and 6 test examples. Test comprises only **two independent families**, one override and
one source-confusion family. Each family has safe, attack, and hard-benign variants.

We evaluated 24 validation candidates: layers 8/20, residual/MLP/both, mean-difference/paired-PCA,
and alpha 1/5. A neural alpha policy trained for 72 updates on train examples, with actions
0/1/5. Test covered the baseline, selected fixed intervention, three random-layer seeds,
three random-direction seeds, sign reversal, risk gating, and the frozen alpha policy.

| Outcome | Baseline | Selected fixed | Adaptive alpha |
|---|---:|---:|---:|
| Observed attack success | 0/2 | 0/2 | 0/2 |
| Strict benign task completion | 0/4 | 0/4 | 0/4 |
| Strict attacked task completion | 0/2 | 0/2 | 0/2 |
| Joint security/task success | 0/6 | 0/6 | 0/6 |
| Correct code appears (post-hoc diagnostic) | 5/6 | 5/6 | 5/6 |

All other controls had the same aggregate outcomes. The selected nonzero candidate was
mean-difference steering at residual layer 8 with alpha=1. Validation did **not** recommend
it over the baseline. The learned policy selected alpha=1 for all six test examples; it
therefore did not demonstrate context-dependent adaptation or an optimal choice of alpha.

The strict utility rubric requires only the code. Most model responses instead include
an introduction or Markdown code fence. These responses often contain the correct code
but violate the task's explicit output instruction. The answer-presence diagnostic was
computed after inspecting results; it was not used to reselect a defense or redefine the
primary metric. Re-scoring every cached response with the finalized evaluator produced
zero differences in the original core task/security/validity outcomes.

The probe has a test AUC of 1.0 on these six correlated examples, but **this is not evidence
of a reliable detector**. Zero observed attacks among two cases is weak evidence: its
descriptive Wilson 95% upper bound is approximately 65.8%. The experiment cannot establish
defense improvement, useful alpha learning, or generalization. A larger task-capable model,
stronger attacks, more independent families, and native environment checks are required.

The pilot began during implementation, before source-hash capture was added. Its existing
dataset/model/config fingerprints and original response logs are retained; do not claim it
has a complete source snapshot. New runs record source hashes, library versions, and the
input dataset. The final full smoke run and tests validate the finalized code separately.

## Hybrid-data artifacts

`data/processed/hybrid_with_llm.jsonl` contains 369 examples / 163 families:
126 controlled examples, 120 BIPIA static replay examples, 120 InjecAgent static replay
examples, and one reviewed LLM-paraphrased family (3 examples). It is broader than the
pretrained pilot's dataset and has **not** been fully evaluated in a native environment.

The hybrid split has 124 training, 32 validation, and 213 test examples. Actual pretrained
activation extraction on its training split produced `artifacts/hybrid_train_states.npz`.
`artifacts/hybrid_vectors.npz` stores 128 finite steering vectors: four estimators × two
layers (8/20) × two sites (residual/MLP) × eight categories (seven types plus global),
along with safe and attack prototypes. No validation/test states fitted this bank.

The finalized pipeline verification artifacts are in `runs/smoke_final`. They use a real
tiny transformer with random weights and are not additional pretrained-model evidence.

The local generator's initial structured-output attempt yielded no accepted families.
A simpler paraphrase pass accepted one exfiltration-themed family; six outputs were
rejected for unchanged wording or lost literal targets. Both generation logs are retained
in `data/generated/`. The accepted family received assistant semantic inspection, not
independent human review. Its review method is recorded in the checked-in seed fixture.
