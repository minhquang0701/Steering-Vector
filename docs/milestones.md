# Milestone implementation and completion boundaries

| Milestone | Software/materials delivered | Research or external work still required |
|---|---|---|
| 1 | Existing local Git repository extended; SEAL read and notes corrected; BIPIA/InjecAgent collected with source hashes; paired data generator and local-LLM augmentation; family/template/payload splits; safe/attack prototypes and four per-layer/site estimators. | Review generated examples, expand task diversity, and establish behavior-level rather than merely prompt-condition separability. |
| 2 | Actual PyTorch hooks; residual/MLP/both placement; unsteered baseline; validation sweeps; random placement/direction and sign controls; held-out logs, statistics, reports and plots; random-weight smoke and pretrained CPU pilot. | Demonstrate improvement on stronger models and diverse held-out tasks; use native benchmark action evaluators before claiming native benchmark results. |
| 3 | Train-only logistic risk probe; risk-gated baseline; small neural contextual bandit trained from response rewards; alpha=0 action; checkpoints and propensity/reward logs; frozen-policy evaluation. | Demonstrate superiority over tuned fixed alpha with multiple policy seeds and enough independent test families. A trained controller is not proof of an optimal alpha. |
| 4 | LaTeX draft, bibliography, report/table tooling, mentor leads, an outreach draft, funding links and budget worksheet. | Confirm authors/affiliation, finish research evidence and manuscript review, secure a mentor, obtain acceptance, and apply for eligible funding. No outreach/applications have been sent. |

Computational execution and scientific validation are separate milestones. Smoke outputs are
explicitly labelled random-weight software tests. Small pretrained runs are exploratory pilots,
not publishable efficacy evidence. Reports preserve failed utility and unknown security outcomes.

The original user dataset/scaffold is preserved. No Git history was rewritten, no remote was
published, and no benchmark tool payload was executed.
