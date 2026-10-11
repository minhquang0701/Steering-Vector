# Source-Aware Security Steering

A runnable research pipeline for prompt-injection defense using frozen-model activation
interventions and a learned alpha controller. This implements the computational work for
Milestones 1–3 and prepares the paper/outreach materials for Milestone 4. It does **not**
claim a proven security improvement, a publication-ready result, secured mentorship, or funding.

| Milestone | Implemented |
|---|---|
| 1: data and vectors | Versioned BIPIA/InjecAgent downloads and adapters; controlled and local-LLM paired data; grouped train/validation/test splits; safe/attack prototypes; four vector estimators by layer, site and class. |
| 2: intervention experiments | Residual, MLP and both; baseline, validation sweeps, matched random-layer and random-direction controls, sign reversal; token timing controls; reproducible response logs and reports. |
| 3: adaptive alpha | Train-only risk probe, risk-gated baseline, two-layer neural contextual bandit with REINFORCE, reward/propensity logs, frozen-policy test evaluation. |
| 4: research package | LaTeX draft, verified references, generated tables/plots, mentor shortlist, outreach draft, current funding links and budget worksheet. |

Read [the protocol](docs/methodology.md) before interpreting an experiment. In particular,
prompt-condition vectors are not automatically vectors of malicious behavior, and low
attack success without task completion is not a useful defense.

## Install and test

From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,ml,plots]'
python -m unittest discover -s tests
python -m ruff check src scripts tests
```

The base package needs NumPy and PyYAML. The `ml` extra adds PyTorch/Transformers; `plots`
adds Matplotlib. All commands also work without installation using `PYTHONPATH=src` and
`python -m security_steering`. Tests with the ML extra instantiate real tiny transformers
locally; no model download or API key is needed.

On the implementation machine, the working interpreter was
`/home/daoqm/opentool_env/bin/python` with `PYTHONNOUSERSITE=1 PYTHONPATH=src`. Its tested
versions were Python 3.11.11, PyTorch 2.8.0, Transformers 4.56.1 and NumPy 1.26.4. The system
Python's pre-existing Torch/typing_extensions conflict is avoided by using an isolated environment.

## Run all computational milestones

```bash
security-steering build-data data/processed/controlled.jsonl
security-steering run --config configs/smoke.yaml
```

This runs data validation, real transformer activation collection, vector fitting, placement
sweeps, controller training, held-out comparisons, and report generation. The smoke model has
random weights: its scores verify software, **not defensive effectiveness**.

For a small pretrained CPU pilot:

```bash
security-steering build-data data/processed/pilot.jsonl --classes override source_confusion
security-steering run --config configs/cpu_pilot.yaml
```

`cpu_pilot.yaml` uses SmolLM2-135M-Instruct with `local_files_only: true`; the implementation
session downloaded it into `artifacts/model-cache`. On a fresh machine set this option to
`false` for the first download. For a larger experiment, build the hybrid data below and edit
[configs/research.yaml](configs/research.yaml) for your hardware and a pinned model revision.
That config sweeps many combinations and assumes a CUDA device. Estimate cost with the
pilot first. Models with unsupported decoder structures fail explicitly.

Paths in configurations are relative to the invocation directory. Existing run directories
are protected; use `--output-dir runs/new_run`, or `--resume` for an interrupted run with identical
configuration, dataset, resolved model and source code. A changed implementation must start
a new run. Completed responses are reused and training trajectories are reproducible.

## Hybrid data

```bash
security-steering fetch-benchmarks data/raw
security-steering build-data data/processed/hybrid.jsonl \
  --bipia-dir data/raw/bipia --injecagent-dir data/raw/injecagent --benchmark-limit 30
security-steering generate-data data/generated/candidates.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --count 21
# Review the generated families and set metadata.review_status=approved.
security-steering build-data data/processed/hybrid_with_llm.jsonl \
  --bipia-dir data/raw/bipia --injecagent-dir data/raw/injecagent \
  --llm-data data/generated/candidates.jsonl
```

Public benchmark contents are treated only as data. Downloads record commit IDs, source
URLs, licenses and hashes. Native AgentDojo/InjecAgent action evaluation is not implemented;
the provided adapters are explicitly marked static replays. Missing security/utility checks
remain ungraded and cannot become RL rewards. See [the dataset card](docs/benchmarks.md).
LLM outputs that fail validation are logged and rejected, never replaced with claimed generations.

## Individual stages

```bash
security-steering collect-states --dataset data/processed/controlled.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cuda \
  --layers 6 12 18 --sites residual mlp --output artifacts/train_states.npz
security-steering compute-vectors artifacts/train_states.npz artifacts/vectors.npz \
  --methods mean_difference paired_mean paired_pca ridge_probe
security-steering report runs/cpu_pilot
```

Vectors use `safe - attack`. Layer indices are zero-based block indices. Each extraction
site has its own vector, including at the final layer. Alpha is intervention strength,
not a base-model learning rate. The controller chooses one alpha per response. The global
vector is the default; true test attack labels never choose a vector or alpha.

Old unlabelled Milestone 1 NPZ files cannot safely be reused; recollect explicit train records
with the new site metadata. The original hand-authored seed remains available for inspection,
but executable experiments use the newer paired dataset with machine-checkable rubrics.

## Outputs and paper

Each run directory contains `manifest.json`, an input `dataset.jsonl` snapshot,
`train_states.npz`, `vectors.npz`,
`selection.json`, `test_plan.json`, `responses.jsonl`, `controller.npz`, controller
trajectories, `summary.json`, `report.md`, CSV/LaTeX tables, and PNG/PDF plots when Matplotlib
is installed. Reports include metric denominators, uncertainty and paired family comparisons.
Large/generated data, models and run artifacts are intentionally ignored by Git.

[paper/main.tex](paper/main.tex) is a research draft. Build it with `make paper` when
LaTeX is installed. [Mentor and funding preparation](docs/mentor_and_funding.md) contains
researched leads and drafts; no outreach or applications are sent. The project already has
a local Git repository; no remote repository or public upload is created automatically.

See [milestone status](docs/milestones.md), [SEAL notes](docs/paper_notes.md), and
[the methodology](docs/methodology.md) for completion boundaries and research limitations.

The implementation session's [pilot findings](docs/pilot_findings.md) document the measured
failure to establish an improvement, including the distinction between answer presence and
strict instruction-following. The reviewed local-LLM family is saved in
`data/seed/llm_reviewed_examples.jsonl` so the hybrid dataset can be reconstructed without
regenerating it (review provenance explicitly identifies assistant inspection).
