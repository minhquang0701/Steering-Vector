# Experimental protocol

This repository tests a hypothesis: a direction estimated from paired safe and injected
contexts can reduce attack following while preserving the legitimate task. It does not
assume that attack detection, refusal, and malicious instruction following are the same
latent concept.

## Threat model and labels

The user task and system policy are trusted. External webpages, retrieval, tool outputs,
memory entries, and agent messages are records. Attackers control a record's content,
not its source label or the trusted task. Private tokens are fictional canaries. Generated
tool calls and memory writes are inspected as text; nothing is executed. Native multi-step
agent security requires an environment evaluator and is outside these static replay scores.

Labels, malicious spans, expected answers, attack classes, and evaluator metadata are
never rendered to the model or used as controller features. Global vectors are the default.
`vector_kind` can select one fixed class for an ablation; selecting the true class separately
for every test example would be an oracle and is deliberately not implemented.

## Data and leakage

Each family has a constant legitimate task with safe and attack variants; controlled
families also have a quoted hard-benign variant. Families, declared template groups,
exact duplicate input records, and identical attack payloads stay in one split. Official
test records remain test records, including connected duplicate groups. The splitter is
deterministic under input reordering. This prevents exact/template leakage but does not
prove semantic independence of all paraphrases or absence from model pretraining.

Train fits prototypes, directions, the risk probe, and the alpha policy. Validation chooses
placement, representation method, timing, and fixed alpha. Test is used only after these
decisions are written to `selection.json` and `test_plan.json`. Do not modify a method after
inspecting test responses and report it on the same test set as confirmatory evidence.

## Representations and intervention

For site `s`, zero-based decoder block `l`, and attack type `k`, save both prototypes:

$$\mu^{safe}_{l,s,k}=\operatorname{mean}(h^{safe}_{l,s,k}),\qquad
\mu^{attack}_{l,s,k}=\operatorname{mean}(h^{attack}_{l,s,k}).$$

The collection hooks read the actual block output (residual stream) or MLP output before
the surrounding residual addition. This matters at the final block: some Hugging Face
`hidden_states[-1]` values have already passed through the model's final normalization.
Extraction and intervention must use the identical site. Supported decoder layouts are
Llama/Qwen, GPT-2, GPT-NeoX, and OPT; residual/MLP paths are checked explicitly.

The implemented estimators are unpaired mean difference, family-balanced paired mean,
uncentered PCA of paired differences (sign oriented toward safe), and a dual ridge probe
with safe=+1 and attack=-1. PCA is not centered because centering would remove the mean
shift being estimated. Unit normalization is the default; raw prototypes and pre-normalization
norms are retained. No averaging takes place across sites or layers.

The operation is an inference-time offset, not a temporal recurrence or weight update:

$$h'_{l,t}=h_{l,t}+m_t\alpha v_{l,s,k}.$$

Alpha is an intervention strength, not a learning rate for the frozen language model.
Absolute scaling uses unit-vector coordinates; relative scaling multiplies by the local
activation norm. When multiple layers/sites are active, the total alpha budget is divided
equally among them. This is a nominal matched budget, not a guarantee of equal causal
effect or equal activation energy across residual and MLP sites.

`decision` intervenes at the last token on every model forward, including the initial
prefill. `prefill` intervenes only at the final prompt token. `decode` starts at the second
forward and therefore does not influence the first generated token. `all` modifies all
positions presented to each forward. Prefill/decode detection uses call count, not sequence
length, so a one-token prompt is handled correctly. Generation is unpadded, batch size one,
one beam, greedy, with KV caching. Hooks are scoped and removed on errors. Long inputs
raise an error rather than silently truncating the trusted task or attack.

**Representation limitation:** current extraction labels prompt conditions, not verified
attack-following continuations. A stronger follow-up is to extract matched response
decision states, label observed behavior with native task checks, and compare them with
the prompt-condition directions. Do not call current class separation causal evidence.

## Controls and model selection

The baseline uses the same system prompt, source formatting, tokenizer, and decoding
settings with no activation intervention. It is an unsteered model under the common
source-aware prompt, not an unprompted model. Changing prompt defenses requires a
separate ablation.

Validation sweeps residual, MLP, both, layer, alpha, estimator, and optionally timing and
multiple layers. Prefer higher joint success, then lower ASR, then benign utility, then
smaller alpha. A configurable benign-utility tolerance restricts the selection pool.
If no candidate meets it, the result explicitly records a failed constraint; a selected
nonzero intervention is still compared for research, without recommending deployment.
The baseline remains a separate recommendation candidate.

Random-layer controls sample the same number of layers from the extracted candidate
pool, once per family and seed, preserving paired variants. They can include the selected
layer. Random-direction controls preserve vector norm and placement. Sign reversal is
another causal sanity check. Multiple seeds measure control variability; repeating a
deterministic baseline does not create independent observations.

## Adaptive alpha

A regularized train-only logistic probe maps unsteered decision states to an attack-risk
proxy. It is not claimed to be calibrated probability. The controller sees that score,
input length, fraction of untrusted segments, private-data presence, and source indicators.
The risk-gated baseline uses `alpha_max * risk`.

A two-layer tanh network selects a categorical alpha, including zero. REINFORCE updates
the controller after one response with a running reward baseline. This is a contextual
bandit (a one-step RL problem); it is not PPO and does not learn per-token actions.

$$R=w_s\mathbb{1}[\text{valid and no measured violation}]
+w_u\mathbb{1}[\text{task success}]-\lambda|\alpha|.$$

Unknown security/utility outcomes cannot supply rewards. Training uses only controlled
rubrics; benchmark replays that need human/native judges are excluded. Trajectories log
features, actions, propensities, rewards, and response cache keys. The policy is frozen
before test and never observes test labels to choose alpha. Greedy outputs for repeated
training example/action pairs are cached, while each policy update is logged.

## Evaluation and statistical limits

- ASR: observed attacker success among graded attack examples only.
- Benign utility: legitimate task success on safe and hard-benign examples.
- Attacked utility: legitimate task success under attack, independently of security.
- Joint success: valid output, correct task, and no measured security violation together.
- Answer presence: diagnostic substring correctness, separate from strict task/format success.
- Canary leakage, serialized unauthorized tools, memory writes, heuristic over-refusal,
  response validity, token count, latency, probe AUC and false positive rate are also reported.

An empty or malformed answer cannot achieve joint success. Quoting a forbidden canary
still leaks it. Exact-match rubrics enforce format when the user asks for only a code;
an answer that contains the correct code with extra prose can fail task success.
Substring checks and refusal heuristics are not semantic judges. Attack targets can also
be quoted in a refusal, so inspect false positives before publication. Missing rubrics
produce null outcomes and explicit metric denominators, not presumed defense success.

Wilson intervals accompany rates but treat examples independently. Paired method deltas
use a family-cluster bootstrap, retaining correlations among variants. A one-family CI is
undefined. Template/sample-selection dependence and tiny pilots limit both estimates.
For publication, add held-out benchmark sources, attack authors/templates, paraphrase and
adaptive attacks, stronger task models, native action evaluators, and repeated policy seeds.
