# SEAL and related-work notes

Paper: SEAL: Steerable Reasoning Calibration of Large Language Models for Free

URL: https://arxiv.org/abs/2504.07986 (read the version reporting COLM 2025)

## Milestone-Relevant Ideas

- SEAL keeps the model weights fixed and computes a steering vector offline.
- The paper observes separation among execution, reflection and transition thought representations.
- It computes a vector from mean hidden states: `S = mean(execution) - mean(reflection_or_transition)`.
- During decoding, it applies hidden-state intervention with `H_tilde = H + alpha * S`.
- It extracts decoder-block outputs at thought boundaries and intervenes at delimiter tokens.
- Its layer ablations are model-specific; they do not establish a universal best security layer.
- The paper ablates steering type, steering layer, and steering strength.

## Adaptation For This Project

Instead of reasoning-thought classes, this project uses security classes:

- override
- reveal
- exfiltration
- tool_misuse
- memory_poisoning
- source_confusion
- obfuscation

The milestone 1 vector target is:

```text
v_security,k = normalize(mean(safe_behavior,k) - mean(attack_following_behavior,k))
```

The milestone 2 steering target is:

```text
h_tilde = h + mask_t * alpha_k,t * v_security,k
```

The important change from SEAL is source awareness: the same instruction-like text has different authority depending on whether it appears in a system message, user task, tool output, webpage, memory entry, retrieved document, or agent message.

The implemented collector currently measures **prompt conditions**, not observed safe/attack-following
behavior. Its metadata records that distinction. Demonstrating causal behavioral control requires
held-out task evaluation and controls, even when a detector separates the conditions.

## Closely related security work (checked September 23, 2026)

[ARGUS](https://arxiv.org/abs/2512.05745) already explores steering instruction-following behavior,
adaptive intervention strength, and utility preservation for multimodal indirect injection.
[ICON](https://arxiv.org/abs/2602.20708) combines latent probing and attention correction for agents.
These papers make a blanket novelty claim for "steering against prompt injection" inappropriate.
Our proposed comparisons concern source-aware paired data, matched residual/MLP placement
controls, representation estimators, and a learned strength policy. Novelty remains to be established.
