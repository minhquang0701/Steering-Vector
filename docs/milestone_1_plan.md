# Milestone 1 Plan

Historical scaffold plan. For the implemented pipeline and current completion criteria, see
[milestones.md](milestones.md) and [methodology.md](methodology.md).

## Goal

Initialize the research codebase and produce the data/vector foundation for source-aware security steering.

## Deliverables In This Scaffold

1. Repository creation outside `Weighted_Memory_Tree`.
2. Source-tagged prompt-injection dataset schema.
3. Seed benchmark with safe, attack, and hard-benign examples.
4. Dataset validator for checking family pairs and labels.
5. Vector utilities for:
   - safe-minus-attack mean vectors
   - probe-based defensive vectors
   - additive steering
   - projection-removal steering
6. Optional hidden-state collection script for Hugging Face models.

## Dataset Collection Plan

Use a hybrid dataset strategy:

- Adapt existing prompt-injection benchmark tasks from AgentDojo, InjecAgent, and BIPIA.
- Generate controlled paired examples with an LLM.
- Add hard-benign examples that mention suspicious instructions without being attacks.
- Keep paired safe and attack examples as similar as possible except for the injected instruction.

## Normalized Example Fields

Each example should include:

- `example_id`
- `family_id`
- `variant`: `safe`, `attack`, or `hard_benign`
- `user_task`
- `context_segments`
- `expected_safe_behavior`
- `expected_tool_action`
- `expected_memory_action`
- `attack_class`
- `malicious_span`
- `metadata`

## Security Classes

- `override`: untrusted content tries to change trusted instructions or the user goal.
- `reveal`: untrusted content asks for hidden prompts, policies, private memory, or secrets.
- `exfiltration`: untrusted content tries to send private data externally.
- `tool_misuse`: untrusted content tries to trigger an unauthorized tool call.
- `memory_poisoning`: untrusted content tries to write untrusted instructions into memory.
- `source_confusion`: untrusted content presents itself as higher-authority instruction.
- `obfuscation`: attack is encoded, translated, indirect, or hidden.

## Vector Extraction Plan

For each attack class `k` and layer `l`:

```text
safe_mean = average(hidden_states for safe behavior)
attack_mean = average(hidden_states for attack-following behavior)
v_security,k,l = normalize(safe_mean - attack_mean)
```

Also keep room for probe-based vectors:

```text
v_security,k,l = -normalize(w_probe,k,l)
```

## Milestone 1 Completion Criteria

- `data/seed/security_steering_seed.jsonl` validates successfully.
- Unit tests pass.
- Hidden-state collection can be run later by selecting a small open-source model.
- Vector computation accepts saved hidden-state records and writes a vector bank.
