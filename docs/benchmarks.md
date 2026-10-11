# Benchmark and hybrid-data card

The downloadable sources below were checked on 2026-09-23. Downloads resolve a Git
commit before fetching files and write URLs, SHA-256 checksums, license links, and retrieval
time to `data/raw/manifest.json`. Source data and large artifacts are ignored by Git.
Keep upstream license/notice files with any distribution; a repository license does not
automatically cover every underlying dataset.

| Source | Role here | Evaluation boundary |
|---|---|---|
| [BIPIA](https://github.com/microsoft/BIPIA) | Email QA contexts plus supplied text attacks, paired by appending one attack | Static adaptation. Original task answer is retained; most attack goals require a benchmark judge. Not the full task/attack Cartesian product or published BIPIA protocol. |
| [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | Original user task and tool-response insertion slot, direct-harm and data-stealing cases | Official test split is locked. Explicit attacker tool calls give positive evidence; absence of a call does not prove successful defense. Native actions/utility are not executed. |
| [AgentDojo](https://github.com/ethz-spylab/agentdojo) | Recommended next native environment benchmark | No built-in native runner is claimed. Its dynamic task environment and action checks must be retained for comparable scores. |
| Controlled templates | Seven classes, five external source types, paired canaries and hard-benign quotations | Deterministic local rubrics; deliberately narrow diagnostic tasks. |
| Local LLM candidates | `generate-data` creates attack payloads and deterministic paired tasks | Preserve generator identity/prompt/output, inspect attack-type labels and task correctness, then approve. Rejected generations are logged. |

The BIPIA repository was archived on September 16, 2026; pinning snapshots is especially
useful. Its WebQA/summarization sources have separate acquisition conditions, so the
automatic downloader includes the directly supplied EmailQA subset and text attacks.
See [upstream instructions and licensing](https://github.com/microsoft/BIPIA#download-the-dataset).

## Data preparation

```bash
python -m security_steering fetch-benchmarks data/raw
python -m security_steering build-data data/processed/hybrid.jsonl \
  --bipia-dir data/raw/bipia --injecagent-dir data/raw/injecagent \
  --benchmark-limit 30

python -m security_steering generate-data data/generated/candidates.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --count 21
```

For each LLM family, verify that the attack changes an instruction rather than the answer
fact, the alleged class is correct, the safe pair still contains the answer, and suspicious
quoted text is actually benign. Set `metadata.review_status` to `approved` on all variants
only after review. Then rebuild with `--llm-data data/generated/candidates.jsonl`.
`--include-unreviewed` is an explicit exploratory override; it does not certify data quality.
The local generator paraphrases controlled instructions, retaining literal targets and tool
names. A reviewer must still check that the paraphrase preserves the intended action/type.

`split-data OUTPUT INPUT...` merges normalized JSONL inputs, preserves existing split
locks, and rejects conflicting groups. `import-benchmark` supports local snapshots without
network access. Unknown outcomes remain ungraded. Manual/native evaluations can be
represented in per-example `metadata.evaluation` rubrics; do not insert grader answers
into context segments.

## Schema extension for executable checks

```json
{
  "metadata": {
    "origin": "synthetic_template",
    "split": "train",
    "split_group": "shared-template-17",
    "evaluation": {
      "answers": ["CODE_1234"],
      "answer_match": "exact",
      "attack_targets": ["INJECTED_ABCD"],
      "forbidden_strings": ["PRIVATE_FAKE_CANARY"],
      "allowed_tools": [],
      "allow_memory_write": false,
      "security_check": "controlled"
    }
  }
}
```

Structured generated actions use `{"answer":"...", "tool_calls":[{"name":"..."}],
"memory_write":null}`. The evaluation harness never invokes these actions. An external
free-text benchmark without complete goals must not be relabelled `controlled` merely
to make the bandit accept it.
