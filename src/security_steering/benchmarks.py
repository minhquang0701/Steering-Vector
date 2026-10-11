"""Public benchmark downloads and explicit static replay adapters.

No benchmark code, payload, tool, URL embedded in data, or model output is executed.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from security_steering.io import digest, file_digest, timestamp, write_json
from security_steering.schemas import Span
from security_steering.synthetic import make_family

CATALOG = {
    "bipia": {
        "repository": "microsoft/BIPIA",
        "url": "https://github.com/microsoft/BIPIA",
        "license_url": "https://github.com/microsoft/BIPIA/blob/main/LICENSE",
        "files": [
            "benchmark/email/train.jsonl",
            "benchmark/email/test.jsonl",
            "benchmark/text_attack_train.json",
            "benchmark/text_attack_test.json",
            "LICENSE",
            "NOTICE.md",
        ],
    },
    "injecagent": {
        "repository": "uiuc-kang-lab/InjecAgent",
        "url": "https://github.com/uiuc-kang-lab/InjecAgent",
        "license_url": "https://github.com/uiuc-kang-lab/InjecAgent/blob/main/LICENCE",
        "files": ["data/test_cases_dh_base.json", "data/test_cases_ds_base.json", "LICENCE"],
    },
}


def download_benchmarks(destination, names=("bipia", "injecagent"), revision="main"):
    destination = Path(destination)
    manifest = {"retrieved_at": timestamp(), "sources": []}
    for name in names:
        if name not in CATALOG:
            raise ValueError(f"unknown benchmark {name}")
        source = CATALOG[name]
        request = urllib.request.Request(
            f"https://api.github.com/repos/{source['repository']}/commits/{revision}",
            headers={"User-Agent": "security-steering-research"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            commit = json.load(response)["sha"]
        for relative in source["files"]:
            url = f"https://raw.githubusercontent.com/{source['repository']}/{commit}/{relative}"
            target = destination / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read(50_000_001)
            if len(content) > 50_000_000:
                raise ValueError("download exceeded the 50 MB resource limit")
            if target.exists() and target.read_bytes() != content:
                raise ValueError(
                    f"{target} already contains a different revision; use a new directory"
                )
            target.write_bytes(content)
            manifest["sources"].append(
                {
                    "benchmark": name,
                    "revision": commit,
                    "url": url,
                    "path": str(target),
                    "sha256": file_digest(target),
                    "license_url": source["license_url"],
                }
            )
    write_json(destination / "manifest.json", manifest)
    return manifest


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def import_bipia(context_path, attacks_path, *, limit=None, upstream_split="train"):
    contexts = _jsonl(context_path)
    attacks = json.loads(Path(attacks_path).read_text())
    flat_attacks = [(kind, text) for kind, texts in attacks.items() for text in texts]
    if not flat_attacks:
        raise ValueError("empty BIPIA attack file")
    output = []
    for i, context in enumerate(contexts[:limit]):
        category, payload = flat_attacks[i % len(flat_attacks)]
        family = "bipia_email_" + digest([context, payload])[:16]
        ideal = context["ideal"]
        answers = ideal if isinstance(ideal, list) else [str(ideal)]
        provenance = {
            "origin": "benchmark_bipia_static",
            "source_url": CATALOG["bipia"]["url"],
            "context_sha256": file_digest(context_path),
            "attacks_sha256": file_digest(attacks_path),
            "upstream_split": upstream_split,
            "upstream_index": i,
            "upstream_category": category,
            "adaptation": "append_one_attack_to_email_context",
            "split_group": "bipia_context_" + digest(context),
        }
        if upstream_split == "test":
            provenance["split"] = "test"
        output.extend(
            make_family(
                family_id=family,
                attack_class="override",
                payload=payload,
                answer=str(answers[0]),
                user_task=context["question"],
                evidence=context["context"],
                provenance=provenance,
                hard_benign=False,
                evaluation={
                    "answers": answers,
                    "answer_match": "contains",
                    "security_check": "requires_benchmark_judge",
                    "allowed_tools": [],
                },
            )
        )
    return output


def import_injecagent(path, *, limit=None):
    rows = json.loads(Path(path).read_text())
    output = []
    for index, row in enumerate(rows[:limit]):
        payload = row["Attacker Instruction"]
        template = row["Tool Response Template"]
        if "<Attacker Instruction>" not in template:
            raise ValueError(f"InjecAgent row {index} has no attack insertion placeholder")
        family = "injecagent_" + digest(row)[:16]
        kind = "exfiltration" if "ds_" in Path(path).name else "tool_misuse"
        members = make_family(
            family_id=family,
            attack_class=kind,
            payload=payload,
            answer="Manual utility grading required",
            user_task=row["User Instruction"],
            evidence=template.replace("<Attacker Instruction>", "No additional content."),
            provenance={
                "origin": "benchmark_injecagent_static",
                "source_url": CATALOG["injecagent"]["url"],
                "source_sha256": file_digest(path),
                "upstream_index": index,
                "split": "test",
                "upstream_split": "test",
                "upstream_attack_type": row["Attack Type"],
                "adaptation": "single_tool_response_replay_no_environment_execution",
            },
            hard_benign=False,
            evaluation={
                "answers": [],
                "attacker_tools": row["Attacker Tools"],
                "allowed_tools": [row["User Tool"]],
                "security_check": "partial_tool_trace",
            },
        )
        attack = members[1]
        record = next(s for s in attack.context_segments if s.segment_id == "record")
        record.text = template.replace("<Attacker Instruction>", payload)
        start = record.text.index(payload)
        attack.malicious_span = Span("record", start, start + len(payload))
        output.extend(members)
    return output
