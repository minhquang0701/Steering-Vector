from __future__ import annotations

import json

from security_steering.schemas import SteeringExample

PROMPT_VERSION = "source-aware-v2"


def render_messages(example: SteeringExample) -> list[dict[str, str]]:
    """Render observable inputs, never labels, rubrics, spans, or expected answers."""
    system = []
    records = []
    for segment in example.context_segments:
        if segment.trust_level == "trusted_instruction":
            system.append(segment.text)
        elif segment.trust_level == "private_data":
            system.append("PRIVATE DATA (do not disclose): " + segment.text)
        elif segment.source_type == "user" and segment.text == example.user_task:
            continue
        else:
            records.append(
                {
                    "source": segment.source_type,
                    "trust": segment.trust_level,
                    "content": segment.text,
                }
            )
    return [
        {"role": "system", "content": "\n".join(system)},
        {
            "role": "user",
            "content": "Task: "
            + example.user_task
            + "\nRecords:\n"
            + json.dumps(records, ensure_ascii=False),
        },
    ]


def model_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"


def render_example_prompt(example: SteeringExample) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in render_messages(example))
