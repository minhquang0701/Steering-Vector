"""Frozen Hugging Face decoder and a real, random tiny decoder for offline tests."""

from __future__ import annotations

import time
from dataclasses import dataclass

from security_steering.intervention import SteeringHooks, transformer_blocks
from security_steering.io import digest
from security_steering.prompting import model_prompt, render_messages


class ByteTokenizer:
    chat_template = None
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        import torch

        ids = torch.tensor([[2, *[b + 3 for b in text.encode("utf-8")]]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, **_kwargs):
        return bytes([i - 3 for i in ids.tolist() if 3 <= i < 259]).decode(
            "utf-8", errors="replace"
        )


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    hook_calls: dict[str, int]


class ModelBackend:
    def __init__(
        self,
        name: str = "tiny-random",
        *,
        device: str = "cpu",
        revision: str = "main",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        seed: int = 42,
        max_input_tokens: int = 2048,
        dtype: str = "auto",
        threads: int = 2,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

        if max_input_tokens < 1 or threads < 1:
            raise ValueError("max_input_tokens and threads must be positive")
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.name, self.device = name, device
        self.max_input_tokens = max_input_tokens
        self.is_smoke = name == "tiny-random"
        if self.is_smoke:
            self.tokenizer = ByteTokenizer()
            self.model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=259,
                    n_positions=max(4096, max_input_tokens + 1024),
                    n_embd=32,
                    n_layer=3,
                    n_head=4,
                    bos_token_id=2,
                    eos_token_id=1,
                    pad_token_id=0,
                    resid_pdrop=0,
                    embd_pdrop=0,
                    attn_pdrop=0,
                )
            )
        else:
            options = dict(
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=False,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(name, **options)
            if dtype not in {"auto", "float32", "float16", "bfloat16"}:
                raise ValueError("dtype must be auto/float32/float16/bfloat16")
            resolved_dtype = (
                (torch.float32 if device == "cpu" else torch.float16)
                if (dtype == "auto")
                else getattr(torch, dtype)
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=resolved_dtype,
                **options,
            )
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.num_layers = len(transformer_blocks(self.model))
        self.hidden_dim = self.model.config.hidden_size
        self.identity = digest(
            {
                "name": name,
                "revision": revision,
                "resolved_commit": getattr(self.model.config, "_commit_hash", None),
                "config": self.model.config.to_dict(),
                "seed": seed if self.is_smoke else None,
                "dtype": str(next(self.model.parameters()).dtype),
            }
        )

    def encode_messages(self, messages, *, reserve_tokens: int = 0):
        prompt = model_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        count = inputs["input_ids"].shape[1]
        context = getattr(self.model.config, "max_position_embeddings", 10**9)
        if count > self.max_input_tokens or count + reserve_tokens > context:
            raise ValueError(
                f"prompt has {count} tokens; exceeds configured/model context limit. "
                "Increase max_input_tokens or explicitly filter long examples; no truncation."
            )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def encode(self, example, *, reserve_tokens: int = 0):
        return self.encode_messages(render_messages(example), reserve_tokens=reserve_tokens)

    def generate(self, example, interventions=(), *, max_new_tokens: int = 48) -> Generation:
        return self.generate_messages(
            render_messages(example), interventions, max_new_tokens=max_new_tokens
        )

    def generate_messages(self, messages, interventions=(), *, max_new_tokens=48):
        import torch

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        inputs = self.encode_messages(messages, reserve_tokens=max_new_tokens)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), SteeringHooks(self.model, list(interventions)) as hooks:
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return Generation(
            self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            inputs["input_ids"].shape[1],
            len(new_tokens),
            time.perf_counter() - started,
            dict(hooks.calls),
        )
