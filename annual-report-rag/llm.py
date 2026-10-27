"""Local LLM only: a Hugging Face model on this machine's GPU. No API keys, no requests to outside services.

The weights are downloaded once into the Hugging Face cache (or copied in by hand). After that, set HF_OFFLINE=1
and the model loads with local_files_only=True, so nothing leaves the laptop. If the model can't load (no GPU,
out of memory, not downloaded), get_llm() returns None and the app answers in extractive mode instead.

.env:
    LLM_MODEL=Qwen/Qwen3.5-4B     # default; Qwen/Qwen2.5-7B-Instruct (4-bit, ~5.5 GB VRAM) also fits 8 GB
    HF_4BIT=1                     # 4-bit weights (needed to fit a 4B-7B model in 8 GB VRAM)
    HF_OFFLINE=1                  # after the first download: never contact the internet
    LLM_PROVIDER=none             # optional: skip the model, extractive mode only
"""
from __future__ import annotations

import os
import re
import threading
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # no usage pings to huggingface.co
if os.getenv("HF_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"                  # hard block on all hub network calls
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import config  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

class LocalLLM:
    """A causal LM on the local GPU (CPU if there is no GPU). Greedy decoding: the task is extraction."""
    provider = "local"

    def __init__(self, model_id: str):
        import torch
        import transformers
        from transformers import AutoTokenizer

        self.model, self.torch, self._lock = model_id, torch, threading.Lock()
        t0 = time.perf_counter()
        offline = os.getenv("HF_OFFLINE", "0") == "1"
        cuda = torch.cuda.is_available()

        kwargs = {"device_map": "auto" if cuda else "cpu",
                  "dtype": torch.bfloat16 if cuda else torch.float32,
                  "local_files_only": offline}
        if cuda and os.getenv("HF_4BIT", "1") == "1":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)

        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=offline)
        errors = []
        # Text-only class first (smaller, faster). Qwen3.5 ships with a vision encoder, so fall back to the
        # multimodal classes if the text-only class doesn't recognise the config.
        for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                self.net = cls.from_pretrained(model_id, **kwargs).eval()
                break
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{name}: {str(exc)[:150]}")
        else:
            raise RuntimeError("no transformers class could load the model: " + " | ".join(errors))

        where = torch.cuda.get_device_name(0) if cuda else "CPU (slow)"
        vram = f", {torch.cuda.memory_allocated() / 2**30:.1f} GB VRAM" if cuda else ""
        print(f"[llm] {model_id} loaded on {where}{vram} in {time.perf_counter() - t0:.0f}s"
              f"{' (offline)' if offline else ''}")

    def _prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if getattr(self.tok, "chat_template", None):
            # enable_thinking=False: Qwen3/3.5 otherwise write a long <think> block before the JSON
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        return f"{system}\n\n{user}\n\nAnswer:\n"

    def complete(self, system: str, user: str, max_tokens: int = 800, schema: dict | None = None) -> str:
        """schema is accepted for interface compatibility; the prompt already asks for JSON and qa.parse_json
        tolerates extra text around it. The static verifier rejects anything that isn't grounded."""
        inputs = self.tok(self._prompt(system, user), return_tensors="pt").to(self.net.device)
        with self._lock, self.torch.inference_mode():
            out = self.net.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,                              # deterministic: same question, same answer
                temperature=None, top_p=None, top_k=None,     # silence sampling warnings from generation_config
                repetition_penalty=1.0,                       # >1 would penalise copying quotes verbatim
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # in case the template still thinks
        return text.split("</think>")[-1].strip()


_LLM: LocalLLM | None = None
_LOADED = False


def get_llm() -> LocalLLM | None:
    """The local model (loaded once per process), or None for extractive mode."""
    global _LLM, _LOADED
    if _LOADED:
        return _LLM
    _LOADED = True
    if config.LLM_PROVIDER == "none":
        print("[llm] LLM_PROVIDER=none; using extractive mode")
        return None
    model_id = config.LLM_MODEL or DEFAULT_MODEL
    try:
        _LLM = LocalLLM(model_id)
    except Exception as exc:  # no GPU memory, model not downloaded, transformers too old...
        print(f"[llm] local model {model_id} failed to load ({type(exc).__name__}: {str(exc)[:200]}); "
              "using extractive mode")
        _LLM = None
    return _LLM