"""Local open-weight verifier client (transformers).

Serves both roles in the pilot:

* **zero-shot** -- Qwen2.5-Coder-7B-Instruct, DeepSeek-Coder-V2-Lite-Instruct,
  and the Qwen2.5-7B-Instruct control that isolates coding specialisation.
* **after SFT** -- pass ``adapter_path`` to load a LoRA adapter trained by
  ``verifier_pilot.sft``.

Decoding is greedy (``do_sample=False``), the open-weight equivalent of
temperature 0, so the comparison against the API models is like-for-like.

The answer is read by **generating it and parsing the text**, matching how the
API models answer (they generate a complete JSON object under a schema) and how
the reward-model evaluations this pilot follows read open-weight models.

``constrained=True`` selects an alternative channel that compares the logits of
the first token of each answer. **Do not use it for reported results.**
``not_aligned`` tokenises to ``['not', '_aligned']``, so that comparison is
between the rare token ``aligned`` and the very common generic token ``not``,
whose much higher prior collapses the verifier: measured on Qwen2.5-Coder-7B it
predicted ``aligned`` for 18.3% of PyMETA pairs and 1.8% of COJ2022 pairs
against a 33.3% base rate, with median logit margins of -2.25 and -4.38. It is
kept only so that artefact stays reproducible.
"""

from __future__ import annotations

from .base import (
    ALIGNED, ERROR, MAX_ANSWER_TOKENS, NOT_ALIGNED, Prediction, VerifierClient,
)

ANSWER_PREFIX = "Answer:"


class HFLocalClient(VerifierClient):
    provider = "local"

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        constrained: bool = False,
        max_new_tokens: int = MAX_ANSWER_TOKENS,
        max_input_tokens: int = 8192,
        seed: int = 42,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.name = name or model_id.split("/")[-1]
        self.constrained = constrained
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self._torch = torch

        torch.manual_seed(seed)
        self._init_params({
            # greedy: the open-weight equivalent of temperature 0, so the
            # comparison against the seed-pinned API models is like-for-like
            "decoding": "greedy (do_sample=False)",
            "temperature": "n/a (greedy)",
            "seed": seed,
            "answer_channel": "first-token logit comparison (BIASED, see docstring)"
                              if constrained else "greedy generation + parse",
            "max_new_tokens": max_new_tokens,
            "max_input_tokens": max_input_tokens,
            "dtype": "4bit-nf4" if load_in_4bit else dtype,
            "adapter": adapter_path or None,
        }, strict_params=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"device_map": device_map, "trust_remote_code": True}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.name = name or f"{model_id.split('/')[-1]}-sft"
        self.model.eval()

        if self.constrained:
            self._aligned_id = self._first_token(ALIGNED)
            self._not_id = self._first_token("not_aligned")
            print(
                f"    [{self.name}] WARNING: constrained first-token channel is biased "
                f"(comparing {self.tokenizer.decode([self._aligned_id])!r} against "
                f"{self.tokenizer.decode([self._not_id])!r}); do not report these numbers",
                flush=True,
            )

    def _first_token(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"empty tokenisation for {text!r}")
        return ids[0]

    def _build_inputs(self, system: str, user: str):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        return {k: v.to(self.model.device) for k, v in encoded.items()}

    def predict(self, system: str, user: str) -> Prediction:
        import time

        started = time.monotonic()
        try:
            inputs = self._build_inputs(system, user)
            n_input = int(inputs["input_ids"].shape[-1])
            if self.constrained:
                prediction = self._predict_constrained(inputs)
            else:
                prediction = self._predict_generate(inputs, n_input)
            prediction.latency_s = round(time.monotonic() - started, 3)
            prediction.usage = {"prompt_tokens": n_input}
            return prediction
        except Exception as exc:  # noqa: BLE001
            return Prediction.failure(type(exc).__name__, f"{type(exc).__name__}: {exc}")

    def _predict_constrained(self, inputs) -> Prediction:
        torch = self._torch
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        aligned_lp = float(log_probs[self._aligned_id])
        not_lp = float(log_probs[self._not_id])
        label = ALIGNED if aligned_lp >= not_lp else NOT_ALIGNED
        return Prediction(
            label=label,
            raw=f'{{"label": "{label}", "margin": {aligned_lp - not_lp:.4f}}}',
            finish_reason="constrained",
        )

    def _predict_generate(self, inputs, n_input: int) -> Prediction:
        from .base import parse_label

        torch = self._torch
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        raw = self.tokenizer.decode(output[0][n_input:], skip_special_tokens=True).strip()
        label = parse_label(raw)
        if label is None:
            return Prediction(
                label=ERROR, raw=raw, ok=False,
                error_kind="ParseFailure", error_detail=raw[:200],
            )
        return Prediction(label=label, raw=raw)

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        try:
            self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - CPU-only box
            pass
