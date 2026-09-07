"""LoRA / QLoRA SFT for the open-weight verifiers.

Loss is computed **only on the answer tokens**. The prompt -- which is long,
contains the entire student program, and is identical across the positive and
negative built from the same submission -- is masked to -100. Training on the
prompt would spend nearly all of the gradient signal on language-modelling the
student's code, which is not the task and washes out the two tokens that matter.

QLoRA (``load_in_4bit=True``) fits a 7B backbone on a single 24GB card. The
pilot's purpose is feasibility, so LoRA is the right cost point; full
fine-tuning only becomes worth it once the zero-shot vs SFT gap is established.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LoraConfig_:
    """Training hyperparameters, recorded next to the adapter for provenance."""

    model_id: str
    output_dir: str
    epochs: float = 2.0
    learning_rate: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 16
    max_seq_len: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    seed: int = 42
    gradient_checkpointing: bool = True


#: Attention + MLP projections. Names cover the Qwen2 and DeepSeek-V2 families;
#: any name absent from a given architecture is silently skipped by peft.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "kv_a_proj_with_mqa", "kv_b_proj",   # DeepSeek-V2 MLA
]


class VerifierSFTDataset:
    """Tokenised examples with the prompt masked out of the loss."""

    def __init__(self, examples: list[dict], tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        messages = example["messages"]
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        answer_text = messages[-1]["content"]

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            answer_ids = answer_ids + [eos]

        # Truncate the *prompt* from the left so the answer always survives.
        budget = self.max_seq_len - len(answer_ids)
        if budget <= 0:
            raise ValueError(f"max_seq_len={self.max_seq_len} too small for the answer")
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[-budget:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


def collate(features: list[dict], pad_token_id: int) -> dict:
    import torch

    width = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}
    for feature in features:
        pad = width - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad)
        batch["labels"].append(feature["labels"] + [-100] * pad)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def train(
    config: LoraConfig_,
    train_examples: list[dict],
    eval_examples: list[dict] | None = None,
) -> Path:
    """Fine-tune a LoRA adapter and save it. Returns the adapter directory."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"trust_remote_code": True, "device_map": "auto"}
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=[],
        seed=config.seed,
        eval_strategy="epoch" if eval_examples else "no",
        per_device_eval_batch_size=config.batch_size,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=VerifierSFTDataset(train_examples, tokenizer, config.max_seq_len),
        eval_dataset=(
            VerifierSFTDataset(eval_examples, tokenizer, config.max_seq_len)
            if eval_examples else None
        ),
        data_collator=lambda f: collate(f, tokenizer.pad_token_id),
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    print(f"adapter saved -> {adapter_dir}")
    return adapter_dir
