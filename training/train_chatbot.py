"""
Fine-tune a chat/instruction model for text generation (SFT + LoRA).

Uses transformers + PEFT (no TRL) so training works on Windows without UTF-8 import issues.

Example:
  python training/train_chatbot.py
  python training/train_chatbot.py --epochs 1 --max-seq-length 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.helpers import (  # noqa: E402
    load_chat_dataset,
    pick_device,
    repo_root,
    tokenize_chat_dataset,
)
from utils.env import get_hf_token, load_project_env  # noqa: E402

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATA = repo_root() / "training" / "data" / "chat_examples.jsonl"
DEFAULT_OUTPUT = repo_root() / "outputs" / "chatbot-lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a chat model with LoRA (SFT).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base instruct/chat model on the Hub.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL with {\"messages\": [...]} rows.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Where adapters and checkpoints go.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Effective batch size = batch_size * this value.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max tokens per training example.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (lower = less VRAM).")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling factor.")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint frequency.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Log frequency.")
    return parser.parse_args()


def lora_target_modules(model_id: str) -> list[str]:
    lowered = model_id.lower()
    if "gpt2" in lowered or "distilgpt" in lowered:
        return ["c_attn", "c_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    args = parse_args()
    load_project_env(REPO_ROOT)
    hf_token = get_hf_token()

    device = pick_device()
    use_cuda = device == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA GPU detected. Training will be slow; use a smaller model if needed.")

    dataset = load_chat_dataset(args.data)
    print(f"Loaded {len(dataset)} training examples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        token=hf_token,
    )
    if use_cuda:
        model = model.to("cuda")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_target_modules(args.model),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = tokenize_chat_dataset(dataset, tokenizer, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=use_cuda,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Adapter saved to: {args.output_dir}")
    print("Try: python training/chat_with_model.py")


if __name__ == "__main__":
    main()
