"""
Chat with your fine-tuned LoRA adapter.

Example:
  python training/chat_with_model.py --prompt "What is fine-tuning?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.helpers import pick_device, repo_root  # noqa: E402
from utils.env import get_hf_token, load_project_env  # noqa: E402

DEFAULT_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER = repo_root() / "outputs" / "chatbot-lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned chat adapter.")
    parser.add_argument("--base-model", default=DEFAULT_BASE, help="Same base model used for training.")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER, help="Folder with saved LoRA adapter.")
    parser.add_argument("--prompt", default="What is a learning rate?", help="User message.")
    parser.add_argument(
        "--system",
        default="You are a friendly coding tutor. Keep answers short and practical.",
        help="System prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_project_env(REPO_ROOT)
    hf_token = get_hf_token()

    device = pick_device()
    use_cuda = device == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32

    if not args.adapter_dir.exists():
        raise SystemExit(
            f"Adapter not found at {args.adapter_dir}. Train first:\n"
            "  python training/train_chatbot.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        token=hf_token,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    if use_cuda:
        model = model.to("cuda")

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if use_cuda else -1,
    )
    outputs = generator(
        prompt_text,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
    )
    print(outputs[0]["generated_text"].strip())


if __name__ == "__main__":
    main()
