from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chat_jsonl(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"Line {line_no} must contain a 'messages' list.")
            rows.append(row)

    if not rows:
        raise ValueError(f"No examples found in {path}")

    return Dataset.from_list(rows)


def load_chat_dataset(data_path: Path) -> Dataset:
    if data_path.suffix == ".jsonl":
        return load_chat_jsonl(data_path)
    return load_dataset("json", data_files=str(data_path), split="train")


def format_chat_text(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_chat_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> Dataset:
    def tokenize_batch(batch: dict[str, Any]) -> dict[str, list[list[int]]]:
        texts = [format_chat_text(tokenizer, messages) for messages in batch["messages"]]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )
