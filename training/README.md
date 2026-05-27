# Train a chatbot (text generation)

This folder is your **first fine-tuning loop** for instruction/chat models. You will:

1. Prepare chat examples (`messages` format)
2. Fine-tune a small pretrained model with **LoRA** (GPU-friendly)
3. Chat with the saved adapter

## Prerequisites

```powershell
cd c:\Users\xavie\Documents\TestWebApps\huggingface-lab
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `HUGGINGFACE_HUB_TOKEN` (gated models need this).

Check GPU + token:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python scripts/check_hf_token.py
```

Install a **CUDA-enabled PyTorch** if `CUDA: False` but you have an NVIDIA card:  
https://pytorch.org/get-started/locally/

## 1) Run your first training job

Uses a tiny sample dataset so you learn the workflow (not production quality):

```powershell
python training/train_chatbot.py
```

Defaults:

| Setting | Value |
|--------|--------|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1B params) |
| Data | `training/data/chat_examples.jsonl` |
| Output | `outputs/chatbot-lora/` |
| Method | SFT + LoRA on GPU (fp16), `transformers` Trainer (no TRL — Windows-safe) |

Tweak for VRAM (8 GB vs 12+ GB):

```powershell
# Less VRAM
python training/train_chatbot.py --batch-size 1 --gradient-accumulation-steps 4 --max-seq-length 256 --lora-r 8

# Faster experiment on tiny data
python training/train_chatbot.py --epochs 1
```

## 2) Talk to your adapter

```powershell
python training/chat_with_model.py --prompt "What is overfitting?"
```

## 3) Use your own data (important)

Add rows to a `.jsonl` file. Each line is one conversation:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "User question here"},
  {"role": "assistant", "content": "Ideal answer you want the model to learn"}
]}
```

Train on your file:

```powershell
python training/train_chatbot.py --data path\to\my_chats.jsonl --output-dir outputs\my-bot
python training/chat_with_model.py --adapter-dir outputs\my-bot --prompt "Hello"
```

**Tips for chat data**

- Hundreds of good examples beat thousands of noisy ones when starting out.
- Keep system prompts consistent.
- Include diverse user phrasings for the same intent.
- Do not put secrets in training files.

## 4) What to learn next

| Topic | Why it matters |
|-------|----------------|
| **Eval set** | Hold out 10–20% of data; measure quality on unseen prompts |
| **Larger datasets** | Hub datasets like `HuggingFaceH4/ultrachat_200k` (subset first) |
| **SFT with causal LM** | Supervised fine-tuning for instruction following (this repo uses `Trainer` + LoRA) |
| **DPO / RLHF** | Align style/preferences after SFT |
| **Quantization (4-bit)** | Train bigger models on the same GPU |
| **Push to Hub** | Share `adapter` + base model id for reuse |

## 5) Suggested learning path (1–2 weeks)

1. **Today** — Run train + chat scripts; change one hyperparameter (`learning_rate`, `epochs`) and observe output.
2. **Day 2–3** — Write 50–100 custom Q&A pairs; retrain; compare before/after.
3. **Week 1** — Add a small eval JSONL; generate answers and score manually.
4. **Week 2** — Try a slightly larger instruct model if VRAM allows; read TRL docs on packing and chat templates.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA: False` | Reinstall PyTorch with CUDA; update NVIDIA drivers |
| Out of memory | Lower `--max-seq-length`, `--lora-r`, or `--batch-size` |
| Slow on CPU | Use `--model gpt2` only for plumbing tests, not quality |
| Gated model error | Accept license on Hugging Face; ensure token in `.env` |
| Gibberish output | More/better data, more epochs, or lower learning rate |
| TRL `UnicodeDecodeError` on Windows | This repo avoids TRL; if you use TRL elsewhere, run with `$env:PYTHONUTF8=1` |
