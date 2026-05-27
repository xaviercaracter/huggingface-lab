# huggingface-lab
A sandbox repo for testing and experimenting with HuggingFace models, pipelines, and APIs

# 🤗 HuggingFace Lab

A personal sandbox for testing and experimenting with HuggingFace models, pipelines, datasets, and APIs.

## 📁 Structure
huggingface-lab/
├── pipelines/ # Quick pipeline experiments (text, image, audio, etc.)
├── models/ # Custom model loading and fine-tuning experiments
├── datasets/ # Dataset loading and preprocessing
├── training/ # Chatbot fine-tuning (SFT + LoRA) and guides
├── notebooks/ # Jupyter notebooks for exploration
└── utils/ # Shared utility functions

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your HuggingFace token
```bash
export HUGGINGFACE_TOKEN=your_token_here
# Or use the HF CLI:
huggingface-cli login
```

### 3. Run a quick test
```bash
python pipelines/text_generation.py
```

### 4. Train a chatbot (GPU + LoRA)
See **[training/README.md](training/README.md)** for the full learning path.

```bash
python training/train_chatbot.py
python training/chat_with_model.py --prompt "What is fine-tuning?"
```

## 🧪 Experiments

| File | Task | Model |
|------|------|-------|
| `pipelines/text_generation.py` | Text Generation | GPT-2 |
| `pipelines/sentiment_analysis.py` | Sentiment Analysis | distilbert-base-uncased |
| `pipelines/image_classification.py` | Image Classification | ViT |
| `pipelines/summarization.py` | Summarization | BART |

## 📚 Resources
- https://huggingface.co/docs
- https://huggingface.co/docs/transformers
- https://huggingface.co/models
- https://huggingface.co/datasets