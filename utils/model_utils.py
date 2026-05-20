from transformers import AutoTokenizer, AutoModel
import torch

def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def get_embeddings(text: str,
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    model, tokenizer = load_model_and_tokenizer(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

if __name__ == "__main__":
    text = "HuggingFace makes AI accessible to everyone."
    emb = get_embeddings(text)
    print(f"Embedding shape: {emb.shape}")
    print(f"First 5 values: {emb[0][:5].tolist()}")