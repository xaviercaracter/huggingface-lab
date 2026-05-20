from transformers import pipeline

def analyze_sentiment(texts,
                      model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    classifier = pipeline("sentiment-analysis", model=model)
    return classifier(texts)

if __name__ == "__main__":
    samples = [
        "I love working with HuggingFace models!",
        "This model takes forever to load.",
        "The results are surprisingly accurate.",
    ]
    results = analyze_sentiment(samples)
    print("Sentiment Analysis Results:")
    for text, res in zip(samples, results):
        print(f"[{res['label']} | {res['score']:.2%}] {text}")