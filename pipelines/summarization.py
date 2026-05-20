from transformers import pipeline

def summarize(text: str,
              model: str = "facebook/bart-large-cnn",
              max_length: int = 130,
              min_length: int = 30):
    summarizer = pipeline("summarization", model=model)
    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return result[0]["summary_text"]

if __name__ == "__main__":
    article = """
    HuggingFace is a company and open-source community that focuses on natural
    language processing (NLP) and machine learning. The company is known for its
    Transformers library, which provides thousands of pre-trained models for tasks
    like text classification, named entity recognition, question answering,
    summarization, translation, and text generation.
    """
    print("Original text snippet:")
    print(article.strip()[:200] + "...\n")
    print("Summary:")
    print(summarize(article))