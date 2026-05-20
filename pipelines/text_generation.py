from transformers import pipeline

def run_text_generation(prompt: str,
                        model: str = "gpt2",
                        max_length: int = 100):
    generator = pipeline("text-generation", model=model)
    results = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        truncation=True,
    )
    return results[0]["generated_text"]

if __name__ == "__main__":
    prompt = "The future of AI in software development is"
    print(f"Prompt: {prompt}\n")
    output = run_text_generation(prompt)
    print("Generated:\n")
    print(output)