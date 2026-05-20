from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

def classify_image(image_url: str,
                   model: str = "google/vit-base-patch16-224",
                   top_k: int = 5):
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    image = Image.open(BytesIO(resp.content))

    classifier = pipeline("image-classification", model=model)
    return classifier(image, top_k=top_k)

if __name__ == "__main__":
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
    print(f"Classifying image from:\n{url}\n")
    preds = classify_image(url)
    print("Top predictions:")
    for p in preds:
        print(f"{p['label']:<30} {p['score']:.2%}")