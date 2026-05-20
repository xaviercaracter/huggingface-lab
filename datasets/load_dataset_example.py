from datasets import load_dataset

def explore_dataset(name: str,
                    split: str = "train",
                    num_samples: int = 5):
    print(f"Loading dataset: {name} ({split})")
    ds = load_dataset(name, split=split)
    print(f"Dataset size: {len(ds)} samples")
    print(f"Features: {ds.features}\n")

    print(f"First {num_samples} samples:")
    for i, sample in enumerate(ds.select(range(num_samples))):
        print(f"[{i}] {sample}")
    return ds

if __name__ == "__main__":
    explore_dataset("imdb", split="train", num_samples=3)