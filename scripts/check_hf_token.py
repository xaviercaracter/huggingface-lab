from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from utils.env import get_hf_token, load_project_env  # noqa: E402


def main() -> None:
    load_project_env(repo_root)
    token = get_hf_token()
    if not token:
        raise SystemExit(
            "No token found. Create a `.env` in the repo root with "
            "HUGGINGFACE_HUB_TOKEN=... (or HF_TOKEN=...). You can copy from `.env.example`."
        )

    print("Hugging Face token loaded.")
    print(f"Prefix: {token[:8]}…  Length: {len(token)}")


if __name__ == "__main__":
    main()

