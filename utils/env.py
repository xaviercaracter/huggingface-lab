from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_project_env(project_root: Optional[str | Path] = None) -> None:
    root = Path(project_root) if project_root is not None else Path.cwd()
    env_path = root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def get_hf_token() -> Optional[str]:
    return os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

