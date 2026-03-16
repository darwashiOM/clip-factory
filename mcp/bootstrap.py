from pathlib import Path
import os
from dotenv import load_dotenv


def resolve_root_and_load_env() -> Path:
    initial_root = Path(
        os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))
    ).resolve()

    load_dotenv(initial_root / ".env")

    root = Path(
        os.environ.get("CLIP_FACTORY_ROOT", str(Path.home() / "clip-factory"))
    ).resolve()

    if root != initial_root and (root / ".env").exists():
        load_dotenv(root / ".env", override=True)

    return root