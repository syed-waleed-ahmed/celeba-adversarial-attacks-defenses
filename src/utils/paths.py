from pathlib import Path

# repo_root/src/utils/paths.py -> repo_root
ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    return ensure_dir(ROOT / "data")


def ckpt_dir() -> Path:
    return ensure_dir(ROOT / "checkpoints")


def outputs_dir() -> Path:
    return ensure_dir(ROOT / "outputs")
