
from pathlib import Path
from typing import Optional


class WorldModelSelection(object):
    def __init__(self, mode, lobs5_ckpt_path):
        self.mode = mode
        self.lobs5_ckpt_path = lobs5_ckpt_path


def validate_world_model_choice(mode, lobs5_ckpt_path):
    m = mode.strip().lower()
    if m not in {"historical", "generative"}:
        raise ValueError(f"Invalid world_model mode: {mode}. Use 'historical' or 'generative'.")

    ckpt = Path(lobs5_ckpt_path).expanduser().resolve() if lobs5_ckpt_path else None
    if m == "generative":
        if ckpt is None:
            raise ValueError("Generative mode requires --lobs5_ckpt_path.")
        if not ckpt.exists():
            raise FileNotFoundError(f"LOBS5 checkpoint path not found: {ckpt}")

    return WorldModelSelection(mode=m, lobs5_ckpt_path=ckpt)
