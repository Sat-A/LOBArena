from pathlib import Path

import pytest

from LOBArena.evaluate.checkpoint_loader import restore_params_with_cpu_fallback


def test_restore_params_with_cpu_fallback_missing_state_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Checkpoint state directory not found"):
        restore_params_with_cpu_fallback(tmp_path / "missing_ckpt", step=1)

