import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from LOBArena.evaluate import pipeline
from LOBArena.evaluate.world_model_selector import validate_world_model_choice


def _base_args():
    root = Path(__file__).resolve().parents[2]
    return SimpleNamespace(
        world_model="historical",
        policy_mode="random",
        fixed_action=0,
        jaxmarl_root=str(root / "JaxMARL-HFT"),
        lobs5_root=str(root / "LOBS5"),
        lobs5_ckpt_path="",
        policy_ckpt_dir="",
        policy_config="",
        policy_handoff="",
        policy_handoff_batch=None,
        policy_handoff_manifest="",
        data_dir=str(root / "LOBArena"),
        sample_index=0,
        checkpoint_step=None,
        test_split=1.0,
        start_date="",
        end_date="",
        n_cond_msgs=64,
        n_steps=2,
        sample_top_n=1,
        seed=42,
        output_root=str(root / "LOBArena" / "tests" / ".tmp_outputs"),
        run_name="rt_args",
        fast_startup=True,
        cpu_safe=False,
        device="auto",
        strict_generative=False,
    )


def test_validate_world_model_choice_requires_ckpt_for_generative():
    with pytest.raises(ValueError, match="lobs5_ckpt_path"):
        validate_world_model_choice("generative", None)


def test_resolve_batch_candidates_rejects_empty_manifest_candidates(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"candidates": []}))
    args = _base_args()
    args.policy_handoff_manifest = str(manifest)
    with pytest.raises(ValueError, match="non-empty 'candidates'"):
        pipeline.resolve_batch_candidates(args)


def test_runtime_metadata_captures_requested_device():
    args = _base_args()
    args.device = "cpu"
    args.cpu_safe = True
    pipeline._configure_runtime(args)
    md = pipeline._runtime_metadata(args, Path(args.data_dir), jax_backend="cpu", jax_devices=["cpu:0"])
    assert md["requested_device"] == "cpu"
    assert md["cpu_safe"] is True
    assert md["thread_env"]["JAX_PLATFORM_NAME"] == "cpu"

