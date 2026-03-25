
from pathlib import Path
import sys


def add_jaxmarl_paths(jaxmarl_root: Path) -> None:
    root = str(jaxmarl_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def add_lobs5_paths(lobs5_root: Path) -> None:
    root = str(lobs5_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_world_model_from_jaxmarl(jaxmarl_root: Path, lobs5_root: Path):
    add_jaxmarl_paths(jaxmarl_root)
    from run_one_step_inference import (  # type: ignore
        _add_python_paths,
        _enable_legacy_token_mode_22,
        _ensure_model_args_defaults,
        _latest_checkpoint_step,
        _load_metadata_robust,
        _prepare_date_filtered_data_dir,
        _restore_params_only,
    )

    _add_python_paths(lobs5_root)
    return {
        "_enable_legacy_token_mode_22": _enable_legacy_token_mode_22,
        "_ensure_model_args_defaults": _ensure_model_args_defaults,
        "_latest_checkpoint_step": _latest_checkpoint_step,
        "_load_metadata_robust": _load_metadata_robust,
        "_prepare_date_filtered_data_dir": _prepare_date_filtered_data_dir,
        "_restore_params_only": _restore_params_only,
    }


def restore_params_with_cpu_fallback(ckpt_path, step):
    """Restore checkpoint params with single-device fallback for topology mismatch."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    state_dir = Path(ckpt_path) / str(step) / "state"
    if not state_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint state directory not found: {state_dir}")

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored = checkpointer.restore(str(state_dir))
    except Exception as direct_err:
        try:
            meta_tree = checkpointer.metadata(str(state_dir)).tree
        except Exception as meta_err:
            raise RuntimeError(
                f"Checkpoint restore failed for {state_dir}. "
                f"Direct restore error: {direct_err}. Metadata load error: {meta_err}."
            ) from meta_err
        single = jax.sharding.SingleDeviceSharding(jax.devices()[0])

        def _to_struct(x):
            if x is None:
                return None
            if hasattr(x, "shape") and hasattr(x, "dtype"):
                return jax.ShapeDtypeStruct(shape=tuple(x.shape), dtype=jnp.dtype(x.dtype), sharding=single)
            return x

        def _to_restore_arg(x):
            if isinstance(x, jax.ShapeDtypeStruct):
                return ocp.ArrayRestoreArgs(sharding=single)
            return None

        target = jax.tree_util.tree_map(_to_struct, meta_tree)
        restore_args = jax.tree_util.tree_map(_to_restore_arg, target)
        try:
            restored = checkpointer.restore(
                str(state_dir),
                args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args),
            )
        except Exception as fallback_err:
            raise RuntimeError(
                f"Checkpoint restore failed for {state_dir} with CPU fallback. "
                f"Direct restore error: {direct_err}. Fallback restore error: {fallback_err}."
            ) from fallback_err
    if not isinstance(restored, dict) or "params" not in restored:
        raise RuntimeError(f"Unexpected checkpoint state format in {state_dir}")
    return restored["params"]
