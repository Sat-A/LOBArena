
from pathlib import Path
import sys
from collections.abc import Mapping


class PolicySelection(object):
    def __init__(self, mode, fixed_action, checkpoint_dir, config_path):
        self.mode = mode
        self.fixed_action = fixed_action
        self.checkpoint_dir = checkpoint_dir
        self.config_path = config_path


def validate_policy_choice(mode, fixed_action, policy_ckpt_dir, policy_config):
    m = mode.strip().lower()
    if m not in {"random", "fixed", "ippo_rnn", "lose_money", "directional"}:
        raise ValueError(f"Invalid policy_mode: {mode}. Use random|fixed|ippo_rnn|lose_money|directional")

    ckpt = Path(policy_ckpt_dir).expanduser().resolve() if policy_ckpt_dir else None
    cfg = Path(policy_config).expanduser().resolve() if policy_config else None

    if m == "ippo_rnn":
        if ckpt is None or cfg is None:
            raise ValueError("ippo_rnn policy mode requires --policy_ckpt_dir and --policy_config")
        if not ckpt.exists():
            raise FileNotFoundError(f"Policy checkpoint dir not found: {ckpt}")
        if not cfg.exists():
            raise FileNotFoundError(f"Policy config file not found: {cfg}")

    return PolicySelection(mode=m, fixed_action=int(fixed_action), checkpoint_dir=ckpt, config_path=cfg)


def _normalize_flax_variables_tree(params):
    """
    Normalize restored train_state params into a Flax variables tree.

    Expected shape for apply_fn is usually {"params": <param_tree>}. Some
    checkpoints restore as a bare param tree or as nested {"params":{"params":...}}.
    """
    tree = params
    while (
        isinstance(tree, Mapping)
        and set(tree.keys()) == {"params"}
        and isinstance(tree.get("params"), Mapping)
        and set(tree["params"].keys()) == {"params"}
    ):
        tree = tree["params"]
    if isinstance(tree, Mapping) and "params" not in tree:
        return {"params": tree}
    return tree


def _stabilize_adapter_train_state(adapter):
    normalized = _normalize_flax_variables_tree(adapter.train_state.params)
    adapter.train_state = adapter.train_state.replace(params=normalized)


def _latest_checkpoint_step_from_dir(checkpoint_dir):
    root = Path(checkpoint_dir).expanduser().resolve()
    step_dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        if (child / "state").is_dir():
            step_dirs.append(int(child.name))
    if not step_dirs:
        raise RuntimeError(f"No checkpoint step/state directories found under {root}")
    return max(step_dirs)


def _extract_policy_params_tree(restored, model_index):
    if hasattr(restored, "params"):
        return restored.params
    if isinstance(restored, Mapping):
        if "model" in restored:
            model = restored.get("model")
            if isinstance(model, (list, tuple)) and len(model) > 0:
                idx = max(0, min(int(model_index), len(model) - 1))
                return _extract_policy_params_tree(model[idx], model_index)
            return _extract_policy_params_tree(model, model_index)
        if "state" in restored:
            return _extract_policy_params_tree(restored.get("state"), model_index)
        if "params" in restored:
            return restored.get("params")
        return restored
    if isinstance(restored, (list, tuple)) and len(restored) > 0:
        idx = max(0, min(int(model_index), len(restored) - 1))
        return _extract_policy_params_tree(restored[idx], model_index)
    raise RuntimeError("Unable to extract policy params from restored checkpoint payload")


def _restore_policy_params_with_cpu_fallback(checkpoint_dir, checkpoint_step, model_index):
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp

    step = int(checkpoint_step) if checkpoint_step is not None else _latest_checkpoint_step_from_dir(checkpoint_dir)
    state_dir = Path(checkpoint_dir).expanduser().resolve() / str(step) / "state"
    if not state_dir.is_dir():
        raise FileNotFoundError(f"Policy checkpoint state dir not found: {state_dir}")

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored = checkpointer.restore(str(state_dir))
    except Exception as direct_err:
        try:
            meta_tree = checkpointer.metadata(str(state_dir)).tree
        except Exception as meta_err:
            raise RuntimeError(
                f"Policy checkpoint restore failed for {state_dir}. "
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
                f"Policy checkpoint restore failed for {state_dir} with CPU fallback. "
                f"Direct restore error: {direct_err}. Fallback restore error: {fallback_err}."
            ) from fallback_err

    params = _extract_policy_params_tree(restored, model_index=model_index)
    return _normalize_flax_variables_tree(params)


class _FallbackLearnedPolicyAdapter:
    def __init__(self, checkpoint_dir, config_path, obs_dim, action_dim, seed, checkpoint_step, deterministic, model_index):
        import jax
        import jax.numpy as jnp
        from omegaconf import OmegaConf
        from gymnax_exchange.jaxrl.MARL.baseline_eval.baseline_JAXMARL import ActorCriticRNN, ScannedRNN

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.deterministic = bool(deterministic)
        self.model_index = int(model_index)
        self.rng = jax.random.PRNGKey(int(seed))

        cfg_raw = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
        if not isinstance(cfg_raw, dict):
            raise ValueError(f"Invalid policy config format: {config_path}")
        self.config = dict(cfg_raw)
        self.config["NUM_ENVS"] = 1
        self.config["GRU_HIDDEN_DIM"] = int(self.config.get("GRU_HIDDEN_DIM", 256))
        self.config["FC_DIM_SIZE"] = int(self.config.get("FC_DIM_SIZE", 256))

        params = _restore_policy_params_with_cpu_fallback(
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=checkpoint_step,
            model_index=self.model_index,
        )
        if isinstance(params, Mapping):
            dense_k = (
                params.get("params", {})
                .get("Dense_0", {})
                .get("kernel", None)
                if isinstance(params.get("params", {}), Mapping)
                else None
            )
            if hasattr(dense_k, "shape") and len(dense_k.shape) == 2:
                self.obs_dim = int(dense_k.shape[0])
            out_k = (
                params.get("params", {})
                .get("SingleActionOutput_0", {})
                .get("Dense_0", {})
                .get("kernel", None)
                if isinstance(params.get("params", {}), Mapping)
                else None
            )
            if hasattr(out_k, "shape") and len(out_k.shape) == 2:
                self.action_dim = int(out_k.shape[1])

        self.network = ActorCriticRNN(self.action_dim, config=self.config)
        self.params = params
        self.hidden = ScannedRNN.initialize_carry(1, int(self.config["GRU_HIDDEN_DIM"]))
        self._apply_jit = jax.jit(self.network.apply)

    def act_with_state(self, obs_vec, hidden, done=False):
        import jax
        import jax.numpy as jnp

        if obs_vec.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Policy obs dim mismatch: expected {self.obs_dim}, got {obs_vec.shape[-1]}"
            )
        dones = jnp.array([done], dtype=jnp.bool_)
        ac_in = (obs_vec.reshape(1, 1, -1), dones.reshape(1, 1))
        new_hidden, pi, _value = self._apply_jit(self.params, hidden, ac_in)
        if self.deterministic and hasattr(pi, "logits"):
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            self.rng, sample_rng = jax.random.split(self.rng)
            action = pi.sample(seed=sample_rng)
        return int(jnp.asarray(action).reshape(-1)[0]), new_hidden

    def fresh_hidden(self):
        from gymnax_exchange.jaxrl.MARL.baseline_eval.baseline_JAXMARL import ScannedRNN

        return ScannedRNN.initialize_carry(1, int(self.config["GRU_HIDDEN_DIM"]))


def _instantiate_policy_adapter(checkpoint_dir, config_path, obs_dim, action_dim, seed, checkpoint_step, deterministic, model_index):
    from run_learned_mm_worldmodel_rollout import LearnedPolicyAdapter  # type: ignore

    try:
        return LearnedPolicyAdapter(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=int(seed),
            checkpoint_step=checkpoint_step,
            deterministic=deterministic,
            model_index=int(model_index),
        )
    except Exception as err:
        print(f"[LOBArena] LearnedPolicyAdapter restore failed, using fallback restore path: {err}")
        return _FallbackLearnedPolicyAdapter(
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            seed=seed,
            checkpoint_step=checkpoint_step,
            deterministic=deterministic,
            model_index=model_index,
        )


def load_ippo_policy_adapter(jaxmarl_root, checkpoint_dir, config_path, seed):
    root = str(jaxmarl_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Keep dims aligned with JaxMARL-HFT defaults.
    adapter = _instantiate_policy_adapter(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        obs_dim=12,
        action_dim=5,
        seed=int(seed),
        checkpoint_step=None,
        deterministic=True,
        model_index=1,
    )
    if hasattr(adapter, "train_state"):
        _stabilize_adapter_train_state(adapter)
    return adapter


def load_ippo_policy_adapter_with_index(jaxmarl_root, checkpoint_dir, config_path, seed, model_index):
    root = str(jaxmarl_root)
    if root not in sys.path:
        sys.path.insert(0, root)

    adapter = _instantiate_policy_adapter(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        obs_dim=12,
        action_dim=5,
        seed=int(seed),
        checkpoint_step=None,
        deterministic=True,
        model_index=int(model_index),
    )
    if hasattr(adapter, "train_state"):
        _stabilize_adapter_train_state(adapter)
    return adapter
