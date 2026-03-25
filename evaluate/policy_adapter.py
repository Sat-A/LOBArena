
from pathlib import Path
import sys


class PolicySelection(object):
    def __init__(self, mode, fixed_action, checkpoint_dir, config_path):
        self.mode = mode
        self.fixed_action = fixed_action
        self.checkpoint_dir = checkpoint_dir
        self.config_path = config_path


def validate_policy_choice(mode, fixed_action, policy_ckpt_dir, policy_config):
    m = mode.strip().lower()
    if m not in {"random", "fixed", "ippo_rnn", "lose_money"}:
        raise ValueError(f"Invalid policy_mode: {mode}. Use random|fixed|ippo_rnn|lose_money")

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


def load_ippo_policy_adapter(jaxmarl_root, checkpoint_dir, config_path, seed):
    root = str(jaxmarl_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from run_learned_mm_worldmodel_rollout import LearnedPolicyAdapter  # type: ignore

    # Keep dims aligned with JaxMARL-HFT defaults.
    return LearnedPolicyAdapter(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        obs_dim=12,
        action_dim=5,
        seed=int(seed),
        checkpoint_step=None,
        deterministic=True,
        model_index=1,
    )


def load_ippo_policy_adapter_with_index(jaxmarl_root, checkpoint_dir, config_path, seed, model_index):
    root = str(jaxmarl_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from run_learned_mm_worldmodel_rollout import LearnedPolicyAdapter  # type: ignore

    return LearnedPolicyAdapter(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        obs_dim=12,
        action_dim=5,
        seed=int(seed),
        checkpoint_step=None,
        deterministic=True,
        model_index=int(model_index),
    )
