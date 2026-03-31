from pathlib import Path
from types import SimpleNamespace

import pytest

from LOBArena.evaluate import policy_adapter


def test_normalize_flax_variables_tree_wraps_bare_param_tree():
    raw = {"Dense_0": {"kernel": [1], "bias": [0]}}
    out = policy_adapter._normalize_flax_variables_tree(raw)
    assert isinstance(out, dict)
    assert "params" in out
    assert out["params"] == raw


def test_normalize_flax_variables_tree_flattens_double_params_wrapper():
    raw = {"params": {"params": {"Dense_0": {"kernel": [1], "bias": [0]}}}}
    out = policy_adapter._normalize_flax_variables_tree(raw)
    assert isinstance(out, dict)
    assert "params" in out
    assert "Dense_0" in out["params"]


def test_load_ippo_policy_adapter_with_index_normalizes_train_state(monkeypatch: pytest.MonkeyPatch):
    class DummyAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.train_state = SimpleNamespace(
                params={"Dense_0": {"kernel": [1], "bias": [0]}},
                replace=lambda **repl: SimpleNamespace(
                    params=repl.get("params"),
                    replace=lambda **repl2: SimpleNamespace(params=repl2.get("params")),
                ),
            )

    monkeypatch.setitem(__import__("sys").modules, "run_learned_mm_worldmodel_rollout", SimpleNamespace(LearnedPolicyAdapter=DummyAdapter))
    adapter = policy_adapter.load_ippo_policy_adapter_with_index(
        jaxmarl_root=Path("."),
        checkpoint_dir=Path("."),
        config_path=Path("."),
        seed=123,
        model_index=2,
    )
    assert "params" in adapter.train_state.params
    assert "Dense_0" in adapter.train_state.params["params"]
    assert int(adapter.kwargs["model_index"]) == 2


def test_extract_policy_params_tree_handles_model_list():
    restored = {
        "model": [
            {"params": {"params": {"Dense_0": {"kernel": [1], "bias": [0]}}}},
            {"params": {"Dense_0": {"kernel": [2], "bias": [0]}}},
        ]
    }
    out = policy_adapter._extract_policy_params_tree(restored, model_index=1)
    assert isinstance(out, dict)
    assert "Dense_0" in out


def test_instantiate_policy_adapter_falls_back(monkeypatch: pytest.MonkeyPatch):
    class BrokenAdapter:
        def __init__(self, **kwargs):
            raise RuntimeError("forced restore failure")

    class DummyFallback:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        __import__("sys").modules,
        "run_learned_mm_worldmodel_rollout",
        SimpleNamespace(LearnedPolicyAdapter=BrokenAdapter),
    )
    monkeypatch.setattr(policy_adapter, "_FallbackLearnedPolicyAdapter", DummyFallback)
    out = policy_adapter._instantiate_policy_adapter(
        checkpoint_dir=Path("."),
        config_path=Path("."),
        obs_dim=12,
        action_dim=5,
        seed=7,
        checkpoint_step=None,
        deterministic=True,
        model_index=0,
    )
    assert isinstance(out, DummyFallback)
    assert int(out.kwargs["seed"]) == 7
