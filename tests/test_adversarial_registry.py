import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from LOBArena.evaluate.adversarial import (
    _resolve_competitors,
    _validate_competitor_spec,
    load_competitor_registry,
)


def test_adversarial_competitor_registry_exists_and_parseable():
    p = Path("/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json")
    assert p.exists()
    data = json.loads(p.read_text())
    assert "competitors" in data
    assert "random_baseline" in data["competitors"]
    assert "fixed_baseline_hold" in data["competitors"]
    assert "ippo_rnn_placeholder" in data["competitors"]
    assert "scripted_directional_placeholder" in data["competitors"]


def test_load_competitor_registry_returns_mapping():
    p = Path("/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json")
    registry = load_competitor_registry(p)
    assert isinstance(registry, dict)
    assert registry["random_baseline"]["policy_mode"] == "random"


def test_validate_competitor_spec_rejects_unsupported_directional_placeholder():
    with pytest.raises(ValueError, match="unsupported"):
        _validate_competitor_spec(
            "scripted_directional_placeholder",
            {"policy_mode": "directional"},
        )


def test_validate_competitor_spec_requires_fixed_action_for_fixed_mode():
    with pytest.raises(ValueError, match="requires 'fixed_action'"):
        _validate_competitor_spec(
            "bad_fixed",
            {"policy_mode": "fixed"},
        )


def test_resolve_competitors_uses_direct_args_when_no_registry_key():
    args = SimpleNamespace(
        competitor_keys=[],
        competitor_registry_config="/unused/path.json",
        competitor_policy_mode="fixed",
        competitor_fixed_action=2,
        competitor_policy_ckpt="",
        competitor_policy_config="",
        competitor_policy_handoff="",
    )

    comps = _resolve_competitors(args)
    assert len(comps) == 1
    assert comps[0]["source"] == "direct_args"
    assert comps[0]["policy_mode"] == "fixed"
    assert comps[0]["fixed_action"] == 2


def test_resolve_competitors_by_registry_key():
    args = SimpleNamespace(
        competitor_keys=["random_baseline", "fixed_baseline_hold"],
        competitor_registry_config="/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json",
        competitor_policy_mode="random",
        competitor_fixed_action=0,
        competitor_policy_ckpt="",
        competitor_policy_config="",
        competitor_policy_handoff="",
    )

    comps = _resolve_competitors(args)
    assert [c["key"] for c in comps] == ["random_baseline", "fixed_baseline_hold"]
    assert all(c["source"] == "registry" for c in comps)


def test_resolve_competitors_unknown_registry_key_raises():
    args = SimpleNamespace(
        competitor_keys=["does_not_exist"],
        competitor_registry_config="/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json",
        competitor_policy_mode="random",
        competitor_fixed_action=0,
        competitor_policy_ckpt="",
        competitor_policy_config="",
        competitor_policy_handoff="",
    )

    with pytest.raises(KeyError, match="not found"):
        _resolve_competitors(args)


def test_validate_competitor_spec_accepts_loss_mode():
    _validate_competitor_spec(
        "loss_seeker",
        {"policy_mode": "lose_money"},
    )
