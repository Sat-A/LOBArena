import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from LOBArena.evaluate.adversarial import (
    _resolve_competitors,
    _validate_competitor_spec,
    load_competitor_registry,
    main,
)
from LOBArena.evaluate.phase2_contract import build_campaign_summary_payload


def test_adversarial_competitor_registry_exists_and_parseable():
    p = Path("/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json")
    assert p.exists()
    data = json.loads(p.read_text())
    assert "competitors" in data
    assert "random_baseline" in data["competitors"]
    assert "fixed_baseline_hold" in data["competitors"]
    assert "ippo_rnn_placeholder" in data["competitors"]
    assert "directional_baseline" in data["competitors"]
    assert "scripted_directional_placeholder" in data["competitors"]


def test_load_competitor_registry_returns_mapping():
    p = Path("/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json")
    registry = load_competitor_registry(p)
    assert isinstance(registry, dict)
    assert registry["random_baseline"]["policy_mode"] == "random"


def test_validate_competitor_spec_accepts_directional_mode():
    spec = {"policy_mode": "directional"}
    _validate_competitor_spec("directional_baseline", spec)


def test_validate_competitor_spec_rejects_unsupported_scripted_mode():
    with pytest.raises(ValueError, match="unsupported"):
        _validate_competitor_spec(
            "scripted_directional_placeholder",
            {"policy_mode": "scripted_directional"},
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


def _write_valid_policy_handoff(tmp_path: Path) -> Path:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg = tmp_path / "policy.yaml"
    cfg.write_text("seed: 1\n")
    handoff = tmp_path / "policy_handoff.generated.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "policy": {
                    "mode": "ippo_rnn",
                    "checkpoint_dir": str(ckpt),
                    "config_path": str(cfg),
                    "model_index": 1,
                },
                "restore_topology": {
                    "restore_strategy": "single_device_fallback",
                    "train_device_count": 1,
                    "eval_device_count": 1,
                },
                "evaluation": {"seed": 7, "date_window": {"start_date": "", "end_date": ""}},
                "provenance": {"run_id": "unit", "git_commit": ""},
            }
        )
    )
    return handoff


def _write_phase2_summary(tmp_path: Path, handoff_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary_path = run_dir / "train_eval_summary.json"
    payload = build_campaign_summary_payload(
        run_name="phase2_unit",
        train_data_dir=str(tmp_path / "train"),
        test_data_dir=str(tmp_path / "test"),
        train_steps=10,
        eval_steps=5,
        eval_rc=0,
        policy_mode="ippo_rnn",
        policy_ckpt_dir=str(tmp_path / "ckpt"),
        policy_config=str(tmp_path / "policy.yaml"),
        input_policy_handoff="",
        generated_policy_handoff=str(handoff_path),
        run_dir=run_dir,
        summary_path=summary_path,
        runtime_sec=0.5,
    )
    summary_path.write_text(json.dumps(payload))
    return summary_path


def test_validate_competitor_spec_accepts_phase2_campaign_summary_handoff(tmp_path: Path):
    handoff = _write_valid_policy_handoff(tmp_path)
    summary_path = _write_phase2_summary(tmp_path, handoff)
    spec = {"policy_handoff": str(summary_path)}

    _validate_competitor_spec("trained_policy", spec)

    assert spec["policy_mode"] == "ippo_rnn"
    assert spec["policy_handoff"] == str(handoff.resolve())
    assert spec["policy_ckpt_dir"] == str((tmp_path / "ckpt").resolve())
    assert spec["policy_config"] == str((tmp_path / "policy.yaml").resolve())


def test_validate_competitor_spec_campaign_summary_missing_handoff_artifact(tmp_path: Path):
    missing_handoff = tmp_path / "missing_policy_handoff.generated.json"
    summary_path = _write_phase2_summary(tmp_path, missing_handoff)
    spec = {"policy_handoff": str(summary_path)}

    with pytest.raises(FileNotFoundError, match="references missing policy handoff artifact"):
        _validate_competitor_spec("trained_policy", spec)


def test_validate_competitor_spec_rejects_invalid_handoff_payload(tmp_path: Path):
    bad = tmp_path / "bad_handoff.json"
    bad.write_text(json.dumps({"not": "a valid handoff or campaign summary"}))
    spec = {"policy_handoff": str(bad)}

    with pytest.raises(ValueError, match="Expected either a policy_handoff artifact payload or a phase2 campaign summary payload"):
        _validate_competitor_spec("trained_policy", spec)


def test_validate_competitor_spec_campaign_summary_without_handoff_raises(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg = tmp_path / "policy.yaml"
    cfg.write_text("seed: 1\n")
    summary_path = run_dir / "train_eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "contract_version": "phase2-alpha/1.0",
                "run_name": "phase2_no_handoff",
                "status": "success",
                "train": {"data_dir": str(tmp_path / "train"), "steps": 10},
                "evaluation": {"data_dir": str(tmp_path / "test"), "steps": 5, "eval_rc": 0, "run_name": "phase2_no_handoff_eval"},
                "policy": {
                    "mode": "ippo_rnn",
                    "checkpoint_dir": str(ckpt),
                    "config_path": str(cfg),
                    "input_handoff": "",
                    "generated_handoff": "",
                },
                "artifacts": {"run_dir": str(run_dir), "summary_path": str(summary_path)},
                "runtime_sec": 0.1,
            }
        )
    )
    spec = {"policy_handoff": str(summary_path)}
    with pytest.raises(ValueError, match="does not include a policy handoff path"):
        _validate_competitor_spec("trained_policy", spec)


def test_validate_competitor_spec_handoff_json_parse_error_is_explicit(tmp_path: Path):
    invalid = tmp_path / "invalid_handoff.json"
    invalid.write_text("{not-json")
    spec = {"policy_handoff": str(invalid)}
    with pytest.raises(ValueError, match="is not valid JSON"):
        _validate_competitor_spec("trained_policy", spec)


def test_adversarial_main_rejects_multi_node_env(monkeypatch, tmp_path: Path):
    args = SimpleNamespace(
        data_dir=str(tmp_path / "data"),
        target_policy_mode="random",
        target_fixed_action=0,
        target_policy_ckpt="",
        target_policy_config="",
        target_policy_handoff="",
        competitor_policy_mode="random",
        competitor_fixed_action=0,
        competitor_policy_ckpt="",
        competitor_policy_config="",
        competitor_policy_handoff="",
        competitor_registry_config="/unused/registry.json",
        competitor_keys=[],
        output_root=str(tmp_path / "outputs"),
        run_name="adv_guard",
        n_steps=1,
        seed=1,
        sample_index=0,
        test_split=1.0,
        start_date="",
        end_date="",
        round_robin=False,
    )
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "3")
    monkeypatch.setattr("LOBArena.evaluate.adversarial.parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="single-node execution only"):
        main()
