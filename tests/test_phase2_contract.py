import json
from pathlib import Path

import pytest

from LOBArena.evaluate.phase2_contract import (
    PHASE2_ALPHA_CONTRACT_VERSION,
    build_campaign_summary_payload,
    generate_policy_handoff_artifact,
    load_phase2_alpha_contract_spec,
    validate_campaign_summary_payload,
)


def _tmp_policy_inputs(tmp_path: Path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg = tmp_path / "policy.yaml"
    cfg.write_text("seed: 1\n")
    return ckpt, cfg


def test_phase2_contract_spec_matches_module_version():
    spec = load_phase2_alpha_contract_spec()
    assert spec["contract_version"] == PHASE2_ALPHA_CONTRACT_VERSION


def test_phase2_contract_spec_version_mismatch_raises(tmp_path: Path, monkeypatch):
    bad_spec_path = tmp_path / "phase2_alpha_contract.json"
    bad_spec_path.write_text(
        json.dumps(
            {
                "contract_name": "LOBArena Phase2 Train-Handoff-Eval Contract",
                "contract_version": "phase2-alpha/9.9",
            }
        )
    )
    monkeypatch.setattr("LOBArena.evaluate.phase2_contract.PHASE2_ALPHA_CONTRACT_SPEC_PATH", bad_spec_path)
    with pytest.raises(ValueError, match="Contract spec version mismatch"):
        load_phase2_alpha_contract_spec()


def test_build_campaign_summary_includes_legacy_fields(tmp_path: Path):
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
        policy_mode="random",
        policy_ckpt_dir="",
        policy_config="",
        input_policy_handoff="",
        generated_policy_handoff="",
        run_dir=run_dir,
        summary_path=summary_path,
        runtime_sec=1.25,
    )
    assert payload["contract_version"] == PHASE2_ALPHA_CONTRACT_VERSION
    assert payload["status"] == "success"
    assert payload["policy_mode"] == "random"
    assert payload["eval_rc"] == 0
    assert payload["artifacts"]["summary_path"] == str(summary_path.resolve())
    assert payload["lineage"] == {"train": {}, "handoff": {}, "evaluation": {}}


def test_build_campaign_summary_captures_lineage_fields(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary_path = run_dir / "train_eval_summary.json"
    payload = build_campaign_summary_payload(
        run_name="phase2_unit",
        train_data_dir=str(tmp_path / "train"),
        test_data_dir=str(tmp_path / "test"),
        train_steps=10,
        eval_steps=5,
        eval_rc=1,
        policy_mode="ippo_rnn",
        policy_ckpt_dir=str(tmp_path / "ckpt"),
        policy_config=str(tmp_path / "policy.yaml"),
        input_policy_handoff="",
        generated_policy_handoff=str(tmp_path / "policy_handoff.generated.json"),
        run_dir=run_dir,
        summary_path=summary_path,
        runtime_sec=2.0,
        lineage={
            "train": {"invoked": True, "status": "success"},
            "handoff": {"status": "generated_handoff"},
            "evaluation": {"status": "failed", "rc": 1},
        },
    )
    assert payload["status"] == "failed"
    assert payload["lineage"]["train"]["invoked"] is True
    assert payload["lineage"]["handoff"]["status"] == "generated_handoff"
    assert payload["lineage"]["evaluation"]["rc"] == 1


def test_validate_campaign_summary_rejects_bad_version(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "contract_version": "phase2-alpha/0.9",
        "run_name": "bad",
        "status": "success",
        "train": {"data_dir": str(tmp_path), "steps": 1},
        "evaluation": {"data_dir": str(tmp_path), "steps": 1, "eval_rc": 0, "run_name": "bad_eval"},
        "policy": {
            "mode": "random",
            "checkpoint_dir": "",
            "config_path": "",
            "input_handoff": "",
            "generated_handoff": "",
        },
        "artifacts": {"run_dir": str(run_dir), "summary_path": str(run_dir / "train_eval_summary.json")},
        "runtime_sec": 0.1,
    }
    with pytest.raises(ValueError, match="Unsupported contract_version"):
        validate_campaign_summary_payload(payload, base_dir=run_dir)


def test_validate_campaign_summary_rejects_missing_required_field(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
        "run_name": "bad",
        "status": "success",
        "train": {"data_dir": str(tmp_path), "steps": 1},
        "policy": {
            "mode": "random",
            "checkpoint_dir": "",
            "config_path": "",
            "input_handoff": "",
            "generated_handoff": "",
        },
        "artifacts": {"run_dir": str(run_dir), "summary_path": str(run_dir / "train_eval_summary.json")},
        "runtime_sec": 0.1,
    }
    with pytest.raises(ValueError, match="missing required fields"):
        validate_campaign_summary_payload(payload, base_dir=run_dir)


def test_validate_campaign_summary_rejects_unsupported_top_level_field(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
        "run_name": "bad_extra",
        "status": "success",
        "train": {"data_dir": str(tmp_path), "steps": 1},
        "evaluation": {"data_dir": str(tmp_path), "steps": 1, "eval_rc": 0, "run_name": "bad_extra_eval"},
        "policy": {
            "mode": "random",
            "checkpoint_dir": "",
            "config_path": "",
            "input_handoff": "",
            "generated_handoff": "",
        },
        "artifacts": {"run_dir": str(run_dir), "summary_path": str(run_dir / "train_eval_summary.json")},
        "runtime_sec": 0.1,
        "unexpected": "nope",
    }
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_campaign_summary_payload(payload, base_dir=run_dir)


def test_validate_campaign_summary_rejects_invalid_status(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
        "run_name": "bad_status",
        "status": "pending",
        "train": {"data_dir": str(tmp_path), "steps": 1},
        "evaluation": {"data_dir": str(tmp_path), "steps": 1, "eval_rc": 0, "run_name": "bad_status_eval"},
        "policy": {
            "mode": "random",
            "checkpoint_dir": "",
            "config_path": "",
            "input_handoff": "",
            "generated_handoff": "",
        },
        "artifacts": {"run_dir": str(run_dir), "summary_path": str(run_dir / "train_eval_summary.json")},
        "runtime_sec": 0.1,
    }
    with pytest.raises(ValueError, match="status must be 'success' or 'failed'"):
        validate_campaign_summary_payload(payload, base_dir=run_dir)


def test_validate_campaign_summary_requires_ippo_materialization(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
        "run_name": "bad_ippo",
        "status": "success",
        "train": {"data_dir": str(tmp_path), "steps": 1},
        "evaluation": {"data_dir": str(tmp_path), "steps": 1, "eval_rc": 0, "run_name": "bad_ippo_eval"},
        "policy": {
            "mode": "ippo_rnn",
            "checkpoint_dir": "",
            "config_path": "",
            "input_handoff": "",
            "generated_handoff": "",
        },
        "artifacts": {"run_dir": str(run_dir), "summary_path": str(run_dir / "train_eval_summary.json")},
        "runtime_sec": 0.1,
    }
    with pytest.raises(ValueError, match="requires a handoff path or both checkpoint_dir and config_path"):
        validate_campaign_summary_payload(payload, base_dir=run_dir)


def test_validate_campaign_summary_normalizes_policy_mode_case(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = validate_campaign_summary_payload(
        {
            "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
            "run_name": "mode_case",
            "status": "success",
            "train": {"data_dir": str(tmp_path / "train"), "steps": 1},
            "evaluation": {"data_dir": str(tmp_path / "test"), "steps": 1, "eval_rc": 0, "run_name": "mode_case_eval"},
            "policy": {
                "mode": "IPPO_RNN",
                "checkpoint_dir": "ckpt",
                "config_path": "policy.yaml",
                "input_handoff": "",
                "generated_handoff": "handoff.json",
            },
            "artifacts": {"run_dir": str(run_dir), "summary_path": str(run_dir / "train_eval_summary.json")},
            "runtime_sec": 0.1,
        },
        base_dir=tmp_path,
    )
    assert payload["policy"]["mode"] == "ippo_rnn"


def test_generate_policy_handoff_artifact_writes_valid_payload(tmp_path: Path):
    ckpt, cfg = _tmp_policy_inputs(tmp_path)
    out = tmp_path / "policy_handoff.generated.json"
    handoff = generate_policy_handoff_artifact(
        {
            "policy_mode": "ippo_rnn",
            "checkpoint_dir": str(ckpt),
            "config_path": str(cfg),
            "model_index": 1,
            "restore_topology": {
                "restore_strategy": "single_device_fallback",
                "train_device_count": 2,
                "eval_device_count": 1,
            },
            "evaluation": {
                "seed": 7,
                "start_date": "2024-01-01",
                "end_date": "2024-01-02",
            },
            "provenance": {"run_id": "unit-run", "git_commit": "abc123"},
            "output_path": str(out),
        },
        base_dir=tmp_path,
    )
    assert out.exists()
    persisted = json.loads(out.read_text())
    assert persisted["schema_version"] == "1.0"
    assert persisted["policy"]["mode"] == "ippo_rnn"
    assert handoff["_artifact_path"] == str(out.resolve())


def test_generate_policy_handoff_artifact_rejects_missing_required_field(tmp_path: Path):
    ckpt, cfg = _tmp_policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="missing required fields"):
        generate_policy_handoff_artifact(
            {
                "policy_mode": "ippo_rnn",
                "checkpoint_dir": str(ckpt),
                "config_path": str(cfg),
                "model_index": 0,
                "evaluation": {"seed": 1, "start_date": "", "end_date": ""},
                "provenance": {"run_id": "unit-run", "git_commit": ""},
                "output_path": str(tmp_path / "x.json"),
            },
            base_dir=tmp_path,
        )


def test_generate_policy_handoff_artifact_rejects_bad_restore_strategy(tmp_path: Path):
    ckpt, cfg = _tmp_policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="restore_strategy must be 'direct' or 'single_device_fallback'"):
        generate_policy_handoff_artifact(
            {
                "policy_mode": "ippo_rnn",
                "checkpoint_dir": str(ckpt),
                "config_path": str(cfg),
                "model_index": 0,
                "restore_topology": {
                    "restore_strategy": "bad_strategy",
                    "train_device_count": 1,
                    "eval_device_count": 1,
                },
                "evaluation": {
                    "seed": 7,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                },
                "provenance": {"run_id": "unit-run", "git_commit": "abc123"},
                "output_path": str(tmp_path / "policy_handoff.generated.json"),
            },
            base_dir=tmp_path,
        )


def test_generate_policy_handoff_artifact_rejects_invalid_model_index(tmp_path: Path):
    ckpt, cfg = _tmp_policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="model_index must be >= 0"):
        generate_policy_handoff_artifact(
            {
                "policy_mode": "ippo_rnn",
                "checkpoint_dir": str(ckpt),
                "config_path": str(cfg),
                "model_index": -1,
                "restore_topology": {
                    "restore_strategy": "single_device_fallback",
                    "train_device_count": 1,
                    "eval_device_count": 1,
                },
                "evaluation": {
                    "seed": 7,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                },
                "provenance": {"run_id": "unit-run", "git_commit": "abc123"},
                "output_path": str(tmp_path / "policy_handoff.generated.json"),
            },
            base_dir=tmp_path,
        )


def test_generate_policy_handoff_artifact_rejects_invalid_paths(tmp_path: Path):
    missing_ckpt = tmp_path / "missing_ckpt"
    missing_cfg = tmp_path / "missing_policy.yaml"
    with pytest.raises(FileNotFoundError, match="Policy checkpoint dir not found"):
        generate_policy_handoff_artifact(
            {
                "policy_mode": "ippo_rnn",
                "checkpoint_dir": str(missing_ckpt),
                "config_path": str(missing_cfg),
                "model_index": 0,
                "restore_topology": {
                    "restore_strategy": "single_device_fallback",
                    "train_device_count": 1,
                    "eval_device_count": 1,
                },
                "evaluation": {
                    "seed": 7,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                },
                "provenance": {"run_id": "unit-run", "git_commit": "abc123"},
                "output_path": str(tmp_path / "policy_handoff.generated.json"),
            },
            base_dir=tmp_path,
        )


def test_generate_policy_handoff_artifact_rejects_non_ippo_mode(tmp_path: Path):
    ckpt, cfg = _tmp_policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="only supports policy_mode='ippo_rnn'"):
        generate_policy_handoff_artifact(
            {
                "policy_mode": "random",
                "checkpoint_dir": str(ckpt),
                "config_path": str(cfg),
                "model_index": 0,
                "restore_topology": {
                    "restore_strategy": "single_device_fallback",
                    "train_device_count": 1,
                    "eval_device_count": 1,
                },
                "evaluation": {"seed": 1, "start_date": "", "end_date": ""},
                "provenance": {"run_id": "unit-run", "git_commit": ""},
                "output_path": str(tmp_path / "x.json"),
            },
            base_dir=tmp_path,
        )


def test_validate_campaign_summary_resolves_relative_paths(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = validate_campaign_summary_payload(
        {
            "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
            "run_name": "relpaths",
            "status": "success",
            "train": {"data_dir": "train_data", "steps": 10},
            "evaluation": {
                "data_dir": "test_data",
                "steps": 5,
                "eval_rc": 0,
                "run_name": "relpaths_test_eval",
            },
            "policy": {
                "mode": "ippo_rnn",
                "checkpoint_dir": "ckpt",
                "config_path": "cfg.yaml",
                "input_handoff": "",
                "generated_handoff": "policy_handoff.generated.json",
            },
            "artifacts": {"run_dir": "run", "summary_path": "run/train_eval_summary.json"},
            "runtime_sec": 0.2,
        },
        base_dir=tmp_path,
    )
    assert payload["train"]["data_dir"] == str((tmp_path / "train_data").resolve())
    assert payload["evaluation"]["data_dir"] == str((tmp_path / "test_data").resolve())
    assert payload["policy"]["checkpoint_dir"] == str((tmp_path / "ckpt").resolve())
    assert payload["policy"]["config_path"] == str((tmp_path / "cfg.yaml").resolve())
    assert payload["policy"]["generated_handoff"] == str(
        (tmp_path / "policy_handoff.generated.json").resolve()
    )
    assert payload["artifacts"]["run_dir"] == str(run_dir.resolve())


def test_validate_campaign_summary_rejects_negative_runtime(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="runtime_sec must be >= 0"):
        validate_campaign_summary_payload(
            {
                "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
                "run_name": "bad_runtime",
                "status": "success",
                "train": {"data_dir": str(tmp_path / "train"), "steps": 10},
                "evaluation": {
                    "data_dir": str(tmp_path / "test"),
                    "steps": 5,
                    "eval_rc": 0,
                    "run_name": "bad_runtime_test_eval",
                },
                "policy": {
                    "mode": "random",
                    "checkpoint_dir": "",
                    "config_path": "",
                    "input_handoff": "",
                    "generated_handoff": "",
                },
                "artifacts": {"run_dir": str(run_dir), "summary_path": str(run_dir / "train_eval_summary.json")},
                "runtime_sec": -0.1,
            },
            base_dir=tmp_path,
        )


def test_generate_policy_handoff_artifact_rejects_missing_nested_fields(tmp_path: Path):
    ckpt, cfg = _tmp_policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="restore_topology missing required fields"):
        generate_policy_handoff_artifact(
            {
                "policy_mode": "ippo_rnn",
                "checkpoint_dir": str(ckpt),
                "config_path": str(cfg),
                "model_index": 0,
                "restore_topology": {"train_device_count": 1, "eval_device_count": 1},
                "evaluation": {"seed": 1, "start_date": "", "end_date": ""},
                "provenance": {"run_id": "unit-run", "git_commit": ""},
                "output_path": str(tmp_path / "x.json"),
            },
            base_dir=tmp_path,
        )


def test_generate_policy_handoff_artifact_rejects_extra_fields(tmp_path: Path):
    ckpt, cfg = _tmp_policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="unsupported fields"):
        generate_policy_handoff_artifact(
            {
                "policy_mode": "ippo_rnn",
                "checkpoint_dir": str(ckpt),
                "config_path": str(cfg),
                "model_index": 0,
                "restore_topology": {
                    "restore_strategy": "single_device_fallback",
                    "train_device_count": 1,
                    "eval_device_count": 1,
                },
                "evaluation": {"seed": 1, "start_date": "", "end_date": ""},
                "provenance": {"run_id": "unit-run", "git_commit": ""},
                "output_path": str(tmp_path / "x.json"),
                "extra": "not-allowed",
            },
            base_dir=tmp_path,
        )
