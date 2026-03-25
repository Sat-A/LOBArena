import json
from pathlib import Path

import pytest

from LOBArena.evaluate import adversarial, train_eval
from LOBArena.scripts import run_phase1_smoke


def test_run_phase1_smoke_defaults_are_repo_relative(monkeypatch):
    monkeypatch.setattr(
        run_phase1_smoke.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "data_dir": "/tmp/data",
                "jaxmarl_root": str(Path(__file__).resolve().parents[2] / "JaxMARL-HFT"),
                "lobs5_root": str(Path(__file__).resolve().parents[2] / "LOBS5"),
                "lobs5_ckpt_path": "",
                "policy_ckpt_dir": "",
                "policy_config": "",
                "policy_handoff": "",
                "output_root": str(Path(__file__).resolve().parents[2] / "LOBArena" / "outputs" / "evaluations"),
                "n_steps": 1,
            },
        )(),
    )
    calls = []
    monkeypatch.setattr(run_phase1_smoke, "run", lambda cmd: calls.append(cmd) or 0)
    rc = run_phase1_smoke.main()
    assert rc == 0
    assert calls
    assert any("--world_model" in c for c in calls[0])


def test_train_eval_parse_defaults_repo_relative(monkeypatch):
    monkeypatch.setattr(
        train_eval.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "train_data_dir": "/tmp/train",
                "test_data_dir": "/tmp/test",
                "jaxmarl_root": str(Path(__file__).resolve().parents[2] / "JaxMARL-HFT"),
                "output_root": str(Path(__file__).resolve().parents[2] / "LOBArena" / "outputs" / "evaluations"),
                "train_steps": 1,
                "eval_steps": 1,
                "policy_ckpt_dir": "",
                "policy_config": "",
                "policy_handoff": "",
                "run_name": "x",
                "fast_startup": False,
                "run_jaxmarl_train": False,
                "jaxmarl_train_cmd": "",
                "jaxmarl_train_timeout_sec": 1800,
            },
        )(),
    )
    args = train_eval.parse_args()
    assert "LOBArena/outputs/evaluations" in args.output_root


def test_adversarial_parse_defaults_repo_relative(monkeypatch):
    monkeypatch.setattr(
        adversarial.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "data_dir": "/tmp/data",
                "target_policy_mode": "random",
                "target_fixed_action": 0,
                "target_policy_ckpt": "",
                "target_policy_config": "",
                "target_policy_handoff": "",
                "competitor_policy_mode": "random",
                "competitor_fixed_action": 0,
                "competitor_policy_ckpt": "",
                "competitor_policy_config": "",
                "competitor_policy_handoff": "",
                "competitor_registry_config": "",
                "competitor_keys": [],
                "output_root": str(Path(__file__).resolve().parents[2] / "LOBArena" / "outputs" / "evaluations"),
                "run_name": "x",
                "n_steps": 1,
                "seed": 1,
                "sample_index": 0,
                "test_split": 1.0,
                "start_date": "",
                "end_date": "",
                "round_robin": False,
            },
        )(),
    )
    args = adversarial.parse_args()
    assert "LOBArena/outputs/evaluations" in args.output_root


def _train_eval_args(tmp_path: Path, **overrides):
    base = {
        "train_data_dir": str(tmp_path / "train"),
        "test_data_dir": str(tmp_path / "test"),
        "jaxmarl_root": str(Path(__file__).resolve().parents[2] / "JaxMARL-HFT"),
        "output_root": str(tmp_path / "outputs"),
        "train_steps": 3,
        "eval_steps": 4,
        "policy_ckpt_dir": "",
        "policy_config": "",
        "policy_handoff": "",
        "run_name": "unit_train_eval",
        "fast_startup": False,
        "run_jaxmarl_train": False,
        "jaxmarl_train_cmd": "",
        "jaxmarl_train_timeout_sec": 1800,
    }
    base.update(overrides)
    return type("Args", (), base)()


def test_train_eval_main_rejects_handoff_and_ckpt_together(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        train_eval,
        "parse_args",
        lambda: _train_eval_args(
            tmp_path,
            policy_ckpt_dir=str(tmp_path / "ckpt"),
            policy_config=str(tmp_path / "policy.yaml"),
            policy_handoff=str(tmp_path / "handoff.json"),
            run_name="bad_combo",
        ),
    )

    with pytest.raises(ValueError, match="either --policy_handoff or --policy_ckpt_dir/--policy_config"):
        train_eval.main()


def test_train_eval_main_rejects_partial_ckpt_inputs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        train_eval,
        "parse_args",
        lambda: _train_eval_args(tmp_path, policy_ckpt_dir=str(tmp_path / "ckpt"), run_name="partial_ckpt"),
    )

    with pytest.raises(ValueError, match="requires both --policy_ckpt_dir and --policy_config"):
        train_eval.main()


def test_train_eval_main_generates_summary_random_mode(tmp_path: Path, monkeypatch):
    run_name = "unit_random"
    output_root = tmp_path / "outputs"
    run_dir = output_root / run_name
    command_calls = []

    monkeypatch.setattr(
        train_eval,
        "parse_args",
        lambda: _train_eval_args(tmp_path, output_root=str(output_root), run_name=run_name, fast_startup=True),
    )
    monkeypatch.setattr(train_eval, "_run_command", lambda cmd, timeout_sec=0: command_calls.append(cmd) or {
        "rc": 0,
        "timed_out": False,
        "duration_sec": 0.01,
        "stdout_tail": "",
        "stderr_tail": "",
        "error": "",
    })

    rc = train_eval.main()
    assert rc == 0
    assert command_calls
    cmd = command_calls[0]
    assert "--policy_mode" in cmd
    assert cmd[cmd.index("--policy_mode") + 1] == "random"
    assert "--fast_startup" in cmd

    summary_path = run_dir / "train_eval_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["run_name"] == run_name
    assert summary["status"] == "success"
    assert summary["policy"]["mode"] == "random"
    assert summary["policy_handoff"] == ""
    assert summary["evaluation"]["eval_rc"] == 0


def test_train_eval_main_uses_input_handoff_no_generation(tmp_path: Path, monkeypatch):
    run_name = "unit_handoff"
    output_root = tmp_path / "outputs"
    run_dir = output_root / run_name
    command_calls = []
    generated_calls = []

    handoff_path = tmp_path / "policy_handoff.valid.json"
    handoff_payload = {
        "_artifact_path": str(handoff_path.resolve()),
        "policy": {
            "checkpoint_dir": str((tmp_path / "ckpt_from_handoff").resolve()),
            "config_path": str((tmp_path / "policy_from_handoff.yaml").resolve()),
        },
    }

    monkeypatch.setattr(
        train_eval,
        "parse_args",
        lambda: _train_eval_args(
            tmp_path,
            output_root=str(output_root),
            policy_handoff=str(handoff_path),
            run_name=run_name,
        ),
    )
    monkeypatch.setattr(train_eval, "load_policy_handoff", lambda _path: handoff_payload)
    monkeypatch.setattr(train_eval, "_validate_checkpoint_expectations", lambda _dir: ["ckpt-1"])
    monkeypatch.setattr(train_eval, "generate_policy_handoff_artifact", lambda *_args, **_kwargs: generated_calls.append(1))
    monkeypatch.setattr(train_eval, "_run_command", lambda cmd, timeout_sec=0: command_calls.append(cmd) or {
        "rc": 0,
        "timed_out": False,
        "duration_sec": 0.01,
        "stdout_tail": "",
        "stderr_tail": "",
        "error": "",
    })

    rc = train_eval.main()
    assert rc == 0
    assert generated_calls == []
    assert command_calls
    cmd = command_calls[0]
    assert "--policy_handoff" in cmd
    assert str(handoff_path.resolve()) in cmd

    summary = json.loads((run_dir / "train_eval_summary.json").read_text())
    assert summary["policy"]["mode"] == "ippo_rnn"
    assert summary["policy"]["checkpoint_dir"] == str((tmp_path / "ckpt_from_handoff").resolve())
    assert summary["policy"]["config_path"] == str((tmp_path / "policy_from_handoff.yaml").resolve())
    assert summary["policy_handoff"] == str(handoff_path.resolve())
    assert summary["policy"]["generated_handoff"] == ""
    assert summary["lineage"]["handoff"]["status"] == "input_handoff"


def test_train_eval_main_generates_handoff_when_ippo_without_input(tmp_path: Path, monkeypatch):
    run_name = "unit_ippo_generate"
    output_root = tmp_path / "outputs"
    run_dir = output_root / run_name
    command_calls = []
    captured_generation = {}
    generated_path = (run_dir / "policy_handoff.generated.json").resolve()

    monkeypatch.setattr(
        train_eval,
        "parse_args",
        lambda: _train_eval_args(
            tmp_path,
            output_root=str(output_root),
            policy_ckpt_dir=str(tmp_path / "ckpt"),
            policy_config=str(tmp_path / "policy.yaml"),
            run_name=run_name,
        ),
    )
    monkeypatch.setattr(train_eval, "_validate_checkpoint_expectations", lambda _dir: ["ckpt-1"])
    monkeypatch.setattr(train_eval, "_run_command", lambda cmd, timeout_sec=0: command_calls.append(cmd) or {
        "rc": 0,
        "timed_out": False,
        "duration_sec": 0.01,
        "stdout_tail": "",
        "stderr_tail": "",
        "error": "",
    })

    def _fake_generate(payload, base_dir):
        captured_generation["payload"] = payload
        captured_generation["base_dir"] = base_dir
        return {"_artifact_path": str(generated_path)}

    monkeypatch.setattr(train_eval, "generate_policy_handoff_artifact", _fake_generate)

    rc = train_eval.main()
    assert rc == 0
    assert command_calls
    cmd = command_calls[0]
    assert "--policy_ckpt_dir" in cmd
    assert "--policy_config" in cmd

    assert captured_generation["payload"]["policy_mode"] == "ippo_rnn"
    assert captured_generation["payload"]["output_path"].endswith("policy_handoff.generated.json")
    assert captured_generation["base_dir"] == run_dir

    summary = json.loads((run_dir / "train_eval_summary.json").read_text())
    assert summary["policy"]["mode"] == "ippo_rnn"
    assert summary["policy"]["generated_handoff"] == str(generated_path)
    assert summary["policy_handoff"] == str(generated_path)
    assert summary["lineage"]["handoff"]["status"] == "generated_handoff"


def test_train_eval_main_skips_handoff_generation_on_eval_failure(tmp_path: Path, monkeypatch):
    run_name = "unit_ippo_fail"
    output_root = tmp_path / "outputs"
    run_dir = output_root / run_name
    generated_calls = []

    monkeypatch.setattr(
        train_eval,
        "parse_args",
        lambda: _train_eval_args(
            tmp_path,
            output_root=str(output_root),
            policy_ckpt_dir=str(tmp_path / "ckpt"),
            policy_config=str(tmp_path / "policy.yaml"),
            run_name=run_name,
        ),
    )
    monkeypatch.setattr(train_eval, "_validate_checkpoint_expectations", lambda _dir: ["ckpt-1"])
    monkeypatch.setattr(train_eval, "_run_command", lambda _cmd, timeout_sec=0: {
        "rc": 5,
        "timed_out": False,
        "duration_sec": 0.01,
        "stdout_tail": "",
        "stderr_tail": "",
        "error": "",
    })
    monkeypatch.setattr(
        train_eval,
        "generate_policy_handoff_artifact",
        lambda *_args, **_kwargs: generated_calls.append(1) or {"_artifact_path": "unused"},
    )

    rc = train_eval.main()
    assert rc == 1
    assert generated_calls == []
    summary = json.loads((run_dir / "train_eval_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["evaluation"]["eval_rc"] == 5
    assert summary["policy"]["generated_handoff"] == ""


def test_discover_checkpoint_candidates_finds_state_dirs_and_files(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpts"
    (ckpt_dir / "001" / "state").mkdir(parents=True)
    (ckpt_dir / "checkpoint_latest").write_text("x")
    (ckpt_dir / "weights.msgpack").write_text("x")

    out = train_eval._discover_checkpoint_candidates(ckpt_dir)
    assert str((ckpt_dir / "001").resolve()) in out
    assert str((ckpt_dir / "checkpoint_latest").resolve()) in out
    assert str((ckpt_dir / "weights.msgpack").resolve()) in out


def test_validate_checkpoint_expectations_raises_when_empty(tmp_path: Path):
    ckpt_dir = tmp_path / "empty_ckpts"
    ckpt_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No checkpoint artifacts discovered"):
        train_eval._validate_checkpoint_expectations(str(ckpt_dir))
