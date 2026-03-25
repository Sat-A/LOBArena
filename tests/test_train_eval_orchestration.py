import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from LOBArena.evaluate import train_eval


def test_run_command_success():
    result = train_eval._run_command(["python3", "-c", "print('ok')"], timeout_sec=5)
    assert result["rc"] == 0
    assert result["timed_out"] is False
    assert "ok" in result["stdout_tail"]


def test_run_command_timeout():
    result = train_eval._run_command(["python3", "-c", "import time; time.sleep(0.2)"], timeout_sec=0.01)
    assert result["timed_out"] is True
    assert result["rc"] is None
    assert "timeout" in result["error"]


def test_validate_checkpoint_expectations_accepts_step_state(tmp_path: Path):
    ckpt_dir = tmp_path / "policy_ckpt"
    (ckpt_dir / "12" / "state").mkdir(parents=True)
    candidates = train_eval._validate_checkpoint_expectations(str(ckpt_dir))
    assert len(candidates) == 1
    assert candidates[0].endswith("/12")


def test_validate_checkpoint_expectations_accepts_checkpoint_file(tmp_path: Path):
    ckpt_dir = tmp_path / "policy_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint_best.ckpt").write_text("x")
    candidates = train_eval._validate_checkpoint_expectations(str(ckpt_dir))
    assert any(c.endswith("checkpoint_best.ckpt") for c in candidates)


def test_validate_checkpoint_expectations_rejects_empty_dir(tmp_path: Path):
    ckpt_dir = tmp_path / "policy_ckpt"
    ckpt_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No checkpoint artifacts discovered"):
        train_eval._validate_checkpoint_expectations(str(ckpt_dir))


def _mk_args(tmp_path: Path, **overrides):
    defaults = {
        "train_data_dir": str(tmp_path / "train_data"),
        "test_data_dir": str(tmp_path / "test_data"),
        "jaxmarl_root": str(tmp_path / "jaxmarl"),
        "output_root": str(tmp_path / "outputs"),
        "train_steps": 10,
        "eval_steps": 5,
        "policy_ckpt_dir": "",
        "policy_config": "",
        "policy_handoff": "",
        "run_name": "phase2_integration",
        "fast_startup": True,
        "run_jaxmarl_train": False,
        "jaxmarl_train_cmd": "",
        "jaxmarl_train_timeout_sec": 30,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _write_valid_policy_handoff(tmp_path: Path) -> tuple[Path, Path, Path]:
    ckpt = tmp_path / "ckpt"
    (ckpt / "1" / "state").mkdir(parents=True)
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
    return handoff, ckpt, cfg


def test_campaign_manager_initializes_run_dir_and_default_lineage(tmp_path: Path):
    args = _mk_args(tmp_path, run_name="manager_init")
    mgr = train_eval.Phase2AlphaCampaignManager(args=args, started_at=0.0)
    assert mgr.run_dir == (Path(args.output_root).resolve() / args.run_name)
    assert mgr.run_dir.exists()
    assert mgr.summary_path == mgr.run_dir / "train_eval_summary.json"
    assert mgr.lineage["train"]["status"] == "skipped"
    assert mgr.lineage["handoff"]["status"] == "none"
    assert mgr.lineage["evaluation"]["status"] == "not_started"


def test_main_end_to_end_random_mode_with_lineage(monkeypatch, tmp_path: Path):
    (tmp_path / "train_data").mkdir()
    (tmp_path / "test_data").mkdir()
    args = _mk_args(tmp_path)
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)

    observed_commands = []

    def _fake_run(cmd, timeout_sec=0):
        observed_commands.append((cmd, timeout_sec))
        return {
            "rc": 0,
            "timed_out": False,
            "duration_sec": 0.01,
            "stdout_tail": "ok",
            "stderr_tail": "",
            "error": "",
        }

    monkeypatch.setattr(train_eval, "_run_command", _fake_run)

    rc = train_eval.main()
    assert rc == 0
    assert len(observed_commands) == 1
    eval_cmd, eval_timeout = observed_commands[0]
    assert eval_timeout == 0
    assert "--policy_mode" in eval_cmd and "random" in eval_cmd
    assert "--policy_ckpt_dir" not in eval_cmd
    assert "--policy_handoff" not in eval_cmd

    summary_path = Path(args.output_root) / args.run_name / "train_eval_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "success"
    assert summary["lineage"]["train"]["status"] == "skipped"
    assert summary["lineage"]["handoff"]["status"] == "none"
    assert summary["lineage"]["evaluation"]["status"] == "success"
    assert summary["lineage"]["evaluation"]["rc"] == 0
    assert summary["policy_handoff"] == ""


def test_main_end_to_end_ippo_mode_generates_handoff_and_lineage(monkeypatch, tmp_path: Path):
    (tmp_path / "train_data").mkdir()
    (tmp_path / "test_data").mkdir()
    ckpt_dir = tmp_path / "policy_ckpt"
    (ckpt_dir / "7" / "state").mkdir(parents=True)
    cfg = tmp_path / "policy.yaml"
    cfg.write_text("seed: 1\n")

    args = _mk_args(
        tmp_path,
        policy_ckpt_dir=str(ckpt_dir),
        policy_config=str(cfg),
    )
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)

    calls = []

    def _fake_run(cmd, timeout_sec=0):
        calls.append((cmd, timeout_sec))
        return {
            "rc": 0,
            "timed_out": False,
            "duration_sec": 0.02,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": "",
        }

    monkeypatch.setattr(train_eval, "_run_command", _fake_run)
    rc = train_eval.main()
    assert rc == 0

    eval_cmd = calls[0][0]
    assert "--policy_ckpt_dir" in eval_cmd
    assert str(ckpt_dir) in eval_cmd
    assert "--policy_config" in eval_cmd
    assert str(cfg) in eval_cmd

    summary_path = Path(args.output_root) / args.run_name / "train_eval_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["lineage"]["handoff"]["status"] == "generated_handoff"
    assert summary["lineage"]["handoff"]["checkpoint_candidate_count"] == 1
    generated = summary["lineage"]["handoff"]["generated_handoff"]
    assert generated
    assert Path(generated).exists()
    assert summary["policy_handoff"] == generated


def test_main_consumes_input_handoff_for_adversarial_handoff_path(monkeypatch, tmp_path: Path):
    (tmp_path / "train_data").mkdir()
    (tmp_path / "test_data").mkdir()
    handoff, ckpt, cfg = _write_valid_policy_handoff(tmp_path)
    args = _mk_args(tmp_path, policy_handoff=str(handoff))
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)

    observed = []

    def _fake_run(cmd, timeout_sec=0):
        observed.append(cmd)
        return {
            "rc": 0,
            "timed_out": False,
            "duration_sec": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": "",
        }

    monkeypatch.setattr(train_eval, "_run_command", _fake_run)
    rc = train_eval.main()
    assert rc == 0

    eval_cmd = observed[0]
    assert "--policy_handoff" in eval_cmd
    assert str(handoff.resolve()) in eval_cmd
    assert "--policy_ckpt_dir" not in eval_cmd
    assert "--policy_config" not in eval_cmd

    summary_path = Path(args.output_root) / args.run_name / "train_eval_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["lineage"]["handoff"]["status"] == "input_handoff"
    assert summary["lineage"]["handoff"]["input_handoff"] == str(handoff.resolve())
    assert summary["lineage"]["handoff"]["checkpoint_candidate_count"] == 1
    assert summary["policy_ckpt_dir"] == str(ckpt.resolve())
    assert summary["policy_config"] == str(cfg.resolve())
    assert summary["lineage"]["handoff"]["generated_handoff"] == ""


def test_train_eval_rejects_multi_node_env(monkeypatch, tmp_path: Path):
    args = _mk_args(tmp_path)
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="single-node execution only"):
        train_eval.main()


def test_train_eval_rejects_multi_node_train_cmd(monkeypatch, tmp_path: Path):
    args = _mk_args(
        tmp_path,
        run_jaxmarl_train=True,
        jaxmarl_train_cmd="python train.py --nnodes 2",
    )
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="single-node execution only"):
        train_eval.main()


def test_main_train_subprocess_failure_writes_failed_summary_and_stops(monkeypatch, tmp_path: Path):
    (tmp_path / "train_data").mkdir()
    (tmp_path / "test_data").mkdir()
    args = _mk_args(
        tmp_path,
        run_jaxmarl_train=True,
        jaxmarl_train_cmd="python3 -c \"print('train')\"",
    )
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)

    calls = []

    def _fake_run(cmd, timeout_sec=0):
        calls.append((cmd, timeout_sec))
        return {
            "rc": 2,
            "timed_out": False,
            "duration_sec": 0.03,
            "stdout_tail": "",
            "stderr_tail": "boom",
            "error": "",
        }

    monkeypatch.setattr(train_eval, "_run_command", _fake_run)
    with pytest.raises(RuntimeError, match="training failed"):
        train_eval.main()

    assert len(calls) == 1
    summary_path = Path(args.output_root) / args.run_name / "train_eval_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "failed"
    assert summary["eval_rc"] == 1
    assert summary["lineage"]["train"]["status"] == "failed"
    assert summary["lineage"]["evaluation"]["status"] == "not_started"


def test_main_train_subprocess_timeout_sets_eval_rc_124(monkeypatch, tmp_path: Path):
    (tmp_path / "train_data").mkdir()
    (tmp_path / "test_data").mkdir()
    args = _mk_args(
        tmp_path,
        run_jaxmarl_train=True,
        jaxmarl_train_cmd="python3 -c \"print('train')\"",
        jaxmarl_train_timeout_sec=1,
    )
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)

    monkeypatch.setattr(
        train_eval,
        "_run_command",
        lambda _cmd, timeout_sec=0: {
            "rc": None,
            "timed_out": True,
            "duration_sec": 1.0,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": "timeout after 1 sec",
        },
    )

    with pytest.raises(RuntimeError, match="timed out"):
        train_eval.main()

    summary_path = Path(args.output_root) / args.run_name / "train_eval_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "failed"
    assert summary["eval_rc"] == 124
    assert summary["lineage"]["train"]["status"] == "timed_out"


def test_main_allows_single_node_env(monkeypatch, tmp_path: Path):
    (tmp_path / "train_data").mkdir()
    (tmp_path / "test_data").mkdir()
    args = _mk_args(tmp_path, run_name="single_node_ok")
    monkeypatch.setenv("SLURM_NNODES", "1")
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)
    monkeypatch.setattr(
        train_eval,
        "_run_command",
        lambda _cmd, timeout_sec=0: {
            "rc": 0,
            "timed_out": False,
            "duration_sec": 0.01,
            "stdout_tail": "",
            "stderr_tail": "",
            "error": "",
        },
    )
    assert train_eval.main() == 0


def test_main_rejects_multi_node_env(monkeypatch, tmp_path: Path):
    args = _mk_args(tmp_path, run_name="multi_node_blocked")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="single-node execution only"):
        train_eval.main()


def test_main_rejects_multi_node_train_command(monkeypatch, tmp_path: Path):
    (tmp_path / "train_data").mkdir()
    (tmp_path / "test_data").mkdir()
    args = _mk_args(
        tmp_path,
        run_name="multi_node_train_cmd",
        run_jaxmarl_train=True,
        jaxmarl_train_cmd="python train.py --nnodes 4 --seed 1",
    )
    monkeypatch.setattr(train_eval, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="single-node execution only"):
        train_eval.main()
