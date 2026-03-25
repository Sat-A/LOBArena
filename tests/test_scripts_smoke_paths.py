from pathlib import Path

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

