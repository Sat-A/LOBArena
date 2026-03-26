import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

from LOBArena.evaluate import adversarial


def test_build_matchups_target_vs_many_and_round_robin():
    participants = ["target", "competitor:a", "competitor:b"]
    assert adversarial._resolve_tournament_mode(round_robin=False, competitor_count=2) == "target-vs-many"
    assert adversarial._resolve_tournament_mode(round_robin=True, competitor_count=2) == "round-robin"
    tvm = adversarial._build_matchups("target-vs-many", participants, target_id="target")
    rr = adversarial._build_matchups("round-robin", participants, target_id="target")

    assert tvm == [("target", "competitor:a"), ("target", "competitor:b")]
    assert rr == [
        ("target", "competitor:a"),
        ("target", "competitor:b"),
        ("competitor:a", "competitor:b"),
    ]


def test_run_eval_passes_fairness_args(monkeypatch):
    captured = {}

    def _fake_call(cmd):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(adversarial.subprocess, "call", _fake_call)

    rc = adversarial._run_eval(
        run_name="run_x",
        output_root="/home/s5e/satyamaga.s5e/LOBArena/tests/out",
        data_dir="/home/s5e/satyamaga.s5e/data",
        policy_mode="fixed",
        fixed_action=1,
        ckpt="",
        cfg="",
        n_steps=33,
        seed=7,
        sample_index=9,
        test_split=0.25,
        start_date="2020-01-01",
        end_date="2020-01-31",
        policy_handoff="",
    )

    assert rc == 0
    cmd = captured["cmd"]
    assert "--n_steps" in cmd and cmd[cmd.index("--n_steps") + 1] == "33"
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "7"
    assert "--sample_index" in cmd and cmd[cmd.index("--sample_index") + 1] == "9"
    assert "--test_split" in cmd and cmd[cmd.index("--test_split") + 1] == "0.25"
    assert "--start_date" in cmd and cmd[cmd.index("--start_date") + 1] == "2020-01-01"
    assert "--end_date" in cmd and cmd[cmd.index("--end_date") + 1] == "2020-01-31"


def test_compute_target_pairwise_summary_includes_win_rate_and_pnl_stats():
    matches = [
        {
            "left": {"participant_id": "target"},
            "right": {"participant_id": "competitor:a"},
            "pnl_delta_left_minus_right": 5.0,
            "winner": "target",
            "metadata": {"start_date": "2021-01-01", "end_date": "2021-01-31"},
        },
        {
            "left": {"participant_id": "competitor:b"},
            "right": {"participant_id": "target"},
            "pnl_delta_left_minus_right": 2.0,
            "winner": "competitor:b",
            "metadata": {"start_date": "2021-01-01", "end_date": "2021-01-31"},
        },
        {
            "left": {"participant_id": "target"},
            "right": {"participant_id": "competitor:c"},
            "pnl_delta_left_minus_right": 0.0,
            "winner": "tie",
            "metadata": {"start_date": "2021-01-01", "end_date": "2021-01-31"},
        },
    ]
    summary = adversarial._compute_target_pairwise_summary(matches, target_id="target")
    assert summary["matches_considered"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 1
    assert summary["win_rate_overall"] == 0.5
    assert summary["pnl_delta_stats"] == {
        "mean": 1.0,
        "median": 0.0,
        "std": 2.943920288775949,
        "min": -2.0,
        "max": 5.0,
    }
    assert summary["per_competitor"]["competitor:a"]["win_rate"] == 1.0
    assert summary["per_competitor"]["competitor:a"]["avg_pnl_delta"] == 5.0
    assert summary["per_competitor"]["competitor:b"]["win_rate"] == 0.0
    assert summary["per_competitor"]["competitor:b"]["avg_pnl_delta"] == -2.0
    assert summary["per_competitor"]["competitor:c"]["win_rate"] == 0.5
    assert summary["per_competitor"]["competitor:c"]["avg_pnl_delta"] == 0.0


def test_compute_regime_date_robustness_is_compact_and_deterministic():
    matches = [
        {
            "left": {"participant_id": "target", "world_model_mode": "historical"},
            "right": {"participant_id": "competitor:a", "world_model_mode": "historical"},
            "pnl_delta_left_minus_right": 1.0,
            "winner": "target",
            "metadata": {"start_date": "2021-01-01", "end_date": "2021-01-31"},
        },
        {
            "left": {"participant_id": "target", "world_model_mode": "historical"},
            "right": {"participant_id": "competitor:b", "world_model_mode": "historical"},
            "pnl_delta_left_minus_right": -3.0,
            "winner": "competitor:b",
            "metadata": {"start_date": "2021-01-01", "end_date": "2021-01-31"},
        },
        {
            "left": {"participant_id": "target", "world_model_mode": "generated"},
            "right": {"participant_id": "competitor:c", "world_model_mode": "generated"},
            "pnl_delta_left_minus_right": 2.0,
            "winner": "target",
            "metadata": {"start_date": "2021-02-01", "end_date": "2021-02-28"},
        },
    ]

    robustness = adversarial._compute_regime_date_robustness(matches, target_id="target")
    assert robustness == [
        {
            "regime": "generated",
            "start_date": "2021-02-01",
            "end_date": "2021-02-28",
            "match_count": 1,
            "win_rate_overall": 1.0,
            "avg_pnl_delta": 2.0,
            "pnl_delta_stats": {
                "mean": 2.0,
                "median": 2.0,
                "std": 0.0,
                "min": 2.0,
                "max": 2.0,
            },
        },
        {
            "regime": "historical",
            "start_date": "2021-01-01",
            "end_date": "2021-01-31",
            "match_count": 2,
            "win_rate_overall": 0.5,
            "avg_pnl_delta": -1.0,
            "pnl_delta_stats": {
                "mean": -1.0,
                "median": -1.0,
                "std": 2.0,
                "min": -3.0,
                "max": 1.0,
            },
        },
    ]


def test_main_round_robin_summary_contains_matches_and_aggregate(monkeypatch):
    test_root = Path("/home/s5e/satyamaga.s5e/LOBArena/tests/.adversarial_test_outputs")
    run_name = f"rr_{uuid.uuid4().hex[:8]}"
    output_root = test_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        data_dir="/home/s5e/satyamaga.s5e/LOBArena",
        target_policy_mode="fixed",
        target_fixed_action=1,
        target_policy_ckpt="",
        target_policy_config="",
        target_policy_handoff="",
        competitor_policy_mode="random",
        competitor_fixed_action=0,
        competitor_policy_ckpt="",
        competitor_policy_config="",
        competitor_policy_handoff="",
        competitor_registry_config="/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json",
        competitor_keys=["random_baseline", "fixed_baseline_hold"],
        output_root=str(output_root),
        run_name="adversarial_rr",
        n_steps=12,
        seed=123,
        sample_index=4,
        test_split=0.5,
        start_date="2021-01-01",
        end_date="2021-01-31",
        round_robin=True,
    )

    pnls = {
        "adversarial_rr_target": 10.0,
        "adversarial_rr_competitor_random_baseline": 5.0,
        "adversarial_rr_competitor_fixed_baseline_hold": 8.0,
    }

    def _fake_parse_args():
        return args

    def _fake_run_eval(
        run_name,
        output_root,
        data_dir,
        policy_mode,
        fixed_action,
        ckpt,
        cfg,
        n_steps,
        seed,
        sample_index,
        test_split,
        start_date,
        end_date,
        policy_handoff="",
    ):
        run_dir = Path(output_root) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "world_model_mode": "historical",
            "metrics": {"pnl": {"total_pnl": pnls[run_name]}},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))
        return 0

    monkeypatch.setattr(adversarial, "parse_args", _fake_parse_args)
    monkeypatch.setattr(adversarial, "_run_eval", _fake_run_eval)

    try:
        rc = adversarial.main()
        assert rc == 0
        out = output_root / "adversarial_rr" / "adversarial_summary.json"
        data = json.loads(out.read_text())
        assert data["tournament_mode"] == "round-robin"
        assert len(data["matches"]) == 3
        assert data["fairness_config"]["seed"] == 123
        assert data["fairness_config"]["start_date"] == "2021-01-01"
        assert data["fairness_config"]["end_date"] == "2021-01-31"
        by_participant = data["aggregate"]["by_participant"]
        assert by_participant["target"]["wins"] == 2
        assert by_participant["target"]["losses"] == 0
        assert data["win_rate_overall"] == 1.0
        assert data["winner"] == "target"
        target_pairwise = data["aggregate"]["target_pairwise"]
        assert target_pairwise["win_rate_overall"] == 1.0
        assert target_pairwise["per_competitor"]["competitor:fixed_baseline_hold"]["avg_pnl_delta"] == 2.0
        assert target_pairwise["pnl_delta_stats"]["mean"] == 3.5
        regime_summary = data["aggregate"]["regime_date_robustness"]
        assert regime_summary == [
            {
                "regime": "historical",
                "start_date": "2021-01-01",
                "end_date": "2021-01-31",
                "match_count": 2,
                "win_rate_overall": 1.0,
                "avg_pnl_delta": 3.5,
                "pnl_delta_stats": {
                    "mean": 3.5,
                    "median": 3.5,
                    "std": 1.5,
                    "min": 2.0,
                    "max": 5.0,
                },
            }
        ]
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_adversarial_summary_schema_sanity_round_robin(monkeypatch):
    test_root = Path("/home/s5e/satyamaga.s5e/LOBArena/tests/.adversarial_test_outputs")
    run_name = f"schema_{uuid.uuid4().hex[:8]}"
    output_root = test_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        data_dir="/home/s5e/satyamaga.s5e/LOBArena",
        target_policy_mode="fixed",
        target_fixed_action=1,
        target_policy_ckpt="",
        target_policy_config="",
        target_policy_handoff="",
        competitor_policy_mode="random",
        competitor_fixed_action=0,
        competitor_policy_ckpt="",
        competitor_policy_config="",
        competitor_policy_handoff="",
        competitor_registry_config="/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json",
        competitor_keys=["random_baseline", "fixed_baseline_hold"],
        output_root=str(output_root),
        run_name="adversarial_schema",
        n_steps=8,
        seed=77,
        sample_index=1,
        test_split=0.4,
        start_date="2022-01-01",
        end_date="2022-01-10",
        round_robin=True,
    )

    pnls = {
        "adversarial_schema_target": 2.0,
        "adversarial_schema_competitor_random_baseline": -1.0,
        "adversarial_schema_competitor_fixed_baseline_hold": 1.5,
    }

    def _fake_parse_args():
        return args

    def _fake_run_eval(
        run_name,
        output_root,
        data_dir,
        policy_mode,
        fixed_action,
        ckpt,
        cfg,
        n_steps,
        seed,
        sample_index,
        test_split,
        start_date,
        end_date,
        policy_handoff="",
    ):
        run_dir = Path(output_root) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "world_model_mode": "historical",
            "metrics": {"pnl": {"total_pnl": pnls[run_name]}},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))
        return 0

    monkeypatch.setattr(adversarial, "parse_args", _fake_parse_args)
    monkeypatch.setattr(adversarial, "_run_eval", _fake_run_eval)

    try:
        rc = adversarial.main()
        assert rc == 0
        out = output_root / "adversarial_schema" / "adversarial_summary.json"
        data = json.loads(out.read_text())

        expected_top_keys = {
            "run_name",
            "tournament_mode",
            "fairness_config",
            "target_run",
            "competitor_run",
            "target_rc",
            "competitor_rc",
            "target_total_pnl",
            "competitor_total_pnl",
            "winner",
            "target_summary_path",
            "competitors",
            "participants",
            "matches",
            "aggregate",
            "win_rate_overall",
            "runtime_sec",
        }
        assert expected_top_keys.issubset(set(data.keys()))

        fairness = data["fairness_config"]
        assert set(fairness.keys()) == {"seed", "start_date", "end_date", "n_steps", "sample_index", "test_split"}

        assert isinstance(data["participants"], list) and len(data["participants"]) == 3
        participant_ids = {p["participant_id"] for p in data["participants"]}
        assert "target" in participant_ids
        assert "competitor:random_baseline" in participant_ids
        assert "competitor:fixed_baseline_hold" in participant_ids

        matches = data["matches"]
        assert isinstance(matches, list) and len(matches) == 3
        for match in matches:
            assert set(match.keys()) == {
                "match_id",
                "mode",
                "left",
                "right",
                "pnl_delta_left_minus_right",
                "winner",
                "metadata",
            }
            assert set(match["metadata"].keys()) == {
                "seed",
                "start_date",
                "end_date",
                "n_steps",
                "sample_index",
                "test_split",
            }

        aggregate = data["aggregate"]
        assert set(aggregate.keys()) == {"by_participant", "target_pairwise", "regime_date_robustness"}
        assert set(aggregate["target_pairwise"]["pnl_delta_stats"].keys()) == {"mean", "median", "std", "min", "max"}
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_adversarial_winner_is_tie_when_target_equals_competitor(monkeypatch):
    test_root = Path("/home/s5e/satyamaga.s5e/LOBArena/tests/.adversarial_test_outputs")
    run_name = f"tie_{uuid.uuid4().hex[:8]}"
    output_root = test_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        data_dir="/home/s5e/satyamaga.s5e/LOBArena",
        target_policy_mode="fixed",
        target_fixed_action=1,
        target_policy_ckpt="",
        target_policy_config="",
        target_policy_handoff="",
        competitor_policy_mode="fixed",
        competitor_fixed_action=1,
        competitor_policy_ckpt="",
        competitor_policy_config="",
        competitor_policy_handoff="",
        competitor_registry_config="/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/adversarial_competitors.json",
        competitor_keys=[],
        output_root=str(output_root),
        run_name="adversarial_tie",
        n_steps=3,
        seed=3,
        sample_index=0,
        test_split=1.0,
        start_date="",
        end_date="",
        round_robin=False,
    )

    def _fake_parse_args():
        return args

    def _fake_run_eval(
        run_name,
        output_root,
        data_dir,
        policy_mode,
        fixed_action,
        ckpt,
        cfg,
        n_steps,
        seed,
        sample_index,
        test_split,
        start_date,
        end_date,
        policy_handoff="",
    ):
        run_dir = Path(output_root) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "world_model_mode": "historical",
            "metrics": {"pnl": {"total_pnl": 5.0}},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))
        return 0

    monkeypatch.setattr(adversarial, "parse_args", _fake_parse_args)
    monkeypatch.setattr(adversarial, "_run_eval", _fake_run_eval)

    try:
        rc = adversarial.main()
        assert rc == 0
        out = output_root / "adversarial_tie" / "adversarial_summary.json"
        data = json.loads(out.read_text())
        assert data["winner"] == "tie"
        assert data["competitors"][0]["winner"] == "tie"
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
