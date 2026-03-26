import json
from pathlib import Path

from LOBArena.leaderboard.aggregator import aggregate


def _write_summary(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def test_weighted_ranking_is_deterministic_with_tiebreakers(tmp_path):
    run_a = tmp_path / "a" / "summary.json"
    run_b = tmp_path / "b" / "summary.json"
    run_c = tmp_path / "c" / "summary.json"

    # A and B intentionally tie on composite + legacy tuple; run_name breaks tie deterministically.
    _write_summary(
        run_a,
        {
            "run_name": "alpha_run",
            "world_model_mode": "historical",
            "policy_mode": "ippo_rnn",
            "metrics": {
                "pnl": {"total_pnl": 10.0, "cash_pnl": 9.0, "inventory": 1.0},
                "drawdown": {"max_drawdown": -2.0},
                "risk": {"pnl_delta_std": 1.0},
            },
        },
    )
    _write_summary(
        run_b,
        {
            "run_name": "beta_run",
            "world_model_mode": "historical",
            "policy_mode": "ippo_rnn",
            "metrics": {
                "pnl": {"total_pnl": 10.0, "cash_pnl": 8.0, "inventory": 1.0},
                "drawdown": {"max_drawdown": -2.0},
                "risk": {"pnl_delta_std": 1.0},
            },
        },
    )
    _write_summary(
        run_c,
        {
            "run_name": "zeta_run",
            "world_model_mode": "generative",
            "policy_mode": "random",
            "metrics": {
                "pnl": {"total_pnl": 5.0, "cash_pnl": 5.0, "inventory": 0.0},
                "drawdown": {"max_drawdown": -1.0},
                "risk": {"pnl_delta_std": 0.1},
            },
        },
    )

    result = aggregate(str(tmp_path / "*" / "summary.json"), weights={"pnl": 1.0, "drawdown": 0.5, "risk": 0.1})

    assert result["n_runs"] == 3
    assert result["leaderboard"][0]["run_name"] == "beta_run"
    assert result["leaderboard"][1]["run_name"] == "alpha_run"
    assert result["leaderboard"][2]["run_name"] == "zeta_run"
    assert result["leaderboard"][0]["rank"] == 1
    assert result["ranking"]["method"] == "weighted_composite"


def test_split_leaderboards_include_world_policy_family_and_date_window(tmp_path):
    r1 = tmp_path / "r1" / "summary.json"
    r2 = tmp_path / "r2" / "summary.json"

    _write_summary(
        r1,
        {
            "run_name": "hist_ippo",
            "world_model_mode": "historical",
            "policy_mode": "ippo_rnn",
            "policy_handoff": {
                "policy": {"family": "ippo", "mode": "ippo_rnn"},
                "evaluation": {"date_window": {"start_date": "2024-01-01", "end_date": "2024-01-03"}},
            },
            "metrics": {
                "pnl": {"total_pnl": 4.0, "cash_pnl": 4.0, "inventory": 0.0},
                "drawdown": {"max_drawdown": -1.0},
                "risk": {"pnl_delta_std": 0.3},
            },
        },
    )
    _write_summary(
        r2,
        {
            "run_name": "gen_random",
            "world_model_mode": "generative",
            "policy_mode": "random",
            "start_date": "2024-02-01",
            "end_date": "2024-02-02",
            "metrics": {
                "pnl": {"total_pnl": 3.0, "cash_pnl": 3.0, "inventory": 0.0},
                "drawdown": {"max_drawdown": -0.2},
                "risk": {"pnl_delta_std": 0.1},
            },
        },
    )

    result = aggregate(str(tmp_path / "*" / "summary.json"))
    splits = result["split_leaderboards"]

    assert set(splits["by_world_model_mode"].keys()) == {"historical", "generative"}
    assert set(splits["by_policy_mode"].keys()) == {"ippo_rnn", "random"}
    assert set(splits["by_policy_family"].keys()) == {"ippo", "random"}

    date_keys = set(splits["by_date_window"].keys())
    assert "2024-01-01::2024-01-03" in date_keys
    assert "2024-02-01::2024-02-02" in date_keys

    assert splits["by_world_model_mode"]["historical"]["leaderboard"][0]["run_name"] == "hist_ippo"
    assert splits["by_policy_family"]["ippo"]["leaderboard"][0]["policy_family"] == "ippo"


def test_split_leaderboard_counts_and_best_run_sanity(tmp_path):
    _write_summary(
        tmp_path / "run1" / "summary.json",
        {
            "run_name": "hist_a",
            "world_model_mode": "historical",
            "policy_mode": "fixed",
            "metrics": {
                "pnl": {"total_pnl": 7.0, "cash_pnl": 7.0, "inventory": 0.0},
                "drawdown": {"max_drawdown": -1.0},
                "risk": {"pnl_delta_std": 0.5},
            },
        },
    )
    _write_summary(
        tmp_path / "run2" / "summary.json",
        {
            "run_name": "hist_b",
            "world_model_mode": "historical",
            "policy_mode": "fixed",
            "metrics": {
                "pnl": {"total_pnl": 5.0, "cash_pnl": 5.0, "inventory": 0.0},
                "drawdown": {"max_drawdown": -0.4},
                "risk": {"pnl_delta_std": 0.2},
            },
        },
    )
    _write_summary(
        tmp_path / "run3" / "summary.json",
        {
            "run_name": "gen_a",
            "world_model_mode": "generative",
            "policy_mode": "random",
            "metrics": {
                "pnl": {"total_pnl": 3.0, "cash_pnl": 3.0, "inventory": 0.0},
                "drawdown": {"max_drawdown": -0.2},
                "risk": {"pnl_delta_std": 0.1},
            },
        },
    )

    result = aggregate(str(tmp_path / "*" / "summary.json"))
    splits = result["split_leaderboards"]
    by_world = splits["by_world_model_mode"]

    assert by_world["historical"]["n_runs"] == 2
    assert by_world["historical"]["best_run"]["run_name"] == "hist_a"
    assert by_world["historical"]["leaderboard"][0]["rank"] == 1
    assert by_world["historical"]["leaderboard"][1]["rank"] == 2
    assert by_world["generative"]["n_runs"] == 1
    assert by_world["generative"]["best_run"]["run_name"] == "gen_a"


def test_aggregate_skips_malformed_summary_files(tmp_path):
    _write_summary(
        tmp_path / "ok" / "summary.json",
        {
            "run_name": "ok_run",
            "world_model_mode": "historical",
            "policy_mode": "random",
            "metrics": {
                "pnl": {"total_pnl": 1.0, "cash_pnl": 1.0, "inventory": 0.0},
                "drawdown": {"max_drawdown": -0.1},
                "risk": {"pnl_delta_std": 0.1},
            },
        },
    )
    bad = tmp_path / "bad" / "summary.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not-json")

    result = aggregate(str(tmp_path / "*" / "summary.json"))
    assert result["n_runs"] == 1
    assert result["leaderboard"][0]["run_name"] == "ok_run"
