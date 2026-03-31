
from LOBArena.metrics.computation import (
    build_phase1_metrics,
    compute_raw_pnl_score,
    compute_risk_adjusted_pnl_score,
    normalize_risk_score_weights,
    risk_score_weights_from_cli,
)


def test_build_phase1_metrics_shapes():
    m = build_phase1_metrics(
        final_pnl={"total_pnl": 1.2, "cash_pnl": 0.5, "inventory": 2.0},
        pnl_trace=[0.0, 1.0, 0.2, 1.2],
        inventory_trace=[0, 1, -1, 2],
        mid_before_action=[100.0, 101.0],
        mid_after_action=[100.5, 100.8],
    )
    assert "pnl" in m and "drawdown" in m and "risk" in m and "inventory" in m and "impact" in m


def test_compute_raw_and_risk_adjusted_scores():
    payload = {
        "metrics": {
            "pnl": {"total_pnl": 10.0, "inventory": -2.0},
            "drawdown": {"max_drawdown": -3.0},
            "risk": {"pnl_delta_std": 1.5},
        }
    }
    assert compute_raw_pnl_score(payload) == 10.0

    score = compute_risk_adjusted_pnl_score(payload, {"pnl": 1.0, "drawdown": 0.5, "risk": 0.1, "inventory": 0.0})
    assert abs(score - (10.0 - 1.5 - 0.15)) < 1e-9


def test_weight_helpers_defaults_and_cli_parse():
    default_w = normalize_risk_score_weights(None)
    assert default_w["pnl"] == 1.0
    assert default_w["drawdown"] == 0.5
    parsed = risk_score_weights_from_cli("pnl=2,drawdown=1.0,risk=0.2,inventory=0.1")
    assert parsed == {"pnl": 2.0, "drawdown": 1.0, "risk": 0.2, "inventory": 0.1}
