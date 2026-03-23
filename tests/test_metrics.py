
from LOBArena.metrics.computation import build_phase1_metrics


def test_build_phase1_metrics_shapes():
    m = build_phase1_metrics(
        final_pnl={"total_pnl": 1.2, "cash_pnl": 0.5, "inventory": 2.0},
        pnl_trace=[0.0, 1.0, 0.2, 1.2],
        inventory_trace=[0, 1, -1, 2],
        mid_before_action=[100.0, 101.0],
        mid_after_action=[100.5, 100.8],
    )
    assert "pnl" in m and "drawdown" in m and "risk" in m and "inventory" in m and "impact" in m
