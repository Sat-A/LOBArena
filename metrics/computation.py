
import math


def max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        mdd = min(mdd, x - peak)
    return float(mdd)


def risk_proxy(pnl_trace):
    if len(pnl_trace) < 2:
        return 0.0
    deltas = [pnl_trace[i] - pnl_trace[i - 1] for i in range(1, len(pnl_trace))]
    mu = sum(deltas) / len(deltas)
    var = sum((x - mu) ** 2 for x in deltas) / max(1, len(deltas) - 1)
    return float(math.sqrt(var))


def inventory_stats(inventory_trace):
    if not inventory_trace:
        return {"mean_abs_inventory": 0.0, "max_abs_inventory": 0.0}
    abs_vals = [abs(x) for x in inventory_trace]
    return {
        "mean_abs_inventory": float(sum(abs_vals) / len(abs_vals)),
        "max_abs_inventory": float(max(abs_vals)),
    }


def impact_proxy(mid_before_action, mid_after_action):
    n = min(len(mid_before_action), len(mid_after_action))
    if n == 0:
        return 0.0
    diffs = [mid_after_action[i] - mid_before_action[i] for i in range(n)]
    return float(sum(abs(x) for x in diffs) / n)


def build_phase1_metrics(final_pnl, pnl_trace, inventory_trace, mid_before_action, mid_after_action):
    return {
        "pnl": final_pnl,
        "drawdown": {"max_drawdown": max_drawdown(pnl_trace)},
        "risk": {"pnl_delta_std": risk_proxy(pnl_trace)},
        "inventory": inventory_stats(inventory_trace),
        "impact": {"mean_abs_midprice_change_after_action": impact_proxy(mid_before_action, mid_after_action)},
    }
