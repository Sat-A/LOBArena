
import math
from typing import Any, Dict


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


RISK_SCORE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "pnl": 1.0,
    "drawdown": 0.5,
    "risk": 0.1,
    "inventory": 0.0,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_risk_score_weights(raw_weights: Dict[str, Any] | None) -> Dict[str, float]:
    out = dict(RISK_SCORE_DEFAULT_WEIGHTS)
    if not raw_weights:
        return out
    for key in ("pnl", "drawdown", "risk", "inventory"):
        if key in raw_weights:
            out[key] = _as_float(raw_weights.get(key), out[key])
    return out


def risk_score_weights_from_cli(weights_arg: str) -> Dict[str, float]:
    if not str(weights_arg or "").strip():
        return dict(RISK_SCORE_DEFAULT_WEIGHTS)
    raw: Dict[str, float] = {}
    for item in str(weights_arg).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight item '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        raw[key.strip()] = _as_float(value.strip())
    return normalize_risk_score_weights(raw)


def compute_raw_pnl_score(summary_or_metrics: Dict[str, Any]) -> float:
    metrics = summary_or_metrics.get("metrics", summary_or_metrics)
    pnl = metrics.get("pnl", {}) if isinstance(metrics, dict) else {}
    return _as_float(pnl.get("total_pnl", 0.0))


def compute_risk_adjusted_pnl_score(
    summary_or_metrics: Dict[str, Any],
    weights: Dict[str, Any] | None = None,
) -> float:
    metrics = summary_or_metrics.get("metrics", summary_or_metrics)
    if not isinstance(metrics, dict):
        metrics = {}
    pnl = metrics.get("pnl", {})
    drawdown = metrics.get("drawdown", {})
    risk = metrics.get("risk", {})
    resolved = normalize_risk_score_weights(weights)

    total_pnl = _as_float(pnl.get("total_pnl", 0.0))
    max_drawdown_abs = abs(_as_float(drawdown.get("max_drawdown", 0.0)))
    risk_std = _as_float(risk.get("pnl_delta_std", 0.0))
    inventory_abs = abs(_as_float(pnl.get("inventory", 0.0)))
    return (
        resolved["pnl"] * total_pnl
        - resolved["drawdown"] * max_drawdown_abs
        - resolved["risk"] * risk_std
        - resolved["inventory"] * inventory_abs
    )
