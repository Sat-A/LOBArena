import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_WEIGHTS: Dict[str, float] = {
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


def _extract_date_window(data: Dict[str, Any]) -> Tuple[str, str]:
    start = ""
    end = ""

    date_window = data.get("date_window")
    if isinstance(date_window, dict):
        start = str(date_window.get("start_date", "") or "")
        end = str(date_window.get("end_date", "") or "")

    if not start:
        start = str(data.get("start_date", "") or "")
    if not end:
        end = str(data.get("end_date", "") or "")

    handoff = data.get("policy_handoff")
    if isinstance(handoff, dict):
        eval_meta = handoff.get("evaluation")
        if isinstance(eval_meta, dict):
            dw = eval_meta.get("date_window")
            if isinstance(dw, dict):
                if not start:
                    start = str(dw.get("start_date", "") or "")
                if not end:
                    end = str(dw.get("end_date", "") or "")

    return start, end


def _date_window_key(start_date: str, end_date: str) -> str:
    if start_date or end_date:
        return f"{start_date or 'unknown'}::{end_date or 'unknown'}"
    return "unknown"


def _resolve_policy_family(data: Dict[str, Any]) -> str:
    if data.get("policy_family"):
        return str(data["policy_family"])

    handoff = data.get("policy_handoff")
    if isinstance(handoff, dict):
        policy_meta = handoff.get("policy")
        if isinstance(policy_meta, dict):
            family = policy_meta.get("family")
            if family:
                return str(family)
            if policy_meta.get("mode"):
                return str(policy_meta["mode"])

    return str(data.get("policy_mode", "unknown"))


def _normalize_weights(raw_weights: Dict[str, Any]) -> Dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    for key in ("pnl", "drawdown", "risk", "inventory"):
        if key in raw_weights:
            w[key] = _as_float(raw_weights.get(key), w[key])

    # Support objective-gate config names additively.
    alias_map = {
        "total_pnl": "pnl",
        "drawdown_penalty": "drawdown",
        "pnl_delta_std": "risk",
        "risk_penalty": "risk",
        "risk_denominator": "risk",
        "inventory_penalty": "inventory",
    }
    for src, dst in alias_map.items():
        if src in raw_weights:
            w[dst] = _as_float(raw_weights.get(src), w[dst])
    return w


def _weights_from_config(config_path: str) -> Dict[str, float]:
    payload = json.loads(Path(config_path).read_text())

    if isinstance(payload, dict) and any(k in payload for k in ("pnl", "drawdown", "risk", "inventory")):
        return _normalize_weights(payload)

    selection = payload.get("selection_policy") if isinstance(payload, dict) else None
    primary = selection.get("primary_objective") if isinstance(selection, dict) else None
    weights = primary.get("weights") if isinstance(primary, dict) else None
    if isinstance(weights, dict):
        return _normalize_weights(weights)

    raise ValueError(f"Could not find supported weights in config: {config_path}")


def _weights_from_cli(weights_arg: str) -> Dict[str, float]:
    raw: Dict[str, float] = {}
    for item in weights_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight item '{item}'. Expected key=value")
        key, val = item.split("=", 1)
        raw[key.strip()] = _as_float(val.strip())
    return _normalize_weights(raw)


def compute_composite_score(row: Dict[str, Any], weights: Dict[str, float]) -> float:
    pnl = _as_float(row.get("total_pnl", 0.0))
    dd_abs = abs(_as_float(row.get("max_drawdown", 0.0)))
    risk = _as_float(row.get("risk_std", 0.0))
    inventory_abs = abs(_as_float(row.get("inventory", 0.0)))
    return (
        weights["pnl"] * pnl
        - weights["drawdown"] * dd_abs
        - weights["risk"] * risk
        - weights["inventory"] * inventory_abs
    )


def _legacy_rank_tuple(row: Dict[str, Any]) -> Tuple[float, float, float]:
    pnl = _as_float(row.get("total_pnl", 0.0))
    dd = _as_float(row.get("max_drawdown", 0.0))
    risk = _as_float(row.get("risk_std", 0.0))
    return (pnl, -abs(dd), -risk)


def _row_sort_key(row: Dict[str, Any]) -> Tuple[float, Tuple[float, float, float], str, str]:
    return (
        _as_float(row.get("composite_score", 0.0)),
        _legacy_rank_tuple(row),
        str(row.get("run_name", "")),
        str(row.get("summary_path", "")),
    )


def _rank_rows(rows: Iterable[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    ranked = []
    for row in rows:
        out = dict(row)
        out["composite_score"] = compute_composite_score(out, weights)
        ranked.append(out)

    ranked = sorted(ranked, key=_row_sort_key, reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def _build_split(groups: Dict[str, List[Dict[str, Any]]], weights: Dict[str, float]) -> Dict[str, Any]:
    out = {}
    for group_key, rows in sorted(groups.items()):
        ranked = _rank_rows(rows, weights)
        out[group_key] = {
            "n_runs": len(ranked),
            "leaderboard": ranked,
            "best_run": ranked[0] if ranked else None,
        }
    return out


def _collect_rows(pattern: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(pattern))
    rows = []
    skipped = 0
    for f in files:
        try:
            data = json.loads(Path(f).read_text())
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        metrics = data.get("metrics", {})
        pnl = metrics.get("pnl", {})
        drawdown = metrics.get("drawdown", {})
        risk = metrics.get("risk", {})

        start_date, end_date = _extract_date_window(data)
        rows.append(
            {
                "run_name": data.get("run_name", Path(f).parent.name),
                "summary_path": str(f),
                "world_model_mode": data.get("world_model_mode", "unknown"),
                "policy_mode": data.get("policy_mode", "unknown"),
                "policy_family": _resolve_policy_family(data),
                "date_window_start": start_date,
                "date_window_end": end_date,
                "date_window_key": _date_window_key(start_date, end_date),
                "total_pnl": _as_float(pnl.get("total_pnl", 0.0)),
                "cash_pnl": _as_float(pnl.get("cash_pnl", 0.0)),
                "inventory": _as_float(pnl.get("inventory", 0.0)),
                "max_drawdown": _as_float(drawdown.get("max_drawdown", 0.0)),
                "risk_std": _as_float(risk.get("pnl_delta_std", 0.0)),
            }
        )
    if skipped:
        print(f"Skipped {skipped} unreadable/malformed summary file(s).")
    return rows


def aggregate(pattern: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    use_weights = _normalize_weights(weights or DEFAULT_WEIGHTS)
    rows = _collect_rows(pattern)

    rows_sorted = _rank_rows(rows, use_weights)

    by_world_model: Dict[str, List[Dict[str, Any]]] = {}
    by_policy_mode: Dict[str, List[Dict[str, Any]]] = {}
    by_policy_family: Dict[str, List[Dict[str, Any]]] = {}
    by_date_window: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        wm = str(row.get("world_model_mode", "unknown"))
        pm = str(row.get("policy_mode", "unknown"))
        pf = str(row.get("policy_family", "unknown"))
        dw = str(row.get("date_window_key", "unknown"))

        by_world_model.setdefault(wm, []).append(row)
        by_policy_mode.setdefault(pm, []).append(row)
        by_policy_family.setdefault(pf, []).append(row)
        if dw != "unknown":
            by_date_window.setdefault(dw, []).append(row)

    return {
        "n_runs": len(rows_sorted),
        "leaderboard": rows_sorted,
        "best_run": rows_sorted[0] if rows_sorted else None,
        "ranking": {
            "method": "weighted_composite",
            "weights": use_weights,
        },
        "split_leaderboards": {
            "by_world_model_mode": _build_split(by_world_model, use_weights),
            "by_policy_mode": _build_split(by_policy_mode, use_weights),
            "by_policy_family": _build_split(by_policy_family, use_weights),
            "by_date_window": _build_split(by_date_window, use_weights),
        },
    }


def export_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "rank",
        "run_name",
        "summary_path",
        "world_model_mode",
        "policy_mode",
        "policy_family",
        "date_window_start",
        "date_window_end",
        "composite_score",
        "total_pnl",
        "cash_pnl",
        "inventory",
        "max_drawdown",
        "risk_std",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def _resolve_weights(weights_arg: str, weights_config: str) -> Dict[str, float]:
    if weights_arg and weights_config:
        raise ValueError("Use one of --weights or --weights-config, not both")
    if weights_arg:
        return _weights_from_cli(weights_arg)
    if weights_config:
        return _weights_from_config(weights_config)
    return dict(DEFAULT_WEIGHTS)


def _iqm(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    q1_idx = int(0.25 * (n - 1))
    q3_idx = int(0.75 * (n - 1))
    core = ordered[q1_idx : q3_idx + 1]
    return float(sum(core) / len(core)) if core else float(sum(ordered) / len(ordered))


def _mean_median_iqm(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "iqm": 0.0}
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mean = float(sum(ordered) / n)
    median = float(ordered[n // 2]) if (n % 2 == 1) else float((ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)
    return {"mean": mean, "median": median, "iqm": _iqm(ordered)}


def aggregate_multi_window_summary(summary_payload: Dict[str, Any]) -> Dict[str, Any]:
    windows = summary_payload.get("windows", [])
    raw = [_as_float(w.get("raw_pnl_score", 0.0)) for w in windows if isinstance(w, dict)]
    risk = [_as_float(w.get("risk_adjusted_pnl_score", 0.0)) for w in windows if isinstance(w, dict)]
    return {
        "n_windows": len(windows),
        "raw_pnl": _mean_median_iqm(raw),
        "risk_adjusted_pnl": _mean_median_iqm(risk),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate LOBArena run summaries into a leaderboard")
    parser.add_argument("--glob", required=True, dest="glob_pattern", help="Glob pattern for summary JSON files")
    parser.add_argument("--output", required=True, help="Output leaderboard JSON path")
    parser.add_argument(
        "--weights",
        default="",
        help="Comma-separated weights, e.g. pnl=1.0,drawdown=0.5,risk=0.1,inventory=0.0",
    )
    parser.add_argument(
        "--weights-config",
        default="",
        help="Path to weights config JSON (simple weights object or objective-gates config)",
    )
    parser.add_argument(
        "--csv-output",
        default="",
        help="Optional CSV export path for overall leaderboard rows",
    )
    args = parser.parse_args()

    weights = _resolve_weights(args.weights, args.weights_config)
    out = aggregate(args.glob_pattern, weights=weights)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    if args.csv_output:
        export_csv(out["leaderboard"], Path(args.csv_output))
        print(f"Wrote leaderboard CSV: {args.csv_output}")

    print(f"Wrote leaderboard: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
