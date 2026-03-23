
import argparse
import glob
import json
from pathlib import Path


def rank_key(row):
    pnl = float(row.get("total_pnl", 0.0))
    dd = float(row.get("max_drawdown", 0.0))
    risk = float(row.get("risk_std", 0.0))
    return (pnl, -abs(dd), -risk)


def aggregate(pattern):
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        data = json.loads(Path(f).read_text())
        metrics = data.get("metrics", {})
        pnl = metrics.get("pnl", {})
        drawdown = metrics.get("drawdown", {})
        risk = metrics.get("risk", {})
        rows.append(
            {
                "run_name": data.get("run_name", Path(f).parent.name),
                "summary_path": str(f),
                "world_model_mode": data.get("world_model_mode", "unknown"),
                "policy_mode": data.get("policy_mode", "unknown"),
                "total_pnl": float(pnl.get("total_pnl", 0.0)),
                "cash_pnl": float(pnl.get("cash_pnl", 0.0)),
                "inventory": float(pnl.get("inventory", 0.0)),
                "max_drawdown": float(drawdown.get("max_drawdown", 0.0)),
                "risk_std": float(risk.get("pnl_delta_std", 0.0)),
            }
        )

    rows_sorted = sorted(rows, key=rank_key, reverse=True)
    for idx, row in enumerate(rows_sorted, start=1):
        row["rank"] = idx

    return {
        "n_runs": len(rows_sorted),
        "leaderboard": rows_sorted,
        "best_run": rows_sorted[0] if rows_sorted else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate LOBArena run summaries into a leaderboard")
    parser.add_argument("--glob", required=True, dest="glob_pattern", help="Glob pattern for summary JSON files")
    parser.add_argument("--output", required=True, help="Output leaderboard JSON path")
    args = parser.parse_args()

    out = aggregate(args.glob_pattern)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote leaderboard: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
