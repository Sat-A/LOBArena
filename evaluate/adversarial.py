import argparse
import itertools
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    workspace_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="LOBArena Phase 2 adversarial evaluation")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--target_policy_mode", choices=["random", "fixed", "ippo_rnn", "lose_money"], default="random")
    p.add_argument("--target_fixed_action", type=int, default=0)
    p.add_argument("--target_policy_ckpt", default="")
    p.add_argument("--target_policy_config", default="")
    p.add_argument("--target_policy_handoff", default="")
    p.add_argument("--competitor_policy_mode", choices=["random", "fixed", "ippo_rnn", "lose_money"], default="random")
    p.add_argument("--competitor_fixed_action", type=int, default=0)
    p.add_argument("--competitor_policy_ckpt", default="")
    p.add_argument("--competitor_policy_config", default="")
    p.add_argument("--competitor_policy_handoff", default="")
    p.add_argument(
        "--competitor_registry_config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "evaluation_configs" / "adversarial_competitors.json"),
        help="Path to competitor registry JSON.",
    )
    p.add_argument(
        "--competitor_keys",
        nargs="*",
        default=[],
        help="Registry keys for one or more competitors to evaluate (overrides direct competitor args).",
    )
    p.add_argument("--output_root", default=str(workspace_root / "LOBArena" / "outputs" / "evaluations"))
    p.add_argument("--run_name", default="phase2_adversarial")
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--test_split", type=float, default=1.0)
    p.add_argument("--start_date", default="")
    p.add_argument("--end_date", default="")
    p.add_argument(
        "--round_robin",
        action="store_true",
        help="When set, compare all participants pairwise; default is target-vs-many.",
    )
    return p.parse_args()


def load_competitor_registry(registry_path: Path) -> Dict[str, dict]:
    path = Path(registry_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Competitor registry not found: {path}")
    raw = json.loads(path.read_text())
    competitors = raw.get("competitors")
    if not isinstance(competitors, dict):
        raise ValueError(f"Invalid registry format in {path}: expected top-level 'competitors' object")
    return competitors


def _validate_competitor_spec(name: str, spec: dict) -> None:
    handoff = str(spec.get("policy_handoff", "")).strip()
    if handoff:
        handoff_path = Path(handoff).expanduser().resolve()
        if not handoff_path.exists():
            raise FileNotFoundError(f"Competitor '{name}' handoff artifact not found: {handoff_path}")
        return

    mode = str(spec.get("policy_mode", "")).strip().lower()
    if mode in {"directional", "scripted", "scripted_directional"}:
        raise ValueError(
            f"Competitor '{name}' uses unsupported policy_mode '{mode}'. "
            "This placeholder is not executable yet."
        )
    if mode not in {"random", "fixed", "ippo_rnn", "lose_money"}:
        raise ValueError(f"Competitor '{name}' has invalid policy_mode '{mode}'")
    if mode == "fixed":
        if "fixed_action" not in spec:
            raise ValueError(f"Competitor '{name}' with policy_mode=fixed requires 'fixed_action'")
        int(spec["fixed_action"])
    if mode == "ippo_rnn":
        ckpt = str(spec.get("policy_ckpt_dir", "")).strip()
        cfg = str(spec.get("policy_config", "")).strip()
        if not ckpt or not cfg:
            raise ValueError(
                f"Competitor '{name}' with policy_mode=ippo_rnn requires "
                "'policy_ckpt_dir' and 'policy_config'"
            )
        ckpt_path = Path(ckpt).expanduser().resolve()
        cfg_path = Path(cfg).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Competitor '{name}' checkpoint dir not found: {ckpt_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Competitor '{name}' config file not found: {cfg_path}")


def _resolve_competitors(args) -> List[dict]:
    if args.competitor_keys:
        registry = load_competitor_registry(Path(args.competitor_registry_config))
        resolved = []
        for key in args.competitor_keys:
            if key not in registry:
                raise KeyError(f"Competitor key '{key}' not found in registry")
            spec = dict(registry[key])
            _validate_competitor_spec(key, spec)
            resolved.append(
                {
                    "key": key,
                    "policy_mode": str(spec["policy_mode"]).strip().lower(),
                    "fixed_action": int(spec.get("fixed_action", 0)),
                    "policy_ckpt_dir": str(spec.get("policy_ckpt_dir", "")),
                    "policy_config": str(spec.get("policy_config", "")),
                    "policy_handoff": str(spec.get("policy_handoff", "")),
                    "source": "registry",
                }
            )
        return resolved

    direct = {
        "key": "competitor",
        "policy_mode": args.competitor_policy_mode,
        "fixed_action": int(args.competitor_fixed_action),
        "policy_ckpt_dir": args.competitor_policy_ckpt,
        "policy_config": args.competitor_policy_config,
        "policy_handoff": args.competitor_policy_handoff,
        "source": "direct_args",
    }
    _validate_competitor_spec("competitor(direct_args)", direct)
    return [direct]


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "competitor"


def _resolve_tournament_mode(round_robin: bool, competitor_count: int) -> str:
    if round_robin:
        return "round-robin"
    if competitor_count > 1:
        return "target-vs-many"
    return "target-vs-many"


def _build_matchups(mode: str, participant_ids: List[str], target_id: str = "target") -> List[Tuple[str, str]]:
    if mode == "round-robin":
        return list(itertools.combinations(participant_ids, 2))
    return [(target_id, pid) for pid in participant_ids if pid != target_id]


def _winner_from_pnl(pnl_left, pnl_right) -> str:
    if pnl_left is None or pnl_right is None:
        return "unknown"
    if pnl_left > pnl_right:
        return "left"
    if pnl_right > pnl_left:
        return "right"
    return "tie"


def _aggregate_match_counts(matches: List[dict], participant_ids: List[str]) -> Dict[str, dict]:
    agg = {
        pid: {"wins": 0, "losses": 0, "ties": 0, "unknown": 0}
        for pid in participant_ids
    }
    for match in matches:
        left = match["left"]["participant_id"]
        right = match["right"]["participant_id"]
        winner = match["winner"]
        if winner == left:
            agg[left]["wins"] += 1
            agg[right]["losses"] += 1
        elif winner == right:
            agg[right]["wins"] += 1
            agg[left]["losses"] += 1
        elif winner == "tie":
            agg[left]["ties"] += 1
            agg[right]["ties"] += 1
        else:
            agg[left]["unknown"] += 1
            agg[right]["unknown"] += 1
    return agg


def _pnl_delta_from_target_perspective(match: dict, target_id: str = "target"):
    left_id = match["left"]["participant_id"]
    right_id = match["right"]["participant_id"]
    pnl_delta_left_minus_right = match.get("pnl_delta_left_minus_right")
    if pnl_delta_left_minus_right is None:
        return None
    if left_id == target_id:
        return float(pnl_delta_left_minus_right)
    if right_id == target_id:
        return float(-pnl_delta_left_minus_right)
    return None


def _compute_win_rate(wins: int, losses: int, ties: int):
    total = wins + losses + ties
    if total <= 0:
        return None
    return float((wins + 0.5 * ties) / total)


def _compute_numeric_stats(values: List[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mean = sum(ordered) / n
    if n % 2 == 1:
        median = ordered[n // 2]
    else:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    variance = sum((v - mean) ** 2 for v in ordered) / n
    return {
        "mean": float(mean),
        "median": float(median),
        "std": float(math.sqrt(variance)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _compute_target_pairwise_summary(matches: List[dict], target_id: str = "target") -> dict:
    totals = {"wins": 0, "losses": 0, "ties": 0, "unknown": 0}
    per_competitor = {}
    pnl_deltas = []

    for match in matches:
        left_id = match["left"]["participant_id"]
        right_id = match["right"]["participant_id"]
        if target_id not in (left_id, right_id):
            continue

        competitor_id = right_id if left_id == target_id else left_id
        winner = match.get("winner")
        if winner == target_id:
            outcome = "wins"
        elif winner == competitor_id:
            outcome = "losses"
        elif winner == "tie":
            outcome = "ties"
        else:
            outcome = "unknown"
        totals[outcome] += 1

        if competitor_id not in per_competitor:
            per_competitor[competitor_id] = {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "unknown": 0,
                "pnl_deltas": [],
            }
        per_competitor[competitor_id][outcome] += 1

        delta = _pnl_delta_from_target_perspective(match, target_id=target_id)
        if delta is not None:
            pnl_deltas.append(delta)
            per_competitor[competitor_id]["pnl_deltas"].append(delta)

    per_competitor_out = {}
    for competitor_id in sorted(per_competitor.keys()):
        rec = per_competitor[competitor_id]
        deltas = rec.pop("pnl_deltas")
        rec["win_rate"] = _compute_win_rate(rec["wins"], rec["losses"], rec["ties"])
        rec["avg_pnl_delta"] = float(sum(deltas) / len(deltas)) if deltas else None
        per_competitor_out[competitor_id] = rec

    return {
        "matches_considered": int(sum(totals.values())),
        "wins": int(totals["wins"]),
        "losses": int(totals["losses"]),
        "ties": int(totals["ties"]),
        "unknown": int(totals["unknown"]),
        "win_rate_overall": _compute_win_rate(totals["wins"], totals["losses"], totals["ties"]),
        "per_competitor": per_competitor_out,
        "pnl_delta_stats": _compute_numeric_stats(pnl_deltas),
    }


def _compute_regime_date_robustness(matches: List[dict], target_id: str = "target") -> List[dict]:
    buckets = {}
    for match in matches:
        left_id = match["left"]["participant_id"]
        right_id = match["right"]["participant_id"]
        if target_id not in (left_id, right_id):
            continue

        target_run = match["left"] if left_id == target_id else match["right"]
        metadata = match.get("metadata", {})
        regime = str(target_run.get("world_model_mode") or "unknown")
        start_date = str(metadata.get("start_date", ""))
        end_date = str(metadata.get("end_date", ""))
        key = (regime, start_date, end_date)
        if key not in buckets:
            buckets[key] = {"wins": 0, "losses": 0, "ties": 0, "unknown": 0, "pnl_deltas": []}

        winner = match.get("winner")
        if winner == target_id:
            outcome = "wins"
        elif winner in (left_id, right_id):
            outcome = "losses"
        elif winner == "tie":
            outcome = "ties"
        else:
            outcome = "unknown"
        buckets[key][outcome] += 1

        delta = _pnl_delta_from_target_perspective(match, target_id=target_id)
        if delta is not None:
            buckets[key]["pnl_deltas"].append(delta)

    out = []
    for regime, start_date, end_date in sorted(buckets.keys()):
        rec = buckets[(regime, start_date, end_date)]
        deltas = rec.pop("pnl_deltas")
        out.append(
            {
                "regime": regime,
                "start_date": start_date,
                "end_date": end_date,
                "match_count": int(rec["wins"] + rec["losses"] + rec["ties"] + rec["unknown"]),
                "win_rate_overall": _compute_win_rate(rec["wins"], rec["losses"], rec["ties"]),
                "avg_pnl_delta": float(sum(deltas) / len(deltas)) if deltas else None,
                "pnl_delta_stats": _compute_numeric_stats(deltas),
            }
        )
    return out


def _run_eval(
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
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_checkpoint.py")
    cmd = [
        sys.executable,
        script,
        "--world_model", "historical",
        "--policy_mode", policy_mode,
        "--fixed_action", str(int(fixed_action)),
        "--data_dir", data_dir,
        "--n_steps", str(n_steps),
        "--seed", str(seed),
        "--sample_index", str(sample_index),
        "--test_split", str(test_split),
        "--output_root", output_root,
        "--run_name", run_name,
        "--fast_startup",
    ]
    if start_date:
        cmd += ["--start_date", start_date]
    if end_date:
        cmd += ["--end_date", end_date]
    if policy_handoff:
        cmd += ["--policy_handoff", policy_handoff]
    elif policy_mode == "ippo_rnn":
        cmd += ["--policy_ckpt_dir", ckpt, "--policy_config", cfg]
    return subprocess.call(cmd)


def main():
    args = parse_args()
    t0 = time.time()

    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    target_run = f"{args.run_name}_target"
    target_spec = {
        "policy_mode": args.target_policy_mode,
        "fixed_action": int(args.target_fixed_action),
        "policy_ckpt_dir": args.target_policy_ckpt,
        "policy_config": args.target_policy_config,
        "policy_handoff": args.target_policy_handoff,
    }
    _validate_competitor_spec("target", target_spec)
    competitors = _resolve_competitors(args)

    tournament_mode = _resolve_tournament_mode(args.round_robin, len(competitors))

    rc_target = _run_eval(
        target_run,
        str(out_root),
        args.data_dir,
        target_spec["policy_mode"],
        target_spec["fixed_action"],
        target_spec["policy_ckpt_dir"],
        target_spec["policy_config"],
        args.n_steps,
        args.seed,
        args.sample_index,
        args.test_split,
        args.start_date,
        args.end_date,
        target_spec["policy_handoff"],
    )

    target_summary = out_root / target_run / "summary.json"

    def _read_pnl(path):
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return float(data.get("metrics", {}).get("pnl", {}).get("total_pnl", 0.0))

    def _read_world_model_mode(path):
        if not path.exists():
            return "unknown"
        data = json.loads(path.read_text())
        return str(data.get("world_model_mode", "unknown"))

    pnl_target = _read_pnl(target_summary)

    participant_specs = [
        {
            "participant_id": "target",
            "role": "target",
            "key": "target",
            "source": "direct_args",
            **target_spec,
        }
    ]

    comp_results = []
    run_records = {
        "target": {
            "participant_id": "target",
            "run_id": target_run,
            "run_dir": str(out_root / target_run),
            "summary_path": str(target_summary),
            "rc": int(rc_target),
            "total_pnl": pnl_target,
            "world_model_mode": _read_world_model_mode(target_summary),
        }
    }
    seen_comp_run_ids = set()
    for comp in competitors:
        participant_id = f"competitor:{comp['key']}"
        if participant_id in run_records:
            suffix = 2
            while f"{participant_id}:{suffix}" in run_records:
                suffix += 1
            participant_id = f"{participant_id}:{suffix}"

        if len(competitors) == 1 and comp["source"] == "direct_args":
            comp_run = f"{args.run_name}_competitor"
        else:
            base_run = f"{args.run_name}_competitor_{_slugify(comp['key'])}"
            comp_run = base_run
            suffix = 2
            while comp_run in seen_comp_run_ids:
                comp_run = f"{base_run}_{suffix}"
                suffix += 1
        seen_comp_run_ids.add(comp_run)

        rc_comp = _run_eval(
            comp_run,
            str(out_root),
            args.data_dir,
            comp["policy_mode"],
            comp["fixed_action"],
            comp["policy_ckpt_dir"],
            comp["policy_config"],
            args.n_steps,
            args.seed,
            args.sample_index,
            args.test_split,
            args.start_date,
            args.end_date,
            comp.get("policy_handoff", ""),
        )
        comp_summary_path = out_root / comp_run / "summary.json"
        pnl_comp = _read_pnl(comp_summary_path)
        participant_specs.append(
            {
                "participant_id": participant_id,
                "role": "competitor",
                **comp,
            }
        )
        run_records[participant_id] = {
            "participant_id": participant_id,
            "run_id": comp_run,
            "run_dir": str(out_root / comp_run),
            "summary_path": str(comp_summary_path),
            "rc": int(rc_comp),
            "total_pnl": pnl_comp,
            "world_model_mode": _read_world_model_mode(comp_summary_path),
        }
        comp_results.append(
            {
                "participant_id": participant_id,
                "key": comp["key"],
                "source": comp["source"],
                "competitor_run": comp_run,
                "competitor_rc": int(rc_comp),
                "competitor_summary_path": str(comp_summary_path),
                "competitor_total_pnl": pnl_comp,
                "winner": (
                    "target"
                    if pnl_target is not None and pnl_comp is not None and pnl_target >= pnl_comp
                    else "competitor"
                ),
            }
        )

    participant_ids = [p["participant_id"] for p in participant_specs]
    matchups = _build_matchups(tournament_mode, participant_ids, target_id="target")
    matches = []
    for i, (left_id, right_id) in enumerate(matchups, start=1):
        left = run_records[left_id]
        right = run_records[right_id]
        winner_side = _winner_from_pnl(left["total_pnl"], right["total_pnl"])
        if winner_side == "left":
            winner = left_id
        elif winner_side == "right":
            winner = right_id
        else:
            winner = winner_side
        pnl_delta = (
            float(left["total_pnl"] - right["total_pnl"])
            if left["total_pnl"] is not None and right["total_pnl"] is not None
            else None
        )
        matches.append(
            {
                "match_id": f"match_{i:03d}",
                "mode": tournament_mode,
                "left": left,
                "right": right,
                "pnl_delta_left_minus_right": pnl_delta,
                "winner": winner,
                "metadata": {
                    "seed": int(args.seed),
                    "start_date": str(args.start_date),
                    "end_date": str(args.end_date),
                    "n_steps": int(args.n_steps),
                    "sample_index": int(args.sample_index),
                    "test_split": float(args.test_split),
                },
            }
        )

    aggregate = _aggregate_match_counts(matches, participant_ids)
    target_pairwise = _compute_target_pairwise_summary(matches, target_id="target")
    regime_date_robustness = _compute_regime_date_robustness(matches, target_id="target")
    first_comp = comp_results[0]

    result = {
        "run_name": args.run_name,
        "tournament_mode": tournament_mode,
        "fairness_config": {
            "seed": int(args.seed),
            "start_date": str(args.start_date),
            "end_date": str(args.end_date),
            "n_steps": int(args.n_steps),
            "sample_index": int(args.sample_index),
            "test_split": float(args.test_split),
        },
        "target_run": target_run,
        "competitor_run": first_comp["competitor_run"],
        "target_rc": int(rc_target),
        "competitor_rc": first_comp["competitor_rc"],
        "target_total_pnl": pnl_target,
        "competitor_total_pnl": first_comp["competitor_total_pnl"],
        "winner": first_comp["winner"],
        "target_summary_path": str(target_summary),
        "competitors": comp_results,
        "participants": participant_specs,
        "matches": matches,
        "aggregate": {
            "by_participant": aggregate,
            "target_pairwise": target_pairwise,
            "regime_date_robustness": regime_date_robustness,
        },
        "win_rate_overall": target_pairwise["win_rate_overall"],
        "runtime_sec": float(time.time() - t0),
    }

    out = out_root / args.run_name / "adversarial_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[LOBArena adversarial] Summary written: {out}")
    return 0 if (rc_target == 0 and all(r["competitor_rc"] == 0 for r in comp_results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
