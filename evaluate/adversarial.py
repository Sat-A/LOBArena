import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="LOBArena Phase 2 adversarial evaluation")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--target_policy_mode", choices=["random", "fixed", "ippo_rnn"], default="random")
    p.add_argument("--target_policy_ckpt", default="")
    p.add_argument("--target_policy_config", default="")
    p.add_argument("--competitor_policy_mode", choices=["random", "fixed", "ippo_rnn"], default="random")
    p.add_argument("--competitor_policy_ckpt", default="")
    p.add_argument("--competitor_policy_config", default="")
    p.add_argument("--output_root", default="/home/s5e/satyamaga.s5e/LOBArena/outputs/evaluations")
    p.add_argument("--run_name", default="phase2_adversarial")
    p.add_argument("--n_steps", type=int, default=10)
    return p.parse_args()


def _run_eval(run_name, data_dir, policy_mode, ckpt, cfg, n_steps):
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_checkpoint.py")
    cmd = [
        sys.executable,
        script,
        "--world_model", "historical",
        "--policy_mode", policy_mode,
        "--data_dir", data_dir,
        "--n_steps", str(n_steps),
        "--run_name", run_name,
        "--fast_startup",
    ]
    if policy_mode == "ippo_rnn":
        cmd += ["--policy_ckpt_dir", ckpt, "--policy_config", cfg]
    return subprocess.call(cmd)


def main():
    args = parse_args()
    t0 = time.time()

    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    target_run = f"{args.run_name}_target"
    comp_run = f"{args.run_name}_competitor"

    rc_target = _run_eval(target_run, args.data_dir, args.target_policy_mode, args.target_policy_ckpt, args.target_policy_config, args.n_steps)
    rc_comp = _run_eval(comp_run, args.data_dir, args.competitor_policy_mode, args.competitor_policy_ckpt, args.competitor_policy_config, args.n_steps)

    target_summary = out_root / target_run / "summary.json"
    comp_summary = out_root / comp_run / "summary.json"

    def _read_pnl(path):
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return float(data.get("metrics", {}).get("pnl", {}).get("total_pnl", 0.0))

    pnl_target = _read_pnl(target_summary)
    pnl_comp = _read_pnl(comp_summary)

    result = {
        "run_name": args.run_name,
        "target_run": target_run,
        "competitor_run": comp_run,
        "target_rc": int(rc_target),
        "competitor_rc": int(rc_comp),
        "target_total_pnl": pnl_target,
        "competitor_total_pnl": pnl_comp,
        "winner": "target" if pnl_target is not None and pnl_comp is not None and pnl_target >= pnl_comp else "competitor",
        "runtime_sec": float(time.time() - t0),
    }

    out = out_root / args.run_name / "adversarial_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[LOBArena adversarial] Summary written: {out}")
    return 0 if (rc_target == 0 and rc_comp == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
