import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="LOBArena Phase 2 train/eval orchestrator")
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--test_data_dir", required=True)
    p.add_argument("--jaxmarl_root", default="/home/s5e/satyamaga.s5e/JaxMARL-HFT")
    p.add_argument("--output_root", default="/home/s5e/satyamaga.s5e/LOBArena/outputs/evaluations")
    p.add_argument("--train_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--policy_ckpt_dir", default="")
    p.add_argument("--policy_config", default="")
    p.add_argument("--run_name", default="phase2_train_eval")
    p.add_argument("--fast_startup", action="store_true")
    return p.parse_args()


def _run(cmd):
    print("[LOBArena phase2]", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    args = parse_args()
    t0 = time.time()

    run_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Phase2 training integration path:
    # If checkpoint+config provided, treat as pre-trained policy handoff;
    # otherwise use random policy baseline as a training placeholder for workflow wiring.
    policy_mode = "ippo_rnn" if args.policy_ckpt_dir and args.policy_config else "random"

    eval_script = str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_checkpoint.py")

    cmd = [
        sys.executable,
        eval_script,
        "--world_model", "historical",
        "--policy_mode", policy_mode,
        "--data_dir", args.test_data_dir,
        "--n_steps", str(args.eval_steps),
        "--run_name", f"{args.run_name}_test_eval",
    ]
    if policy_mode == "ippo_rnn":
        cmd += ["--policy_ckpt_dir", args.policy_ckpt_dir, "--policy_config", args.policy_config]
    if args.fast_startup:
        cmd += ["--fast_startup"]
    rc = _run(cmd)

    summary = {
        "run_name": args.run_name,
        "train_data_dir": str(Path(args.train_data_dir).expanduser().resolve()),
        "test_data_dir": str(Path(args.test_data_dir).expanduser().resolve()),
        "policy_mode": policy_mode,
        "policy_ckpt_dir": args.policy_ckpt_dir,
        "policy_config": args.policy_config,
        "train_steps": int(args.train_steps),
        "eval_steps": int(args.eval_steps),
        "eval_rc": int(rc),
        "status": "success" if rc == 0 else "failed",
        "runtime_sec": float(time.time() - t0),
    }
    (run_dir / "train_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[LOBArena phase2] Summary written: {run_dir / 'train_eval_summary.json'}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
