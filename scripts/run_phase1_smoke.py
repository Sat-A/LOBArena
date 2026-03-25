#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run(cmd):
    print("[LOBArena smoke] Running:")
    print(" ", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    workspace_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Run Phase 1 smoke scenarios for LOBArena")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--jaxmarl_root", default=str(workspace_root / "JaxMARL-HFT"))
    p.add_argument("--lobs5_root", default=str(workspace_root / "LOBS5"))
    p.add_argument("--lobs5_ckpt_path", default="")
    p.add_argument("--policy_ckpt_dir", default="")
    p.add_argument("--policy_config", default="")
    p.add_argument("--policy_handoff", default="")
    p.add_argument("--output_root", default=str(workspace_root / "LOBArena" / "outputs" / "evaluations"))
    p.add_argument("--n_steps", type=int, default=5)
    args = p.parse_args()

    script = str((Path(__file__).resolve().parents[1] / "scripts" / "evaluate_checkpoint.py"))

    random_cmd = [
        sys.executable, script,
        "--world_model", "historical",
        "--policy_mode", "random",
        "--data_dir", args.data_dir,
        "--jaxmarl_root", args.jaxmarl_root,
        "--lobs5_root", args.lobs5_root,
        "--output_root", args.output_root,
        "--n_steps", str(args.n_steps),
        "--run_name", "phase1_random_policy",
    ]
    if args.policy_handoff:
        random_cmd += ["--policy_handoff", args.policy_handoff]
    rc1 = run(random_cmd)

    rc2 = 0
    if args.lobs5_ckpt_path and args.policy_ckpt_dir and args.policy_config:
        ippo_cmd = [
            sys.executable, script,
            "--world_model", "generative",
            "--policy_mode", "ippo_rnn",
            "--lobs5_ckpt_path", args.lobs5_ckpt_path,
            "--policy_ckpt_dir", args.policy_ckpt_dir,
            "--policy_config", args.policy_config,
            "--data_dir", args.data_dir,
            "--jaxmarl_root", args.jaxmarl_root,
            "--lobs5_root", args.lobs5_root,
            "--output_root", args.output_root,
            "--n_steps", str(args.n_steps),
            "--run_name", "phase1_ippo_rnn_policy",
        ]
        if args.policy_handoff:
            ippo_cmd += ["--policy_handoff", args.policy_handoff]
        rc2 = run(ippo_cmd)
    else:
        print("[LOBArena smoke] Skipping IPPO-RNN smoke run (missing checkpoint/config args).")

    return 0 if (rc1 == 0 and rc2 == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
