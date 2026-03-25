import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from LOBArena.evaluate.phase2_contract import (
    build_campaign_summary_payload,
    generate_policy_handoff_artifact,
)
from LOBArena.evaluate.policy_handoff import load_policy_handoff
from LOBArena.evaluate.single_node_guard import enforce_single_node_context


def parse_args():
    workspace_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="LOBArena Phase 2 train/eval orchestrator")
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--test_data_dir", required=True)
    p.add_argument("--jaxmarl_root", default=str(workspace_root / "JaxMARL-HFT"))
    p.add_argument("--output_root", default=str(workspace_root / "LOBArena" / "outputs" / "evaluations"))
    p.add_argument("--train_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--policy_ckpt_dir", default="")
    p.add_argument("--policy_config", default="")
    p.add_argument("--policy_handoff", default="")
    p.add_argument("--run_name", default="phase2_train_eval")
    p.add_argument("--fast_startup", action="store_true")
    p.add_argument(
        "--run_jaxmarl_train",
        action="store_true",
        help="Invoke optional JaxMARL training subprocess before evaluation.",
    )
    p.add_argument(
        "--jaxmarl_train_cmd",
        default="",
        help="Quoted command string for JaxMARL training subprocess.",
    )
    p.add_argument(
        "--jaxmarl_train_timeout_sec",
        type=int,
        default=1800,
        help="Timeout for JaxMARL training subprocess; <=0 disables timeout.",
    )
    return p.parse_args()


def _run_command(cmd, timeout_sec=0):
    print("[LOBArena phase2]", " ".join(cmd))
    t0 = time.time()
    timeout = timeout_sec if timeout_sec and timeout_sec > 0 else None
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "rc": int(completed.returncode),
            "timed_out": False,
            "duration_sec": float(time.time() - t0),
            "stdout_tail": (completed.stdout or "")[-4000:],
            "stderr_tail": (completed.stderr or "")[-4000:],
            "error": "",
        }
    except subprocess.TimeoutExpired as e:
        return {
            "rc": None,
            "timed_out": True,
            "duration_sec": float(time.time() - t0),
            "stdout_tail": ((e.stdout or "") if isinstance(e.stdout, str) else "")[-4000:],
            "stderr_tail": ((e.stderr or "") if isinstance(e.stderr, str) else "")[-4000:],
            "error": f"timeout after {timeout_sec} sec",
        }
    except Exception as e:
        return {
            "rc": None,
            "timed_out": False,
            "duration_sec": float(time.time() - t0),
            "stdout_tail": "",
            "stderr_tail": "",
            "error": repr(e),
        }


def _discover_checkpoint_candidates(ckpt_dir: Path):
    candidates = []
    for child in sorted(ckpt_dir.iterdir()):
        if child.is_dir() and child.name.isdigit() and (child / "state").is_dir():
            candidates.append(str(child.resolve()))
    patterns = (
        "checkpoint*",
        "*.ckpt",
        "*.pth",
        "*.pt",
        "*.msgpack",
        "*.safetensors",
        "*.orbax*",
    )
    for pat in patterns:
        for m in sorted(ckpt_dir.glob(pat)):
            candidates.append(str(m.resolve()))
    # Preserve order while de-duplicating.
    return list(dict.fromkeys(candidates))


def _validate_checkpoint_expectations(checkpoint_dir: str):
    ckpt = Path(checkpoint_dir).expanduser().resolve()
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Policy checkpoint dir not found: {ckpt}")
    candidates = _discover_checkpoint_candidates(ckpt)
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint artifacts discovered in "
            f"{ckpt}. Expected at least one of: <step>/state directory, "
            "checkpoint* files, *.ckpt, *.pth, *.pt, *.msgpack, *.safetensors, *.orbax*."
        )
    return candidates


@dataclass
class Phase2AlphaCampaignManager:
    args: object
    started_at: float = None

    def __post_init__(self):
        self.started_at = float(self.started_at if self.started_at is not None else time.time())
        self.run_dir = Path(self.args.output_root).expanduser().resolve() / self.args.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.run_dir / "train_eval_summary.json"
        self.lineage = {
            "train": {"invoked": False, "status": "skipped"},
            "handoff": {"status": "none"},
            "evaluation": {"status": "not_started"},
        }

    def _write_summary(
        self,
        *,
        eval_rc: int,
        policy_mode: str,
        policy_ckpt_dir: str,
        policy_config: str,
        input_policy_handoff: str,
        generated_policy_handoff: str,
    ):
        summary = build_campaign_summary_payload(
            run_name=self.args.run_name,
            train_data_dir=self.args.train_data_dir,
            test_data_dir=self.args.test_data_dir,
            train_steps=self.args.train_steps,
            eval_steps=self.args.eval_steps,
            eval_rc=eval_rc,
            policy_mode=policy_mode,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_config=policy_config,
            input_policy_handoff=input_policy_handoff,
            generated_policy_handoff=generated_policy_handoff,
            run_dir=self.run_dir,
            summary_path=self.summary_path,
            runtime_sec=float(time.time() - self.started_at),
            lineage=self.lineage,
        )
        self.summary_path.write_text(json.dumps(summary, indent=2))
        return summary

    def _run_optional_training(self):
        if not self.args.run_jaxmarl_train:
            return
        if not self.args.jaxmarl_train_cmd.strip():
            raise ValueError("When --run_jaxmarl_train is set, --jaxmarl_train_cmd is required.")
        enforce_single_node_context(
            context_name="phase2 train subprocess",
            args=self.args,
            command_strings=[self.args.jaxmarl_train_cmd],
        )

        train_cmd = shlex.split(self.args.jaxmarl_train_cmd)
        train_result = _run_command(train_cmd, timeout_sec=self.args.jaxmarl_train_timeout_sec)
        train_status = (
            "timed_out"
            if train_result["timed_out"]
            else ("success" if train_result["rc"] == 0 else "failed")
        )
        self.lineage["train"] = {
            "invoked": True,
            "status": train_status,
            "command": train_cmd,
            "timeout_sec": int(self.args.jaxmarl_train_timeout_sec),
            "rc": train_result["rc"],
            "duration_sec": train_result["duration_sec"],
            "stdout_tail": train_result["stdout_tail"],
            "stderr_tail": train_result["stderr_tail"],
            "error": train_result["error"],
        }
        if train_status == "success":
            return

        self._write_summary(
            eval_rc=124 if train_result["timed_out"] else 1,
            policy_mode="random",
            policy_ckpt_dir="",
            policy_config="",
            input_policy_handoff="",
            generated_policy_handoff="",
        )
        if train_result["timed_out"]:
            raise RuntimeError(
                f"JaxMARL training timed out after {self.args.jaxmarl_train_timeout_sec} sec. "
                f"Command: {' '.join(train_cmd)}"
            )
        raise RuntimeError(
            f"JaxMARL training failed with rc={train_result['rc']}. "
            f"Command: {' '.join(train_cmd)}"
        )

    def _resolve_policy_materialization(self):
        if self.args.policy_handoff and (self.args.policy_ckpt_dir or self.args.policy_config):
            raise ValueError(
                "Use either --policy_handoff or --policy_ckpt_dir/--policy_config, not both."
            )
        if bool(self.args.policy_ckpt_dir) ^ bool(self.args.policy_config):
            raise ValueError(
                "ippo_rnn setup requires both --policy_ckpt_dir and --policy_config."
            )

        policy_ckpt_dir = str(self.args.policy_ckpt_dir or "")
        policy_config = str(self.args.policy_config or "")
        input_policy_handoff = str(self.args.policy_handoff or "")
        if input_policy_handoff:
            handoff = load_policy_handoff(input_policy_handoff)
            input_policy_handoff = handoff["_artifact_path"]
            policy_ckpt_dir = handoff["policy"]["checkpoint_dir"]
            policy_config = handoff["policy"]["config_path"]

        policy_mode = "ippo_rnn" if input_policy_handoff or (policy_ckpt_dir and policy_config) else "random"
        discovered_ckpts = _validate_checkpoint_expectations(policy_ckpt_dir) if policy_mode == "ippo_rnn" else []
        self.lineage["handoff"] = {
            "status": (
                "input_handoff"
                if input_policy_handoff
                else ("direct_checkpoint" if policy_mode == "ippo_rnn" else "none")
            ),
            "input_handoff": input_policy_handoff,
            "generated_handoff": "",
            "policy_mode": policy_mode,
            "checkpoint_dir": policy_ckpt_dir,
            "config_path": policy_config,
            "checkpoint_candidates": discovered_ckpts[:10],
            "checkpoint_candidate_count": len(discovered_ckpts),
        }
        return policy_mode, policy_ckpt_dir, policy_config, input_policy_handoff

    def _build_eval_command(self, policy_mode, policy_ckpt_dir, policy_config, input_policy_handoff):
        eval_script = str(Path(__file__).resolve().parents[1] / "scripts" / "evaluate_checkpoint.py")
        cmd = [
            sys.executable,
            eval_script,
            "--world_model", "historical",
            "--policy_mode", policy_mode,
            "--data_dir", self.args.test_data_dir,
            "--n_steps", str(self.args.eval_steps),
            "--run_name", f"{self.args.run_name}_test_eval",
        ]
        if input_policy_handoff:
            cmd += ["--policy_handoff", input_policy_handoff]
        elif policy_mode == "ippo_rnn":
            cmd += ["--policy_ckpt_dir", policy_ckpt_dir, "--policy_config", policy_config]
        if self.args.fast_startup:
            cmd += ["--fast_startup"]
        return cmd

    def _run_evaluation(self, cmd):
        eval_result = _run_command(cmd)
        rc = 124 if eval_result["timed_out"] else int(eval_result["rc"] if eval_result["rc"] is not None else 1)
        self.lineage["evaluation"] = {
            "status": (
                "timed_out"
                if eval_result["timed_out"]
                else ("success" if rc == 0 else "failed")
            ),
            "command": cmd,
            "rc": rc,
            "duration_sec": eval_result["duration_sec"],
            "stdout_tail": eval_result["stdout_tail"],
            "stderr_tail": eval_result["stderr_tail"],
            "error": eval_result["error"],
        }
        return rc

    def _generate_handoff_if_needed(self, rc, policy_mode, policy_ckpt_dir, policy_config, input_policy_handoff):
        if not (rc == 0 and not input_policy_handoff and policy_mode == "ippo_rnn"):
            return ""
        generated = generate_policy_handoff_artifact(
            {
                "policy_mode": "ippo_rnn",
                "checkpoint_dir": policy_ckpt_dir,
                "config_path": policy_config,
                "model_index": 1,
                "restore_topology": {
                    "restore_strategy": "single_device_fallback",
                    "train_device_count": int(os.getenv("TRAIN_DEVICE_COUNT", "1")),
                    "eval_device_count": int(os.getenv("EVAL_DEVICE_COUNT", "1")),
                },
                "evaluation": {
                    "seed": 42,
                    "start_date": "",
                    "end_date": "",
                },
                "provenance": {
                    "run_id": self.args.run_name,
                    "git_commit": os.getenv("GIT_COMMIT", ""),
                },
                "output_path": str(self.run_dir / "policy_handoff.generated.json"),
            },
            base_dir=self.run_dir,
        )
        generated_handoff = generated["_artifact_path"]
        self.lineage["handoff"]["status"] = "generated_handoff"
        self.lineage["handoff"]["generated_handoff"] = generated_handoff
        return generated_handoff

    def run(self):
        enforce_single_node_context(context_name="phase2 train/eval orchestration", args=self.args)
        self._run_optional_training()
        policy_mode, policy_ckpt_dir, policy_config, input_policy_handoff = self._resolve_policy_materialization()
        cmd = self._build_eval_command(
            policy_mode=policy_mode,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_config=policy_config,
            input_policy_handoff=input_policy_handoff,
        )
        rc = self._run_evaluation(cmd)
        generated_handoff = self._generate_handoff_if_needed(
            rc=rc,
            policy_mode=policy_mode,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_config=policy_config,
            input_policy_handoff=input_policy_handoff,
        )
        self._write_summary(
            eval_rc=rc,
            policy_mode=policy_mode,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_config=policy_config,
            input_policy_handoff=input_policy_handoff,
            generated_policy_handoff=generated_handoff,
        )
        print(f"[LOBArena phase2] Summary written: {self.summary_path}")
        return 0 if rc == 0 else 1


def main():
    args = parse_args()
    enforce_single_node_context(context_name="phase2 train/eval entrypoint", args=args)
    return Phase2AlphaCampaignManager(args=args).run()


if __name__ == "__main__":
    raise SystemExit(main())
