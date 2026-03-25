import json
from pathlib import Path

from LOBArena.evaluate.policy_handoff import validate_policy_handoff_payload


PHASE2_ALPHA_CONTRACT_VERSION = "phase2-alpha/1.0"
PHASE2_ALPHA_CONTRACT_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "evaluation_configs" / "phase2_alpha_contract.json"
)

CAMPAIGN_SUMMARY_REQUIRED_FIELDS = {
    "contract_version",
    "run_name",
    "status",
    "train",
    "evaluation",
    "policy",
    "artifacts",
    "runtime_sec",
}
CAMPAIGN_SUMMARY_LEGACY_FIELDS = {
    "train_data_dir",
    "test_data_dir",
    "policy_mode",
    "policy_ckpt_dir",
    "policy_config",
    "policy_handoff",
    "train_steps",
    "eval_steps",
    "eval_rc",
}
CAMPAIGN_SUMMARY_OPTIONAL_FIELDS = {
    "lineage",
}
HANDOFF_GENERATION_REQUIRED_FIELDS = {
    "policy_mode",
    "checkpoint_dir",
    "config_path",
    "model_index",
    "restore_topology",
    "evaluation",
    "provenance",
    "output_path",
}
HANDOFF_GENERATION_RESTORE_REQUIRED_FIELDS = {
    "restore_strategy",
    "train_device_count",
    "eval_device_count",
}
HANDOFF_GENERATION_EVALUATION_REQUIRED_FIELDS = {
    "seed",
    "start_date",
    "end_date",
}
HANDOFF_GENERATION_PROVENANCE_REQUIRED_FIELDS = {
    "run_id",
    "git_commit",
}


def _require_dict(name, value):
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be an object")
    return value


def _require_non_empty_string(name, value) -> str:
    out = str(value or "").strip()
    if not out:
        raise ValueError(f"'{name}' must be a non-empty string")
    return out


def _resolve_path(raw_path, base_dir: Path) -> Path:
    p = Path(str(raw_path)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return p


def _normalize_lineage(value):
    lineage = value if isinstance(value, dict) else {}
    train = lineage.get("train", {}) if isinstance(lineage.get("train"), dict) else {}
    handoff = lineage.get("handoff", {}) if isinstance(lineage.get("handoff"), dict) else {}
    evaluation = lineage.get("evaluation", {}) if isinstance(lineage.get("evaluation"), dict) else {}
    return {
        "train": train,
        "handoff": handoff,
        "evaluation": evaluation,
    }


def load_phase2_alpha_contract_spec():
    payload = json.loads(PHASE2_ALPHA_CONTRACT_SPEC_PATH.read_text())
    version = str(payload.get("contract_version", ""))
    if version != PHASE2_ALPHA_CONTRACT_VERSION:
        raise ValueError(
            f"Contract spec version mismatch: '{version}' != '{PHASE2_ALPHA_CONTRACT_VERSION}'"
        )
    return payload


def validate_campaign_summary_payload(payload, base_dir: Path):
    root = _require_dict("campaign_summary", payload)
    allowed = CAMPAIGN_SUMMARY_REQUIRED_FIELDS | CAMPAIGN_SUMMARY_LEGACY_FIELDS | CAMPAIGN_SUMMARY_OPTIONAL_FIELDS
    missing = sorted(CAMPAIGN_SUMMARY_REQUIRED_FIELDS - set(root.keys()))
    if missing:
        raise ValueError(f"campaign_summary missing required fields: {missing}")
    extra = sorted(set(root.keys()) - allowed)
    if extra:
        raise ValueError(f"campaign_summary has unsupported fields: {extra}")

    contract_version = _require_non_empty_string("contract_version", root.get("contract_version"))
    if contract_version != PHASE2_ALPHA_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported contract_version '{contract_version}'. "
            f"Expected '{PHASE2_ALPHA_CONTRACT_VERSION}'."
        )
    run_name = _require_non_empty_string("run_name", root.get("run_name"))
    status = str(root.get("status", "")).strip().lower()
    if status not in {"success", "failed"}:
        raise ValueError("status must be 'success' or 'failed'")

    train = _require_dict("train", root.get("train"))
    train_data_dir = str(_resolve_path(_require_non_empty_string("train.data_dir", train.get("data_dir")), base_dir))
    train_steps = int(train.get("steps"))

    evaluation = _require_dict("evaluation", root.get("evaluation"))
    test_data_dir = str(
        _resolve_path(_require_non_empty_string("evaluation.data_dir", evaluation.get("data_dir")), base_dir)
    )
    eval_steps = int(evaluation.get("steps"))
    eval_rc = int(evaluation.get("eval_rc"))
    eval_run_name = _require_non_empty_string("evaluation.run_name", evaluation.get("run_name"))

    policy = _require_dict("policy", root.get("policy"))
    policy_mode = _require_non_empty_string("policy.mode", policy.get("mode")).lower()
    policy_ckpt_dir_raw = str(policy.get("checkpoint_dir", "")).strip()
    policy_config_raw = str(policy.get("config_path", "")).strip()
    input_handoff_raw = str(policy.get("input_handoff", "")).strip()
    generated_handoff_raw = str(policy.get("generated_handoff", "")).strip()

    policy_ckpt_dir = str(_resolve_path(policy_ckpt_dir_raw, base_dir)) if policy_ckpt_dir_raw else ""
    policy_config = str(_resolve_path(policy_config_raw, base_dir)) if policy_config_raw else ""
    input_handoff = str(_resolve_path(input_handoff_raw, base_dir)) if input_handoff_raw else ""
    generated_handoff = str(_resolve_path(generated_handoff_raw, base_dir)) if generated_handoff_raw else ""
    if policy_mode == "ippo_rnn" and not (generated_handoff or input_handoff or (policy_ckpt_dir and policy_config)):
        raise ValueError(
            "policy.mode='ippo_rnn' requires a handoff path or both checkpoint_dir and config_path"
        )

    artifacts = _require_dict("artifacts", root.get("artifacts"))
    run_dir = str(_resolve_path(_require_non_empty_string("artifacts.run_dir", artifacts.get("run_dir")), base_dir))
    summary_path = str(
        _resolve_path(_require_non_empty_string("artifacts.summary_path", artifacts.get("summary_path")), base_dir)
    )

    runtime_sec = float(root.get("runtime_sec"))
    if runtime_sec < 0:
        raise ValueError("runtime_sec must be >= 0")
    lineage = _normalize_lineage(root.get("lineage"))

    return {
        "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
        "run_name": run_name,
        "status": status,
        "train": {
            "data_dir": train_data_dir,
            "steps": train_steps,
        },
        "evaluation": {
            "data_dir": test_data_dir,
            "steps": eval_steps,
            "eval_rc": eval_rc,
            "run_name": eval_run_name,
        },
        "policy": {
            "mode": policy_mode,
            "checkpoint_dir": policy_ckpt_dir,
            "config_path": policy_config,
            "input_handoff": input_handoff,
            "generated_handoff": generated_handoff,
        },
        "artifacts": {
            "run_dir": run_dir,
            "summary_path": summary_path,
        },
        "runtime_sec": runtime_sec,
        "lineage": lineage,
        "train_data_dir": train_data_dir,
        "test_data_dir": test_data_dir,
        "policy_mode": policy_mode,
        "policy_ckpt_dir": policy_ckpt_dir,
        "policy_config": policy_config,
        "policy_handoff": generated_handoff or input_handoff,
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "eval_rc": eval_rc,
    }


def build_campaign_summary_payload(
    *,
    run_name: str,
    train_data_dir: str,
    test_data_dir: str,
    train_steps: int,
    eval_steps: int,
    eval_rc: int,
    policy_mode: str,
    policy_ckpt_dir: str,
    policy_config: str,
    input_policy_handoff: str,
    generated_policy_handoff: str,
    run_dir: Path,
    summary_path: Path,
    runtime_sec: float,
    lineage=None,
):
    payload = {
        "contract_version": PHASE2_ALPHA_CONTRACT_VERSION,
        "run_name": str(run_name),
        "status": "success" if int(eval_rc) == 0 else "failed",
        "train": {
            "data_dir": str(Path(train_data_dir).expanduser().resolve()),
            "steps": int(train_steps),
        },
        "evaluation": {
            "data_dir": str(Path(test_data_dir).expanduser().resolve()),
            "steps": int(eval_steps),
            "eval_rc": int(eval_rc),
            "run_name": f"{run_name}_test_eval",
        },
        "policy": {
            "mode": str(policy_mode),
            "checkpoint_dir": str(policy_ckpt_dir or ""),
            "config_path": str(policy_config or ""),
            "input_handoff": str(input_policy_handoff or ""),
            "generated_handoff": str(generated_policy_handoff or ""),
        },
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "summary_path": str(summary_path.resolve()),
        },
        "runtime_sec": float(runtime_sec),
        "lineage": _normalize_lineage(lineage),
    }
    return validate_campaign_summary_payload(payload, base_dir=run_dir)


def generate_policy_handoff_artifact(payload, base_dir: Path):
    root = _require_dict("handoff_generation", payload)
    missing = sorted(HANDOFF_GENERATION_REQUIRED_FIELDS - set(root.keys()))
    if missing:
        raise ValueError(f"handoff_generation missing required fields: {missing}")
    extra = sorted(set(root.keys()) - HANDOFF_GENERATION_REQUIRED_FIELDS)
    if extra:
        raise ValueError(f"handoff_generation has unsupported fields: {extra}")

    mode = _require_non_empty_string("policy_mode", root.get("policy_mode")).lower()
    if mode != "ippo_rnn":
        raise ValueError("handoff generation only supports policy_mode='ippo_rnn'")

    checkpoint_dir = _resolve_path(root.get("checkpoint_dir"), base_dir)
    config_path = _resolve_path(root.get("config_path"), base_dir)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Policy checkpoint dir not found: {checkpoint_dir}")
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Policy config file not found: {config_path}")

    model_index = int(root.get("model_index"))
    if model_index < 0:
        raise ValueError("model_index must be >= 0")

    restore_topology = _require_dict("restore_topology", root.get("restore_topology"))
    restore_missing = sorted(HANDOFF_GENERATION_RESTORE_REQUIRED_FIELDS - set(restore_topology.keys()))
    if restore_missing:
        raise ValueError(f"restore_topology missing required fields: {restore_missing}")
    evaluation = _require_dict("evaluation", root.get("evaluation"))
    eval_missing = sorted(HANDOFF_GENERATION_EVALUATION_REQUIRED_FIELDS - set(evaluation.keys()))
    if eval_missing:
        raise ValueError(f"evaluation missing required fields: {eval_missing}")
    provenance = _require_dict("provenance", root.get("provenance"))
    provenance_missing = sorted(HANDOFF_GENERATION_PROVENANCE_REQUIRED_FIELDS - set(provenance.keys()))
    if provenance_missing:
        raise ValueError(f"provenance missing required fields: {provenance_missing}")

    handoff_payload = {
        "schema_version": "1.0",
        "policy": {
            "mode": mode,
            "checkpoint_dir": str(checkpoint_dir),
            "config_path": str(config_path),
            "model_index": model_index,
        },
        "restore_topology": {
            "restore_strategy": str(restore_topology.get("restore_strategy", "single_device_fallback")),
            "train_device_count": int(restore_topology.get("train_device_count", 1)),
            "eval_device_count": int(restore_topology.get("eval_device_count", 1)),
        },
        "evaluation": {
            "seed": int(evaluation.get("seed", 42)),
            "date_window": {
                "start_date": str(evaluation.get("start_date", "")),
                "end_date": str(evaluation.get("end_date", "")),
            },
        },
        "provenance": {
            "run_id": _require_non_empty_string("provenance.run_id", provenance.get("run_id")),
            "git_commit": str(provenance.get("git_commit", "")),
        },
    }

    output_path = _resolve_path(root.get("output_path"), base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = validate_policy_handoff_payload(handoff_payload, base_dir=output_path.parent)
    output_path.write_text(json.dumps(normalized, indent=2))
    normalized["_artifact_path"] = str(output_path)
    return normalized
