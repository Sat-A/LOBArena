import json
from pathlib import Path


_ALLOWED_TOP_LEVEL = {"schema_version", "policy", "restore_topology", "evaluation", "provenance"}
_ALLOWED_POLICY = {"mode", "checkpoint_dir", "config_path", "model_index"}
_ALLOWED_TOPOLOGY = {"restore_strategy", "train_device_count", "eval_device_count"}
_ALLOWED_EVAL = {"seed", "date_window"}
_ALLOWED_DATE_WINDOW = {"start_date", "end_date"}
_ALLOWED_PROVENANCE = {"run_id", "git_commit"}


def _require_dict(name, value):
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be an object")
    return value


def _ensure_exact_keys(name, obj, allowed_keys):
    extra = sorted(set(obj.keys()) - set(allowed_keys))
    if extra:
        raise ValueError(f"'{name}' has unsupported fields: {extra}")


def _ensure_required_keys(name, obj, required_keys):
    missing = sorted(set(required_keys) - set(obj.keys()))
    if missing:
        raise ValueError(f"'{name}' is missing required fields: {missing}")


def _resolve_path(raw_path, base_dir: Path) -> Path:
    p = Path(str(raw_path)).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return p


def validate_policy_handoff_payload(payload, base_dir: Path):
    root = _require_dict("handoff", payload)
    _ensure_exact_keys("handoff", root, _ALLOWED_TOP_LEVEL)
    _ensure_required_keys("handoff", root, _ALLOWED_TOP_LEVEL)

    schema_version = root.get("schema_version")
    if schema_version != "1.0":
        raise ValueError(f"Unsupported schema_version '{schema_version}'. Expected '1.0'.")

    policy = _require_dict("policy", root.get("policy"))
    _ensure_exact_keys("policy", policy, _ALLOWED_POLICY)
    _ensure_required_keys("policy", policy, _ALLOWED_POLICY)
    mode = str(policy.get("mode", "")).strip().lower()
    if mode not in {"ippo_rnn", "random", "fixed", "lose_money"}:
        raise ValueError(f"Invalid policy.mode '{mode}'. Use ippo_rnn|random|fixed|lose_money.")
    model_index = int(policy.get("model_index"))
    if model_index < 0:
        raise ValueError("policy.model_index must be >= 0")

    checkpoint_dir = _resolve_path(policy.get("checkpoint_dir"), base_dir)
    config_path = _resolve_path(policy.get("config_path"), base_dir)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Policy checkpoint dir not found: {checkpoint_dir}")
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Policy config file not found: {config_path}")

    topology = _require_dict("restore_topology", root.get("restore_topology"))
    _ensure_exact_keys("restore_topology", topology, _ALLOWED_TOPOLOGY)
    _ensure_required_keys("restore_topology", topology, _ALLOWED_TOPOLOGY)
    strategy = str(topology.get("restore_strategy", "")).strip().lower()
    if strategy not in {"direct", "single_device_fallback"}:
        raise ValueError(
            "restore_topology.restore_strategy must be 'direct' or 'single_device_fallback'"
        )
    train_devices = int(topology.get("train_device_count"))
    eval_devices = int(topology.get("eval_device_count"))
    if train_devices < 1 or eval_devices < 1:
        raise ValueError("restore_topology device counts must be >= 1")

    evaluation = _require_dict("evaluation", root.get("evaluation"))
    _ensure_exact_keys("evaluation", evaluation, _ALLOWED_EVAL)
    _ensure_required_keys("evaluation", evaluation, _ALLOWED_EVAL)
    seed = int(evaluation.get("seed"))
    date_window = _require_dict("evaluation.date_window", evaluation.get("date_window"))
    _ensure_exact_keys("evaluation.date_window", date_window, _ALLOWED_DATE_WINDOW)
    _ensure_required_keys("evaluation.date_window", date_window, _ALLOWED_DATE_WINDOW)
    start_date = str(date_window.get("start_date", ""))
    end_date = str(date_window.get("end_date", ""))

    provenance = _require_dict("provenance", root.get("provenance"))
    _ensure_exact_keys("provenance", provenance, _ALLOWED_PROVENANCE)
    _ensure_required_keys("provenance", provenance, {"run_id"})
    run_id = str(provenance.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("provenance.run_id is required")
    git_commit = str(provenance.get("git_commit", ""))

    return {
        "schema_version": "1.0",
        "policy": {
            "mode": mode,
            "checkpoint_dir": str(checkpoint_dir),
            "config_path": str(config_path),
            "model_index": model_index,
        },
        "restore_topology": {
            "restore_strategy": strategy,
            "train_device_count": train_devices,
            "eval_device_count": eval_devices,
        },
        "evaluation": {
            "seed": seed,
            "date_window": {
                "start_date": start_date,
                "end_date": end_date,
            },
        },
        "provenance": {
            "run_id": run_id,
            "git_commit": git_commit,
        },
    }


def load_policy_handoff(path):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Policy handoff artifact not found: {p}")
    payload = json.loads(p.read_text())
    normalized = validate_policy_handoff_payload(payload, base_dir=p.parent)
    normalized["_artifact_path"] = str(p)
    return normalized
