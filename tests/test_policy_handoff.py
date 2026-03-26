import json
from pathlib import Path

import pytest

from LOBArena.evaluate.policy_handoff import load_policy_handoff, validate_policy_handoff_payload


def test_policy_handoff_fixture_loads_and_normalizes_paths():
    fixture = Path(__file__).resolve().parent / "fixtures" / "policy_handoff_valid.json"
    handoff = load_policy_handoff(fixture)
    assert handoff["schema_version"] == "1.0"
    assert handoff["policy"]["mode"] == "ippo_rnn"
    assert Path(handoff["policy"]["checkpoint_dir"]).is_dir()
    assert Path(handoff["policy"]["config_path"]).exists()
    assert handoff["policy"]["model_index"] == 1
    assert handoff["evaluation"]["seed"] == 123
    assert handoff["provenance"]["run_id"] == "unit-test-run"
    assert handoff["_artifact_path"].endswith("policy_handoff_valid.json")


def test_policy_handoff_rejects_missing_required_top_level_field():
    fixture = Path(__file__).resolve().parent / "fixtures" / "policy_handoff_valid.json"
    payload = json.loads(fixture.read_text())
    payload.pop("provenance")
    with pytest.raises(ValueError, match="provenance"):
        validate_policy_handoff_payload(payload, base_dir=fixture.parent)


def test_policy_handoff_rejects_unsupported_extra_fields():
    fixture = Path(__file__).resolve().parent / "fixtures" / "policy_handoff_valid.json"
    payload = json.loads(fixture.read_text())
    payload["extra"] = "not-allowed"
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_policy_handoff_payload(payload, base_dir=fixture.parent)


def test_policy_handoff_rejects_invalid_mode():
    fixture = Path(__file__).resolve().parent / "fixtures" / "policy_handoff_valid.json"
    payload = json.loads(fixture.read_text())
    payload["policy"]["mode"] = "bad_mode"
    with pytest.raises(ValueError, match="Invalid policy.mode"):
        validate_policy_handoff_payload(payload, base_dir=fixture.parent)


def test_policy_handoff_accepts_loss_mode():
    fixture = Path(__file__).resolve().parent / "fixtures" / "policy_handoff_valid.json"
    payload = json.loads(fixture.read_text())
    payload["policy"]["mode"] = "lose_money"
    payload["policy"]["checkpoint_dir"] = ""
    payload["policy"]["config_path"] = ""
    out = validate_policy_handoff_payload(payload, base_dir=fixture.parent)
    assert out["policy"]["mode"] == "lose_money"
    assert out["policy"]["checkpoint_dir"] == ""
    assert out["policy"]["config_path"] == ""


def test_policy_handoff_rejects_missing_paths():
    fixture = Path(__file__).resolve().parent / "fixtures" / "policy_handoff_valid.json"
    payload = json.loads(fixture.read_text())
    payload["policy"]["checkpoint_dir"] = "./does-not-exist"
    with pytest.raises(FileNotFoundError, match="checkpoint dir"):
        validate_policy_handoff_payload(payload, base_dir=fixture.parent)
