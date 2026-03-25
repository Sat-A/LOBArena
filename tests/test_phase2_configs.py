import json
from pathlib import Path


def test_phase1_default_config_exists_and_parseable():
    p = Path(__file__).resolve().parents[1] / 'config' / 'evaluation_configs' / 'phase1_default.json'
    assert p.exists()
    data = json.loads(p.read_text())
    assert 'world_model' in data and 'policy_mode' in data


def test_policy_handoff_schema_and_template_exist_and_parseable():
    base = Path(__file__).resolve().parents[1] / 'config' / 'evaluation_configs'
    schema = base / 'policy_handoff_schema.json'
    template = base / 'policy_handoff_template.json'
    assert schema.exists()
    assert template.exists()
    schema_data = json.loads(schema.read_text())
    template_data = json.loads(template.read_text())
    assert schema_data.get("title") == "LOBArena Policy Checkpoint Handoff"
    assert template_data.get("schema_version") == "1.0"
    assert "policy" in template_data and "provenance" in template_data
