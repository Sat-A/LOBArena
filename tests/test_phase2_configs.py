import json
from pathlib import Path


def test_phase1_default_config_exists_and_parseable():
    p = Path('/home/s5e/satyamaga.s5e/LOBArena/config/evaluation_configs/phase1_default.json')
    assert p.exists()
    data = json.loads(p.read_text())
    assert 'world_model' in data and 'policy_mode' in data
