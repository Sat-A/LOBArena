import json
from pathlib import Path

import jax.numpy as jnp

from LOBArena.evaluate import pipeline
from LOBArena.evaluate.policy_adapter import validate_policy_choice
from LOBArena.evaluate.policy_handoff import validate_policy_handoff_payload


def test_validate_policy_choice_accepts_lose_money():
    sel = validate_policy_choice("lose_money", fixed_action=0, policy_ckpt_dir=None, policy_config=None)
    assert sel.mode == "lose_money"


def test_force_marketable_lossy_orders_reprices_limit_adds():
    msgs = jnp.array(
        [
            [1, 1, 0, 100_000, -101, -101, 0, 0],   # buy limit add
            [1, -1, 5, 99_900, -101, -101, 0, 0],   # sell limit add
            [2, 1, 3, 100_100, -101, -101, 0, 0],   # not limit add
        ],
        dtype=jnp.int32,
    )
    out = pipeline._force_marketable_lossy_orders(msgs)
    assert int(out[0, 3]) == 1_000_000_000
    assert int(out[1, 3]) == 1
    assert int(out[0, 2]) >= 1
    assert int(out[2, 3]) == 100_100


def test_policy_handoff_accepts_lose_money_mode():
    fixture = Path(__file__).resolve().parent / "fixtures" / "policy_handoff_valid.json"
    payload = json.loads(fixture.read_text())
    payload["policy"]["mode"] = "lose_money"
    out = validate_policy_handoff_payload(payload, base_dir=fixture.parent)
    assert out["policy"]["mode"] == "lose_money"
