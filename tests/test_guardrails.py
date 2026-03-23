try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp

from LOBArena.guardrails.order_validators import sanitize_action_messages


def test_sanitize_negative_price_limit_order_to_noop():
    msgs = jnp.array([
        [1, 1, 10, -5, -100, -101, 0, 0],
        [1, -1, 5, 100, -101, -101, 0, 1],
    ], dtype=jnp.int32)
    out = sanitize_action_messages(msgs)
    assert int(out[0, 0]) == 0
    assert int(out[1, 0]) == 1
