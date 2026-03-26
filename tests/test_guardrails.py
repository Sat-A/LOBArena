try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp

from LOBArena.guardrails.order_validators import book_quotes_valid, sanitize_action_messages


def test_sanitize_negative_price_limit_order_to_noop():
    msgs = jnp.array([
        [1, 1, 10, -5, -100, -101, 0, 0],
        [1, -1, 5, 100, -101, -101, 0, 1],
    ], dtype=jnp.int32)
    out = sanitize_action_messages(msgs)
    assert int(out[0, 0]) == 0
    assert int(out[1, 0]) == 1


def test_book_quotes_valid_rejects_crossed_or_zero_quotes():
    assert book_quotes_valid(100, 101) is True
    assert book_quotes_valid(0, 101) is False
    assert book_quotes_valid(100, 0) is False
    assert book_quotes_valid(101, 101) is False
    assert book_quotes_valid(102, 101) is False
