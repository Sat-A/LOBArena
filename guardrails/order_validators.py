try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - fallback for non-JAX environments
    import numpy as jnp


def sanitize_action_messages(msgs: jnp.ndarray) -> jnp.ndarray:
    """Nullify invalid limit-add orders (price <= 0) as no-ops."""
    if msgs.size == 0:
        return msgs
    types = msgs[:, 0]
    prices = msgs[:, 3]
    nullify = (types == 1) & (prices <= 0)
    updated_types = jnp.where(nullify, jnp.zeros_like(types), types)
    if hasattr(msgs, "at"):  # JAX array update path
        return msgs.at[:, 0].set(updated_types)
    out = msgs.copy()  # NumPy fallback path
    out[:, 0] = updated_types
    return out


def book_quotes_valid(best_bid: int, best_ask: int) -> bool:
    return best_bid > 0 and best_ask > 0
