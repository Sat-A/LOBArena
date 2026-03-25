import os
import shlex
from typing import Any, Iterable, Optional


class SingleNodeGuardError(RuntimeError):
    """Raised when phase2 entrypoints detect a multi-node execution context."""


_NODE_COUNT_ENV_VARS = (
    "SLURM_NNODES",
    "SLURM_JOB_NUM_NODES",
)

_NODE_COUNT_ARG_FIELDS = (
    "nnodes",
    "num_nodes",
    "nodes",
    "n_nodes",
)

_MULTI_NODE_BOOL_FIELDS = (
    "multi_node",
    "multi_nodes",
)

_NODE_COUNT_FLAGS = (
    "--nnodes",
    "--num_nodes",
    "--nodes",
    "--n_nodes",
)


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _extract_node_count_from_tokens(tokens: list[str]) -> Optional[int]:
    for idx, token in enumerate(tokens):
        for flag in _NODE_COUNT_FLAGS:
            if token == flag and idx + 1 < len(tokens):
                parsed = _parse_int(tokens[idx + 1])
                if parsed is not None:
                    return parsed
            if token.startswith(f"{flag}="):
                parsed = _parse_int(token.split("=", 1)[1])
                if parsed is not None:
                    return parsed
    return None


def enforce_single_node_context(
    *,
    context_name: str,
    args: Any = None,
    command_strings: Optional[Iterable[str]] = None,
) -> None:
    violations: list[str] = []

    for env_name in _NODE_COUNT_ENV_VARS:
        parsed = _parse_int(os.getenv(env_name))
        if parsed is not None and parsed > 1:
            violations.append(f"{env_name}={parsed}")

    if args is not None:
        for field in _NODE_COUNT_ARG_FIELDS:
            if hasattr(args, field):
                parsed = _parse_int(getattr(args, field))
                if parsed is not None and parsed > 1:
                    violations.append(f"{field}={parsed}")
        for field in _MULTI_NODE_BOOL_FIELDS:
            if hasattr(args, field) and bool(getattr(args, field)):
                violations.append(f"{field}=true")

    for command in command_strings or ():
        command = str(command or "").strip()
        if not command:
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        parsed = _extract_node_count_from_tokens(tokens)
        if parsed is not None and parsed > 1:
            violations.append(f"command:{parsed} ({command})")

    if violations:
        details = "; ".join(violations)
        raise SingleNodeGuardError(
            f"{context_name} supports single-node execution only. "
            f"Detected multi-node context: {details}"
        )
