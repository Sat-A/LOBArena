from types import SimpleNamespace

import pytest

from LOBArena.evaluate.single_node_guard import enforce_single_node_context


def test_enforce_single_node_context_allows_local_defaults(monkeypatch):
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
    args = SimpleNamespace(nnodes=1, multi_node=False)
    enforce_single_node_context(context_name="phase2 local test", args=args)


def test_enforce_single_node_context_rejects_multi_node_env(monkeypatch):
    monkeypatch.setenv("SLURM_NNODES", "2")
    with pytest.raises(RuntimeError, match="single-node execution only"):
        enforce_single_node_context(context_name="phase2 local test")


def test_enforce_single_node_context_rejects_multi_node_args():
    args = SimpleNamespace(num_nodes=3)
    with pytest.raises(RuntimeError, match="num_nodes=3"):
        enforce_single_node_context(context_name="phase2 local test", args=args)


def test_enforce_single_node_context_rejects_multi_node_command_flag():
    cmd = "python train.py --nnodes 4 --seed 1"
    with pytest.raises(RuntimeError, match="Detected multi-node context"):
        enforce_single_node_context(
            context_name="phase2 train subprocess",
            command_strings=[cmd],
        )
