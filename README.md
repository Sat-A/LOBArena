# LOBArena

LOBArena is a sibling project to `JaxMARL-HFT` focused on checkpoint-based arena evaluation and comparison of trading agents.

## Phase 1

- Unified evaluation pipeline with world model choice:
  - `historical` replay
  - `generative` (requires LOBS5 checkpoint)
- Inputs: world model mode, trading policy mode/checkpoint, test dataset.
- Matching engine: JAX-LOB stack from `JaxMARL-HFT`.
- Guardrails: invalid-order sanitization and book-integrity rollback checks.
- Metrics: PnL, drawdown, risk proxy, inventory stats, impact proxy.
- Leaderboard: aggregate and rank runs by primary performance metrics.

## Repository structure

- `evaluate/` core execution logic (pipeline, world-model selection, policy adapter, checkpoint loading).
- `guardrails/` message/order safety checks.
- `metrics/` performance and risk metric computations.
- `leaderboard/` run aggregation and ranking.
- `scripts/` CLI entrypoints.
- `config/evaluation_configs/` default run configuration templates.
- `tests/` unit checks for guardrails and metrics.

## Quick usage

```bash
python3 LOBArena/scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode random \
  --data_dir /path/to/test_data \
  --run_name phase1_random_smoke

python3 LOBArena/scripts/evaluate_checkpoint.py \
  --world_model generative \
  --policy_mode ippo_rnn \
  --lobs5_ckpt_path /path/to/lobs5_ckpt \
  --policy_ckpt_dir /path/to/marl_ckpt \
  --policy_config /path/to/ippo_config.yaml \
  --data_dir /path/to/test_data \
  --run_name phase1_ippo_smoke

python3 LOBArena/scripts/build_leaderboard.py \
  --glob 'LOBArena/outputs/evaluations/*/summary.json' \
  --output LOBArena/outputs/evaluations/leaderboard.json
```

## Phase 2 workflows

```bash
python3 LOBArena/scripts/train_eval_phase2.py \
  --train_data_dir /path/to/train_data \
  --test_data_dir /path/to/test_data \
  --run_name phase2_train_eval

python3 LOBArena/scripts/adversarial_eval_phase2.py \
  --data_dir /path/to/test_data \
  --target_policy_mode random \
  --competitor_policy_mode fixed \
  --run_name phase2_adversarial
```

## Runtime notes

- Generative mode requires `--lobs5_ckpt_path` (and optional `--checkpoint_step`).
- `ippo_rnn` policy mode requires `--policy_ckpt_dir` and `--policy_config`.
- On CPU-limited nodes, set strict thread caps before runs:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export JAX_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
```

## Phase 2 (next)

- Train-on-train / eval-on-test workflow.
- Adversarial competitors (MM + directional agents) in arena evaluation.
