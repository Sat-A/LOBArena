# LOBArena Technical Details

This document covers runtime behavior, contracts, and production operations.

## Repository map

- `evaluate/`
  - `pipeline.py`: single-run and batch evaluation orchestration
  - `policy_handoff.py`: handoff schema validation and normalization
  - `policy_adapter.py`: trained policy adapter loading
  - `checkpoint_loader.py`: checkpoint restore with CPU fallback behavior
  - `adversarial.py`: tournament/adversarial runner
  - `train_eval.py`: train-then-evaluate orchestration wrapper
  - `phase2_contract.py`: train/eval campaign contract validation
  - `single_node_guard.py`: single-node execution enforcement
- `leaderboard/`
  - `aggregator.py`: weighted ranking and CSV export
- `guardrails/`
  - `order_validators.py`: action sanitization and quote validity guards
- `metrics/`
  - `computation.py`: PnL, drawdown, risk, inventory, impact metrics
- `scripts/`
  - `evaluate_checkpoint.py`, `build_leaderboard.py`, `adversarial_eval_phase2.py`, `train_eval_phase2.py`, `run_phase1_smoke.py`

## Command reference

### Evaluate own trained checkpoint

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode ippo_rnn \
  --policy_ckpt_dir /path/to/your/checkpoint_dir \
  --policy_config /path/to/your/config.yaml \
  --data_dir /path/to/test_data \
  --run_name my_checkpoint_eval
```

### Evaluate fixed policy

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode fixed \
  --fixed_action 0 \
  --data_dir /path/to/test_data \
  --run_name fixed_policy_eval
```

### Evaluate random policy

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode random \
  --data_dir /path/to/test_data \
  --run_name random_policy_eval
```

### Evaluate generative world model with trained policy

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model generative \
  --lobs5_ckpt_path /path/to/lobs5_ckpt \
  --policy_mode ippo_rnn \
  --policy_ckpt_dir /path/to/your/checkpoint_dir \
  --policy_config /path/to/your/config.yaml \
  --data_dir /path/to/test_data \
  --run_name generative_checkpoint_eval
```

### Batch evaluate multiple policies

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_handoff_manifest /path/to/manifest.json \
  --data_dir /path/to/test_data \
  --run_name batch_eval
```

### Build leaderboard

```bash
python3 scripts/build_leaderboard.py \
  --glob 'outputs/evaluations/*/summary.json' \
  --output outputs/evaluations/leaderboard.json \
  --csv-output outputs/evaluations/leaderboard.csv
```

## Runtime flags

`evaluate_checkpoint.py` supports:
- `--cpu_safe`: conservative thread/runtime settings
- `--device auto|cpu|gpu`: backend control with GPU fail-fast validation
- `--allow_generative_fallback`: opt-in fallback to historical replay
- `--strict_generative`: explicit strict mode (generative mode is strict by default)

## Policy modes

- `ippo_rnn`: actions come from a trained checkpoint
- `fixed`: one constant action for all steps
- `random`: random action each step
- `lose_money`: adverse-action stress behavior

## Artifacts

Per run:
- `outputs/evaluations/<run_name>/summary.json`
- `outputs/evaluations/<run_name>/step_trace.csv`

Batch runs:
- `outputs/evaluations/<run_name>/batch_summary.json`

Adversarial runs:
- `outputs/evaluations/<run_name>/adversarial_summary.json`

Train/eval campaign runs:
- `outputs/evaluations/<run_name>/train_eval_summary.json`

## Safety behavior

- Invalid limit-add messages with non-positive prices are converted to no-op.
- Invalid quote states trigger rollback to prior valid state.
- Quote validity requires `best_bid > 0`, `best_ask > 0`, and `best_bid < best_ask`.
- Batch manifest relative paths are constrained to manifest directory scope.

## Leaderboard scoring

Default composite score in `leaderboard/aggregator.py`:

`score = pnl - drawdown_weight*abs(drawdown) - risk_weight*risk_std - inventory_weight*abs(inventory)`

Default weights:
- `pnl=1.0`
- `drawdown=0.5`
- `risk=0.1`
- `inventory=0.0`

## Cluster guidance

For CPU-constrained nodes:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=1 \
XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
```

Single-node guards are enforced in evaluation, train/eval, and adversarial entrypoints.
