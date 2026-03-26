# LOBArena Technical Details

This document describes architecture, runtime behavior, and production operations.

## Repository map

- `evaluate/`
  - `pipeline.py`: single-run and batch evaluation orchestration
  - `policy_handoff.py`: handoff schema validation/normalization
  - `policy_adapter.py`: trained policy adapter loading
  - `checkpoint_loader.py`: model checkpoint restore with CPU fallback
  - `adversarial.py`: tournament/adversarial runner
  - `train_eval.py`: train-then-evaluate orchestration wrapper
  - `phase2_contract.py`: train/eval campaign contract validation
  - `single_node_guard.py`: single-node enforcement for cluster safety
- `leaderboard/`
  - `aggregator.py`: ranking, split leaderboards, CSV export
- `guardrails/`
  - `order_validators.py`: action sanitization and quote validity guards
- `metrics/`
  - `computation.py`: PnL, drawdown, risk, inventory, and impact metrics
- `scripts/`
  - `evaluate_checkpoint.py`, `build_leaderboard.py`, `adversarial_eval_phase2.py`, `train_eval_phase2.py`, `run_phase1_smoke.py`
- `config/evaluation_configs/`
  - defaults, handoff schema/template, objective gates, competitor registry
- `tests/`
  - policy handoff, batch, leaderboard, adversarial, metrics, guardrails, runtime/args, checkpoint loader, smoke-path tests

## Runtime behavior

`evaluate_checkpoint.py` supports:
- `--cpu_safe`: conservative thread/runtime settings.
- `--device auto|cpu|gpu`: backend control with GPU fail-fast validation.
- `--allow_generative_fallback`: opt-in fallback to historical replay on generative errors.
- `--strict_generative`: explicit strict mode (also implied in generative mode by default).

### Safety semantics

- In generative world-model mode, generation errors fail the run by default.
- To continue on generation errors, set `--allow_generative_fallback`.
- Invalid limit-add messages with non-positive prices are converted to no-op.
- Invalid quotes trigger rollback to the previous valid state.
- Quote validity requires `best_bid > 0`, `best_ask > 0`, and `best_bid < best_ask`.
- Batch manifest relative paths are constrained to the manifest directory root.

## Core execution patterns

Single evaluation:

```bash
python3 scripts/evaluate_checkpoint.py   --world_model historical   --policy_mode random   --data_dir /path/to/test_data   --run_name eval_hist_random
```

Trained policy evaluation:

```bash
python3 scripts/evaluate_checkpoint.py   --world_model historical   --policy_mode ippo_rnn   --policy_ckpt_dir /path/to/policy_ckpt   --policy_config /path/to/config.yaml   --data_dir /path/to/test_data   --run_name eval_hist_ippo
```

Batch handoff evaluation:

```bash
python3 scripts/evaluate_checkpoint.py   --world_model historical   --policy_handoff_manifest /path/to/manifest.json   --data_dir /path/to/test_data   --run_name eval_batch
```

Adversarial/tournament evaluation:

```bash
python3 scripts/adversarial_eval_phase2.py   --data_dir /path/to/test_data   --target_policy_mode random   --competitor_policy_mode fixed   --run_name eval_tournament
```

Train/eval orchestration wrapper:

```bash
python3 scripts/train_eval_phase2.py   --train_data_dir /path/to/train_data   --test_data_dir /path/to/test_data   --run_name train_eval_campaign
```

## Artifacts and contracts

Per evaluation run:
- `outputs/evaluations/<run_name>/summary.json`
- `outputs/evaluations/<run_name>/step_trace.csv`

Batch runs:
- `outputs/evaluations/<run_name>/batch_summary.json`

Adversarial runs:
- `outputs/evaluations/<run_name>/adversarial_summary.json`

Train/eval campaign runs:
- `outputs/evaluations/<run_name>/train_eval_summary.json`

Contract/config references:
- `config/evaluation_configs/policy_handoff_schema.json`
- `config/evaluation_configs/phase2_alpha_contract.json`
- `config/evaluation_configs/champion_objective_gates.json`

## Leaderboard scoring

Default weighted composite in `leaderboard/aggregator.py`:

`score = pnl - drawdown_weight*abs(drawdown) - risk_weight*risk_std - inventory_weight*abs(inventory)`

Default weights:
- `pnl=1.0`
- `drawdown=0.5`
- `risk=0.1`
- `inventory=0.0`

Custom weights can be passed via `--weights` or `--weights-config`.

## Cluster guidance

On CPU-constrained nodes, cap thread settings:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=1 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
```

Single-node guardrails are enforced in evaluation, train/eval, and adversarial entrypoints.

## Troubleshooting

- `lobs5_ckpt_path` required error:
  - Use `--lobs5_ckpt_path` when `--world_model generative`.
- GPU backend mismatch:
  - Use `--device auto|cpu` or fix CUDA/JAX setup.
- `ippo_rnn` missing config/checkpoint:
  - Provide both `--policy_ckpt_dir` and `--policy_config`.
- Handoff validation errors:
  - Align payload to `policy_handoff_schema.json`.
