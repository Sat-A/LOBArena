# LOBArena Technical Details

This document contains implementation-level details, architecture notes, and operational guidance.

For setup and common commands, use the public README: [`README.md`](README.md).

## Repository map

- `evaluate/`
  - `pipeline.py`: Phase 1 single-run and batch evaluation orchestrator
  - `policy_handoff.py`: schema-enforced policy artifact loading
  - `policy_adapter.py`: IPPO-RNN adapter integration
  - `checkpoint_loader.py`: checkpoint restore + CPU fallback
  - `adversarial.py`: adversarial/arena runner (supports direct handoff artifacts and phase2 campaign-summary handoff references)
  - `train_eval.py`: Phase2-alpha orchestration with optional JaxMARL subprocess training, handoff generation, and lineage summaries
  - `phase2_contract.py`: phase2-alpha contract loader/validator and handoff generation helpers
- `leaderboard/`
  - `aggregator.py`: weighted ranking + split leaderboards + CSV export
- `guardrails/`
  - `order_validators.py`: message sanitization and quote validity checks
- `metrics/`
  - `computation.py`: PnL, drawdown, risk proxy, inventory, impact proxy
- `scripts/`
  - `evaluate_checkpoint.py`, `build_leaderboard.py`, `adversarial_eval_phase2.py`, `train_eval_phase2.py`, `run_phase1_smoke.py`
- `config/evaluation_configs/`
  - phase defaults, handoff schema/template, phase2-alpha orchestration contract, champion objective config, adversarial competitor registry
- `tests/`
  - policy handoff, batch, leaderboard, adversarial, metrics, guardrails, runtime/args, checkpoint loader, script smoke paths

## Runtime and production behavior

`scripts/evaluate_checkpoint.py` supports:

- `--cpu_safe`: applies conservative thread/runtime environment defaults.
- `--device auto|cpu|gpu`:
  - `cpu` enforces CPU backend.
  - `gpu` validates JAX backend is GPU and fails early otherwise.
- `--strict_generative`: on generative inference error, fails immediately (no silent fallback to historical messages).

Example:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode random \
  --data_dir /path/to/test_data \
  --cpu_safe \
  --device cpu \
  --run_name prod_safe_hist_random
```

## Execution path matrix

### A) Phase 1 single evaluation (`evaluate_checkpoint.py`)

Historical + random:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode random \
  --data_dir /path/to/test_data \
  --run_name phase1_hist_random
```

Historical + fixed:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode fixed \
  --fixed_action 0 \
  --data_dir /path/to/test_data \
  --run_name phase1_hist_fixed
```

Historical + lose_money:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode lose_money \
  --data_dir /path/to/test_data \
  --run_name phase1_hist_loss
```

Historical + ippo_rnn:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode ippo_rnn \
  --policy_ckpt_dir /path/to/policy_ckpt \
  --policy_config /path/to/config.yaml \
  --data_dir /path/to/test_data \
  --run_name phase1_hist_ippo
```

Generative + ippo_rnn:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model generative \
  --policy_mode ippo_rnn \
  --lobs5_ckpt_path /path/to/lobs5_ckpt \
  --policy_ckpt_dir /path/to/policy_ckpt \
  --policy_config /path/to/config.yaml \
  --data_dir /path/to/test_data \
  --run_name phase1_gen_ippo
```

### B) Batch/handoff evaluation

Single handoff:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_handoff /path/to/handoff.json \
  --data_dir /path/to/test_data \
  --run_name phase1_handoff_single
```

Batch list:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_handoff_batch /path/a.json /path/b.json \
  --data_dir /path/to/test_data \
  --run_name phase1_handoff_batch
```

Manifest batch:

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_handoff_manifest /path/to/manifest.json \
  --data_dir /path/to/test_data \
  --run_name phase1_handoff_manifest
```

### C) Leaderboard

```bash
python3 scripts/build_leaderboard.py \
  --glob 'outputs/evaluations/*/summary.json' \
  --output outputs/evaluations/leaderboard.json \
  --weights-config config/evaluation_configs/champion_objective_gates.json \
  --csv-output outputs/evaluations/leaderboard.csv
```

### D) Adversarial path (Phase 2 orchestration)

```bash
python3 scripts/adversarial_eval_phase2.py \
  --data_dir /path/to/test_data \
  --target_policy_mode random \
  --competitor_policy_mode fixed \
  --run_name phase2_adversarial
```

### E) Train/eval path (Phase 2α orchestration wrapper)

```bash
python3 scripts/train_eval_phase2.py \
  --train_data_dir /path/to/train_data \
  --test_data_dir /path/to/test_data \
  --run_name phase2_train_eval
```

Optional training subprocess path:

```bash
python3 scripts/train_eval_phase2.py \
  --train_data_dir /path/to/train_data \
  --test_data_dir /path/to/test_data \
  --run_name phase2_train_eval_with_train \
  --run_jaxmarl_train \
  --jaxmarl_train_cmd "python3 /path/to/JaxMARL-HFT/run_gen_worldmodel_pg_train.py --policy_arch ippo_rnn --n_envs 1 --n_steps 2 --n_updates 1 --gpu_id 0" \
  --jaxmarl_train_timeout_sec 1800
```

## Outputs and contracts

Per-run artifacts:

- `outputs/evaluations/<run_name>/summary.json`
- `outputs/evaluations/<run_name>/step_trace.csv`

Includes run metadata, metrics, action histogram, and traces. Runtime metadata and generation fallback metadata are also captured.

Batch runs emit `batch_summary.json` with candidate rows and baseline rank.

Adversarial runs emit `adversarial_summary.json` with participants, matches, and aggregate statistics.

Phase 2 train→handoff→eval uses `config/evaluation_configs/phase2_alpha_contract.json` as the source-of-truth contract, implemented by `evaluate/phase2_contract.py`.

The generated `train_eval_summary.json` now includes structured lineage:
- `lineage.train` (invoked/status/command/timeout/rc/stdout/stderr tail)
- `lineage.handoff` (input/generated handoff and checkpoint candidate metadata)
- `lineage.evaluation` (status/rc/command + output tails)

## Guardrails behavior

- Invalid limit-add orders with non-positive prices are converted to a no-op message type.
- Invalid quotes after action/world step trigger rollback to prior valid state in the pipeline loop.

## CPU-constrained host guidance

Set thread caps to avoid JAX thread-creation failures:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=1 \
XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
```

## Troubleshooting

- `AssertionError` in dataset loader:
  - Cause: `--data_dir` does not contain expected LOB message files.
  - Fix: provide a valid test dataset directory.

- `Requested --device gpu but JAX backend is 'cpu'`:
  - Cause: GPU backend unavailable.
  - Fix: correct CUDA/JAX environment or use `--device auto|cpu`.

- `ippo_rnn policy mode requires --policy_ckpt_dir and --policy_config`:
  - Cause: missing policy inputs.
  - Fix: provide both fields or use a non-IPPO mode.

- Policy handoff validation errors:
  - Cause: schema strict checks (exact keys, required fields, file existence).
  - Fix: align artifact with `config/evaluation_configs/policy_handoff_schema.json`.

## Phase2 single-node execution results (latest)

- One-node guardrails are enforced in phase2 entrypoints/helpers (`evaluate/single_node_guard.py`) and validated by tests.
- Curated adversary candidates from JaxMARL outputs were materialized into:
  - `config/evaluation_configs/phase2_adversary_candidates.json`
  - `config/evaluation_configs/phase2_adversary_handoffs_manifest.json`
  - `config/evaluation_configs/phase2_handoff_candidate_001.json` ... `005.json`
- Adversarial registry now includes `phase2_candidate_001..005` handoff-backed entries.
- Refreshed leaderboard artifacts:
  - `outputs/evaluations/phase2_refreshed_leaderboard.json`
  - `outputs/evaluations/phase2_refreshed_leaderboard.csv`
  - `outputs/evaluations/phase2_top3_snapshot.json`
- Profit-seeking experiment run:
  - `outputs/evaluations/phase2_profit_fixed4_cpu_hist/summary.json`
  - Provenance command: `outputs/evaluations/phase2_profit_fixed4_cpu_hist/provenance_command.txt`
  - Outcome: positive PnL not achieved in this environment (total_pnl remained 0.0).

## Trained-checkpoint pipeline timing (current measured baseline)

Timing was computed from trained-checkpoint summaries in:
`JaxMARL-HFT/outputs/gen_worldmodel_pg_train/*/summary.json`
using `timing_breakdown.total_runtime_sec`.

- Sample size: `37` runs
- Average end-to-end runtime: `286.922s` (~4m47s)
- Median runtime: `276.007s` (~4m36s)
- Min/Max runtime: `259.689s` / `393.247s`

Recent trained/aggressive subset (`train_best_*` + `aggr_profit_*`) average runtime:
- `306.175s` (~5m06s)

Latest aggressive run:
- `aggr_profit_tiny_cpu`: `307.073s`, with non-zero trade incidence but negative average PnL.
