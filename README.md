# LOBArena

LOBArena is an evaluation harness for trading policies against limit-order-book world models.

This README is public-facing and focused on quick setup and how to run tests/evaluations.

For architecture, runtime internals, execution matrices, and troubleshooting details, see [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md).

## Quick setup

Use the `lobs5` conda environment (recommended for JAX/JaxMARL-HFT/LOBS5 compatibility), then install dependencies:

```bash
pip install -r requirements.txt
```

## Basic evaluation run (Phase 1)

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode random \
  --data_dir /path/to/test_data \
  --run_name phase1_smoke
```

Artifacts are written to `outputs/evaluations/<run_name>/` (including `summary.json` and `step_trace.csv`).

## Leaderboard build

```bash
python3 scripts/build_leaderboard.py \
  --glob 'outputs/evaluations/*/summary.json' \
  --output outputs/evaluations/leaderboard.json
```

## Smoke wrapper

```bash
python3 scripts/run_phase1_smoke.py --data_dir /path/to/test_data
```

## Phase 2 orchestration scripts

```bash
python3 scripts/train_eval_phase2.py \
  --train_data_dir /path/to/train \
  --test_data_dir /path/to/test \
  --run_name phase2_train_eval
```

```bash
python3 scripts/adversarial_eval_phase2.py \
  --data_dir /path/to/test \
  --target_policy_mode random \
  --competitor_policy_mode fixed \
  --run_name phase2_adversarial
```

## Running tests

Run full suite:

```bash
python -m pytest tests -v
```

Run one test file:

```bash
python -m pytest tests/test_guardrails.py -v
```

Run one test:

```bash
python -m pytest tests/test_metrics.py::test_build_phase1_metrics_shapes -v
```

## Notes

- `generative` world model requires `--lobs5_ckpt_path`.
- `ippo_rnn` policy mode requires `--policy_ckpt_dir` and `--policy_config`.
- If your host is CPU-constrained, see thread/JAX runtime settings in [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md).
