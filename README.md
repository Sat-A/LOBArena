# LOBArena

LOBArena is a production-oriented evaluation harness for trading policies in limit-order-book environments.

It is designed for three practical tasks:
- evaluate a single trained checkpoint,
- compare many checkpoints consistently,
- generate leaderboard artifacts for model selection.

For implementation internals, see [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md).

## Quick setup

Use the `lobs5` environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Most important commands

### 1) Evaluate your own trained checkpoint (IPPO-RNN)

Use this when you want policy actions to come from your trained model weights.

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode ippo_rnn \
  --policy_ckpt_dir /path/to/your/checkpoint_dir \
  --policy_config /path/to/your/config.yaml \
  --data_dir /path/to/test_data \
  --run_name my_checkpoint_eval
```

### 2) Evaluate a fixed baseline policy

Use this to benchmark against a constant action policy.

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode fixed \
  --fixed_action 0 \
  --data_dir /path/to/test_data \
  --run_name fixed_policy_eval
```

### 3) Evaluate a random baseline policy

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode random \
  --data_dir /path/to/test_data \
  --run_name random_policy_eval
```

## What `policy_mode` means

`policy_mode` defines where the action comes from at each step:
- `ippo_rnn`: your trained checkpoint (`--policy_ckpt_dir` + `--policy_config`)
- `fixed`: one constant action (`--fixed_action`)
- `random`: random action sampling
- `lose_money`: stress-test policy that intentionally takes adverse actions

## Compare multiple checkpoints

### Batch via handoff manifest

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

## Outputs

Each run writes:
- `outputs/evaluations/<run_name>/summary.json`
- `outputs/evaluations/<run_name>/step_trace.csv`

## Reliability defaults

- Generative mode is fail-fast by default.
- Historical fallback for generation errors is opt-in via `--allow_generative_fallback`.
- Batch manifest relative paths are constrained to the manifest directory.
- Quote validation enforces `best_bid < best_ask`.

## Tests

Focused suites:

```bash
python -m pytest -q tests/test_policy_handoff.py tests/test_batch_evaluation.py
```

Full suite:

```bash
python -m pytest tests -v
```
