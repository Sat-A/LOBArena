# LOBArena

LOBArena is a production-oriented evaluation harness for trading policies on limit-order-book environments.

It helps teams do three things:
- run reproducible checkpoint evaluations,
- compare agents with consistent scoring,
- generate leaderboard artifacts for decision-making.

For implementation details and operational internals, see [`TECHNICAL_DETAILS.md`](TECHNICAL_DETAILS.md).

## Quick start

Use the `lobs5` conda environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run a baseline evaluation:

```bash
python3 scripts/evaluate_checkpoint.py   --world_model historical   --policy_mode random   --data_dir /path/to/test_data   --run_name smoke_eval
```

Outputs are written to `outputs/evaluations/<run_name>/`.

## What `policy_mode` means

`policy_mode` selects how the trading action is chosen at each step:
- `random`: samples a random action.
- `fixed`: always uses `--fixed_action`.
- `ippo_rnn`: loads a trained policy checkpoint (`--policy_ckpt_dir` + `--policy_config`).
- `lose_money`: stress-test policy that intentionally chooses adverse actions.

## Common workflows

Single checkpoint evaluation:

```bash
python3 scripts/evaluate_checkpoint.py   --world_model historical   --policy_mode ippo_rnn   --policy_ckpt_dir /path/to/policy_ckpt   --policy_config /path/to/config.yaml   --data_dir /path/to/test_data   --run_name eval_trained_policy
```

Batch evaluation from handoff manifest:

```bash
python3 scripts/evaluate_checkpoint.py   --world_model historical   --policy_handoff_manifest /path/to/manifest.json   --data_dir /path/to/test_data   --run_name eval_batch
```

Adversarial/tournament comparison:

```bash
python3 scripts/adversarial_eval_phase2.py   --data_dir /path/to/test_data   --target_policy_mode ippo_rnn   --target_policy_ckpt /path/to/target_ckpt   --target_policy_config /path/to/target_config.yaml   --competitor_policy_mode fixed   --competitor_fixed_action 0   --run_name adversarial_eval
```

Leaderboard build:

```bash
python3 scripts/build_leaderboard.py   --glob 'outputs/evaluations/*/summary.json'   --output outputs/evaluations/leaderboard.json   --csv-output outputs/evaluations/leaderboard.csv
```

## Reliability and safety defaults

- Generative evaluation now fails fast by default on generation errors.
- Historical fallback is opt-in with `--allow_generative_fallback`.
- Batch manifest relative paths are constrained to the manifest directory.
- Quote validation enforces `best_bid < best_ask`.

## Tests

Run focused regression tests:

```bash
python -m pytest -q tests/test_policy_handoff.py tests/test_batch_evaluation.py
```

Run full suite:

```bash
python -m pytest tests -v
```
