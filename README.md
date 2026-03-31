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
- `directional`: simple adversarial policy that alternates forced marketable buy/sell flow

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

### Multi-window evaluation (4 windows in parallel)

Runs 4 windows in 2026 (month/week × adversarial on/off) concurrently and writes aggregate stats (Mean/Median/IQM) for raw and risk-adjusted PnL.

Adversarial windows are regime/opponent conditions around the same evaluated policy. They do **not** replace the evaluated policy with an adversarial policy; reported PnL is computed for the evaluated agent only.

Lesson learned: earlier multi-window wiring accidentally swapped the evaluated policy in adversarial windows. This is fixed; adversarial mode now changes market/opponent conditions only.

```bash
python3 scripts/evaluate_checkpoint.py \
  --world_model historical \
  --policy_mode ippo_rnn \
  --policy_ckpt_dir /path/to/your/checkpoint_dir \
  --policy_config /path/to/your/config.yaml \
  --data_dir /path/to/test_data \
  --multi_window \
  --risk_weights pnl=1.0,drawdown=0.5,risk=0.1,inventory=0.0 \
  --run_name multi_window_eval
```

### Plot multi-window results separately

Generate plots later from an existing `multi_window_summary.json` (separate from scoring/evaluation):

```bash
python3 scripts/plot_multi_window.py \
  --summary_path outputs/evaluations/<run_name>/multi_window_summary.json
```

### GPU cluster runs (Slurm, single-node)

Run tests on one GPU node:

```bash
sbatch slurm/sbatch_gpu_tests.sh
```

Run parallel multi-window evaluation on one GPU node:

```bash
sbatch --export=ALL,DATA_DIR=/path/to/test_data,POLICY_MODE=random,RUN_NAME=phase1_gpu_parallel \
  slurm/sbatch_gpu_parallel_eval.sh
```

## Outputs

Each run writes:
- `outputs/evaluations/<run_name>/summary.json`
- `outputs/evaluations/<run_name>/step_trace.csv`

Multi-window runs additionally write:
- `outputs/evaluations/<run_name>/multi_window_summary.json`
- `outputs/evaluations/<run_name>/multi_window_scores.csv`
- `outputs/evaluations/<run_name>/plots/multi_window_scores_by_window.png` (after running `scripts/plot_multi_window.py`)
- `outputs/evaluations/<run_name>/plots/multi_window_aggregate_stats.png` (after running `scripts/plot_multi_window.py`)

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
