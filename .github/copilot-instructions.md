# Copilot Instructions

## Build, test, and lint commands

Use the `lobs5` conda environment for LOBArena runtime/tests because this project depends on JAX/JaxMARL-HFT/LOBS5.

Install dependencies:

```bash
pip install -r requirements.txt
```

Phase 1 evaluation entrypoint:

```bash
python3 scripts/evaluate_checkpoint.py --world_model historical --policy_mode random --data_dir /path/to/test_data --run_name phase1_smoke
```

Leaderboard aggregation:

```bash
python3 scripts/build_leaderboard.py --glob 'outputs/evaluations/*/summary.json' --output outputs/evaluations/leaderboard.json
```

Smoke workflow wrapper:

```bash
python3 scripts/run_phase1_smoke.py --data_dir /path/to/test_data
```

Phase 2 workflow scripts:

```bash
python3 scripts/train_eval_phase2.py --train_data_dir /path/to/train --test_data_dir /path/to/test --run_name phase2_train_eval
python3 scripts/adversarial_eval_phase2.py --data_dir /path/to/test --target_policy_mode random --competitor_policy_mode fixed --run_name phase2_adversarial
```

Run full test suite:

```bash
python -m pytest tests -v
```

Run a single test file:

```bash
python -m pytest tests/test_guardrails.py -v
```

Run a single test:

```bash
python -m pytest tests/test_metrics.py::test_build_phase1_metrics_shapes -v
```

Linting:
- No repository-wide lint command/config is defined in this repo.

## High-level architecture

LOBArena is an evaluation harness around external model/simulator stacks:

- `scripts/evaluate_checkpoint.py` calls `LOBArena.evaluate.pipeline.main`, the core Phase 1 runner.
- `evaluate/pipeline.py` orchestrates:
  - world model selection (`historical` replay vs `generative` LOBS5 checkpoint),
  - policy selection (`random`, `fixed`, `ippo_rnn`),
  - guardrail application,
  - step simulation through JaxLOB/JaxMARL-HFT adapters,
  - metrics and artifact writing.
- `evaluate/world_model_selector.py` and `evaluate/policy_adapter.py` enforce mode-specific required inputs.
- `evaluate/checkpoint_loader.py` bridges into JaxMARL-HFT helper functions and includes CPU-safe Orbax restore fallback.
- `guardrails/order_validators.py` sanitizes invalid outbound messages before they reach the simulator.
- `metrics/computation.py` computes Phase 1 metrics from step traces.
- `leaderboard/aggregator.py` merges many run summaries into ranked output JSON.
- `scripts/train_eval_phase2.py` and `scripts/adversarial_eval_phase2.py` are orchestration wrappers for Phase 2 workflows, both reusing the Phase 1 evaluator.

Typical flow: parse CLI -> validate mode/config -> bootstrap external paths -> run simulation loop -> apply guardrails -> compute metrics -> write `summary.json` and `step_trace.csv` -> aggregate with leaderboard script.

## Key conventions

- Path bootstrapping is intentional: scripts/modules insert repo roots into `sys.path` to import sibling projects (`JaxMARL-HFT`, `LOBS5`) at runtime.
- `generative` world model always requires `--lobs5_ckpt_path`; checkpoint step defaults to latest if not provided.
- `ippo_rnn` policy always requires both `--policy_ckpt_dir` and `--policy_config`.
- `--fast_startup` changes JAX memory env defaults (`XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_MEM_FRACTION`) for lighter startup.
- On CPU-constrained hosts, set thread caps before running to avoid JAX thread-creation failures:
  - `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=1`
  - `XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'`
- Guardrails prioritize simulator integrity:
  - invalid limit-add orders with non-positive prices are converted to no-op message type,
  - invalid quotes after action/world step trigger rollback to prior valid state in pipeline loop.
- Outputs are run-directory based under `outputs/evaluations/<run_name>/` with stable artifact names:
  - `summary.json`
  - `step_trace.csv`
- Leaderboard ranking uses `(total_pnl, -abs(max_drawdown), -risk_std)` as sort key (`leaderboard/aggregator.py`).
- Config defaults for Phase 1 are kept in `config/evaluation_configs/phase1_default.json`; tests assert this file is present and parseable.
