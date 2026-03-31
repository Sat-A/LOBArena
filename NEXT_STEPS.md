# LOBArena Next Steps Plan

## 1) Stabilize runtime execution profiles

- Add an explicit `--cpu_safe` flag to `scripts/evaluate_checkpoint.py` that applies conservative thread/env defaults internally.
- Add optional `--device` (`cpu|gpu`) CLI argument and enforce deterministic backend selection.
- Capture platform/runtime metadata (backend, thread env vars, host) in every `summary.json`.

## 2) Improve policy coverage

- Add first-class support for loading and evaluating multiple `ippo_rnn` checkpoints in batch.
- Add policy registry/config file for reusable named policies.
- Add validation for checkpoint/config compatibility before rollout starts.

## 3) Strengthen metrics

- Extend risk section beyond delta-std:
  - rolling volatility
  - downside deviation
  - max adverse excursion
- Add richer impact metrics:
  - short-horizon midprice response
  - signed impact by action side
- Record per-step execution-quality metrics for debugging.

## 4) Leaderboard upgrades

- Add normalized multi-objective score with configurable weights.
- Support split leaderboards by:
  - world model mode
  - policy family
  - date window
- Add CSV export and lightweight HTML render.

## 5) Phase 2 training/eval hardening

- Replace placeholder orchestration with full train-run integration:
  - launch training jobs
  - locate produced checkpoints
  - auto-run test-set evaluation
- Store train/eval linkage metadata (train run -> checkpoint -> eval run).

## 6) Adversarial framework expansion

- Add configurable competitor catalog (MM, directional, scripted baselines).
- Support multiple competitors in one tournament run.
- Add pairwise win-rate and robustness summaries.

## 7) Test & CI coverage

- Add unit tests for pipeline argument validation and fallback behavior.
- Add integration smoke tests with tiny fixture data.
- Add CI target for:
  - syntax checks
  - unit tests
  - leaderboard aggregation sanity check

## 8) Documentation polish

- Keep benchmark commands explicit and decoupled: run scoring first, then `scripts/plot_multi_window.py` for visuals.
- Keep `.github/copilot-instructions.md` and `README.md` synced with CLI changes.
- Add examples for common run recipes:
  - historical random baseline
  - generative checkpoint evaluation
  - adversarial comparison run
