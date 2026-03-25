# JaxMARL-HFT → LOBArena Phase 2 Integration Contract

**Date:** 2026-03-24  
**Status:** ACTIVE  
**Scope:** Generative world-model policy training integration with LOBArena evaluation pipeline

---

## 1. TRAINING ENTRYPOINTS & COMMANDS

### 1.1 Primary Training Script
- **Path:** `/home/s5e/satyamaga.s5e/JaxMARL-HFT/run_gen_worldmodel_pg_train.py`
- **Language:** Python 3.10+
- **Entry Point:** `main()` → returns exit code 0 on success, non-zero on failure

### 1.2 Minimum Reliable Smoke Command (Single GPU)
```bash
cd /home/s5e/satyamaga.s5e/JaxMARL-HFT
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python run_gen_worldmodel_pg_train.py \
  --policy_arch ippo_rnn \
  --checkpoint_restore_topology single-device-remap \
  --n_envs 1 \
  --n_steps 2 \
  --n_updates 1 \
  --n_cond_msgs 8 \
  --gpu_id 0 \
  --seed 42 \
  --output_root ./outputs/gen_worldmodel_pg_train \
  --fast_startup
```
**Expected Runtime:** 5–10 minutes on single GPU  
**Success Indicator:** Exit code 0 + summary.json written to `--output_root/<run_name>/`

### 1.3 Slurm Smoke Submission Script
- **Path:** `/home/s5e/satyamaga.s5e/JaxMARL-HFT/slurm/sbatch_smoke_gen_worldmodel_train.sh`
- **Required Env Vars:** `POLICY_ARCH`, `CHECKPOINT_RESTORE_TOPOLOGY`, `N_ENVS`, `N_UPDATES`
- **Job Spec:** 1 node, 1 GPU, 8 CPUs, 30 min timeout
- **Output:** `slurm/logs/{job_name}_{job_id}.{out,err}`

### 1.4 Sweep Command (Profile Search)
```bash
bash /home/s5e/satyamaga.s5e/JaxMARL-HFT/run_sweep_gen_worldmodel_train_single_node.sh
```
**Variables Required:**
- `GPU_ID`: GPU device ID (0, 1, 2, ...)
- `N_ENVS_CANDIDATES`: Space-separated list (e.g., "1 2 4")
- `SWEEP_TAG`: Unique identifier for run family
- `CHECKPOINT_RESTORE_TOPOLOGY`: "auto", "strict", or "single-device-remap"
- `RETRY_PER_PROFILE`: Integer (e.g., 2)

### 1.5 Multi-Seed Training (Train-Best)
- **Path:** `/home/s5e/satyamaga.s5e/JaxMARL-HFT/slurm/sbatch_train_gen_worldmodel_best.sh`
- **Launches:** 4 parallel training jobs (seeds 42–45)
- **Output:** Individual `train_best_seed{S}_{SLURM_JOB_ID}/summary.json` + aggregate
- **Aggregation:** Uses `aggregate_gen_worldmodel_pnl.py`

---

## 2. CONFIGURATION FILES & PROFILES

### 2.1 Profile Format (Recommended for LOBArena)
- **Location:** `config/gen_worldmodel_profiles/*.env`
- **Format:** Bash-compatible key-value pairs (sourced with `set -a`)
- **Example:** `/home/s5e/satyamaga.s5e/JaxMARL-HFT/config/gen_worldmodel_profiles/aggressive_pnl.env`

**Required Profile Keys:**
```bash
POLICY_ARCH=ippo_rnn                         # "mlp" or "ippo_rnn"
CHECKPOINT_RESTORE_TOPOLOGY=single-device-remap  # restore mode
MM_ACTION_SPACE=bobRL                        # market-making action space
MM_BOB_V0=10                                 # bob variant selector
MM_FIXED_QUANT_VALUE=10                      # fixed order quantity
LR=3e-4                                      # learning rate
ENTROPY_COEF=5e-3                            # entropy coefficient
VALUE_COEF=0.5                               # value loss weight
N_COND_MSGS=8                                # conditioning messages
N_STEPS=32                                   # rollout horizon
N_UPDATES=20                                 # gradient updates per run
SEED=42                                      # PRNG seed
```

**Optional Sweep/Train Keys:**
```bash
N_ENVS=2                                     # smoke mode
N_ENVS_CANDIDATES="1 2 4"                    # sweep search space
GPU_IDS="0 1 2 3"                            # available GPUs
N_ENVS_BEST=2                                # train-best setting
```

### 2.2 Alternative: Direct CLI Arguments
All profile keys can be passed as `--<key>` CLI arguments to `run_gen_worldmodel_pg_train.py`:
```bash
python run_gen_worldmodel_pg_train.py \
  --policy_arch ippo_rnn \
  --mm_action_space bobRL \
  --lr 3e-4 \
  --entropy_coef 5e-3 \
  ... (see argparse in script)
```

### 2.3 World-Model Checkpoint Location
- **Source:** `--ckpt_path` (default: env var `WORLD_MODEL_CKPT`)
- **Default Value:** `/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440`
- **Format:** Orbax checkpoint directory with structure:
  ```
  <ckpt_path>/
    <step>/
      state/              # PyTreeCheckpointer state
  ```
- **Restore Strategies:** `auto` (try native, fallback to single-device), `strict`, `single-device-remap`

### 2.4 Data Directory Structure
- **Source:** `--data_dir` (default: env var `LOB_PREPROC_DATA_DIR`)
- **Expected Layout:**
  ```
  <data_dir>/
    <stock>_<date>_34200000_57600000_message_10.csv
    <stock>_<date>_34200000_57600000_orderbook_10.csv
  ```
- **Date Filtering:** Applied via `--start_date` and `--end_date` (ISO 8601)

---

## 3. OUTPUT DIRECTORY STRUCTURE & CHECKPOINT FORMATS

### 3.1 Output Root & Run Directory
```
--output_root/
  <run_name>/
    summary.json             # Main training results artifact
```

**Run Name Convention:**
- Auto-generated: `YYYYMMDD_HHMMSS` (default)
- Custom: via `--run_name "my_run_name"`
- Slurm jobs: `{prefix}_{SLURM_JOB_ID}` (smoke: `genwm_train_smoke_*`, train-best: `train_best_seed*_job*`)

### 3.2 Summary JSON Format (Single Training Run)
**Path:** `<run_dir>/summary.json`

**Key Sections:**
```json
{
  "run_name": "genwm_train_smoke_2915942",
  "run_dir": "/absolute/path/to/run_dir",
  "policy_arch": "ippo_rnn",
  "checkpoint_path": "/path/to/world_model_ckpt",
  "checkpoint_step": 135458,
  "checkpoint_restore": {
    "state_dir": "/path/to/ckpt/135458/state",
    "requested_mode": "single-device-remap",
    "effective_mode": "single-device-remap",
    "used_single_device_remap": true,
    "fallback_from_auto_used": false,
    "topology_mismatch_detected": false,
    "native_restore_error": null
  },
  "data_dir": "/path/to/lob_data",
  "dataset_effective_dir": "/tmp/filtered_data",
  "start_date": "2026-01-01",
  "end_date": "2026-01-31",
  "seed": 42,
  "n_envs": 2,
  "n_steps": 20,
  "n_updates": 12,
  "n_cond_msgs": 8,
  "action_dim": 21,
  "mm_action_space": "bobRL",
  "mm_bob_v0": 10,
  "mm_fixed_quant_value": 10,
  
  "selection_metrics": {
    "quality_primary": "pnl.mean_avg_pnl",
    "quality_guardrail": "pnl.final_avg_pnl > 0 and trade_incidence > 0",
    "throughput_tiebreaker": "throughput.updates_mean_steps_per_sec"
  },
  
  "throughput": {
    "updates_mean_steps_per_sec": 3.895723318461456,
    "updates_max_steps_per_sec": 4.275102667643557
  },
  
  "pnl": {
    "updates_avg_pnl": [0.0, 0.0, ..., 0.0],  # per-update PnL
    "mean_avg_pnl": 0.0,
    "best_avg_pnl": 0.0,
    "final_avg_pnl": 0.0
  },
  
  "trade_incidence": {
    "episodes_with_trades": 0,
    "fraction": 0.0,
    "mean_agent_trade_count": 0.0
  },
  
  "timing_breakdown": {
    "restore_sec": 3.311,
    "dataset_load_sec": 0.071,
    "sim_init_sec": 2.601,
    "total_runtime_sec": 393.247
  },
  
  "update_logs": [
    {
      "update": 1,
      "avg_pnl": 0.0,
      "pnl_std": 0.0,
      "avg_agent_trade_count": 0.0,
      "episodes_with_agent_trades": 0,
      "agent_trade_fraction": 0.0,
      "loss": 0.058385651,
      "entropy": 2.883821249,
      "policy_arch": "ippo_rnn",
      "steps_per_sec": 0.265604964,
      "step_latency_ms_p50": 134.78,
      "step_latency_ms_p95": 591.61,
      "policy_loss": -0.014419104,
      "value_loss": 0.145609513
    },
    ...
  ]
}
```

### 3.3 Sweep Aggregate JSON
**Path:** `outputs/gen_worldmodel_pg_train/single_node_sweep_aggregate_{SLURM_JOB_ID}.json`

**Format:**
```json
{
  "n_runs": 4,
  "runs": [
    {
      "run_name": "sweep_gen_train_daystart_20260317_a_g0_n1",
      "path": "outputs/gen_worldmodel_pg_train/sweep_gen_train_daystart_20260317_a_g0_n1/summary.json",
      "seed": 42,
      "n_envs": 1,
      "final_avg_pnl": 0.0,
      "mean_steps_per_sec": 3.65
    },
    ...
  ],
  "pnl": {
    "mean": 0.0,
    "median": 0.0,
    "std": 0.0,
    "min": 0.0,
    "max": 0.0
  },
  "throughput": {
    "mean_steps_per_sec": 3.66,
    "median_steps_per_sec": 3.69,
    "max_steps_per_sec": 3.71
  }
}
```

### 3.4 Train-Best Aggregate JSON
**Path:** `outputs/gen_worldmodel_pg_train/train_best_aggregate_{SLURM_JOB_ID}.json`

**Format:** Same structure as sweep aggregate (multiple seeds, mean/std statistics)

### 3.5 Campaign Summary (Orchestration Output)
**Path:** `outputs/gen_worldmodel_pg_train/campaigns/{campaign_id}/campaign_summary.json`

**Format:**
```json
{
  "campaign_id": "local_validation_v2",
  "created_at_utc": "2026-03-24T09:23:37+00:00",
  "campaign_status": "success|running|blocked",
  "mode": "dry-run|submit",
  "profile": {
    "path": "/home/s5e/satyamaga.s5e/JaxMARL-HFT/config/gen_worldmodel_profiles/aggressive_pnl.env",
    "name": "aggressive_pnl.env"
  },
  "stages": [
    {
      "stage": "smoke",
      "status": "success|failed|running",
      "job_id": "2915942",
      "rc": "0"
    },
    {
      "stage": "sweep",
      "status": "success|failed|running",
      "job_id": "2915160",
      "rc": "0"
    },
    {
      "stage": "train-best",
      "status": "success|failed|running|skipped",
      "job_id": "2915160",
      "rc": "0"
    }
  ],
  "discovered_artifacts": {
    "smoke_summaries": [...],
    "sweep_aggregates": [...],
    "train_best_aggregates": [...],
    "latest_sweep_aggregate": "...",
    "latest_train_best_aggregate": "..."
  },
  "champion_candidates": {
    "sweep": {...},
    "train_best": {...},
    "proposed_champion": {...}
  }
}
```

---

## 4. SUCCESS & FAILURE INDICATORS

### 4.1 Success Criteria
1. **Exit Code:** Script returns 0 to shell
2. **Summary File Exists:** `<run_dir>/summary.json` present and valid JSON
3. **Required Fields:** All keys in section 3.2 populated (no null/missing)
4. **Timing:** `total_runtime_sec > 0`
5. **Optional Quality:** `trade_incidence.fraction > 0` and `pnl.final_avg_pnl ≥ 0` (guardrail for production)

### 4.2 Failure Modes & Recovery

| Failure Mode | Exit Code | Recovery |
|---|---|---|
| Checkpoint restore mismatch | 1 | Retry with `--checkpoint_restore_topology single-device-remap` |
| Data directory missing | 1 | Verify `--data_dir` path exists and contains CSV files |
| CUDA/GPU unavailable | 1 | Check `--gpu_id` is valid; retry on different GPU |
| JAX/JAX-LOB import error | 1 | Verify `LOBS5_ROOT` is set; check conda env has JAX installed |
| Timeout (cluster job) | 124 | Increase `--n_updates` and `--n_steps` in job script |
| OOM (out of memory) | 137 | Reduce `--n_envs` or `--n_steps` |
| NaN loss / policy divergence | 0 but `pnl=NaN` | Reduce `--lr` or `--entropy_coef` |

### 4.3 Fallback Behavior
- **Checkpoint restore:** When `--checkpoint_restore_topology=auto`, first attempts native restore; on topology mismatch, automatically falls back to single-device remap.
- **Dataset filtering:** If date range yields empty dataset, raises `RuntimeError("Dataset is empty")`.
- **JIT warmup:** If `--jit_message_build` is set but fails, silently disables JIT for that run (logs warning).

---

## 5. REQUIRED ARGUMENTS FOR LOBArena INTEGRATION

### 5.1 Mandatory Arguments (No Defaults)
```
--ckpt_path              Path to world-model Orbax checkpoint directory
--data_dir               Path to LOB data directory (preprocessed LOBSTER CSVs)
--lobs5_root             Path to LOBS5 repository root
```

### 5.2 Recommended Arguments (With Defaults)
```
--policy_arch ippo_rnn            # Use RNN-based policy (required for LOBArena)
--checkpoint_restore_topology auto  # Auto-fallback for device mismatch
--mm_action_space bobRL           # Rich action space for market-making
--n_cond_msgs 8                   # Conditioning horizon (LOBArena default)
--gpu_id 0                        # Single GPU (LOBArena smoke tests)
--seed 42                         # Reproducible seed
--output_root ./outputs/gen_worldmodel_pg_train
```

### 5.3 Tunable Arguments (Experiment-Dependent)
```
--n_envs {1,2,4}         # Parallel environments (throughput/quality tradeoff)
--n_steps {10,20,32}     # Rollout horizon (longer = more gradient signal)
--n_updates {5,10,20}    # Training iterations
--lr {1e-4,3e-4,1e-3}    # Learning rate
--entropy_coef {1e-3,5e-3}  # Exploration coefficient
--value_coef 0.5         # Critic loss weight (usually fixed)
```

### 5.4 Profile Invocation (Recommended for LOBArena)
```bash
# Load all settings from profile
source config/gen_worldmodel_profiles/aggressive_pnl.env

# Train single run
python run_gen_worldmodel_pg_train.py \
  --policy_arch "${POLICY_ARCH}" \
  --checkpoint_restore_topology "${CHECKPOINT_RESTORE_TOPOLOGY}" \
  --n_envs "${N_ENVS}" \
  --lr "${LR}" \
  ...
```

---

## 6. LOBArena INTEGRATION POINTS

### 6.1 Checkpoint Handoff
**LOBArena will:**
1. Discover trained policy checkpoint in `<run_dir>/summary.json` → extract `run_dir`
2. Infer policy network parameters from `policy_arch` and `action_dim`
3. Use LOBArena's `evaluate/checkpoint_loader.py::load_ippo_policy_adapter()` to instantiate policy
4. Run rollouts via `evaluate/pipeline.py` with discovered checkpoint

**JaxMARL-HFT Requirement:**
- All training hyperparameters must be stored in `summary.json` so LOBArena can reconstruct the policy network without additional config lookups.

### 6.2 Policy Artifact Manifest
**Location:** `<run_dir>/policy_handoff.json` (optional, generated by LOBArena)

**Format:**
```json
{
  "training_run_id": "genwm_train_smoke_2915942",
  "training_summary": "/home/s5e/satyamaga.s5e/JaxMARL-HFT/outputs/gen_worldmodel_pg_train/genwm_train_smoke_2915942/summary.json",
  "policy_type": "ippo_rnn",
  "world_model_checkpoint": "/lus/lfs1aip2/projects/s5e/quant/AlphaTrade/experiments/exp_H1-scaling-law/checkpoints/j2514440_bkotgtm5_2514440",
  "world_model_step": 135458,
  "policy_network_params": {
    "fc_dim_size": 128,
    "gru_hidden_dim": 128,
    "action_dim": 21
  },
  "training_config": {
    "seed": 42,
    "n_envs": 2,
    "lr": 3e-4,
    "entropy_coef": 1e-3,
    "value_coef": 0.5
  },
  "eval_ready": true,
  "ready_at_utc": "2026-03-17T10:13:42+00:00"
}
```

### 6.3 Campaign Orchestration (Optional)
**Helper Script:** `scripts/experiments/run_genwm_training_campaign.sh`

**Usage:**
```bash
# Dry-run (logs intended commands, no job submission)
bash scripts/experiments/run_genwm_training_campaign.sh \
  config/gen_worldmodel_profiles/aggressive_pnl.env

# Real submission with dependency chain (smoke → sweep → train-best)
bash scripts/experiments/run_genwm_training_campaign.sh \
  config/gen_worldmodel_profiles/aggressive_pnl.env \
  --submit
```

**Produces:**
- `outputs/gen_worldmodel_pg_train/campaigns/{campaign_id}/campaign_summary.json`
- Dependency-linked Slurm jobs

---

## 7. TROUBLESHOOTING FOR LOBARENA

### Issue: Checkpoint restore fails with "topology mismatch"
**Solution:**
```bash
python run_gen_worldmodel_pg_train.py \
  --checkpoint_restore_topology single-device-remap \
  ...
```
(This is the default recommended mode; LOBArena should always use this.)

### Issue: Policy network reconstruction fails in LOBArena
**Debugging:**
1. Verify `summary.json` contains all required keys: `policy_arch`, `action_dim`, `fc_dim_size`, `gru_hidden_dim`
2. Verify `checkpoint_path` and `checkpoint_step` point to valid Orbax checkpoint
3. Test locally: `python run_one_step_inference.py --ckpt_path ... --checkpoint_step ...`

### Issue: Data directory validation fails in LOBArena
**Check:**
- `--data_dir` contains `*.csv` files matching pattern `*_message_*.csv` and `*_orderbook_*.csv`
- `--start_date` and `--end_date` overlap with available data

### Issue: JaxMARL imports fail in LOBArena
**Solution:**
1. Ensure `--lobs5_root` and `--jaxmarl_root` are set correctly
2. Activate conda env: `conda activate lobs5`
3. Set `PYTHONPATH`: `export PYTHONPATH=/home/s5e/satyamaga.s5e/JaxMARL-HFT:/home/s5e/satyamaga.s5e/LOBS5:${PYTHONPATH}`

---

## 8. SUMMARY TABLE: QUICK REFERENCE

| Item | Value |
|---|---|
| **Training Script** | `/home/s5e/satyamaga.s5e/JaxMARL-HFT/run_gen_worldmodel_pg_train.py` |
| **Main Output** | `<output_root>/<run_name>/summary.json` |
| **Smoke Command** | `python run_gen_worldmodel_pg_train.py --policy_arch ippo_rnn --n_envs 1 --n_steps 2 --n_updates 1 --gpu_id 0` |
| **Expected Runtime** | 5–10 min (smoke), 30–60 min (sweep), 2–4 hours (train-best multi-seed) |
| **Success Exit Code** | 0 |
| **Policy Architecture** | IPPO RNN (Flax LocalActorCriticRNN) |
| **Action Dimension** | 21 (bobRL action space) |
| **Checkpoint Format** | Orbax PyTreeCheckpointer (`<ckpt>/step/state/`) |
| **Profile Format** | Bash env file (key=value pairs) |
| **LOBArena Integration** | Load policy via `checkpoint_loader.py::load_ippo_policy_adapter()` + run via `evaluate/pipeline.py` |
| **Key Metrics** | `pnl.final_avg_pnl`, `throughput.updates_mean_steps_per_sec`, `trade_incidence.fraction` |

---

## DOCUMENT HISTORY

| Version | Date | Author | Summary |
|---|---|---|---|
| 1.0 | 2026-03-24 | Exploration Agent | Initial phase2 integration contract; smoke/sweep/train-best validated; output schemas documented; LOBArena integration points identified |

