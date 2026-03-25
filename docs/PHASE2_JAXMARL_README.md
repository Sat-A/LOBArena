# Phase 2: JaxMARL-HFT Integration Overview

## Status: INTEGRATION CONTRACT COMPLETE ✅

**Contract Document:** [`phase2_jaxmarl_integration_contract.md`](./phase2_jaxmarl_integration_contract.md)

This file summarizes the JaxMARL-HFT integration contract for LOBArena phase 2 evaluation pipeline.

---

## Quick Start for LOBArena

### 1. Run JaxMARL-HFT Training (Minimal Smoke Test)

```bash
cd /home/s5e/satyamaga.s5e/JaxMARL-HFT
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python run_gen_worldmodel_pg_train.py \
  --policy_arch ippo_rnn \
  --checkpoint_restore_topology single-device-remap \
  --n_envs 1 --n_steps 2 --n_updates 1 \
  --gpu_id 0 --seed 42 --fast_startup
```

**Expected Output:** `outputs/gen_worldmodel_pg_train/<timestamp>/summary.json`

### 2. Load Policy into LOBArena

```python
from LOBArena.evaluate.checkpoint_loader import load_ippo_policy_adapter

policy = load_ippo_policy_adapter(
    jaxmarl_root="/home/s5e/satyamaga.s5e/JaxMARL-HFT",
    checkpoint_dir="<run_dir>",  # e.g., outputs/gen_worldmodel_pg_train/20260324_120000
    config_path=None,             # Inferred from summary.json
    seed=42
)
```

### 3. Run Evaluation

```bash
python LOBArena/scripts/evaluate_checkpoint.py \
  --policy_handoff_manifest <run_dir>/summary.json \
  --eval_dataset_path <lob_data_path> \
  --world_model_checkpoint <ckpt_path> \
  --world_model_step <step>
```

---

## Key Files

| File | Purpose |
|---|---|
| [`phase2_jaxmarl_integration_contract.md`](./phase2_jaxmarl_integration_contract.md) | **MAIN CONTRACT** – Full integration spec, schemas, examples |
| `PHASE2_JAXMARL_README.md` | This file – Quick start guide |

---

## Training Entrypoints

### Smoke Test (Minimal, ~5-10 min)
```bash
python run_gen_worldmodel_pg_train.py --policy_arch ippo_rnn --n_envs 1 --n_steps 2 --n_updates 1 --gpu_id 0
```

### Sweep Test (Profile Search, ~1 hour)
```bash
GPU_ID=0 bash run_sweep_gen_worldmodel_train_single_node.sh
```

### Train-Best (Multi-Seed, ~2-4 hours)
```bash
sbatch slurm/sbatch_train_gen_worldmodel_best.sh
```

### Profile-Driven Campaign (Recommended)
```bash
bash scripts/experiments/run_genwm_training_campaign.sh \
  config/gen_worldmodel_profiles/aggressive_pnl.env --submit
```

---

## Output Schema

### Per-Run: `summary.json`
Key sections:
- `run_name`, `run_dir`, `seed`: Run metadata
- `checkpoint_path`, `checkpoint_step`: World-model checkpoint
- `policy_arch`, `action_dim`, `fc_dim_size`, `gru_hidden_dim`: Network architecture
- `throughput`: Steps/sec metrics
- `pnl`: Profit/loss metrics
- `trade_incidence`: Trading frequency
- `update_logs`: Per-iteration metrics (loss, entropy, etc.)

### Per-Sweep: `single_node_sweep_aggregate_*.json`
Aggregated stats across `n_envs` profiles:
- `n_runs`, `runs[]`: Individual run metadata
- `pnl`: Mean/median/std/min/max
- `throughput`: Aggregated steps/sec

### Per-Campaign: `campaigns/{id}/campaign_summary.json`
Multi-stage orchestration:
- `stages[]`: Smoke, sweep, train-best status
- `champion_candidates`: Selected best policies
- `discovered_artifacts`: Paths to all outputs

---

## Configuration Profiles

**Location:** `config/gen_worldmodel_profiles/*.env`

**Example:** `aggressive_pnl.env`
```bash
POLICY_ARCH=ippo_rnn
CHECKPOINT_RESTORE_TOPOLOGY=single-device-remap
MM_ACTION_SPACE=bobRL
N_ENVS=2
LR=3e-4
ENTROPY_COEF=5e-3
N_STEPS=32
N_UPDATES=20
SEED=42
```

**Usage:**
```bash
source config/gen_worldmodel_profiles/aggressive_pnl.env
python run_gen_worldmodel_pg_train.py \
  --policy_arch "${POLICY_ARCH}" \
  --lr "${LR}" \
  ... (export all keys as CLI args)
```

---

## Checkpoint Restoration

**Recommended Mode:** `single-device-remap`

This mode ensures checkpoint portability across different device topologies (single GPU, multi-GPU, CPU).

```bash
python run_gen_worldmodel_pg_train.py \
  --checkpoint_restore_topology single-device-remap \
  --ckpt_path <path> \
  ...
```

**LOBArena Integration:** Always use this mode for deterministic evaluation.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Checkpoint restore fails | Use `--checkpoint_restore_topology single-device-remap` |
| Policy reconstruction fails in LOBArena | Verify `summary.json` contains `policy_arch`, `action_dim`, `fc_dim_size`, `gru_hidden_dim` |
| Data dir validation fails | Check CSV files match pattern `*_message_*.csv` and `*_orderbook_*.csv` |
| Import errors in LOBArena | Set `PYTHONPATH=/home/s5e/satyamaga.s5e/JaxMARL-HFT:$PYTHONPATH` |
| Out of memory (OOM) | Reduce `--n_envs` or `--n_steps` |

---

## Integration Points

1. **Training → Output:** JaxMARL-HFT produces `summary.json` with full training config + metrics
2. **LOBArena Discovery:** Reads `summary.json` to extract policy architecture + hyperparameters
3. **Policy Instantiation:** Uses `evaluate/checkpoint_loader.py::load_ippo_policy_adapter()` (no external config needed)
4. **Evaluation:** Runs rollouts via `evaluate/pipeline.py` with discovered policy
5. **Lineage:** Stores training → eval linkage for full reproducibility

---

## For More Details

See [`phase2_jaxmarl_integration_contract.md`](./phase2_jaxmarl_integration_contract.md) for:
- Complete CLI reference
- All output JSON schemas
- Success/failure criteria
- Recovery strategies
- 5 common troubleshooting scenarios

---

**Document Date:** 2026-03-24  
**Status:** ACTIVE  
**Task ID:** phase2-contract-spec-jaxmarl (COMPLETE)
