import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from LOBArena.evaluate.checkpoint_loader import load_world_model_from_jaxmarl, restore_params_with_cpu_fallback
from LOBArena.evaluate.policy_adapter import (
    load_ippo_policy_adapter,
    load_ippo_policy_adapter_with_index,
    validate_policy_choice,
)
from LOBArena.evaluate.policy_handoff import load_policy_handoff
from LOBArena.evaluate.single_node_guard import enforce_single_node_context
from LOBArena.evaluate.world_model_selector import validate_world_model_choice
from LOBArena.guardrails.order_validators import book_quotes_valid, sanitize_action_messages
from LOBArena.metrics.computation import build_phase1_metrics


class RunArtifacts(object):
    def __init__(self, run_dir, summary_json, step_trace_csv):
        self.run_dir = run_dir
        self.summary_json = summary_json
        self.step_trace_csv = step_trace_csv


@dataclass(frozen=True)
class BatchCandidate:
    candidate_id: str
    policy_handoff_path: str


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="LOBArena Phase 1 evaluation pipeline")
    p.add_argument("--world_model", choices=["historical", "generative"], required=True)
    p.add_argument("--policy_mode", choices=["random", "fixed", "ippo_rnn", "lose_money"], default="random")
    p.add_argument("--fixed_action", type=int, default=0)

    p.add_argument("--jaxmarl_root", default=str(workspace_root / "JaxMARL-HFT"))
    p.add_argument("--lobs5_root", default=str(workspace_root / "LOBS5"))
    p.add_argument("--lobs5_ckpt_path", default="")
    p.add_argument("--policy_ckpt_dir", default="")
    p.add_argument("--policy_config", default="")
    p.add_argument("--policy_handoff", default="")
    p.add_argument(
        "--policy_handoff_batch",
        nargs="+",
        default=None,
        help="List of policy handoff artifacts to evaluate in one invocation.",
    )
    p.add_argument(
        "--policy_handoff_manifest",
        default="",
        help="JSON manifest with policy handoff candidates for batch evaluation.",
    )

    p.add_argument("--data_dir", required=True)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--checkpoint_step", type=int, default=None)
    p.add_argument("--test_split", type=float, default=1.0)
    p.add_argument("--start_date", default="")
    p.add_argument("--end_date", default="")

    p.add_argument("--n_cond_msgs", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=25)
    p.add_argument("--sample_top_n", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--output_root", default=str(workspace_root / "LOBArena" / "outputs" / "evaluations"))
    p.add_argument("--run_name", default="")
    p.add_argument("--cpu_safe", action="store_true", help="Apply conservative CPU-thread runtime limits.")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument(
        "--strict_generative",
        action="store_true",
        help="Fail run on generative world-model inference error instead of falling back to historical replay.",
    )
    p.add_argument(
        "--allow_generative_fallback",
        action="store_true",
        help=(
            "Allow fallback to historical replay when generative inference fails. "
            "By default, generative runs fail fast on generation errors."
        ),
    )
    p.add_argument("--fast_startup", action="store_true")
    return p.parse_args()


def _configure_runtime(args):
    if args.cpu_safe:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        os.environ.setdefault("JAX_NUM_THREADS", "1")
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

    if args.device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    if args.fast_startup:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")


def _runtime_metadata(args, selected_data_dir: Path, jax_backend: str, jax_devices):
    tracked_env = {
        k: os.environ.get(k, "")
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
            "JAX_NUM_THREADS",
            "XLA_FLAGS",
            "JAX_PLATFORMS",
            "JAX_PLATFORM_NAME",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
        )
    }
    return {
        "host": platform.node(),
        "python_version": sys.version.split()[0],
        "requested_device": args.device,
        "cpu_safe": bool(args.cpu_safe),
        "fast_startup": bool(args.fast_startup),
        "jax_backend": jax_backend,
        "jax_device_count": len(jax_devices),
        "jax_devices": [str(d) for d in jax_devices],
        "selected_data_dir": str(selected_data_dir),
        "thread_env": tracked_env,
    }


def _prepare_artifacts(args):
    root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name.strip() if args.run_name.strip() else time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=run_dir,
        summary_json=run_dir / "summary.json",
        step_trace_csv=run_dir / "step_trace.csv",
    )


def _build_obs(step_i, bid, ask, agent_state, world_time):
    spread = int(ask - bid)
    mid = (float(bid) + float(ask)) / 2.0
    import jax.numpy as jnp

    return jnp.asarray(
        [
            float(step_i),
            float(agent_state.inventory),
            float(agent_state.cash_balance),
            float(agent_state.total_PnL),
            float(bid),
            float(ask),
            float(spread),
            float(mid),
            float(world_time[0]),
            float(world_time[1]),
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )


def _choose_loss_seeking_action(mm_agent, n_actions, step_world_state, agent_state, agent_params):
    """Pick action that creates most marketable volume to force adverse fills."""
    import jax.numpy as jnp

    best_idx = 0
    best_score = -1
    for action_i in range(int(n_actions)):
        action_msgs, _cancel_msgs, _extras = mm_agent.get_messages(
            jnp.int32(action_i),
            step_world_state,
            agent_state,
            agent_params,
        )
        msgs = sanitize_action_messages(action_msgs)
        if msgs.size == 0:
            score = 0
        else:
            msg_types = msgs[:, 0]
            side = msgs[:, 1]
            qty = msgs[:, 2]
            price = msgs[:, 3]
            buy_cross = (msg_types == 1) & (side == 1) & (price > 10_000_000)
            sell_cross = (msg_types == 1) & (side == -1) & (price < 10_000)
            marketable = buy_cross | sell_cross
            score = int(jnp.sum(jnp.where(marketable, qty, 0)))
        if score > best_score:
            best_idx = action_i
            best_score = score
    return jnp.int32(best_idx)


def _force_marketable_lossy_orders(action_msgs):
    """Mutate limit-add order prices to aggressively cross the spread."""
    import jax.numpy as jnp

    if action_msgs.size == 0:
        return action_msgs
    msg_types = action_msgs[:, 0]
    side = action_msgs[:, 1]
    qty = action_msgs[:, 2]
    price = action_msgs[:, 3]
    is_limit_add = msg_types == 1
    forced_price = jnp.where(side == 1, jnp.int32(1_000_000_000), jnp.int32(1))
    forced_qty = jnp.maximum(qty, jnp.ones_like(qty))
    new_price = jnp.where(is_limit_add, forced_price, price)
    new_qty = jnp.where(is_limit_add, forced_qty, qty)
    out = action_msgs.at[:, 2].set(new_qty)
    out = out.at[:, 3].set(new_price)
    return out


def _write_csv(path, header, rows):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _is_batch_mode(args: argparse.Namespace) -> bool:
    return bool(args.policy_handoff_batch) or bool(str(args.policy_handoff_manifest).strip())


def _load_policy_handoff_manifest(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Policy handoff manifest not found: {p}")
    payload = json.loads(p.read_text())
    if isinstance(payload, list):
        payload = {"candidates": payload}
    if not isinstance(payload, dict):
        raise ValueError("Policy handoff manifest must be a JSON object or JSON list")

    candidates_raw = payload.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("Policy handoff manifest must include non-empty 'candidates' list")
    fairness = payload.get("fairness") or {}
    if fairness and not isinstance(fairness, dict):
        raise ValueError("Manifest fairness must be an object when provided")
    return payload, candidates_raw, fairness, p.parent


def _normalize_batch_candidate(
    candidate_raw,
    idx: int,
    base_dir: Path,
    trusted_base_dir: Path | None = None,
) -> BatchCandidate:
    if isinstance(candidate_raw, str):
        candidate_id = f"candidate_{idx + 1:03d}"
        raw_path = candidate_raw
    elif isinstance(candidate_raw, dict):
        raw_path = (
            candidate_raw.get("policy_handoff")
            or candidate_raw.get("handoff")
            or candidate_raw.get("path")
            or ""
        )
        if not raw_path:
            raise ValueError(f"Batch candidate {idx} is missing handoff path field")
        candidate_id = str(candidate_raw.get("name", "")).strip() or f"candidate_{idx + 1:03d}"
    else:
        raise ValueError(f"Batch candidate {idx} must be string path or object")

    handoff_path = Path(str(raw_path)).expanduser()
    raw_is_absolute = handoff_path.is_absolute()
    if not handoff_path.is_absolute():
        handoff_path = (base_dir / handoff_path).resolve()
    else:
        handoff_path = handoff_path.resolve()
    if trusted_base_dir is not None and not raw_is_absolute:
        trusted_root = trusted_base_dir.resolve()
        try:
            handoff_path.relative_to(trusted_root)
        except ValueError as e:
            raise ValueError(
                f"Batch candidate {idx} path escapes manifest directory: {handoff_path}. "
                f"Expected under: {trusted_root}"
            ) from e
    return BatchCandidate(candidate_id=candidate_id, policy_handoff_path=str(handoff_path))


def resolve_batch_candidates(args: argparse.Namespace):
    if args.policy_handoff and _is_batch_mode(args):
        raise ValueError("Use either --policy_handoff (single) or batch inputs, not both")

    candidates = []
    fairness_overrides = {}
    if args.policy_handoff_batch:
        for i, raw in enumerate(args.policy_handoff_batch):
            candidates.append(_normalize_batch_candidate(raw, i, Path.cwd()))
    if args.policy_handoff_manifest:
        _manifest, candidates_raw, fairness, base_dir = _load_policy_handoff_manifest(args.policy_handoff_manifest)
        for i, raw in enumerate(candidates_raw):
            candidates.append(_normalize_batch_candidate(raw, i, base_dir, trusted_base_dir=base_dir))
        fairness_overrides = {
            "seed": fairness.get("seed"),
            "start_date": fairness.get("start_date"),
            "end_date": fairness.get("end_date"),
        }

    deduped = []
    seen = set()
    for c in candidates:
        key = (c.candidate_id, c.policy_handoff_path)
        if key in seen:
            continue
        deduped.append(c)
        seen.add(key)
    return deduped, fairness_overrides


def _sanitize_run_suffix(value: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "candidate"


def _ranking_key(row):
    pnl = float(row.get("total_pnl", 0.0))
    return (-pnl, str(row.get("candidate_id", "")))


def _run_single_evaluation(args) -> dict:
    artifacts = _prepare_artifacts(args)
    jaxmarl_root = Path(args.jaxmarl_root).expanduser().resolve()
    lobs5_root = Path(args.lobs5_root).expanduser().resolve()

    handoff = None
    if args.policy_handoff:
        handoff = load_policy_handoff(args.policy_handoff)
        args.policy_mode = handoff["policy"]["mode"]
        args.policy_ckpt_dir = handoff["policy"]["checkpoint_dir"]
        args.policy_config = handoff["policy"]["config_path"]
        if args.seed is None:
            args.seed = int(handoff["evaluation"]["seed"])
        if not args.start_date:
            args.start_date = str(handoff["evaluation"]["date_window"]["start_date"])
        if not args.end_date:
            args.end_date = str(handoff["evaluation"]["date_window"]["end_date"])
    if args.seed is None:
        args.seed = 42

    wm_sel = validate_world_model_choice(args.world_model, args.lobs5_ckpt_path or None)
    pol_sel = validate_policy_choice(args.policy_mode, args.fixed_action, args.policy_ckpt_dir or None, args.policy_config or None)
    strict_generation = bool(
        args.strict_generative
        or (wm_sel.mode == "generative" and not bool(getattr(args, "allow_generative_fallback", False)))
    )

    tools = load_world_model_from_jaxmarl(jaxmarl_root, lobs5_root)

    import jax
    import jax.numpy as jnp

    jax_devices = jax.devices()
    jax_backend = jax.default_backend()
    if args.device == "gpu" and jax_backend != "gpu":
        raise RuntimeError(
            f"Requested --device gpu but JAX backend is '{jax_backend}'. "
            "Check CUDA/JAX setup or use --device auto/cpu."
        )

    sys.path.insert(0, str(jaxmarl_root))
    from minimal_agent_generative_step import _best_quotes, _build_world_state, _compute_agent_pnl_from_trades  # type: ignore

    from lob.encoding import Message_Tokenizer, Vocab  # type: ignore
    from lob.init_train import init_train_state  # type: ignore
    from lob import inference_no_errcorr as inference  # type: ignore
    from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration, MarketMaking_EnvironmentConfig, World_EnvironmentConfig  # type: ignore
    from gymnax_exchange.jaxen.StatesandParams import MMEnvParams, MMEnvState  # type: ignore
    from gymnax_exchange.jaxen.mm_env import MarketMakingAgent  # type: ignore

    rng = jax.random.key(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    selected_data_dir, temp_data_ctx = tools["_prepare_date_filtered_data_dir"](data_dir, args.start_date, args.end_date)

    ckpt_path = wm_sel.lobs5_ckpt_path if wm_sel.mode == "generative" else (
        Path(args.lobs5_ckpt_path).expanduser().resolve() if args.lobs5_ckpt_path else None
    )

    if wm_sel.mode == "generative":
        step = args.checkpoint_step if args.checkpoint_step is not None else tools["_latest_checkpoint_step"](ckpt_path)
        params = restore_params_with_cpu_fallback(ckpt_path, step)
        ckpt_vocab_size = int(params["message_encoder"]["encoder"]["embedding"].shape[0])
        if ckpt_vocab_size >= 10000:
            tools["_enable_legacy_token_mode_22"]()
        model_args = tools["_load_metadata_robust"](ckpt_path)
        model_args = tools["_ensure_model_args_defaults"](model_args)
        model_args.num_devices = 1
        model_args.bsz = 1
        model_args.micro_bsz = 1
        model_args.global_bsz = 1
        if ckpt_vocab_size >= 10000:
            model_args.token_mode = 22

        vocab = Vocab()
        n_eval_messages = max(args.n_steps + 2, args.n_cond_msgs + args.n_steps + 2)
        eval_seq_len = (n_eval_messages - 1) * Message_Tokenizer.MSG_LEN
        init_state, model_cls = init_train_state(
            model_args,
            n_classes=ckpt_vocab_size,
            seq_len=eval_seq_len,
            book_dim=503,
            book_seq_len=eval_seq_len,
        )
        state = init_state.replace(params=params, step=step)
        model = model_cls(training=False, step_rescale=1.0)
        init_hidden = model.initialize_carry(
            1,
            hidden_size=(model_args.ssm_size_base // pow(2, int(model_args.conj_sym))),
            n_message_layers=model_args.n_message_layers,
            n_book_pre_layers=model_args.n_book_pre_layers,
            n_book_post_layers=model_args.n_book_post_layers,
            n_fused_layers=model_args.n_layers,
            h_size_ema=model_args.ssm_size_base,
        )
    else:
        step = None
        state = None
        model = None
        init_hidden = None
        vocab = Vocab()

    n_eval_messages = max(args.n_steps + 2, args.n_cond_msgs + args.n_steps + 2)
    ds = inference.get_dataset(str(selected_data_dir), args.n_cond_msgs, n_eval_messages, test_split=args.test_split)
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty")
    idx = max(0, min(args.sample_index, len(ds) - 1))

    m_seq, _, b_seq_pv, msg_seq_raw, book_l2_init = ds[[idx]]
    m_seq = jnp.array(m_seq)
    b_seq_pv = jnp.array(b_seq_pv)
    msg_seq_raw = jnp.array(msg_seq_raw)
    book_l2_init = jnp.array(book_l2_init)

    try:
        b_seq = inference.transform_L2_state_batch(b_seq_pv, 500, 100)
    except Exception:
        b_seq = b_seq_pv
    m_seq_inp = m_seq[:, : args.n_cond_msgs * Message_Tokenizer.MSG_LEN + 1]
    b_seq_inp = b_seq[:, : args.n_cond_msgs + 1]
    m_seq_raw_all = msg_seq_raw[0]
    m_seq_raw_cond = m_seq_raw_all[: args.n_cond_msgs]
    init_time_batched = b_seq_pv[:, 0, 1:3]

    sim = inference.OrderBook(cfg=JAXLOB_Configuration())
    sim_states = inference.get_sims_vmap(book_l2_init, jnp.expand_dims(m_seq_raw_cond, axis=0), init_time_batched, sim)
    sim_state = jax.tree_util.tree_map(lambda x: x[0], sim_states)

    world_cfg = World_EnvironmentConfig(tick_size=100)
    mm_cfg = MarketMaking_EnvironmentConfig(action_space="bobStrategy", fixed_quant_value=10, bob_v0=10)
    mm_agent = MarketMakingAgent(cfg=mm_cfg, world_config=world_cfg)

    agent_state = MMEnvState(posted_distance_bid=0, posted_distance_ask=0, inventory=0, total_PnL=0.0, cash_balance=0.0)
    agent_params = MMEnvParams(trader_id=jnp.int32(-101), time_delay_obs_act=jnp.int32(0), normalize=jnp.bool_(True))

    policy = None
    policy_hidden = None
    if pol_sel.mode == "ippo_rnn":
        if handoff is not None:
            policy = load_ippo_policy_adapter_with_index(
                jaxmarl_root,
                checkpoint_dir=pol_sel.checkpoint_dir,
                config_path=pol_sel.config_path,
                seed=args.seed,
                model_index=handoff["policy"]["model_index"],
            )
        else:
            policy = load_ippo_policy_adapter(
                jaxmarl_root,
                checkpoint_dir=pol_sel.checkpoint_dir,
                config_path=pol_sel.config_path,
                seed=args.seed,
            )
        policy_hidden = policy.fresh_hidden()

    current_sim_state = sim_state
    current_world_time = jnp.array(init_time_batched[0], dtype=jnp.int32)

    step_rows = []
    pnl_trace = []
    inventory_trace = []
    mid_before_action = []
    mid_after_action = []
    action_hist = {}
    generation_fallback_count = 0
    generation_error_last = None

    for step_i in range(args.n_steps):
        bid_before, ask_before = _best_quotes(sim, current_sim_state)
        if not book_quotes_valid(bid_before, ask_before):
            break

        if pol_sel.mode == "fixed":
            action_this_step = jnp.int32(pol_sel.fixed_action)
        elif pol_sel.mode == "random":
            rng, rng_action = jax.random.split(rng)
            action_this_step = jax.random.randint(
                rng_action,
                shape=(),
                minval=0,
                maxval=int(mm_cfg.n_actions),
                dtype=jnp.int32,
            )
        elif pol_sel.mode == "ippo_rnn":
            obs = _build_obs(step_i, int(bid_before), int(ask_before), agent_state, current_world_time)
            action_i, policy_hidden = policy.act_with_state(obs, policy_hidden, done=False)
            action_this_step = jnp.int32(action_i)
        else:
            step_world_state = _build_world_state(sim, current_sim_state, current_world_time)
            action_this_step = _choose_loss_seeking_action(
                mm_agent,
                mm_cfg.n_actions,
                step_world_state,
                agent_state,
                agent_params,
            )

        step_world_state = _build_world_state(sim, current_sim_state, current_world_time)
        action_msgs, cancel_msgs, _extras = mm_agent.get_messages(action_this_step, step_world_state, agent_state, agent_params)
        if pol_sel.mode == "lose_money":
            action_msgs = _force_marketable_lossy_orders(action_msgs)

        action_msgs = sanitize_action_messages(action_msgs)
        cancel_msgs = sanitize_action_messages(cancel_msgs)
        combined_agent_msgs = jnp.concatenate([cancel_msgs, action_msgs], axis=0)

        sim_state_after_action = sim.process_orders_array(current_sim_state, combined_agent_msgs)
        bid_after_action, ask_after_action = _best_quotes(sim, sim_state_after_action)
        if not book_quotes_valid(bid_after_action, ask_after_action):
            sim_state_after_action = current_sim_state
            bid_after_action, ask_after_action = bid_before, ask_before

        if wm_sel.mode == "generative":
            rng, rng_gen = jax.random.split(rng)
            try:
                msgs_decoded, _l2_states, _num_errors, _msg_tokens = inference.generate(
                    sim,
                    state,
                    model,
                    model_args.batchnorm,
                    vocab.ENCODING,
                    args.sample_top_n,
                    100,
                    m_seq_inp[0],
                    b_seq_inp[0],
                    1,
                    sim_state_after_action,
                    rng_gen,
                    init_hidden,
                    True,
                    jnp.asarray(current_world_time),
                    False,
                    None,
                )
                first_msg = msgs_decoded[0]
                sim_msg = inference.msg_to_jnp(first_msg)
            except Exception as e:
                generation_fallback_count += 1
                generation_error_last = repr(e)
                if strict_generation:
                    raise RuntimeError(
                        f"Generative inference failed at step={step_i}: {e}. "
                        "Use --allow_generative_fallback to permit historical replay fallback."
                    ) from e
                hist_idx = args.n_cond_msgs + step_i
                if hist_idx >= int(m_seq_raw_all.shape[0]):
                    break
                first_msg = m_seq_raw_all[hist_idx]
                sim_msg = inference.msg_to_jnp(first_msg)
        else:
            hist_idx = args.n_cond_msgs + step_i
            if hist_idx >= int(m_seq_raw_all.shape[0]):
                break
            first_msg = m_seq_raw_all[hist_idx]
            sim_msg = inference.msg_to_jnp(first_msg)

        sim_state_after_step = sim.process_order_array(sim_state_after_action, sim_msg)
        bid_after_step, ask_after_step = _best_quotes(sim, sim_state_after_step)
        if not book_quotes_valid(bid_after_step, ask_after_step):
            sim_state_after_step = sim_state_after_action
            bid_after_step, ask_after_step = bid_after_action, ask_after_action

        mid_bef = (float(bid_before) + float(ask_before)) / 2.0
        mid_aft = (float(bid_after_step) + float(ask_after_step)) / 2.0

        step_pnl = _compute_agent_pnl_from_trades(
            sim_state_after_step.trades,
            trader_id=int(agent_params.trader_id),
            tick_size=int(world_cfg.tick_size),
            final_midprice=mid_aft,
        )
        agent_state = MMEnvState(
            posted_distance_bid=0,
            posted_distance_ask=0,
            inventory=int(round(step_pnl["inventory"])),
            total_PnL=float(step_pnl["total_pnl"]),
            cash_balance=float(step_pnl["cash_pnl"]),
        )

        action_int = int(action_this_step)
        action_hist[action_int] = action_hist.get(action_int, 0) + 1
        pnl_trace.append(float(step_pnl["total_pnl"]))
        inventory_trace.append(float(step_pnl["inventory"]))
        mid_before_action.append(mid_bef)
        mid_after_action.append(mid_aft)

        step_rows.append(
            [
                int(step_i + 1),
                action_int,
                int(bid_before),
                int(ask_before),
                int(bid_after_action),
                int(ask_after_action),
                int(bid_after_step),
                int(ask_after_step),
                float(mid_aft),
                int(first_msg[1]) if len(first_msg) > 1 else -1,
                int(first_msg[4]) if len(first_msg) > 4 else -1,
                int(first_msg[5]) if len(first_msg) > 5 else -1,
                float(step_pnl["total_pnl"]),
                float(step_pnl["inventory"]),
            ]
        )

        msg_time_s = int(first_msg[8]) if len(first_msg) > 8 else -1
        msg_time_ns = int(first_msg[9]) if len(first_msg) > 9 else -1
        if msg_time_s >= 0 and msg_time_ns >= 0:
            current_world_time = jnp.array([msg_time_s, msg_time_ns], dtype=jnp.int32)

        current_sim_state = sim_state_after_step

    final_mid = mid_after_action[-1] if mid_after_action else math.nan
    final_pnl = _compute_agent_pnl_from_trades(
        current_sim_state.trades,
        trader_id=int(agent_params.trader_id),
        tick_size=int(world_cfg.tick_size),
        final_midprice=float(final_mid) if final_mid == final_mid else 0.0,
    )

    metrics = build_phase1_metrics(
        final_pnl=final_pnl,
        pnl_trace=pnl_trace,
        inventory_trace=inventory_trace,
        mid_before_action=mid_before_action,
        mid_after_action=mid_after_action,
    )

    summary = {
        "run_name": artifacts.run_dir.name,
        "run_dir": str(artifacts.run_dir),
        "world_model_mode": wm_sel.mode,
        "policy_mode": pol_sel.mode,
        "policy_handoff": handoff if handoff is not None else None,
        "checkpoint_path": str(ckpt_path) if ckpt_path is not None else "",
        "checkpoint_step": int(step) if step is not None else None,
        "data_dir": str(data_dir),
        "dataset_effective_dir": str(selected_data_dir),
        "sample_index": int(idx),
        "seed": int(args.seed),
        "n_steps_requested": int(args.n_steps),
        "n_steps_executed": int(len(step_rows)),
        "runtime": _runtime_metadata(args, selected_data_dir, jax_backend, jax_devices),
        "generation": {
            "strict_mode": bool(strict_generation),
            "fallback_allowed": bool(not strict_generation),
            "fallback_used": bool(generation_fallback_count > 0),
            "fallback_count": int(generation_fallback_count),
            "last_error": generation_error_last,
        },
        "action_histogram": {str(k): int(v) for k, v in sorted(action_hist.items())},
        "metrics": metrics,
        "pnl_trace": pnl_trace,
        "inventory_trace": inventory_trace,
    }

    _write_csv(
        artifacts.step_trace_csv,
        [
            "step_index",
            "action_taken",
            "best_bid_before_action",
            "best_ask_before_action",
            "best_bid_after_action",
            "best_ask_after_action",
            "best_bid_after_step",
            "best_ask_after_step",
            "midprice_after_step",
            "world_event_type",
            "world_price",
            "world_size",
            "total_pnl",
            "inventory",
        ],
        step_rows,
    )

    artifacts.summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[LOBArena] Summary written: {artifacts.summary_json}")

    if temp_data_ctx is not None:
        temp_data_ctx.cleanup()
    return summary


def run_batch_evaluation(args, eval_runner=None) -> int:
    candidates, fairness_overrides = resolve_batch_candidates(args)
    if not candidates:
        raise ValueError("No batch candidates provided")

    base_run_name = args.run_name.strip() if args.run_name.strip() else time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser().resolve()
    batch_root = output_root / base_run_name
    batch_root.mkdir(parents=True, exist_ok=True)

    shared_seed = args.seed if args.seed is not None else fairness_overrides.get("seed")
    shared_start_date = args.start_date if args.start_date else (fairness_overrides.get("start_date") or "")
    shared_end_date = args.end_date if args.end_date else (fairness_overrides.get("end_date") or "")
    runner = eval_runner or _run_single_evaluation

    batch_rows = []
    for i, candidate in enumerate(candidates, start=1):
        candidate_args = copy.deepcopy(args)
        candidate_args.policy_handoff = candidate.policy_handoff_path
        candidate_args.policy_handoff_batch = None
        candidate_args.policy_handoff_manifest = ""
        candidate_args.output_root = str(batch_root)
        candidate_args.run_name = f"{i:03d}_{_sanitize_run_suffix(candidate.candidate_id)}"
        candidate_args.seed = shared_seed
        candidate_args.start_date = shared_start_date
        candidate_args.end_date = shared_end_date
        summary = runner(candidate_args)
        metrics = summary.get("metrics", {})
        pnl = metrics.get("pnl", {})
        batch_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "policy_handoff": candidate.policy_handoff_path,
                "run_name": summary.get("run_name", candidate_args.run_name),
                "run_dir": summary.get("run_dir", ""),
                "summary_path": str(Path(summary.get("run_dir", "")) / "summary.json"),
                "total_pnl": float(pnl.get("total_pnl", 0.0)),
                "metrics": metrics,
            }
        )

    ranked_rows = sorted(batch_rows, key=_ranking_key)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank_baseline"] = rank

    batch_summary = {
        "batch_run_name": base_run_name,
        "batch_root": str(batch_root),
        "world_model_mode": args.world_model,
        "n_candidates": len(ranked_rows),
        "shared_fairness": {
            "seed": shared_seed,
            "start_date": shared_start_date,
            "end_date": shared_end_date,
        },
        "candidates": ranked_rows,
    }
    batch_summary_path = batch_root / "batch_summary.json"
    batch_summary_path.write_text(json.dumps(batch_summary, indent=2))
    print(f"[LOBArena] Batch summary written: {batch_summary_path}")
    return 0


def main() -> int:
    args = parse_args()
    enforce_single_node_context(context_name="phase2 evaluation entrypoint", args=args)
    _configure_runtime(args)
    if _is_batch_mode(args):
        return run_batch_evaluation(args)

    _run_single_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
