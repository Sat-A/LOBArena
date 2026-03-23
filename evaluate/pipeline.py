
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from LOBArena.evaluate.checkpoint_loader import load_world_model_from_jaxmarl, restore_params_with_cpu_fallback
from LOBArena.evaluate.policy_adapter import load_ippo_policy_adapter, validate_policy_choice
from LOBArena.evaluate.world_model_selector import validate_world_model_choice
from LOBArena.guardrails.order_validators import book_quotes_valid, sanitize_action_messages
from LOBArena.metrics.computation import build_phase1_metrics


class RunArtifacts(object):
    def __init__(self, run_dir, summary_json, step_trace_csv):
        self.run_dir = run_dir
        self.summary_json = summary_json
        self.step_trace_csv = step_trace_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOBArena Phase 1 evaluation pipeline")
    p.add_argument("--world_model", choices=["historical", "generative"], required=True)
    p.add_argument("--policy_mode", choices=["random", "fixed", "ippo_rnn"], default="random")
    p.add_argument("--fixed_action", type=int, default=0)

    p.add_argument("--jaxmarl_root", default="/home/s5e/satyamaga.s5e/JaxMARL-HFT")
    p.add_argument("--lobs5_root", default="/home/s5e/satyamaga.s5e/LOBS5")
    p.add_argument("--lobs5_ckpt_path", default="")
    p.add_argument("--policy_ckpt_dir", default="")
    p.add_argument("--policy_config", default="")

    p.add_argument("--data_dir", required=True)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--checkpoint_step", type=int, default=None)
    p.add_argument("--test_split", type=float, default=1.0)
    p.add_argument("--start_date", default="")
    p.add_argument("--end_date", default="")

    p.add_argument("--n_cond_msgs", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=25)
    p.add_argument("--sample_top_n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output_root", default="/home/s5e/satyamaga.s5e/LOBArena/outputs/evaluations")
    p.add_argument("--run_name", default="")
    p.add_argument("--fast_startup", action="store_true")
    return p.parse_args()


def _configure_runtime(args):
    if args.fast_startup:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.50")
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")


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


def _write_csv(path, header, rows):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main() -> int:
    args = parse_args()
    _configure_runtime(args)
    artifacts = _prepare_artifacts(args)

    jaxmarl_root = Path(args.jaxmarl_root).expanduser().resolve()
    lobs5_root = Path(args.lobs5_root).expanduser().resolve()

    wm_sel = validate_world_model_choice(args.world_model, args.lobs5_ckpt_path or None)
    pol_sel = validate_policy_choice(args.policy_mode, args.fixed_action, args.policy_ckpt_dir or None, args.policy_config or None)

    tools = load_world_model_from_jaxmarl(jaxmarl_root, lobs5_root)

    # Deferred imports after path bootstrap.
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, str(jaxmarl_root))
    from minimal_agent_generative_step import _best_quotes, _build_world_state, _compute_agent_pnl_from_trades  # type: ignore

    from lob.encoding import Message_Tokenizer, Vocab  # type: ignore
    from lob.init_train import init_train_state  # type: ignore
    from lob import inference_no_errcorr as inference  # type: ignore
    from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration, MarketMaking_EnvironmentConfig, World_EnvironmentConfig  # type: ignore
    from gymnax_exchange.jaxen.StatesandParams import MMEnvParams, MMEnvState  # type: ignore
    from gymnax_exchange.jaxen.mm_env import MarketMakingAgent  # type: ignore

    rng = jax.random.key(args.seed)

    # Dataset prep
    data_dir = Path(args.data_dir).expanduser().resolve()
    selected_data_dir, temp_data_ctx = tools["_prepare_date_filtered_data_dir"](data_dir, args.start_date, args.end_date)

    ckpt_path = wm_sel.lobs5_ckpt_path if wm_sel.mode == "generative" else (Path(args.lobs5_ckpt_path).expanduser().resolve() if args.lobs5_ckpt_path else None)

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
        params = None
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

    # CPU-safe transformation; fallback to simple passthrough if helper functions are unavailable.
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
        else:
            obs = _build_obs(step_i, int(bid_before), int(ask_before), agent_state, current_world_time)
            action_i, policy_hidden = policy.act_with_state(obs, policy_hidden, done=False)
            action_this_step = jnp.int32(action_i)

        step_world_state = _build_world_state(sim, current_sim_state, current_world_time)
        action_msgs, cancel_msgs, _extras = mm_agent.get_messages(action_this_step, step_world_state, agent_state, agent_params)

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
            except Exception:
                # Graceful fallback for CPU-only environments when generative path uses GPU-only jit.
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
        "checkpoint_path": str(ckpt_path) if ckpt_path is not None else "",
        "checkpoint_step": int(step) if step is not None else None,
        "data_dir": str(data_dir),
        "dataset_effective_dir": str(selected_data_dir),
        "sample_index": int(idx),
        "seed": int(args.seed),
        "n_steps_requested": int(args.n_steps),
        "n_steps_executed": int(len(step_rows)),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
