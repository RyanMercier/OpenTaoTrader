"""Train a PPO policy across multiple Bittensor subnets.

Adaptation of train_ppo.py from
https://github.com/ZiadFrancis/Reinforcement_Trading_Part_2

We collapse their MultiDiscrete (direction × SL × TP) action space into
a long-only Discrete(3) since the AMM doesn't support shorts, and use a
SubprocVecEnv across subnets so PPO sees mixed trajectories in one run.

Two-phase split:
  - Training subnets cycle through episodes of historical bars in the
    train window (default Apr 15 → May 31 2026).
  - Validation: a separate held-out window (Jun 1 → Jun 21) is replayed
    deterministically with the learned policy and per-subnet PnL is
    reported. The model is saved regardless; the operator decides
    whether the OOS numbers warrant live deployment.

Usage:
    python -m trading.rl.train_ppo \\
        --db /home/ryan/bittensor/OpenTaoAPI/TaoOpenAPI/data/opentao.db \\
        --train-start 2026-04-15 --train-end 2026-05-31 \\
        --val-start 2026-06-01 --val-end 2026-06-21 \\
        --subnets 1,8,19,51,64,95,120 \\
        --total-steps 200000 \\
        --output models/ppo_bittensor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Make the package importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from trading.data import DataLoader
from trading.rl.bittensor_env import BittensorSubnetEnv, EnvConfig, make_env_factory


def _load_snapshots(db_path: str, subnets: list[int], start: str, end: str):
    loader = DataLoader(db_path)
    all_snaps = loader.load_all_snapshots(start=start, end=end, netuids=subnets)
    # Filter to subnets that actually have enough data for features
    out = {}
    for nid in subnets:
        snaps = all_snaps.get(nid, [])
        if len(snaps) < 400:  # need >= ~7d of 30-min bars for stable features
            print(f"  SN{nid}: skipped, only {len(snaps)} bars")
            continue
        out[nid] = snaps
    return out


def build_vec_env(train_data: dict[int, list], cfg: EnvConfig, num_envs: int):
    """Round-robin assign training subnets to env workers."""
    nids = list(train_data.keys())
    if not nids:
        raise RuntimeError("No training subnets with sufficient data")
    factories = []
    for i in range(num_envs):
        nid = nids[i % len(nids)]
        factory = make_env_factory(train_data[nid], cfg)
        # Wrap with Monitor so SB3 sees episode rewards/lengths
        factories.append(lambda f=factory: Monitor(f()))
    if num_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories)


def evaluate_on_subnet(model, snapshots, cfg: EnvConfig) -> dict:
    """Run the policy deterministically through one subnet's OOS bars."""
    env = BittensorSubnetEnv(snapshots, cfg)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    n_trades = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        total_reward += reward
        if info.get("entered") or info.get("exited"):
            n_trades += 1
        done = term or trunc
    final_value = env._portfolio_value(snapshots[-1])
    return {
        "final_value": final_value,
        "return_pct": (final_value / cfg.initial_capital - 1) * 100.0,
        "trades": n_trades,
        "total_reward": total_reward,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--train-start", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-start", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--subnets", default="1,8,19,51,64,95,120",
                   help="Comma-separated netuids to train on")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--capital", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="models/ppo_bittensor")
    args = p.parse_args()

    subnets = [int(x.strip()) for x in args.subnets.split(",") if x.strip()]
    print(f"Loading data for subnets {subnets}...")
    train_data = _load_snapshots(args.db, subnets, args.train_start, args.train_end)
    val_data = _load_snapshots(args.db, subnets, args.val_start, args.val_end)
    print(f"Train subnets with data: {sorted(train_data.keys())}")
    print(f"Val subnets with data:   {sorted(val_data.keys())}")

    cfg = EnvConfig(initial_capital=args.capital)
    vec_env = build_vec_env(train_data, cfg, args.num_envs)

    print(f"\nTraining PPO for {args.total_steps:,} steps on {args.num_envs} parallel envs...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,           # encourage exploration; pure greedy can collapse on flat reward
        clip_range=0.2,
        verbose=1,
        seed=args.seed,
        device="auto",
    )
    model.learn(total_timesteps=args.total_steps, progress_bar=False)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"\nSaved model to {args.output}.zip")

    print("\nOOS evaluation per subnet:")
    val_results = {}
    for nid, snaps in val_data.items():
        if len(snaps) < 50:
            continue
        res = evaluate_on_subnet(model, snaps, cfg)
        val_results[nid] = res
        print(f"  SN{nid}: final={res['final_value']:.2f} TAO ({res['return_pct']:+.2f}%), trades={res['trades']}")

    summary = {
        "subnets_trained": sorted(train_data.keys()),
        "subnets_evaluated": sorted(val_data.keys()),
        "train_window": [args.train_start, args.train_end],
        "val_window": [args.val_start, args.val_end],
        "total_steps": args.total_steps,
        "per_subnet_val": val_results,
        "model_path": f"{args.output}.zip",
    }
    summary_path = f"{args.output}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    # Average OOS return as quick fitness number
    if val_results:
        avg = sum(r["return_pct"] for r in val_results.values()) / len(val_results)
        print(f"Average OOS return across {len(val_results)} subnets: {avg:+.2f}%")


if __name__ == "__main__":
    main()
