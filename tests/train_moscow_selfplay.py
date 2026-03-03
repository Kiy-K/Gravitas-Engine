#!/usr/bin/env python3
"""
train_moscow_selfplay.py — Self-play training for Battle of Moscow (CoW-native).

Uses the CoW-native military system directly:
  - MilitaryWrapper with MultiDiscrete action space
  - Built-in production, building, research, combat, supply dynamics
  - No legacy plugin system dependency — all mechanics in military_dynamics

6-phase curriculum:
  Phase 1: Operation Typhoon   — Axis learns to advance and produce
  Phase 2: Mozhaisk Defense     — Soviet learns to hold and build
  Phase 3: General Winter       — Both sides, winter attrition amplified
  Phase 4: Partisan Escalation  — Both sides, contested territory focus
  Phase 5: Zhukov Counterattack — Both sides, Soviet reinforcement wave
  Phase 6: Final Self-Play      — Full dynamics, LR annealed

Usage:
    python tests/train_moscow_selfplay.py --total-rounds 20
    python tests/train_moscow_selfplay.py --resume-from logs/moscow_selfplay/phase1_round_003
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Project root on path ─────────────────────────────────────────────────── #
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from extensions.military.military_wrapper import MilitaryWrapper
from extensions.military.military_dynamics import (
    ActionType, N_ACTION_TYPES, N_UNIT_TYPES, N_BUILDING_TYPES,
    world_to_obs_array, obs_size, check_victory,
)
from extensions.military.cow_combat import CowTerrain, CowUnitType, cow_type_from_legacy

# ── Optional imports ──────────────────────────────────────────────────────── #
try:
    from sb3_contrib import RecurrentPPO
    HAS_RPPO = True
except ImportError:
    HAS_RPPO = False

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gymnasium_shim as gym          # type: ignore[no-redef]
    from gymnasium_shim import spaces     # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────────────────── #
# Constants                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

AXIS   = 0
SOVIET = 1
SIDE_NAMES = {AXIS: "Axis", SOVIET: "Soviet"}

# Terrain mapping: YAML terrain names → CowTerrain enum names
_TERRAIN_MAP = {
    "urban": "URBAN",
    "fortified": "URBAN",       # fortified → urban with bunker buildings
    "forest": "FOREST",
    "rail": "PLAINS",           # rail hub → plains
    "open": "PLAINS",
    "mountains": "MOUNTAINS",
}


# ─────────────────────────────────────────────────────────────────────────── #
# Scenario builder                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def build_moscow_scenario(yaml_path: str) -> Dict[str, Any]:
    """Parse moscow.yaml and build a scenario config dict for MilitaryWrapper."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    sectors = raw.get("sectors", [])
    n_clusters = len(sectors)

    # ── Adjacency from logistics_links ────────────────────────────────────
    links = raw.get("logistics_links", [])
    adj = np.eye(n_clusters, dtype=bool)
    for link in links:
        i, j = int(link["from"]), int(link["to"])
        if i < n_clusters and j < n_clusters:
            adj[i, j] = adj[j, i] = True

    # ── Terrains ──────────────────────────────────────────────────────────
    cluster_terrains = []
    for sec in sectors:
        t = sec.get("terrain", "open")
        cluster_terrains.append(_TERRAIN_MAP.get(t, "PLAINS"))

    # ── Owners ────────────────────────────────────────────────────────────
    cluster_owners: List[Optional[int]] = []
    for sec in sectors:
        side = sec.get("side", "").lower()
        if side == "axis":
            cluster_owners.append(AXIS)
        elif side == "soviet":
            cluster_owners.append(SOVIET)
        else:
            cluster_owners.append(None)

    # ── Buildings from CoW config or defaults ─────────────────────────────
    cow_cfg = raw.get("cow_military", {})
    cluster_buildings_raw = cow_cfg.get("cluster_buildings")
    if cluster_buildings_raw is None:
        # Generate defaults from terrain + sector role
        cluster_buildings_raw = []
        for sec in sectors:
            t = sec.get("terrain", "open")
            side = sec.get("side", "").lower()
            bld: Dict[str, int] = {}
            if t in ("urban", "fortified"):
                bld["BARRACKS"] = 2
                bld["BUNKER"] = 1 if t == "fortified" else 0
            elif t == "rail":
                bld["BARRACKS"] = 1
                bld["SUPPLY_DEPOT"] = 1
            elif side in ("axis", "soviet"):
                bld["BARRACKS"] = 1
            cluster_buildings_raw.append(bld)

    # ── Factions ──────────────────────────────────────────────────────────
    axis_clusters = [i for i, o in enumerate(cluster_owners) if o == AXIS]
    soviet_clusters = [i for i, o in enumerate(cluster_owners) if o == SOVIET]

    factions = [
        {
            "faction_id": AXIS,
            "name": "German Army Group Center",
            "doctrine": "AXIS",
            "controlled_clusters": axis_clusters,
            "base_income": cow_cfg.get("axis_income", [4.0, 3.5, 2.5, 2.0, 1.0]),
        },
        {
            "faction_id": SOVIET,
            "name": "Soviet Western Front",
            "doctrine": "COMINTERN",
            "controlled_clusters": soviet_clusters,
            "base_income": cow_cfg.get("soviet_income", [3.5, 4.0, 2.5, 1.5, 1.5]),
        },
    ]

    # ── Objectives ────────────────────────────────────────────────────────
    objectives = cow_cfg.get("objectives", [
        {"objective_id": 0, "name": "Capture Moscow",
         "objective_type": "capture", "target_cluster_id": 0,
         "faction_id": AXIS, "required_strength": 30},
        {"objective_id": 1, "name": "Hold Mozhaisk Line",
         "objective_type": "hold", "target_cluster_id": 2,
         "faction_id": SOVIET, "required_strength": 20,
         "hold_required": 30},
        {"objective_id": 2, "name": "Defend Moscow",
         "objective_type": "hold", "target_cluster_id": 0,
         "faction_id": SOVIET, "required_strength": 30,
         "hold_required": 50},
        {"objective_id": 3, "name": "Secure Vyazma Supply Line",
         "objective_type": "hold", "target_cluster_id": 5,
         "faction_id": AXIS, "required_strength": 15,
         "hold_required": 20},
    ])

    # ── Initial units ─────────────────────────────────────────────────────
    initial_units_raw = raw.get("initial_units", {})
    initial_units: Dict[int, List[Dict]] = {}
    for k, v in initial_units_raw.items():
        cluster_id = int(k)
        units_for_cluster: List[Dict] = []
        for defn in v:
            # Map unit type name to CowUnitType
            utype = defn.get("unit_type", "INFANTRY").upper()
            # Try direct enum lookup first, then legacy alias
            try:
                CowUnitType[utype]
            except KeyError:
                mapped = cow_type_from_legacy(utype)
                utype = mapped.name if mapped else "INFANTRY"

            faction_raw = defn.get("faction", 0)
            if isinstance(faction_raw, str):
                faction_id = AXIS if faction_raw.lower() == "axis" else SOVIET
            else:
                faction_id = int(faction_raw)

            units_for_cluster.append({
                "unit_type": utype,
                "count": int(defn.get("count", 1)),
                "faction": faction_id,
            })
        initial_units[cluster_id] = units_for_cluster

    # ── Faction modifiers (national spirit + government) ─────────────────
    faction_modifiers = cow_cfg.get("faction_modifiers", {})

    # ── Physics engine config ──────────────────────────────────────────
    map_physics = cow_cfg.get("map_physics")
    physics_enabled = map_physics is not None

    return {
        "n_clusters": n_clusters,
        "adjacency": adj.tolist(),
        "cluster_terrains": cluster_terrains,
        "cluster_owners": cluster_owners,
        "cluster_buildings": cluster_buildings_raw,
        "factions": factions,
        "objectives": objectives,
        "initial_units": initial_units,
        "faction_modifiers": faction_modifiers,
        "physics_enabled": physics_enabled,
        "map_physics": map_physics,
        # Pass raw for reference
        "_raw": raw,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Environment factory                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def make_moscow_env(
    scenario_cfg: Dict[str, Any],
    faction_id: int = AXIS,
    opponent_faction_id: int = SOVIET,
    opponent_policy: Optional[Any] = None,
    max_steps: int = 200,
    seed: int = 42,
) -> MilitaryWrapper:
    """Create a single MilitaryWrapper env for one faction."""
    env = MilitaryWrapper(
        scenario_cfg=scenario_cfg,
        faction_id=faction_id,
        opponent_faction_id=opponent_faction_id,
        opponent_policy=opponent_policy,
        max_steps=max_steps,
        max_clusters=12,
        seed=seed,
    )
    return env


def make_vec_env(
    scenario_cfg: Dict[str, Any],
    faction_id: int,
    opponent_faction_id: int,
    opponent_policy: Optional[Any] = None,
    n_envs: int = 4,
    max_steps: int = 200,
    seed: int = 42,
) -> VecNormalize:
    """Create vectorized + normalized env for SB3 training.

    Uses SubprocVecEnv for true multiprocess parallelism.
    """
    import tempfile as _tmpmod

    _opp_path: Optional[str] = None
    if opponent_policy is not None and hasattr(opponent_policy, "save"):
        _tmp = _tmpmod.NamedTemporaryFile(suffix=".zip", delete=False)
        _opp_path = _tmp.name
        _tmp.close()
        opponent_policy.save(_opp_path)

    def _make(i: int):
        def _thunk():
            opp_policy = None
            if _opp_path is not None:
                from sb3_contrib import RecurrentPPO as _RPPO
                opp_model = _RPPO.load(_opp_path, device="cpu")
                # Wrap model.predict as a callable policy
                def _opp_fn(obs):
                    import numpy as _np
                    act, _ = opp_model.predict(obs.reshape(1, -1), deterministic=False)
                    return act.flatten()
                opp_policy = _opp_fn
            env = make_moscow_env(
                scenario_cfg, faction_id=faction_id,
                opponent_faction_id=opponent_faction_id,
                opponent_policy=opp_policy, max_steps=max_steps,
                seed=seed + i * 1000,
            )
            return Monitor(env)
        return _thunk

    vec = SubprocVecEnv([_make(i) for i in range(n_envs)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
    vec._opponent_tmp_path = _opp_path  # type: ignore
    return vec


# ─────────────────────────────────────────────────────────────────────────── #
# Self-play training loop                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

def train_selfplay(args: argparse.Namespace) -> None:
    """
    6-phase self-play training for Battle of Moscow (CoW-native).

    Phase 1: Operation Typhoon   — 3 rounds, Axis vs random
    Phase 2: Mozhaisk Defense     — 3 rounds, Soviet vs frozen Axis
    Phase 3: General Winter       — 3 rounds, both sides
    Phase 4: Partisan Escalation  — 3 rounds, both sides
    Phase 5: Zhukov Counterattack — 4 rounds, both sides
    Phase 6: Final Self-Play      — 4 rounds, both sides, LR annealed
    """
    assert HAS_RPPO, "sb3-contrib required: pip install sb3-contrib"
    assert HAS_SB3,  "stable-baselines3 required"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load scenario
    regime_path = args.regime_file or str(_ROOT / "gravitas" / "scenarios" / "moscow.yaml")
    scenario_cfg = build_moscow_scenario(regime_path)

    # ── 6-Phase schedule ───────────────────────────────────────────────── #
    PHASES = [
        {
            "name": "Phase 1: Operation Typhoon (Axis Advance)",
            "rounds": 3, "steps": args.steps_per_round,
            "train_sides": [AXIS],
            "lr": args.lr, "ent_coef": 0.02,
            "max_steps": 150,
        },
        {
            "name": "Phase 2: Mozhaisk Defense Line (Soviet Defense)",
            "rounds": 3, "steps": args.steps_per_round,
            "train_sides": [SOVIET],
            "lr": args.lr, "ent_coef": 0.02,
            "max_steps": 150,
        },
        {
            "name": "Phase 3: General Winter (Both Sides)",
            "rounds": 3, "steps": args.steps_per_round,
            "train_sides": [AXIS, SOVIET],
            "lr": args.lr, "ent_coef": 0.015,
            "max_steps": 200,
        },
        {
            "name": "Phase 4: Partisan Escalation (Contested Focus)",
            "rounds": 3, "steps": args.steps_per_round,
            "train_sides": [AXIS, SOVIET],
            "lr": args.lr * 0.8, "ent_coef": 0.012,
            "max_steps": 200,
        },
        {
            "name": "Phase 5: Zhukov Counteroffensive (Soviet Push)",
            "rounds": 4, "steps": args.steps_per_round,
            "train_sides": [AXIS, SOVIET],
            "lr": args.lr * 0.5, "ent_coef": 0.01,
            "max_steps": 250,
        },
        {
            "name": "Phase 6: Final Self-Play (Full Dynamics)",
            "rounds": 4, "steps": args.steps_per_round,
            "train_sides": [AXIS, SOVIET],
            "lr": args.lr * 0.3, "ent_coef": 0.008,
            "max_steps": 300,
        },
    ]

    total_rounds = sum(p["rounds"] for p in PHASES)
    total_steps = sum(p["rounds"] * p["steps"] * len(p["train_sides"]) for p in PHASES)

    # ── Print banner ──────────────────────────────────────────────────────
    print("+" + "=" * 58 + "+")
    print("|    BATTLE OF MOSCOW — 6-PHASE SELF-PLAY (CoW-native)    |")
    print("+" + "=" * 58 + "+")
    print(f"  Clusters:       {scenario_cfg['n_clusters']}")
    print(f"  Action space:   MultiDiscrete (produce/upgrade/move/attack/build/research/reinforce/retreat)")
    print(f"  Total phases:   6")
    print(f"  Total rounds:   {total_rounds}")
    print(f"  Steps/round:    {args.steps_per_round}")
    print(f"  Total steps:    ~{total_steps:,}")
    print(f"  N envs:         {args.n_envs}")
    print(f"  LSTM hidden:    {args.lstm_hidden}")
    print(f"  Device:         {args.device}")
    print(f"  Log dir:        {log_dir}")
    print("+" + "-" * 58 + "+")
    for i, p in enumerate(PHASES, 1):
        sides = "+".join(SIDE_NAMES[s] for s in p["train_sides"])
        print(f"  P{i}: {p['rounds']}r x {p['steps']//1000}K  lr={p['lr']:.1e}  [{sides}]")
    print("+" + "=" * 58 + "+")

    # WandB
    if args.wandb and HAS_WANDB:
        wandb.init(
            project="gravitas-moscow-cow",
            config={**vars(args), "n_phases": 6, "total_rounds": total_rounds},
            name=f"moscow_cow_6phase_{total_rounds}r",
        )

    # ── Initialize models ─────────────────────────────────────────────────
    axis_model: Optional[RecurrentPPO] = None
    soviet_model: Optional[RecurrentPPO] = None

    policy_kwargs = dict(
        lstm_hidden_size=args.lstm_hidden,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    _resume_from = args.resume_from

    # ── Phase loop ────────────────────────────────────────────────────────
    global_round = 0
    t_start = time.time()

    for phase_idx, phase in enumerate(PHASES):
        phase_num = phase_idx + 1
        phase_name = phase["name"]
        phase_lr = phase["lr"]
        phase_ent = phase["ent_coef"]
        phase_max_steps = phase["max_steps"]

        print(f"\n{'#' * 60}")
        print(f"  {phase_name}")
        print(f"  Rounds: {phase['rounds']}  Steps: {phase['steps']}  LR: {phase_lr:.1e}")
        print(f"{'#' * 60}")

        for rnd_in_phase in range(1, phase["rounds"] + 1):
            global_round += 1
            print(f"\n{'=' * 60}")
            print(f"  Round {global_round}/{total_rounds}  "
                  f"(Phase {phase_num}, round {rnd_in_phase}/{phase['rounds']})")
            print(f"{'=' * 60}")

            for side in phase["train_sides"]:
                side_name = SIDE_NAMES[side]
                opponent_side = SOVIET if side == AXIS else AXIS
                opponent_model = soviet_model if side == AXIS else axis_model
                current_model = axis_model if side == AXIS else soviet_model

                print(f"\n  Training {side_name} "
                      f"(vs {'model' if opponent_model else 'random'})")
                t0 = time.time()

                env = make_vec_env(
                    scenario_cfg,
                    faction_id=side,
                    opponent_faction_id=opponent_side,
                    opponent_policy=opponent_model,
                    n_envs=args.n_envs,
                    max_steps=phase_max_steps,
                    seed=args.seed + global_round * 100 + side * 50,
                )

                if current_model is None:
                    if _resume_from:
                        rdir = Path(_resume_from)
                        side_file = "axis_model.zip" if side == AXIS else "soviet_model.zip"
                        ckpt = rdir / side_file
                        if ckpt.exists():
                            current_model = RecurrentPPO.load(
                                str(ckpt), env=env, device=args.device,
                                verbose=0, learning_rate=phase_lr,
                                ent_coef=phase_ent,
                            )
                            print(f"    Resumed from {ckpt}")

                    if current_model is None:
                        current_model = RecurrentPPO(
                            "MultiInputLstmPolicy" if isinstance(
                                env.observation_space, spaces.Dict
                            ) else "MlpLstmPolicy",
                            env,
                            verbose=0,
                            learning_rate=phase_lr,
                            n_steps=args.n_steps,
                            batch_size=args.batch_size,
                            n_epochs=args.n_epochs,
                            gamma=args.gamma,
                            gae_lambda=0.95,
                            ent_coef=phase_ent,
                            clip_range=0.2,
                            max_grad_norm=0.5,
                            device=args.device,
                            policy_kwargs=policy_kwargs,
                        )
                else:
                    current_model.set_env(env)
                    current_model.learning_rate = phase_lr
                    current_model.ent_coef = phase_ent

                current_model.learn(
                    total_timesteps=phase["steps"],
                    reset_num_timesteps=False,
                    progress_bar=True,
                )
                elapsed = time.time() - t0
                print(f"  Done {side_name} ({elapsed:.1f}s)")

                # Cleanup
                _tmp_path = getattr(env, "_opponent_tmp_path", None)
                env.close()
                if _tmp_path and os.path.exists(_tmp_path):
                    os.unlink(_tmp_path)

                if side == AXIS:
                    axis_model = current_model
                else:
                    soviet_model = current_model

            # ── Save checkpoint ───────────────────────────────────────────
            ckpt_dir = log_dir / f"phase{phase_num}_round_{global_round:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            if axis_model is not None:
                axis_model.save(str(ckpt_dir / "axis_model"))
            if soviet_model is not None:
                soviet_model.save(str(ckpt_dir / "soviet_model"))
            print(f"  Saved -> {ckpt_dir}")

            # ── Quick eval ────────────────────────────────────────────────
            if (axis_model is not None and soviet_model is not None
                    and global_round % args.eval_every == 0):
                eval_results = quick_eval(
                    scenario_cfg, axis_model, soviet_model,
                    max_steps=phase_max_steps,
                    n_episodes=5, seed=args.seed + global_round,
                )
                print(f"\n  Eval (5 eps):")
                print(f"    Axis  R={eval_results['axis_reward_mean']:+.1f}")
                print(f"    Soviet R={eval_results['soviet_reward_mean']:+.1f}")
                print(f"    AvgLen: {eval_results['avg_length']:.0f}  "
                      f"Axis wins: {eval_results['axis_wins']}")

                if args.wandb and HAS_WANDB:
                    wandb.log({
                        "phase": phase_num,
                        "global_round": global_round,
                        "lr": phase_lr,
                        **{f"eval/{k}": v for k, v in eval_results.items()},
                    })

        elapsed_total = time.time() - t_start
        print(f"\n  {phase_name} complete  ({elapsed_total / 60:.1f}min total)")

    # ── Final save ────────────────────────────────────────────────────────
    if axis_model is not None:
        axis_model.save(str(log_dir / "axis_final"))
    if soviet_model is not None:
        soviet_model.save(str(log_dir / "soviet_final"))
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  MOSCOW 6-PHASE TRAINING COMPLETE (CoW-native)")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Models: {log_dir}/axis_final.zip, soviet_final.zip")
    print(f"{'=' * 60}")

    if args.wandb and HAS_WANDB:
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────── #
# Quick evaluation                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def quick_eval(
    scenario_cfg: Dict[str, Any],
    axis_model: Any,
    soviet_model: Any,
    max_steps: int = 200,
    n_episodes: int = 5,
    seed: int = 99999,
) -> Dict[str, float]:
    """Run evaluation episodes. Axis acts, Soviet is opponent."""
    import torch

    axis_rewards, soviet_rewards = [], []
    ep_lengths = []
    axis_wins = 0

    with torch.no_grad():
        for ep in range(n_episodes):
            # Wrap soviet model predict as opponent policy
            sov_lstm_state = None
            sov_starts = np.ones((1,), dtype=bool)

            def soviet_opp(obs):
                nonlocal sov_lstm_state, sov_starts
                act, sov_lstm_state = soviet_model.predict(
                    obs.reshape(1, -1), state=sov_lstm_state,
                    episode_start=sov_starts, deterministic=True,
                )
                sov_starts = np.zeros((1,), dtype=bool)
                return act.flatten()

            env = make_moscow_env(
                scenario_cfg, faction_id=AXIS,
                opponent_faction_id=SOVIET,
                opponent_policy=soviet_opp,
                max_steps=max_steps,
                seed=seed + ep,
            )

            obs, info = env.reset()
            ax_lstm = None
            ax_starts = np.ones((1,), dtype=bool)
            ax_total_r = 0.0
            sov_lstm_state = None
            sov_starts = np.ones((1,), dtype=bool)
            done, trunc = False, False
            steps = 0

            while not (done or trunc):
                ax_act, ax_lstm = axis_model.predict(
                    obs.reshape(1, -1), state=ax_lstm,
                    episode_start=ax_starts, deterministic=True,
                )
                obs, reward, done, trunc, info = env.step(ax_act.flatten())
                ax_starts = np.zeros((1,), dtype=bool)
                ax_total_r += reward
                steps += 1

            victory = info.get("victory", {})
            if victory.get("winner") == AXIS:
                axis_wins += 1

            axis_rewards.append(ax_total_r)
            soviet_rewards.append(-ax_total_r)
            ep_lengths.append(steps)

    return {
        "axis_reward_mean": float(np.mean(axis_rewards)),
        "soviet_reward_mean": float(np.mean(soviet_rewards)),
        "avg_length": float(np.mean(ep_lengths)),
        "axis_wins": axis_wins,
        "soviet_wins": n_episodes - axis_wins,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# CLI                                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Battle of Moscow Self-Play Training (CoW-native)")
    p.add_argument("--regime-file", type=str, default=None,
                   help="Path to moscow.yaml (auto-detected if None)")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Directory with round checkpoint to resume from")
    p.add_argument("--log-dir", type=str, default="logs/moscow_selfplay",
                   help="Output directory for models and logs")
    p.add_argument("--steps-per-round", type=int, default=50000,
                   help="Training steps per side per round")
    p.add_argument("--n-envs", type=int, default=8,
                   help="Parallel envs for training")
    p.add_argument("--n-steps", type=int, default=512,
                   help="Rollout steps per env per update")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=1,
                   help="Run quick eval every N rounds")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_selfplay(parse_args())
