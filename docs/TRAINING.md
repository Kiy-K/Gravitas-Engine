# Training Guide

How to train RL agents with GRAVITAS Engine using PPO and RecurrentPPO.

## Overview

Training scripts in this repository:

| Script | Environment | Algorithm | Use Case |
|--------|------------|-----------|---------|
| `tests/train_rppo.py` | `GravitasEnv` (single-agent) | RecurrentPPO | Regime stabilization |
| `tests/train_stalingrad_selfplay.py` | `SelfPlayEnv` (multi-agent) | RecurrentPPO | Battle of Stalingrad |

All scripts use `sb3-contrib`'s `RecurrentPPO` (LSTM policies) because the environments are partially observable — agents only see their own sectors clearly, and temporal context matters for long-horizon strategies.

## Dependencies

```bash
pip install stable-baselines3 sb3-contrib
# Optional for TensorBoard logging:
pip install tensorboard
```

## Single-Agent Training (`train_rppo.py`)

Trains a single PPO agent on `GravitasEnv` to stabilize a political system.

```bash
python tests/train_rppo.py
```

The agent controls governance stances (7 discrete options) and is rewarded for:
- High average cluster stability σ
- Low exhaustion E
- Low fragmentation Φ

Models are saved to `logs/` as `.zip` files (SB3 format).

## Multi-Agent Self-Play (`train_stalingrad_selfplay.py`)

Trains Axis and Soviet agents alternately via iterative self-play on the Stalingrad scenario.

```bash
# Basic run
python tests/train_stalingrad_selfplay.py --total-rounds 20

# Resume from checkpoint
python tests/train_stalingrad_selfplay.py \
    --resume-from logs/stalingrad_selfplay/round_005

# Full options
python tests/train_stalingrad_selfplay.py \
    --total-rounds 6 \
    --steps-per-round 25000 \
    --n-envs 4 \
    --log-dir logs/stalingrad_selfplay
```

Each round trains one side against a frozen opponent, then swaps sides.

### How Self-Play Works

`SelfPlayEnv` wraps `StalingradMultiAgentEnv` to expose a single-agent Gymnasium interface:
- The **training side** takes actions and receives rewards
- The **frozen side** runs the previously saved policy
- After each round, the frozen policy is updated with the latest checkpoint

This means each side learns to beat an increasingly competent opponent.

## Battle Evaluation

After training, evaluate trained models:

```bash
python tests/eval_stalingrad_battle.py \
    --axis-model logs/stalingrad_selfplay/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay/soviet_final.zip \
    --n-episodes 30
```

Outputs per-episode stats: winner, turns survived, final stability scores, shock events triggered.

## Running Trained Models via CLI

```bash
python cli.py run stalingrad --episodes 30 \
    --axis-model logs/stalingrad_selfplay/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay/soviet_final.zip

# Detailed battle replay
python cli.py replay \
    --axis-model logs/stalingrad_selfplay/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay/soviet_final.zip
```

## Observation Space

Per-side observations are padded fixed-size vectors. For the current Gravitas/Stalingrad stack, the shape is:

`(10 * n_clusters_max + 8 + action_dim,)`

Agents see distorted values (via `media_bias`) rather than raw `ClusterState` — this is by design. Use `FogOfWarWrapper` to add additional polarization-scaled noise on top:

```python
from extensions.fog_of_war.wrapper import FogOfWarWrapper
from gravitas_engine.agents.gravitas_env import GravitasEnv

env = GravitasEnv()
env = FogOfWarWrapper(env, bias_scale=0.05)
```

## Monitoring Exhaustion During Training

Use `ExhaustionMonitor` to track and warn when global exhaustion exceeds the critical threshold:

```python
from extensions.exhaustion.monitor import ExhaustionMonitor
from sb3_contrib import RecurrentPPO

monitor = ExhaustionMonitor(critical_threshold=0.8, verbose=1)
model = RecurrentPPO("MlpLstmPolicy", env)
model.learn(total_timesteps=500_000, callback=monitor)
```

At the end of training, a summary is printed with mean exhaustion, peak exhaustion, and the fraction of steps spent in the critical zone.

## Visualizing Results

Plot global state evolution after an episode:

```python
from extensions.topology.visualizer import plot_topology, plot_global_timeseries

# Topology snapshot
plot_topology(env.world)

# Time-series across an episode (collect world snapshots in a list)
plot_global_timeseries(world_history)
```

## Model Format

Models are saved in SB3 `.zip` format and can be loaded with:

```python
from sb3_contrib import RecurrentPPO

model = RecurrentPPO.load("logs/stalingrad_selfplay/axis_final.zip")
obs, info = env.reset()
lstm_states = None
episode_starts = True
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_starts = terminated or truncated
    if terminated or truncated:
        break
```
