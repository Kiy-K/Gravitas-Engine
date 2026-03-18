# Training Guide

How to train RL agents with GRAVITAS Engine using PPO and RecurrentPPO.

## Overview

Three training scripts are provided:

| Script | Environment | Algorithm | Use Case |
|--------|------------|-----------|---------|
| `tests/train_rppo.py` | `GravitasEnv` (single-agent) | RecurrentPPO | Regime stabilization |
| `tests/train_moscow_selfplay.py` | `SelfPlayEnv` (multi-agent) | RecurrentPPO | Battle of Moscow |
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

## Multi-Agent Self-Play (`train_moscow_selfplay.py`)

Trains Axis and Soviet agents alternately via a 6-phase curriculum on the Battle of Moscow scenario.

```bash
# Basic run
python tests/train_moscow_selfplay.py --total-rounds 20

# Resume from checkpoint
python tests/train_moscow_selfplay.py \
    --resume-from logs/moscow_selfplay/phase1_round_003

# Full options
python tests/train_moscow_selfplay.py \
    --total-rounds 6 \
    --steps-per-round 25000 \
    --n-envs 4 \
    --log-dir logs/moscow_selfplay
```

### 6-Phase Curriculum

| Phase | Name | Focus |
|-------|------|-------|
| 1 | Operation Typhoon | Axis learns to advance and produce |
| 2 | Mozhaisk Defense | Soviet learns to hold and build |
| 3 | General Winter | Both sides; winter attrition amplified |
| 4 | Partisan Escalation | Both sides; contested territory focus |
| 5 | Zhukov Counterattack | Both sides; Soviet reinforcement wave |
| 6 | Final Self-Play | Full dynamics; learning rate annealed |

Each phase runs for `--steps-per-round` steps per round, rotating which side is trained while the other runs a frozen policy.

### How Self-Play Works

`SelfPlayEnv` wraps `StalingradMultiAgentEnv` to expose a single-agent Gymnasium interface:
- The **training side** takes actions and receives rewards
- The **frozen side** runs the previously saved policy
- After each round, the frozen policy is updated with the latest checkpoint

This means each side learns to beat an increasingly competent opponent.

## Battle Evaluation

After training, evaluate trained models:

```bash
python tests/eval_moscow_battle.py \
    --axis-model logs/moscow_selfplay/axis_final.zip \
    --soviet-model logs/moscow_selfplay/soviet_final.zip \
    --n-episodes 30
```

Outputs per-episode stats: winner, turns survived, final stability scores, shock events triggered.

## Running Trained Models via CLI

```bash
python cli.py run moscow --episodes 30 \
    --axis-model logs/moscow_selfplay/axis_final.zip \
    --soviet-model logs/moscow_selfplay/soviet_final.zip

# Detailed battle replay
python cli.py replay \
    --axis-model logs/moscow_selfplay/axis_final.zip \
    --soviet-model logs/moscow_selfplay/soviet_final.zip
```

## Observation Space

Per-side observations are shaped `(5N + 8 + action_dim,)` where N = number of clusters.

Agents see distorted values (via `media_bias`) rather than raw `ClusterState` — this is by design. Use `FogOfWarWrapper` to add additional polarization-scaled noise on top:

```python
from extensions.fog_of_war.wrapper import FogOfWarWrapper

env = MilitaryWrapper(...)
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

model = RecurrentPPO.load("logs/moscow_selfplay/axis_final.zip")
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
