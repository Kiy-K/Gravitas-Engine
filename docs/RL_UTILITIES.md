# RL Utilities

Three utility modules for reinforcement learning with GRAVITAS Engine:

- `FogOfWarWrapper` — observation noise scaled by polarization
- `ExhaustionMonitor` — SB3 callback tracking exhaustion during training
- `TopologyVisualizer` — matplotlib/networkx network graph for GravitasWorld

---

## FogOfWarWrapper (`extensions/fog_of_war/wrapper.py`)

A `gymnasium.Wrapper` that adds polarization-scaled Gaussian noise to observations, simulating Clausewitz's fog of war: information quality degrades as systemic polarization rises.

### Noise Model

```
noise_std = bias_scale × (1 + Π)
```

Where `Π` is the current global polarization (read from `env.world.global_state.polarization`, range [0,1]).

At `Π=0` (no polarization): noise std = `bias_scale`.
At `Π=1` (maximum polarization): noise std = `2 × bias_scale`.

This is layered **on top of** the media bias already baked into `GravitasEnv`. They serve different purposes:
- **GravitasEnv internal bias**: structured, directional, per-cluster distortion
- **FogOfWarWrapper**: unstructured, isotropic chaos noise

### Usage

```python
from gravitas_engine.agents.gravitas_env import GravitasEnv
from extensions.fog_of_war.wrapper import FogOfWarWrapper

env = GravitasEnv()
env = FogOfWarWrapper(env, bias_scale=0.1)

obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env` | — | A `GravitasEnv` instance |
| `bias_scale` | `0.1` | Base noise magnitude; total std = `bias_scale × (1 + Π)` |

Observations are clipped to the declared `observation_space` bounds after noise is applied. Note: GravitasEnv observations legitimately span `[-2, 6]`, not `[0, 1]`.

---

## ExhaustionMonitor (`extensions/exhaustion/monitor.py`)

A Stable Baselines3 `BaseCallback` that monitors global exhaustion during training and logs warnings when it exceeds a critical threshold.

### What it tracks

| Attribute | Description |
|-----------|-------------|
| `exhaustion_history` | Raw exhaustion value per step |
| `n_critical_steps` | Steps where `E > critical_threshold` |
| `peak_exhaustion` | Highest exhaustion seen during training |

TensorBoard metrics: `gravitas/exhaustion`, `gravitas/peak_exhaustion` (logged each step).

### Usage

```python
from stable_baselines3 import PPO
from extensions.exhaustion.monitor import ExhaustionMonitor

monitor = ExhaustionMonitor(
    penalty=0.1,            # propagated to env.update_config() if supported
    critical_threshold=0.8, # log warning above this level
    verbose=1,              # 0=silent, 1=warnings, 2=all steps
)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100_000, callback=monitor)
```

### Training Summary (printed at end)

```
[ExhaustionMonitor] Training summary:
  Mean exhaustion   : 0.342
  Peak exhaustion   : 0.891
  Critical fraction : 4.2% (4200 steps)
```

### Design note

Actual **reward shaping** for exhaustion should be done inside the environment (`env.update_config(exhaustion_penalty=...)`) rather than in this callback. SB3's rollout buffer rewards are read-only by the time `_on_step` is called. This callback is a **monitor + logger** only; the penalty value is passed to the env via `update_config` at training start.

---

## TopologyVisualizer (`extensions/topology/visualizer.py`)

Matplotlib/NetworkX visualization of GravitasWorld cluster topology.

### `plot_topology(world, ...)`

Renders a network graph where:
- **Node color**: cluster stability σ (green = stable, red = unstable; uses RdYlGn colormap)
- **Node size**: hazard h (larger = higher hazard)
- **Node labels**: cluster index, σ, h values
- **Green dashed edges**: proximity/trade links (from `world.adjacency`)
- **Red solid edges**: conflict linkage (from `world.conflict`; thickness = conflict weight)

Layout is computed via NetworkX spring layout (seeded at 42 for reproducibility). Falls back to circular layout if the graph has no edges.

```python
from extensions.topology.visualizer import plot_topology
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
plot_topology(world, ax=ax, show_conflict=True, show_proximity=True)
plt.show()
```

**Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `world` | — | A `GravitasWorld` instance |
| `ax` | `None` | Matplotlib Axes; created if None |
| `show_conflict` | `True` | Draw conflict edges |
| `show_proximity` | `True` | Draw proximity/trade edges |
| `title` | `None` | Plot title; defaults to "Cluster Topology (Step N)" |

**Returns**: The matplotlib `Axes` object.

### `plot_global_timeseries(world_history, ...)`

Plots global state variables over time from a list of `GravitasWorld` snapshots.

Traces plotted:
- **Exhaustion (E)** — orange
- **Polarization (Π)** — red
- **Fragmentation (Φ)** — purple
- **Coherence (Ψ)** — blue
- Horizontal dashed line at E = 0.6 (exhaustion threshold)

```python
from extensions.topology.visualizer import plot_global_timeseries

worlds = []  # collect world snapshots during episode
for _ in range(200):
    obs, reward, term, trunc, info = env.step(action)
    worlds.append(env.world)

plot_global_timeseries(worlds)
plt.show()
```

**Dependencies**: `matplotlib`, `networkx`. Install with:
```bash
pip install matplotlib networkx
```
