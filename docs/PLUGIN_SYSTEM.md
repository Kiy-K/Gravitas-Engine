# Plugin System Guide

The GRAVITAS plugin system enables modular, hot-pluggable extensions to the simulation engine. Plugins hook into the step loop to modify world state, inject historical mechanics, or log events — without touching core engine code.

## Table of Contents

- [Overview](#overview)
- [Writing a Plugin](#writing-a-plugin)
- [Plugin Lifecycle](#plugin-lifecycle)
- [Configuration](#configuration)
- [Built-in Plugins](#built-in-plugins)
- [Event Logging](#event-logging)
- [Advanced Usage](#advanced-usage)

## Overview

Plugins are Python classes that inherit from `GravitasPlugin` and implement the `on_step(world, turn)` method. They are:

- **Standalone**: No circular imports — plugins only depend on `gravitas_engine.core`.
- **Configurable**: Per-plugin settings via `configs/custom.yaml`.
- **Discoverable**: Automatically found by the engine from the config or CLI.
- **Composable**: Multiple plugins run sequentially on each step.

## Writing a Plugin

Create a new file in `gravitas/plugins/` (e.g., `my_mechanic.py`):

```python
from gravitas.plugins import GravitasPlugin

class Plugin(GravitasPlugin):
    """Your plugin must export a class named 'Plugin'."""

    name = "my_mechanic"
    description = "Brief description of what this plugin does."
    version = "1.0.0"

    DEFAULTS = {
        "some_param": 42,
        "threshold": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._cfg = {**self.DEFAULTS, **(config or {})}

    def on_step(self, world, turn, **kwargs):
        """Called every simulation step. Must return world."""
        from gravitas_engine.core.gravitas_state import ClusterState

        c_arr = world.cluster_array()
        N = world.n_clusters

        # Your mechanic logic here
        if turn % self._cfg["some_param"] == 0:
            # Modify cluster state
            c_arr_mod = c_arr.copy()
            c_arr_mod[0, 2] += 0.01  # Boost resources in cluster 0
            new_clusters = [
                ClusterState.from_array(c_arr_mod[i]) for i in range(N)
            ]
            world = world.copy_with_clusters(new_clusters)
            self.log_event(turn, "Applied my mechanic!")

        return world
```

Then register it in `gravitas/plugins/__init__.py`:

```python
_BUILTIN_PLUGINS: Dict[str, str] = {
    "nonlinear_combat": "gravitas.plugins.nonlinear_combat",
    "logistics_network": "gravitas.plugins.logistics_network",
    "partisan_warfare": "gravitas.plugins.partisan_warfare",
    "soviet_reinforcements": "gravitas.plugins.soviet_reinforcements",
    "axis_airlift": "gravitas.plugins.axis_airlift",
    "my_mechanic": "gravitas.plugins.my_mechanic",  # Add this
}
```

## Plugin Lifecycle

Plugins have four lifecycle hooks:

| Hook | When Called | Required | Purpose |
| ---- | ---------- | -------- | ------- |
| `on_init(world)` | Engine initialization | No | One-time setup |
| `on_reset(world)` | Episode start | No | Reset internal counters |
| `on_step(world, turn)` | After each env step | **Yes** | Core mechanic logic |
| `on_episode_end(world, turn)` | Episode termination | No | Cleanup, summary logging |

### Execution Order

1. Environment `step()` runs (actions, ODE integration, shocks, media/alliance/pop/economy updates).
2. Each plugin's `on_step()` is called **in order** (as listed in config).
3. Plugins receive the world **after** all engine updates.
4. Each plugin can modify the world; the next plugin sees the modified state.

This means plugin order matters. Put prerequisite plugins first in the config.

## Configuration

### Via `configs/custom.yaml`

```yaml
scenario: stalingrad
plugins:
  - nonlinear_combat
  - logistics_network

plugin_configs:
  nonlinear_combat:
    terrain_modifier: 1.2
    fortification_bonus: 0.35
  logistics_network:
    disruption_threshold: 0.4
    repair_rate: 0.02
```

### Via Python API

```python
from gravitas import GravitasEngine

engine = GravitasEngine(scenario="stalingrad")
engine.load_plugins(
    ["nonlinear_combat"],
    plugin_configs={
        "nonlinear_combat": {
            "trigger_turn_interval": 1,
            "breakthrough_threshold": 0.58,
        }
    }
)
```

### Via CLI

```bash
python cli.py run stalingrad --plugins nonlinear_combat logistics_network
```

## Built-in Plugins

### `nonlinear_combat`

Applies nonlinear combat dynamics (Lanchester square-law style balance, diminishing returns, breakthrough sigmoids, combat fatigue, terrain multipliers).

### `logistics_network`

Adds supply-network dynamics with nonlinear production/consumption, saturating logistics flow, and disruption cascades.

### `partisan_warfare`

Models autonomous partisan behavior (recruitment, sabotage, ambushes, movement, detection) as a stochastic third force.

### `soviet_reinforcements`

**Historical basis**: Soviet forces received nightly reinforcements via barge crossings over the Volga River. These resupply operations were only possible when the crossing points were securely held.

**Mechanic**: Every N steps, if the Volga Crossing sector is stable (σ > threshold) and not under heavy fire (hazard < threshold), the Soviet Strategic Reserve receives military and resource boosts.

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `trigger_turn_interval` | 50 | Check frequency (turns) |
| `volga_cluster` | 3 | Volga Crossing sector index |
| `reserve_cluster` | 6 | Soviet Reserve sector index |
| `sigma_threshold` | 0.5 | Min stability for crossing |
| `hazard_threshold` | 0.7 | Max hazard before barges blocked |
| `military_boost` | 0.10 | Military increase per delivery |
| `resource_boost` | 0.05 | Resource increase per delivery |
| `military_cap` | 0.95 | Max military (prevent invincibility) |

### `axis_airlift`

**Historical basis**: After Soviet encirclement of the German 6th Army, Göring promised resupply by air. The Luftwaffe delivered only a fraction of the required 300 tons/day, with losses mounting over time.

**Mechanic**: Every N steps, if the Axis Supply Corridor is not collapsed, a diminishing resource boost is delivered to encircled Axis sectors. The boost decays each delivery to simulate historical degradation.

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `trigger_turn_interval` | 40 | Check frequency (turns) |
| `supply_corridor_cluster` | 5 | Axis Supply Corridor index |
| `target_clusters` | [0, 1, 2] | Encircled sectors to supply |
| `sigma_threshold` | 0.3 | Min corridor stability for flights |
| `hazard_threshold` | 0.8 | Max corridor hazard for flights |
| `base_resource_boost` | 0.04 | Starting resource boost |
| `decay_rate` | 0.005 | Boost reduction per delivery |

## Event Logging

Plugins can log structured events via `self.log_event()`:

```python
self.log_event(
    turn=150,
    message="Volga reinforcements arrive!",
    data={"military_before": 0.54, "military_after": 0.64},
)
```

Events are collected per-episode and returned in the results:

```python
results = engine.run(episodes=5)
for r in results:
    for event in r["plugin_events"]:
        print(f"Turn {event['turn']}: [{event['plugin']}] {event['message']}")
```

## Advanced Usage

### Disabling a Plugin at Runtime

```python
for plugin in engine.plugins:
    if plugin.name == "nonlinear_combat":
        plugin.enabled = False
```

### Accessing Plugin State

```python
for plugin in engine.plugins:
    if plugin.name == "logistics_network":
        print(f"Tracked sectors: {len(plugin._sector_production)}")
```

### Custom Plugin Discovery

For plugins outside the built-in directory, import and instantiate directly:

```python
from my_custom_plugins import WeatherPlugin

weather = WeatherPlugin(config={"blizzard_chance": 0.1})
engine.plugins.append(weather)
```
