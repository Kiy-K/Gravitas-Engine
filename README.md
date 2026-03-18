# GRAVITAS Engine

> **G**overnance under **R**ecursive **A**nd **V**olatile **I**nstability **T**hrough **A**daptive **S**imulation

High-fidelity simulation and optimization of hierarchical political systems, adversarial multi-agent warfare, and systemic survival under spatial instability.

## Overview

GRAVITAS Engine is a research-grade simulation framework for modeling the dynamics of political regimes, military campaigns, and systemic collapse. It combines non-linear ODE dynamics, Hawkes shock processes, media bias modeling, population demographics, and a full economic subsystem into a unified Gymnasium environment.

The engine supports both **single-agent governance** (PPO stabilization) and **multi-agent adversarial warfare** (RecurrentPPO self-play), with a modular **plugin system** for extending simulation mechanics. It now includes a **real-time strategic map GUI** for visualizing the Air Strip One 1984 scenario.

## Key Features

- **CoW-Native Military System**: Full Call of War-style combat with 34 unit types, terrain bonuses, morale dynamics, physics integration, and per-sector land garrisons.
- **Physics-Driven Simulation**: Realistic terrain, weather, supply logistics, and line-of-sight modeling integrated with combat mechanics.
- **Advanced Unit Traits**: Elite units, terrain specialization, weather adaptation, engineering capabilities, and combat specials (suppression, breakthrough, ambush).
- **Multi-Agent Warfare**: Adversarial Axis vs. Soviet self-play with per-side observations, actions, and rewards (Battle of Stalingrad scenario).
- **Air Strip One 1984**: Three-faction strategic simulation across 35 sectors (British Isles, France, Benelux, Netherlands) with LLM-driven AI players.
- **Real-Time GUI**: Interactive strategic map viewer with fleet positions, land garrisons, BLF resistance, and war correspondent dispatches.
- **Plugin Architecture**: Modular, hot-pluggable simulation extensions with standardized `on_step(world, turn)` interface.
- **Historical Scenarios**: YAML-defined scenarios (Stalingrad 1942-43 and Air Strip One 1984).
- **Hierarchical Modeling**: Districts nested within provinces, with custom adjacency and diffusion rates.
- **Reinforcement Learning**: Trained PPO/RecurrentPPO agents for regime stabilization and adversarial warfare.
- **Economic Subsystem**: Per-cluster GDP, unemployment, debt, and industrial capacity with bidirectional military feedback.
- **Population & Military**: Multi-class demographics, ethnic tension, conscription/desertion dynamics, and 34 military unit types.
- **Spatial Dynamics**: Domino effects, cascade failures, and alliance diplomacy across geographical units.
- **CLI Interface**: `python cli.py run stalingrad --episodes 30` for quick scenario execution.

## Architecture

```text
GravitasEngine/
├── gravitas/                    # High-level orchestration
│   ├── __init__.py              # GravitasEngine + Engine alias
│   ├── engine.py                # Core engine with plugin loader
│   ├── plugins/                 # Plugin system
│   │   ├── __init__.py          # GravitasPlugin base class + discovery
│   │   ├── nonlinear_combat.py  # Nonlinear combat dynamics
│   │   ├── logistics_network.py # Supply network dynamics
│   │   ├── partisan_warfare.py  # Autonomous partisan mechanics
│   │   ├── soviet_reinforcements.py  # Volga barge crossing mechanic
│   │   └── axis_airlift.py      # Luftwaffe airlift mechanic
│   ├── scenarios/               # Scenario YAML files
│   │   └── airstrip_one.yaml    # 35-sector 1984 scenario
│   ├── llm_game.py              # Air Strip One LLM game engine
│   └── weather_bridge.py        # Weather system integration
├── gravitas_engine/             # Core simulation engine
│   ├── core/                    # State, params, integrator
│   ├── agents/                  # Environments (GravitasEnv, StalingradMA)
│   ├── systems/                 # ODE dynamics, shocks, media, economy
│   └── analysis/                # Metrics, logging
├── extensions/                  # Military wrapper, political interface
│   ├── military/                # CoW-native military system
│   │   ├── cow_combat.py        # 34 unit types, traits, combat resolution
│   │   ├── land_bridge.py       # Per-sector land garrisons
│   │   ├── military_dynamics.py # Combat, production, research, morale
│   │   ├── military_state.py    # State dataclasses
│   │   ├── military_wrapper.py  # Gymnasium environment
│   │   ├── physics.py           # Physics engine (terrain, weather, supply)
│   │   ├── physics_bridge.py    # Integration layer
│   │   └── unit_types.py        # Legacy unit type mappings
│   ├── economy_v2/              # Economy system (GDP, factories)
│   ├── pop/                     # Population dynamics
│   ├── research/                # 10×5 tech tree system
│   ├── governance/              # Budget, corruption, bureaucracy
│   ├── ministries/              # Autonomous ministry system
│   ├── resistance/              # BLF resistance system
│   ├── naval/                   # Naval warfare, 6 sea zones
│   ├── air_force/               # Air operations, 10 aircraft types
│   ├── intelligence/            # Espionage, fog of war
│   ├── war_economy/             # Legacy resource model + manpower
│   ├── exhaustion/              # SB3 exhaustion monitor callback
│   ├── fog_of_war/              # Observation noise wrapper
│   ├── topology/                # Network graph visualization
│   └── audits/                  # System validation
├── gui/                         # Real-time strategic map GUI
│   ├── main.py                  # Pygame-based map viewer
│   ├── generate_map.py          # Geographic asset generation
│   ├── assets/                  # Map images, sector positions
│   └── __init__.py              # GUI package
├── configs/                     # Unified YAML configuration
│   ├── custom.yaml              # Plugin + scenario config
│   └── (use custom.yaml)        # Unified scenario + plugin config
├── cli.py                       # CLI entry point
├── tests/                       # Training, evaluation, replay scripts
│   ├── benchmark_llm.py         # LLM benchmark for Air Strip One
│   ├── train_stalingrad_selfplay.py  # Stalingrad self-play training
│   └── train_rppo.py            # RecurrentPPO single-agent training
└── docs/                        # Documentation
```

## Installation

```bash
git clone https://github.com/Kiy-K/Gravitas-Engine.git
cd Gravitas-Engine
pip install -e ".[dev]"        # Install with dev dependencies
# pip install -e ".[all]"      # Install with all optional dependencies
```

## Quick Start

### CLI (Recommended)

```bash
# Run Stalingrad scenario
python cli.py run stalingrad --episodes 5

# Run from unified config
python cli.py run --config configs/custom.yaml --episodes 30

# List available scenarios and plugins
python cli.py list scenarios
python cli.py list plugins

# Run with trained agents
python cli.py run stalingrad --episodes 30 \
    --axis-model logs/stalingrad_selfplay/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay/soviet_final.zip

# Detailed battle replay with sector-by-sector commentary
python cli.py replay \
    --axis-model logs/stalingrad_selfplay/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay/soviet_final.zip
```

### Python API

```python
from gravitas import GravitasEngine

# From config file
engine = GravitasEngine.from_config("configs/custom.yaml")
results = engine.run(episodes=10)

# Manual setup
engine = GravitasEngine(scenario="stalingrad", seed=42)
engine.load_plugins(["nonlinear_combat", "logistics_network"])
results = engine.run(episodes=30)

# Access environment directly
env = engine.env
obs = env.reset(seed=0)
```

### Air Strip One 1984 (LLM-Driven)

```bash
# Run 100-week game with AI players
python tests/benchmark_llm.py \
    --oceania-model claude-haiku-4-5-20251001 \
    --eurasia-model claude-haiku-4-5-20251001 \
    --winston-model claude-haiku-4-5-20251001 \
    --commentary-model mistral-medium-latest \
    --turns 100 --seed 451

# Real-time GUI visualization
.venv/bin/python gui/main.py --seed 451 --turns 100 --speed 1.0
```

### Single-Agent Training

```bash
python tests/train_rppo.py
```

### Multi-Agent Self-Play Training

```bash
python tests/train_stalingrad_selfplay.py \
    --total-rounds 6 --steps-per-round 25000 --n-envs 4 \
    --log-dir logs/stalingrad_selfplay
```

### Battle Evaluation

```bash
python tests/eval_stalingrad_battle.py \
    --axis-model logs/stalingrad_selfplay/axis_final.zip \
    --soviet-model logs/stalingrad_selfplay/soviet_final.zip \
    --n-episodes 30
```

## Plugin System

Plugins extend simulation behavior by hooking into the step loop. Each plugin implements a standardized interface:

```python
from gravitas.plugins import GravitasPlugin

class Plugin(GravitasPlugin):
    name = "my_plugin"
    description = "What this plugin does."

    def on_step(self, world, turn, **kwargs):
        # Inspect and modify world state
        c_arr = world.cluster_array()
        # ... apply mechanic ...
        return world  # Return modified world
```

### Built-in Plugins

| Plugin | Description |
|--------|-------------|
| `nonlinear_combat` | Nonlinear combat dynamics (Lanchester, breakthrough, fatigue, terrain multipliers) |
| `logistics_network` | Graph-based logistics with saturating flow, consumption pressure, disruptions |
| `partisan_warfare` | Autonomous partisan actions (recruitment, sabotage, ambush, movement) |
| `soviet_reinforcements` | Volga-crossing reinforcement mechanic |
| `axis_airlift` | Diminishing Axis air resupply mechanic |

### Plugin Lifecycle

| Hook | When Called | Purpose |
|------|-----------|---------|
| `on_init(world)` | Engine init | One-time setup |
| `on_reset(world)` | Episode start | Reset plugin state |
| `on_step(world, turn)` | Each step | Core mechanic (required) |
| `on_episode_end(world, turn)` | Episode end | Cleanup, logging |

### Configuration

Plugins are configured via `configs/custom.yaml`:

```yaml
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

## Battle of Stalingrad Scenario

The flagship training scenario simulates the decisive Eastern Front battle (Aug 23, 1942 – Feb 2, 1943) with **9 operational sectors**:

| ID | Sector | Controller | Terrain | Role |
|----|--------|-----------|---------|------|
| 0 | Stalingrad City Center | Axis | urban | Contested city ruins, block-by-block combat |
| 1 | Tractor Factory District | Axis | urban | Industrial fighting zone |
| 2 | Mamayev Kurgan | Contested | urban | Strategic hill dominating the city |
| 3 | Volga Crossing | Soviet | riverfront | Soviet lifeline across the Volga |
| 4 | Northern Don River Line | Axis | open | Axis defensive flank |
| 5 | Axis Supply Corridor | Axis | open | Overextended logistics corridor |
| 6 | Soviet Strategic Reserve | Soviet | open | Build-up for Operation Uranus |
| 7 | Romanian/Italian Sector | Axis | open | Weak Axis southern flank |
| 8 | Wintergewitter Corridor | Axis | open | Hoth's relief push route |

### Unit Types & Traits

- **Infantry**: Guards (elite, urban bonus), Ski Troops (winter hardened), Shock Troops (breakthrough)
- **Specialists**: Engineers (fortifications, mines), Snipers (suppression), Mountain Troops (terrain bonus)
- **Armor**: Light/Medium/Heavy tanks, Tank Destroyers, Flame Tanks (anti-structure)
- **Support**: Artillery, Mortars, Rocket Artillery, Anti-Air, Supply Trucks
- **Recon**: Armored Cars, Recon Infantry (detection)

### Historical Shock Events

- **Winter Blizzard**: Severe cold shock that strains both sides
- **Luftwaffe Airlift**: Partial, diminishing Axis resupply attempt
- **Operation Uranus**: Soviet double envelopment event
- **Axis Kessel**: Encirclement and logistics collapse
- **Operation Winter Storm**: German relief thrust from the south
- **Operation Little Saturn**: Soviet move that cuts relief momentum

### Training Results (Self-Play)

| Phase | Features | Balance |
|-------|----------|---------|
| Base | 34 unit types, terrain | Soviet defensive advantage |
| + Physics | Weather, supply, attrition | Winter favors Soviets |
| + Traits | Elite units, specialists | Dynamic tactical depth |

## Documentation

- [Air Strip One Scenario](docs/AIRSTRIP_ONE.md) — 35-sector 1984 strategic simulation
- [Systems Documentation](docs/AIRSTRIP_ONE_SYSTEMS.md) — Complete 13-system technical reference
- [Architecture Overview](docs/ARCHITECTURE.md) — Engine internals and design
- [GUI Documentation](docs/GUI.md) — Real-time strategic map viewer
- [Military System Guide](docs/MILITARY_SYSTEM.md) — CoW-native combat, unit types, and traits
- [Physics Integration](docs/PHYSICS_INTEGRATION.md) — Terrain, weather, and supply modeling
- [Plugin System Guide](docs/PLUGIN_SYSTEM.md) — Writing and configuring plugins
- [Ministries System](docs/MINISTRIES.md) — Autonomous government ministries
- [Training Guide](docs/TRAINING.md) — PPO/RecurrentPPO training and self-play
- [RL Utilities](docs/RL_UTILITIES.md) — FogOfWarWrapper, ExhaustionMonitor, TopologyVisualizer
- [War Economy (Legacy)](docs/WAR_ECONOMY.md) — Leontief resource model

## Population + Military Modeling

The `PopWrapper` enables multi-class demographics, ethnic tension, and the Soldier archetype with morale/conscription dynamics:

```python
from extensions.pop.pop_wrapper import PopWrapper
from gravitas_engine.agents.gravitas_env import GravitasEnv

env = GravitasEnv()
pop_env = PopWrapper(env)
obs, info = pop_env.reset()
```

See `docs/AIRSTRIP_ONE_SYSTEMS.md` (section 3) for full details.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
