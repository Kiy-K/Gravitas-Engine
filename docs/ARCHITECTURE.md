# Architecture Overview

GRAVITAS Engine is organized into three layers: the **core simulation** (`gravitas_engine/`), the **military/political extensions** (`extensions/`), and the **high-level orchestration** (`gravitas/`). This document describes how the layers interact and the design decisions behind each.

## Table of Contents

- [Layer Diagram](#layer-diagram)
- [Core Simulation Layer](#core-simulation-layer)
- [Extensions Layer](#extensions-layer)
- [Orchestration Layer](#orchestration-layer)
- [Data Flow](#data-flow)
- [Configuration System](#configuration-system)
- [Plugin Integration](#plugin-integration)
- [Key Design Decisions](#key-design-decisions)

## Layer Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                    CLI (cli.py)                              │
├─────────────────────────────────────────────────────────────┤
│              Orchestration (gravitas/)                       │
│   ┌──────────┐  ┌────────────┐  ┌──────────────────┐       │
│   │ engine.py│  │ plugins/   │  │ scenarios/*.yaml │       │
│   │          │──│ on_step()  │  │                  │       │
│   └──────────┘  └────────────┘  └──────────────────┘       │
├─────────────────────────────────────────────────────────────┤
│            Core Simulation (gravitas_engine/)                │
│   ┌────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│   │ core/  │  │ agents/  │  │ systems/ │  │ analysis/  │  │
│   │ params │  │ env      │  │ ODE      │  │ metrics    │  │
│   │ state  │  │ actions  │  │ shocks   │  │ logging    │  │
│   │ integr.│  │ MA env   │  │ media    │  │            │  │
│   └────────┘  └──────────┘  │ economy  │  └────────────┘  │
│                              │ pop      │                   │
│                              └──────────┘                   │
├─────────────────────────────────────────────────────────────┤
│              Extensions (extensions/)                        │
│   ┌──────────────┐ ┌─────────────┐ ┌──────────────┐        │
│   │ naval/       │ │ air_force/  │ │ resistance/  │        │
│   │ 14 ship cls  │ │ 10 ac types │ │ BLF 7 levels │        │
│   │ 6 sea zones  │ │ 7 bases/side│ │ Winston Smith│        │
│   │ 3 inv types  │ │             │ │              │        │
│   ├──────────────┤ ├─────────────┤ ├──────────────┤        │
│   │ economy_v2/  │ │ pop/        │ │ intelligence/│        │
│   │ 10 factories │ │ real numbers│ │ fog of war   │        │
│   │ GDP model    │ │ 1984 classes│ │ espionage    │        │
│   ├──────────────┤ ├─────────────┤ ├──────────────┤        │
│   │ research/    │ │ governance/ │ │ war_economy/ │        │
│   │ 10×5 tech tree│ │ budget+corr │ │ legacy model │        │
│   │ 50 techs     │ │ bureaucracy │ │              │        │
│   │ 13 prereqs   │ │             │ │              │        │
│   ├──────────────┤ ├─────────────┤ ├──────────────┤        │
│   │ military/    │ │ manpower/   │ │ audits/      │        │
│   │ land combat  │ │ 15 conscrip │ │ system audit │        │
│   │ 30+ units    │ │ training    │ │              │        │
│   └──────────────┘ └─────────────┘ └──────────────┘        │
│                                                          │
│   ┌─────────────────────────────────────────────────────┐ │
│   │ GUI Layer (gui/)                                    │ │
│   │ main.py — Real-time strategic map viewer             │ │
│   │ generate_map.py — Geographic asset generation       │ │
│   │ assets/ — Map images, sector positions              │ │
│   └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Simulation Layer

**Package**: `gravitas_engine/`

The foundational simulation engine implementing the GRAVITAS model — a non-linear ODE system with Hawkes shock processes, media bias, and partial observability.

### `core/`

- **`gravitas_params.py`** — Immutable `GravitasParams` dataclass with all simulation hyperparameters (100+ fields covering ODE coefficients, shock rates, economy, demographics).
- **`gravitas_state.py`** — `GravitasWorld` container holding all simulation state: cluster states, global state, alliance matrix, population, economy. Uses immutable copy-on-write pattern (`copy_with_*` methods).
- **`integrator.py`** — RK4 numerical integration for the ODE system.

### `agents/`

- **`gravitas_env.py`** — Single-agent Gymnasium environment (`GravitasEnv`). Supports `Discrete` and hierarchical action spaces with 7 stances.
- **`stalingrad_ma.py`** — Multi-agent environment (`StalingradMultiAgentEnv`) for adversarial two-player Stalingrad. Wraps `GravitasWorld` with per-side observations, actions, rewards, and LSTM-compatible state. Also contains `SelfPlayEnv` for training.
- **`gravitas_actions.py`** — `HierarchicalAction` and `Stance` definitions.

### `systems/`

- **`hawkes_shock.py`** — Hawkes self-exciting point process for stochastic shocks.
- **`media_bias.py`** — Media bias dynamics with propaganda effects.
- **`economy.py`** — Per-cluster economic subsystem (GDP, unemployment, debt, industrial capacity).

### `analysis/`

- **`metrics.py`** — Summary statistics for evaluation.
- **`logging.py`** — State logging for trajectory analysis.

## Extensions Layer

**Package**: `extensions/`

Optional wrappers that add military and population modeling on top of the core simulation.

### Military Extension (`extensions/military/`)

- **`cow_combat.py`** — CoW-native combat engine with 34 unit types, UnitTraits system, and combat resolution.
- **`military_dynamics.py`** — Combat dynamics, production, research, morale, and trait-based mechanics.
- **`military_state.py`** — Unit and cluster military state dataclasses with physics integration.
- **`military_wrapper.py`** — `MilitaryWrapper` Gymnasium environment with physics-enabled combat.
- **`physics.py`** — Physics engine modeling terrain, weather, supply logistics, and line-of-sight.
- **`physics_bridge.py`** — Integration layer between CoW military system and physics engine.
- **`unit_types.py`** — Legacy unit type mappings for backward compatibility.

### Population Extension (`extensions/pop/`)

- **`PopWrapper`** — Adds multi-class demographics, ethnic tension, and Soldier archetype with morale/conscription/desertion dynamics.

### Land Combat Extension (`extensions/military/`)

- **`land_bridge.py`** — Per-sector land garrisons using CoW combat system (30+ unit types). Resolves combat in contested sectors each turn. Spawns beachhead units for successful invasions.

### Manpower Extension (`extensions/manpower/`)

- **`manpower.py`** — Conscription system with 15 laws, training pipeline, and recruitment from population pools.

### Audits Extension (`extensions/audits/`)

- **`system.py`** — System integrity checks and validation utilities.

## Orchestration Layer

**Package**: `gravitas/`

The newest layer providing high-level scenario management, plugin support, and a clean API.

### `engine.py` — `GravitasEngine`

The central class that:

1. **Loads scenarios** from YAML files (searches `gravitas/scenarios/` then `training/regimes/`).
2. **Discovers and loads plugins** from config or CLI arguments.
3. **Runs episodes** with trained or random agents, invoking plugin hooks at each step.
4. **Collects results** including plugin events, per-episode stats, and aggregate summaries.

### `plugins/` — Plugin System

- **`__init__.py`** — `GravitasPlugin` ABC, discovery, and loading utilities.
- **`soviet_reinforcements.py`** — Volga barge crossing mechanic.
- **`axis_airlift.py`** — Luftwaffe airlift mechanic.

See [Plugin System Guide](PLUGIN_SYSTEM.md) for details.

### `scenarios/` — Scenario YAML Files

Scenario definitions including sector configs, initial states, alliances, shock events, and training parameters. Available scenarios: `moscow.yaml` (9-sector Battle of Moscow), `stalingrad.yaml` (9-sector Battle of Stalingrad), `airstrip_one.yaml` (35-sector 1984 strategic simulation).

## Data Flow

### Single Episode Execution

```text
1. GravitasEngine.run()
   │
2. env.reset(seed)
   │  → GravitasWorld initialized with cluster states, alliances
   │  → Physics states initialized (terrain, weather, supply)
   │  → Military units created from scenario config
   │  → Plugin.on_reset(world) called for each plugin
   │
3. Loop: env.step(actions)
   │  a. Decode Axis + Soviet actions → HierarchicalActions
   │  b. Apply actions to world state
   │  c. RK4 integration (ODE backbone)
   │  d. Shock sampling + application (Hawkes process)
   │  e. Media bias update
   │  f. Alliance decay
   │  g. Population step (if enabled)
   │  h. Economy step (if enabled)
   │  i. Physics step (terrain, weather, supply attrition)
   │  j. Military step (CoW combat, production, movement)
   │  k. Physics bridge integration (combat modifiers, LOS)
   │  l. Advance step counter
   │  m. Compute per-side rewards
   │  n. Build per-side observations (including physics)
   │  │
   │  → Plugin.on_step(world, turn) called for each plugin
   │
4. Episode ends (collapse or max_steps reached)
   │  → Plugin.on_episode_end(world, turn)
   │
5. Collect results + plugin events
```

### Observation Space

Per-side observations include:

- Own cluster states (σ, h, r, m, τ, p) for controlled sectors
- Enemy cluster states (partial/noisy)
- Global state (exhaustion, media bias, etc.)
- Alliance information
- Previous action encoding
- Economy indicators (if enabled)
- Military units (34 types, HP, XP, traits)
- Physics observations (terrain, weather, supply levels)
- Line of sight and detection data

## Configuration System

### Scenario YAML (`gravitas/scenarios/moscow.yaml`, `stalingrad.yaml`)

Defines the full scenario: GravitasParams overrides, sector definitions with initial states, alliance matrix, custom shock events, physics configuration, and training parameters. Moscow scenario includes physics-driven terrain/weather and 34 unit types.

### Unified Config (`configs/custom.yaml`)

High-level config specifying which scenario to load, which plugins to activate, and per-plugin parameter overrides. Used by the CLI and `GravitasEngine.from_config()`.

### Precedence

1. CLI arguments (highest priority)
2. `configs/custom.yaml` settings
3. Scenario YAML defaults
4. `GravitasParams` dataclass defaults (lowest priority)

## Plugin Integration

Plugins are standalone modules that:

1. **Import only from `gravitas_engine.core`** — no circular dependencies.
2. **Receive the world after all engine updates** — plugins modify post-step state.
3. **Execute sequentially in config order** — each plugin sees the previous plugin's modifications.
4. **Log structured events** — accessible in results for analysis.

### Why Plugins Instead of Hardcoded Mechanics?

- **Modularity**: Enable/disable mechanics without code changes.
- **Configurability**: Tune parameters via YAML without retraining.
- **Testability**: Test each mechanic in isolation.
- **Extensibility**: Add new historical mechanics (weather, logistics, morale events) as plugins.

---

## GUI Layer

**Package**: `gui/`

Real-time strategic map visualization for the Air Strip One scenario.

### `main.py` — Strategic Map GUI

A Pygame-based real-time viewer showing:
- **35-sector map** with real geographic positions
- **6 sea zones** with fleet positions
- **Land garrisons** and contested sectors
- **BLF resistance** activity levels
- **Faction scores** and military forces
- **War correspondent** dispatches
- **Interactive controls**: pause/play, speed, sector selection

### `generate_map.py` — Asset Generation

Generates map assets from Natural Earth geographic data:
- **Western Europe map** with sector boundaries
- **Sector positions** mapped from real coordinates
- **Sea zone centers** for fleet indicators

### Controls

- **SPACE** — Pause/Resume auto-play
- **N** — Next turn (when paused)
- **+/-** — Speed up/slow down
- **S** — Toggle sector names
- **F** — Toggle fleet display
- **ESC/Q** — Quit

---

## Key Design Decisions

### Immutable World State

`GravitasWorld` uses a copy-on-write pattern. All modifications create new instances via `copy_with_*` methods. This enables:

- Safe plugin execution (plugins can't corrupt state)
- Easy rollback and diffing between steps
- Thread-safe parallel episode execution

### Separate Orchestration Package

The `gravitas/` package is deliberately separate from `gravitas_engine/` to:

- Avoid circular imports (the original motivation)
- Keep the core simulation pure and dependency-free
- Allow the orchestration layer to evolve independently
- Enable backward compatibility (existing code imports from `gravitas_engine/` unchanged)

### YAML-Driven Scenarios

Scenarios are defined in YAML rather than Python to:

- Allow non-programmers to create and modify scenarios
- Enable parameter sweeps without code changes
- Separate data (sector definitions) from logic (simulation engine)
- Support complex physics and military configurations

### CoW-Native Military System

The military system uses Call of War-style mechanics to:

- Provide realistic tactical depth with 34 unit types and traits
- Enable terrain and weather effects through physics integration
- Support complex combat dynamics (suppression, breakthrough, morale)
- Maintain compatibility with reinforcement learning training

### Physics Integration

Physics modeling adds realism by:

- Simulating terrain effects on movement and combat
- Modeling weather attrition and equipment reliability
- Providing supply logistics and line-of-sight calculations
- Creating dynamic environmental constraints for tactical decisions

### RecurrentPPO for Multi-Agent

LSTM-based policies (RecurrentPPO from sb3-contrib) are used because:

- The environment is partially observable (each side sees limited info)
- Temporal context matters (reinforcement timing, shock patterns)
- Memory helps agents learn long-horizon strategies (defend Moscow for 50+ turns)
- Complex physics and military systems require temporal reasoning.
