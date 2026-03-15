# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install
```bash
pip install -e ".[dev]"        # Install with dev dependencies
pip install -e ".[all]"        # Install with all optional dependencies
```

### Build Cython kernels
```bash
./build_cython.sh              # Compile in-place (required for performance)
./build_cython.sh --clean      # Remove artifacts
python setup.py build_ext --inplace  # Alternative
```
The compiled kernel (`gravitas_engine/core/_kernels.so`) is not required — the engine falls back to pure Python — but is ~10–50x faster on ODE-heavy workloads.

### Run tests
```bash
pytest tests/                            # All tests
pytest tests/smoke_test.py               # Core sanity check (fastest)
pytest tests/test_env.py                 # Gymnasium compliance check
pytest tests/test_military_extension.py  # Military system
pytest -v tests/test_war_economy.py      # War economy
```
Note: `tests/train_*.py`, `tests/evaluate_*.py`, and `tests/benchmark_*.py` are scripts, not test suites.

### Lint
```bash
ruff check gravitas_engine tests         # Check
ruff check --fix gravitas_engine tests   # Auto-fix
mypy gravitas_engine                     # Type checking
```

### Run scenarios
```bash
python cli.py run stalingrad --episodes 5
python cli.py run --config configs/custom.yaml --episodes 10
python cli.py list scenarios
python cli.py list plugins
```

---

## Architecture

### Package layout

There are two separate Python packages:

- **`gravitas_engine/`** — the core simulation engine (ODE dynamics, RL environments, subsystems)
- **`gravitas/`** — high-level orchestration layer (engine wrapper, plugin system, scenarios, LLM game)

`extensions/` contains optional add-on modules (military, war economy, naval, air force, intelligence, resistance) that are not installed by default via `pyproject.toml`.

### State model

All simulation state flows through a three-tier hierarchy:

| Type | Scope | Key Fields |
|------|-------|-----------|
| `ClusterState` | Per-region | `sigma` (stability), `hazard`, `resource`, `military`, `trust`, `polar` |
| `GlobalState` | World-wide | `exhaustion`, `fragmentation`, `polarization`, `media_bias`, `shock_rate` |
| `GravitasWorld` | Container | `clusters`, `global_state`, `adjacency`, `conflict`, `alliance`, `economy` |

### Core dynamics (`gravitas_engine/core/`)

The simulation integrates a system of coupled ODEs via RK4:
- **Hazard index** — algebraic, re-derived each step from stability, polarization, and neighbor cascade
- **Stability** (`dσ/dt`) — recovers via institutional trust + resources; dragged by hazard; diffuses spatially
- **Exhaustion** (`dE/dt`) — accumulates with military use; high exhaustion freezes all recovery
- **Polarization / fragmentation / trust** — feedback loops between these global and per-cluster scalars

The hot path is `_kernels.pyx` (Cython). If the `.so` is not built, `gravitas_dynamics.py` provides identical pure-Python fallback.

### RL environments (`gravitas_engine/agents/`)

| Class | Type | Description |
|-------|------|-------------|
| `GravitasEnv` | `gymnasium.Env` | Single-agent governance. Action = stance (6–8 options). Reward = stability minus fragmentation/exhaustion penalties. |
| `StalingradMultiAgentEnv` | Custom | Two-player Axis vs. Soviet. Same action/obs structure; reward is stability advantage over opponent. Used for self-play training. |
| `SelfPlayEnv` | `gymnasium.Env` | Wraps `StalingradMultiAgentEnv` for SB3 training — trains one side while the other runs a frozen policy. |

Observation shape is `(5N + 8 + action_dim,)` where N = number of clusters. Agents never observe raw `ClusterState` — they see distorted values via `media_bias`.

### Plugin system (`gravitas/plugins/`)

Plugins hook into the episode loop via three lifecycle methods:

```python
class GravitasPlugin(ABC):
    def on_reset(self, world, **kwargs) -> GravitasWorld: ...
    def on_step(self, world, turn, **kwargs) -> GravitasWorld: ...   # required
    def on_episode_end(self, world, turn, **kwargs) -> None: ...
```

Built-in plugins: `soviet_reinforcements`, `axis_airlift`, `nonlinear_combat`, `logistics_network`, `partisan_warfare`. Plugins are loaded by name from YAML config or passed explicitly to `GravitasEngine`.

### Scenario definition

Scenarios are YAML files in `gravitas/scenarios/` or `training/regimes/`. The key top-level sections are:

- `agents` — list of factions with `side` and `controlled_clusters`
- `sectors` — cluster definitions with terrain and initial values
- `logistics_links`, `terrain` — physics configuration
- Standard params: `topology`, `shocks`, `military`, `media`, `reward`, `episode`

`regime_loader.py` converts YAML → `GravitasParams` dataclass. `GravitasEngine._load_scenario()` handles cluster assignment and env construction.

### Extensions

The `extensions/` modules are largely independent subsystems that share state through `GravitasWorld` or their own state containers:

- **`military/`** — Call of War-style combat. 34 unit types, terrain/weather physics (`physics.py` is ~88KB). `physics_bridge.py` maps CoW unit state → terrain modifiers for the core engine.
- **`war_economy/`** — 7-sector Leontief input-output economy with conscription and manpower pipelines.
- **`naval/`** / **`air_force/`** — standalone combat subsystems consumed by the LLM game.
- **`intelligence/`** / **`resistance/`** — fog-of-war and insurgency mechanics for the Air Strip One scenario.

### LLM game (`gravitas/llm_game.py`)

`llm_game.py` integrates all extensions into a turn-based game for language model agents (Air Strip One scenario). It exposes ~24 text-parseable action types and produces ~500–800 token state summaries per turn. The `GameState` dataclass is the unified container for all extension states.

### Trained models

Training scripts live in `tests/train_*.py`. Models are saved to `logs/` as `.zip` files (SB3 format) and can be loaded via `RecurrentPPO.load()`. The CLI `--axis-model` / `--soviet-model` flags accept these paths.
