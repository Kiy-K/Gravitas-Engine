"""
military_wrapper.py — Gymnasium wrapper for the CoW-native military system.

Exposes a MultiDiscrete action space:
  [action_type, source_cluster, target_cluster, unit_type_idx, aux_idx]

Observation is a flat float32 vector produced by world_to_obs_array.

Supports two-faction self-play: the wrapper manages a *single* faction's
perspective; pair two wrappers (or alternate turns) for self-play training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gymnasium_shim as gym  # type: ignore
    from gymnasium_shim import spaces  # type: ignore

from .cow_combat import (
    CowTerrain, CowDoctrine, CowBuildingType, CowUnitType,
)
from .military_state import (
    CowWorldState, CowClusterState, CowFactionState,
    CowExternalModifiers, merge_modifiers,
    init_world_state, spawn_initial_units,
)
from .military_dynamics import (
    step_world, check_victory, world_to_obs_array, obs_size,
    ActionType, N_ACTION_TYPES, N_UNIT_TYPES, N_BUILDING_TYPES,
)
from .physics_bridge import (
    init_physics_for_world, MapPhysicsConfig, load_map_physics,
)


# ─────────────────────────────────────────────────────────────────────────── #
# Wrapper                                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class MilitaryWrapper(gym.Env):
    """Gymnasium env wrapping the CoW military system for a single faction.

    For self-play: instantiate two MilitaryWrapper instances sharing the same
    CowWorldState, one per faction.  Alternatively, use the ``opponent_policy``
    callback so the wrapper automatically steps the opponent.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        scenario_cfg: Dict[str, Any],
        faction_id: int = 0,
        opponent_faction_id: int = 1,
        opponent_policy: Optional[Any] = None,
        max_steps: int = 200,
        max_clusters: int = 12,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.scenario_cfg = scenario_cfg
        self.faction_id = faction_id
        self.opponent_id = opponent_faction_id
        self.opponent_policy = opponent_policy
        self.max_steps = max_steps
        self.max_clusters = max_clusters
        self.render_mode = render_mode

        self._rng = np.random.default_rng(seed)

        # Determine n_clusters from config
        self.n_clusters = scenario_cfg.get("n_clusters", 6)
        n_c = min(self.n_clusters, self.max_clusters)

        # Action space: MultiDiscrete
        #   [action_type, source_cluster, target_cluster, unit_type_idx, aux_idx]
        # aux_idx encodes: building type for BUILD, level for PRODUCE/RESEARCH
        self.aux_size = max(N_BUILDING_TYPES, 5)  # covers building types and unit levels 1-4
        self.action_space = spaces.MultiDiscrete([
            N_ACTION_TYPES,    # 0: action type
            n_c,               # 1: source cluster
            n_c,               # 2: target cluster
            N_UNIT_TYPES,      # 3: unit type index
            self.aux_size,     # 4: auxiliary (building type / level)
        ])

        # Physics enabled flag
        self._physics_enabled = scenario_cfg.get("physics_enabled", False)

        # Observation space
        self._obs_dim = obs_size(self.max_clusters, physics_enabled=self._physics_enabled)
        self.observation_space = spaces.Box(
            low=-1.0, high=5.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self.world: Optional[CowWorldState] = None
        self._step_count = 0

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = self.scenario_cfg

        # Build adjacency
        adj_raw = cfg.get("adjacency")
        if adj_raw is not None:
            adj = np.array(adj_raw, dtype=bool)
        else:
            n = self.n_clusters
            adj = np.eye(n, dtype=bool)
            for i in range(n - 1):
                adj[i, i + 1] = adj[i + 1, i] = True

        # Terrains
        terrain_names = cfg.get("cluster_terrains")
        terrains = None
        if terrain_names:
            terrains = [CowTerrain[t] for t in terrain_names]

        # Owners
        owners = cfg.get("cluster_owners")

        # Buildings
        buildings = cfg.get("cluster_buildings")

        self.world = init_world_state(
            n_clusters=self.n_clusters,
            faction_configs=cfg["factions"],
            adjacency=adj,
            objectives=cfg.get("objectives"),
            cluster_terrains=terrains,
            cluster_owners=owners,
            cluster_buildings=buildings,
        )

        # Apply external modifiers from spirit / government if configured
        self._apply_external_modifiers(cfg)

        # Initialize physics engine if enabled
        if self._physics_enabled:
            self._init_physics(cfg)

        # Spawn initial units
        unit_specs = cfg.get("initial_units")
        if unit_specs:
            # Convert string keys to int keys if needed
            specs = {}
            for k, v in unit_specs.items():
                specs[int(k)] = v
            self.world = spawn_initial_units(self.world, specs)

        self._step_count = 0

        # Reset opponent LSTM hidden states on episode boundary
        if self.opponent_policy is not None and hasattr(self.opponent_policy, '_reset_states'):
            self.opponent_policy._reset_states()

        obs = world_to_obs_array(self.world, self.faction_id, self.max_clusters)
        info = {"step": 0, "victory": check_victory(self.world)}
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────

    def step(
        self, action: NDArray[np.int64],
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        assert self.world is not None, "Call reset() first"

        # Decode action
        at = int(action[0])
        src = int(action[1])
        tgt = int(action[2])
        ut_idx = int(action[3])
        aux = int(action[4])

        # Build faction_actions dict
        faction_actions: Dict[int, Tuple[int, int, int, int, int]] = {
            self.faction_id: (at, src, tgt, ut_idx, aux),
        }

        # Opponent action
        if self.opponent_id in self.world.factions:
            opp_action = self._get_opponent_action()
            faction_actions[self.opponent_id] = opp_action

        # Step world
        self.world, rewards, metrics = step_world(self.world, faction_actions, self._rng)
        self._step_count += 1

        # Reward for our faction
        reward = float(rewards.get(self.faction_id, 0.0))

        # Check termination
        victory = check_victory(self.world)
        terminated = bool(victory["done"])
        truncated = self._step_count >= self.max_steps

        # Terminal reward shaping
        if terminated:
            if victory["winner"] == self.faction_id:
                reward += 20.0
            elif victory["winner"] is not None:
                reward -= 20.0

        obs = world_to_obs_array(self.world, self.faction_id, self.max_clusters)
        info = {
            "step": self._step_count,
            "metrics": metrics,
            "victory": victory,
        }

        return obs, reward, terminated, truncated, info

    # ── External Modifiers ──────────────────────────────────────────────

    def _apply_external_modifiers(self, cfg: Dict[str, Any]) -> None:
        """Load national spirit and government modifiers and apply to factions."""
        faction_modifiers = cfg.get("faction_modifiers", {})

        for fid, faction in self.world.factions.items():
            fmod = faction_modifiers.get(fid, faction_modifiers.get(str(fid), {}))
            if not fmod:
                continue

            mods = CowExternalModifiers()

            # Load national spirit modifiers
            spirit_cfg = fmod.get("national_spirit")
            if spirit_cfg:
                try:
                    from gravitas_engine.systems.national_spirit import (
                        load_spirit_from_yaml, spirit_to_cow_modifiers,
                    )
                    spirit = load_spirit_from_yaml(spirit_cfg)
                    spirit_mods = spirit_to_cow_modifiers(spirit)
                    mods = merge_modifiers(mods, spirit_mods)
                except Exception:
                    pass  # graceful fallback if spirit system unavailable

            # Load government modifiers
            gov_cfg = fmod.get("government_type")
            if gov_cfg:
                try:
                    from gravitas_engine.systems.government import (
                        GovernmentType, government_to_cow_modifiers,
                    )
                    gov_type = GovernmentType[gov_cfg]
                    gov_mods = government_to_cow_modifiers(gov_type)
                    mods = merge_modifiers(mods, gov_mods)
                except Exception:
                    pass  # graceful fallback if government system unavailable

            # Direct overrides (for testing / scenario tuning)
            direct = fmod.get("direct_modifiers", {})
            if direct:
                for k, v in direct.items():
                    if hasattr(mods, k):
                        setattr(mods, k, v)

            faction.external_modifiers = mods

    # ── Physics Initialization ─────────────────────────────────────────

    def _init_physics(self, cfg: Dict[str, Any]) -> None:
        """Initialize physics engine from scenario config."""
        physics_cfg = cfg.get("map_physics")
        yaml_path = cfg.get("physics_yaml_path")

        config = None
        if physics_cfg:
            # Build MapPhysicsConfig from inline config dict
            config = MapPhysicsConfig(
                name=physics_cfg.get("name", "default"),
                n_sectors=physics_cfg.get("n_sectors", self.n_clusters),
                climate_type=physics_cfg.get("climate", {}).get("type", "continental"),
                temperature_curve={
                    int(k): float(v)
                    for k, v in physics_cfg.get("climate", {}).get(
                        "temperature_curve", {0: 5.0, 50: 2.0, 100: -5.0, 150: -20.0, 200: -15.0}
                    ).items()
                },
                base_humidity=physics_cfg.get("climate", {}).get("humidity", 60.0),
                base_wind_ms=physics_cfg.get("climate", {}).get("wind_ms", 5.0),
                steps_per_day=physics_cfg.get("climate", {}).get("steps_per_day", 4),
                sectors={
                    int(k): v for k, v in physics_cfg.get("sectors", {}).items()
                },
                supply_routes=physics_cfg.get("supply_routes", []),
                factions=physics_cfg.get("factions", {}),
            )

        states, los, config = init_physics_for_world(
            self.world,
            physics_config=config,
            yaml_path=yaml_path,
        )
        self.world.physics_states = states
        self.world.los_state = los
        self.world.physics_config = config

    # ── Opponent ──────────────────────────────────────────────────────────

    def _get_opponent_action(self) -> Tuple[int, int, int, int, int]:
        """Get opponent action from policy or use random."""
        if self.opponent_policy is not None:
            opp_obs = world_to_obs_array(self.world, self.opponent_id, self.max_clusters)
            opp_action = self.opponent_policy(opp_obs)
            return (int(opp_action[0]), int(opp_action[1]), int(opp_action[2]),
                    int(opp_action[3]), int(opp_action[4]))
        else:
            # Random valid-ish action: mostly NOOP with occasional produce/move
            r = self._rng.random()
            if r < 0.5:
                return (0, 0, 0, 0, 0)  # NOOP
            elif r < 0.7:
                # Random produce
                src = self._rng.integers(0, self.n_clusters)
                ut = self._rng.integers(0, N_UNIT_TYPES)
                return (ActionType.PRODUCE.value, int(src), 0, int(ut), 1)
            elif r < 0.85:
                # Random move
                src = self._rng.integers(0, self.n_clusters)
                tgt = self._rng.integers(0, self.n_clusters)
                return (ActionType.MOVE.value, int(src), int(tgt), 0, 0)
            else:
                # Random reinforce
                src = self._rng.integers(0, self.n_clusters)
                return (ActionType.REINFORCE.value, int(src), 0, 0, 0)

    # ── Render ────────────────────────────────────────────────────────────

    def render(self) -> Optional[str]:
        if self.world is None:
            return None

        lines = [f"=== Step {self._step_count} ==="]
        for fid, f in self.world.factions.items():
            hp = self.world.faction_total_hp(fid)
            nc = self.world.faction_cluster_count(fid)
            res = f.resources.round(1)
            lines.append(f"  F{fid} ({f.name}): HP={hp:.0f} clusters={nc} res={res}")

        for c in self.world.clusters:
            owner = c.owner_faction if c.owner_faction is not None else "none"
            n_units = len(c.alive_units)
            lines.append(f"  C{c.cluster_id} [{c.terrain.name}] owner={owner} units={n_units} supply={c.supply:.1f}")

        for obj in self.world.objectives:
            status = "DONE" if obj.is_completed else f"{obj.completion_progress:.0%}"
            lines.append(f"  Obj{obj.objective_id} ({obj.name}): {status}")

        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text

    # ── Utility ───────────────────────────────────────────────────────────

    def get_action_mask(self) -> NDArray[np.bool_]:
        """Return a flat boolean mask over the action space.

        For simplicity, this returns a per-dimension mask rather than
        a full combinatorial mask.  Invalid actions get a small negative
        reward from apply_action, which is sufficient for PPO training.
        """
        # For now, return all-True (no masking)
        # TODO: implement per-dimension action masking for curriculum
        return np.ones(self.action_space.nvec.sum(), dtype=bool)

    @property
    def action_meanings(self) -> List[str]:
        return [at.name for at in ActionType]

    def set_opponent_policy(self, policy) -> None:
        """Hot-swap the opponent policy (for self-play league)."""
        self.opponent_policy = policy

    def get_world_state(self) -> Optional[CowWorldState]:
        """Direct access to world state for debugging / advanced reward."""
        return self.world
