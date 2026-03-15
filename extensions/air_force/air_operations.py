"""
air_operations.py — Air force step function, actions, observation, initialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .air_state import (
    AircraftType, AircraftRole, AircraftSquadron, AirWing, AirZone,
    AirZoneControl, AirWorld, RadarStation, AIRCRAFT_STATS, N_AIRCRAFT_TYPES,
)
from .air_combat import (
    resolve_air_battle, strategic_bombing, close_air_support,
    anti_ship_strike, airborne_drop,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Actions                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class AirAction(Enum):
    NOOP               = 0
    ASSIGN_MISSION     = 1   # Set squadron mission (CAP/ESCORT/BOMB/CAS/RECON)
    STRATEGIC_BOMB     = 2   # Launch bombing raid on target cluster
    CAS_SUPPORT        = 3   # Close air support for ground battle
    ANTI_SHIP_STRIKE   = 4   # Attack enemy naval fleet
    AIRBORNE_DROP      = 5   # Paratroop operation
    AIR_SUPPLY         = 6   # Supply besieged cluster by air
    BUILD_SQUADRON     = 7   # Queue aircraft production
    REBASE             = 8   # Move squadron to different airfield
    REPAIR_REFIT       = 9   # Repair + resupply squadron at base


N_AIR_ACTIONS = len(AirAction)


def apply_air_action(
    aw: AirWorld,
    faction_id: int,
    action_type: int,
    target_zone_id: int,
    squadron_idx: int,
    aircraft_type_idx: int,
    rng: np.random.Generator,
) -> Tuple[AirWorld, float]:
    """Apply an air action. Returns (updated_world, reward)."""
    try:
        action = AirAction(action_type)
    except ValueError:
        return aw, 0.0

    if action == AirAction.NOOP:
        return aw, 0.0

    faction_sqs = []
    for wing in aw.air_wings:
        if wing.faction_id == faction_id:
            faction_sqs.extend(wing.squadrons)

    if action == AirAction.ASSIGN_MISSION:
        missions = ["STANDBY", "CAP", "ESCORT", "INTERCEPT", "BOMB", "CAS", "RECON", "TRANSPORT"]
        mission = missions[min(target_zone_id, len(missions) - 1)]
        if squadron_idx < len(faction_sqs):
            faction_sqs[squadron_idx].mission = mission
            return aw, 0.1
        return aw, -0.05

    elif action == AirAction.STRATEGIC_BOMB:
        if target_zone_id >= len(aw.air_zones):
            return aw, -0.1
        bombers = [sq for sq in faction_sqs if sq.stats.role == AircraftRole.BOMBER and sq.can_sortie]
        escorts = [sq for sq in faction_sqs if sq.stats.can_escort and sq.can_sortie]
        defenders = []
        for wing in aw.air_wings:
            if wing.faction_id != faction_id:
                defenders.extend([sq for sq in wing.squadrons
                                  if sq.stats.role == AircraftRole.FIGHTER and sq.is_operational])
        result = strategic_bombing(bombers, escorts, aw.air_zones[target_zone_id], defenders, rng)
        return aw, result["damage_dealt"] * 0.05 - result["bomber_losses"] * 2.0

    elif action == AirAction.CAS_SUPPORT:
        if target_zone_id >= len(aw.air_zones):
            return aw, -0.1
        cas_sqs = [sq for sq in faction_sqs
                   if sq.stats.role in (AircraftRole.CAS, AircraftRole.BOMBER) and sq.can_sortie]
        result = close_air_support(cas_sqs, aw.air_zones[target_zone_id], rng)
        return aw, result["ground_damage"] * 0.1 - result["cas_losses"] * 1.5

    elif action == AirAction.ANTI_SHIP_STRIKE:
        if target_zone_id >= len(aw.air_zones):
            return aw, -0.1
        strikers = [sq for sq in faction_sqs if sq.stats.naval_attack > 1.0 and sq.can_sortie]
        result = anti_ship_strike(strikers, aw.air_zones[target_zone_id], 3.0, rng)
        return aw, result["naval_damage"] * 0.08 - result["aircraft_losses"] * 2.0

    elif action == AirAction.AIRBORNE_DROP:
        if target_zone_id >= len(aw.air_zones):
            return aw, -0.1
        transports = [sq for sq in faction_sqs if sq.stats.carry_capacity > 0 and sq.can_sortie]
        defenders = []
        for wing in aw.air_wings:
            if wing.faction_id != faction_id:
                defenders.extend([sq for sq in wing.squadrons
                                  if sq.stats.role == AircraftRole.FIGHTER])
        result = airborne_drop(transports, aw.air_zones[target_zone_id], defenders, rng)
        return aw, result["troops_dropped"] * 0.01 - result["transport_losses"] * 3.0

    elif action == AirAction.BUILD_SQUADRON:
        at_idx = min(aircraft_type_idx, N_AIRCRAFT_TYPES - 1)
        at = AircraftType(at_idx)
        stats = AIRCRAFT_STATS[at]
        if faction_id not in aw.production_queues:
            aw.production_queues[faction_id] = []
        if len(aw.production_queues[faction_id]) >= 5:
            return aw, -0.1
        aw.production_queues[faction_id].append((at, stats.build_time))
        return aw, 0.2

    elif action == AirAction.REPAIR_REFIT:
        if squadron_idx < len(faction_sqs):
            sq = faction_sqs[squadron_idx]
            sq.strength = min(1.0, sq.strength + 0.05)
            sq.fuel = min(1.0, sq.fuel + 0.3)
            sq.ammo = min(1.0, sq.ammo + 0.3)
            sq.crew_fatigue = max(0.0, sq.crew_fatigue - 0.1)
            return aw, 0.1
        return aw, -0.05

    return aw, 0.0


# ═══════════════════════════════════════════════════════════════════════════ #
# Air Zone Control Update                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

def update_air_zone_control(aw: AirWorld) -> None:
    """Update air zone control based on fighter presence."""
    for zone in aw.air_zones:
        faction_power: Dict[int, float] = {}
        for wing in aw.air_wings:
            fighter_power = sum(
                sq.combat_power * sq.stats.air_attack
                for sq in wing.squadrons
                if sq.stats.role == AircraftRole.FIGHTER and sq.is_operational
                and sq.mission in ("CAP", "INTERCEPT", "STANDBY")
            )
            if fighter_power > 0:
                faction_power[wing.faction_id] = faction_power.get(wing.faction_id, 0) + fighter_power

        if not faction_power:
            zone.control = AirZoneControl.UNCONTESTED
            zone.controlling_faction = None
        elif len(faction_power) == 1:
            fid = list(faction_power.keys())[0]
            power = list(faction_power.values())[0]
            if power > 10.0:
                zone.control = AirZoneControl.SUPREMACY
            elif power > 3.0:
                zone.control = AirZoneControl.SUPERIORITY
            else:
                zone.control = AirZoneControl.UNCONTESTED
            zone.controlling_faction = fid
        else:
            powers = sorted(faction_power.items(), key=lambda x: x[1], reverse=True)
            top_fid, top_power = powers[0]
            second_power = powers[1][1]

            if top_power > second_power * 3.0:
                zone.control = AirZoneControl.SUPREMACY
                zone.controlling_faction = top_fid
            elif top_power > second_power * 1.5:
                zone.control = AirZoneControl.SUPERIORITY
                zone.controlling_faction = top_fid
            else:
                zone.control = AirZoneControl.CONTESTED
                zone.controlling_faction = None


# ═══════════════════════════════════════════════════════════════════════════ #
# Main Step                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_air(
    aw: AirWorld,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[AirWorld, Dict[str, float]]:
    """Advance air force simulation by one step."""
    feedback: Dict[str, float] = {
        "air_battles": 0, "bombing_damage": 0.0,
        "cas_damage": 0.0, "aircraft_lost": 0.0,
    }

    # ── 1. Weather update ─────────────────────────────────────────────── #
    for zone in aw.air_zones:
        zone.cloud_cover = float(np.clip(
            zone.cloud_cover + rng.normal(0, 0.05) * dt, 0.0, 1.0))

    # ── 2. Reset daily sorties ────────────────────────────────────────── #
    for wing in aw.air_wings:
        for sq in wing.squadrons:
            sq.sorties_today = 0

    # ── 3. CAP engagements (fighters intercept enemy aircraft) ─────── #
    # STANDBY fighters scramble when enemy detected — they don't just sit
    # at the airfield while bombers fly overhead. CAP/INTERCEPT have full
    # strength; STANDBY scrambles at 70% effectiveness (reaction delay).
    _ENGAGE_MISSIONS = ("CAP", "INTERCEPT", "STANDBY")
    for zone in aw.air_zones:
        faction_wings: Dict[int, List[AircraftSquadron]] = {}
        for wing in aw.air_wings:
            for sq in wing.squadrons:
                if sq.is_operational and sq.stats.role == AircraftRole.FIGHTER:
                    faction_wings.setdefault(wing.faction_id, []).append(sq)

        fids = list(faction_wings.keys())
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                fighters_a = [sq for sq in faction_wings[fids[i]]
                              if sq.mission in _ENGAGE_MISSIONS]
                fighters_b = [sq for sq in faction_wings[fids[j]]
                              if sq.mission in _ENGAGE_MISSIONS]
                if fighters_a and fighters_b:
                    result = resolve_air_battle(fighters_a, fighters_b, zone.cloud_cover, rng, dt)
                    feedback["air_battles"] += 1
                    feedback["aircraft_lost"] += result["attacker_losses"] + result["defender_losses"]

    # ── 4. Fuel consumption + fatigue ─────────────────────────────────── #
    for wing in aw.air_wings:
        for sq in wing.squadrons:
            if sq.mission != "STANDBY":
                sq.fuel = max(0.0, sq.fuel - sq.stats.fuel_per_sortie * 0.003 * dt)
                sq.crew_fatigue = min(1.0, sq.crew_fatigue + 0.005 * dt)
            else:
                # Rest at base
                sq.crew_fatigue = max(0.0, sq.crew_fatigue - 0.01 * dt)
                sq.morale = min(1.0, sq.morale + 0.002 * dt)

    # ── 5. Production queue ───────────────────────────────────────────── #
    for fid, queue in aw.production_queues.items():
        completed = []
        for i, (at, steps_left) in enumerate(queue):
            queue[i] = (at, steps_left - 1)
            if steps_left - 1 <= 0:
                completed.append((i, at))

        for idx, at in reversed(completed):
            queue.pop(idx)
            new_sq = AircraftSquadron(
                squadron_id=aw.alloc_squadron_id(),
                aircraft_type=at, faction_id=fid,
                strength=1.0, experience=0.0,
                based_at=0,  # assign to first available base
            )
            # Add to faction's first wing
            for wing in aw.air_wings:
                if wing.faction_id == fid:
                    wing.squadrons.append(new_sq)
                    break

    # ── 6. Zone control update ────────────────────────────────────────── #
    update_air_zone_control(aw)

    # ── 7. Remove destroyed squadrons ─────────────────────────────────── #
    for wing in aw.air_wings:
        wing.squadrons = [sq for sq in wing.squadrons if sq.strength > 0.02]

    aw.step += 1
    return aw, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_air(
    n_clusters: int,
    n_sea_zones: int,
    faction_configs: Dict[int, Dict],
    rng: np.random.Generator,
) -> AirWorld:
    """
    Initialize air world.

    faction_configs: dict of faction_id → {
        "base_clusters": [int],
        "squadrons": [{"type": str, "count": int}],
        "radar_clusters": [int],
    }
    """
    # Create air zones for all clusters + sea zones
    zones = []
    for i in range(n_clusters):
        zones.append(AirZone(zone_id=i, zone_type="LAND",
                             flak_density=rng.uniform(0.1, 0.4)))
    for i in range(n_sea_zones):
        zones.append(AirZone(zone_id=n_clusters + i, zone_type="SEA"))

    wings = []
    radars = []
    next_sq_id = 0
    next_wing_id = 0

    for fid, cfg in faction_configs.items():
        bases = cfg.get("base_clusters", [0])
        sq_configs = cfg.get("squadrons", [])
        radar_clusters = cfg.get("radar_clusters", [])

        for base in bases:
            squadrons = []
            for sq_cfg in sq_configs:
                try:
                    at = AircraftType[sq_cfg["type"]]
                except KeyError:
                    continue
                count = sq_cfg.get("count", 1)
                for _ in range(count):
                    squadrons.append(AircraftSquadron(
                        squadron_id=next_sq_id, aircraft_type=at,
                        faction_id=fid, strength=0.9 + rng.uniform(0, 0.1),
                        experience=rng.uniform(0.0, 0.3),
                        fuel=0.8 + rng.uniform(0, 0.2),
                        ammo=0.9 + rng.uniform(0, 0.1),
                        morale=0.7 + rng.uniform(0, 0.2),
                        based_at=base,
                    ))
                    next_sq_id += 1

            if squadrons:
                wings.append(AirWing(
                    wing_id=next_wing_id, faction_id=fid,
                    base_cluster=base, squadrons=squadrons,
                ))
                next_wing_id += 1

        for rc in radar_clusters:
            radars.append(RadarStation(
                cluster_id=rc, faction_id=fid,
                detection_range_km=150.0 + rng.uniform(-30, 30),
                effectiveness=0.7 + rng.uniform(0, 0.2),
            ))

    return AirWorld(
        air_zones=zones, air_wings=wings, radar_stations=radars,
        production_queues={fid: [] for fid in faction_configs},
        next_squadron_id=next_sq_id,
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# Observation                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

def air_obs(
    aw: AirWorld,
    faction_id: int,
    max_zones: int = 15,
) -> NDArray[np.float32]:
    """
    Flat observation vector for air state.

    Per zone (8 floats × max_zones):
      - control (one-hot 4)
      - flak_density, cloud_cover
      - friendly_fighter_power, enemy_fighter_power

    Global (12 floats):
      - total fighters, bombers, cas, recon, transport (5)
      - avg fuel, avg fatigue, avg experience, avg morale (4)
      - production queue length (1)
      - total enemy air power (1)
      - radar coverage (1)

    Total: 8 × max_zones + 12
    """
    n_per = 8
    n_global = 12
    obs = np.zeros(n_per * max_zones + n_global, dtype=np.float32)

    for i, zone in enumerate(aw.air_zones):
        if i >= max_zones:
            break
        o = i * n_per
        obs[o + min(zone.control.value, 3)] = 1.0
        obs[o + 4] = zone.flak_density
        obs[o + 5] = zone.cloud_cover
        # Fighter power near this zone (simplified)
        for wing in aw.air_wings:
            fp = sum(sq.combat_power * sq.stats.air_attack
                     for sq in wing.squadrons
                     if sq.stats.role == AircraftRole.FIGHTER and sq.is_operational)
            if wing.faction_id == faction_id:
                obs[o + 6] += fp / 20.0
            else:
                obs[o + 7] += fp / 20.0

    # Global
    go = n_per * max_zones
    sqs = aw.faction_squadrons(faction_id)
    by_role = {r: [] for r in AircraftRole}
    for sq in sqs:
        by_role[sq.stats.role].append(sq)

    obs[go + 0] = sum(sq.combat_power for sq in by_role[AircraftRole.FIGHTER]) / 10.0
    obs[go + 1] = sum(sq.combat_power for sq in by_role[AircraftRole.BOMBER]) / 10.0
    obs[go + 2] = sum(sq.combat_power for sq in by_role[AircraftRole.CAS]) / 10.0
    obs[go + 3] = sum(sq.combat_power for sq in by_role[AircraftRole.RECON]) / 5.0
    obs[go + 4] = sum(sq.combat_power for sq in by_role[AircraftRole.TRANSPORT]) / 5.0

    if sqs:
        obs[go + 5] = np.mean([sq.fuel for sq in sqs])
        obs[go + 6] = np.mean([sq.crew_fatigue for sq in sqs])
        obs[go + 7] = np.mean([sq.experience for sq in sqs])
        obs[go + 8] = np.mean([sq.morale for sq in sqs])

    obs[go + 9] = len(aw.production_queues.get(faction_id, [])) / 5.0

    enemy_power = sum(
        sq.combat_power * sq.stats.air_attack
        for w in aw.air_wings if w.faction_id != faction_id
        for sq in w.squadrons if sq.is_operational
    )
    obs[go + 10] = min(enemy_power / 30.0, 1.0)

    friendly_radar = sum(r.effectiveness for r in aw.radar_stations
                         if r.faction_id == faction_id and r.is_operational)
    obs[go + 11] = min(friendly_radar / 5.0, 1.0)

    return np.clip(obs, 0.0, 1.0)


def air_obs_size(max_zones: int = 15) -> int:
    return 8 * max_zones + 12
