"""
economy_core.py — Realistic GDP-based economy with factory vectors and population pools.

═══════════════════════════════════════════════════════════════════════════════
FACTORY TYPES (10-element vector per cluster)

  Each cluster has factories represented as a vector F[10]:
    0  POWER_PLANT      — Electricity. Enables all other factories. No power = nothing works.
    1  CIVIL_FACTORY     — Consumer goods, textiles, housing. → GDP + civilian morale.
    2  MIL_FACTORY       — Arms, ammunition, vehicles, equipment. → Military production.
    3  DOCKYARD          — Ships (naval production). Slow but powerful.
    4  AIRFIELD          — Aircraft assembly + maintenance. → Air production.
    5  STEEL_MILL        — Processes raw ore → steel. Feeds MIL_FACTORY + DOCKYARD.
    6  REFINERY          — Crude oil → fuel. Fuel is lifeblood of mechanized war.
    7  FARM              — Food production. Starvation = revolt.
    8  HOSPITAL          — Medical care. Reduces attrition, heals wounded, lowers death rate.
    9  INFRASTRUCTURE    — Roads, rail, ports, telecom. Logistics capacity.

  Factory Properties:
    level: int (0-20) — capacity. More levels = more output.
    damage: float (0-1) — from bombing. Reduces effective output.
    workers: int — assigned workforce. Understaffed = reduced output.
    efficiency: float (0-1) — tech level + management. Improves slowly.

  Output = level × (1 - damage) × min(workers/required_workers, 1.0) × efficiency
  Power dependency: all factories except FARM operate at power_ratio effectiveness
  Anti-exploitation: building factories takes turns, bombing is fast, repair is slow

═══════════════════════════════════════════════════════════════════════════════
GDP MODEL

  GDP = Σ(factory_output × output_value) for all clusters in faction
  GDP components:
    - Industrial GDP (MIL_FACTORY + STEEL_MILL + REFINERY output)
    - Agricultural GDP (FARM output)
    - Services GDP (CIVIL_FACTORY + HOSPITAL output)
    - Military GDP (MIL_FACTORY + DOCKYARD + AIRFIELD — counts for war score not civilian)

  GDP per capita = GDP / population → affects morale, recruitment
  War economy ratio = military_gdp / total_gdp → too high = civilian suffering

═══════════════════════════════════════════════════════════════════════════════
POPULATION POOLS (per cluster)

  total_population      — everyone
  working_age           — 16-60, subset of total (~60%)
  employed_civilian     — working in factories
  unemployed            — available for conscription or factory work
  active_military       — fighting troops at the front
  reserves              — trained but not deployed (can be activated quickly)
  garrison              — defending this cluster (home guard)
  in_training           — raw recruits going through pipeline
  wounded               — in hospitals, healing (depends on HOSPITAL capacity)
  prisoners_of_war      — captured (neither working nor fighting)

  Conservation: working_age = employed + unemployed + active + reserves + garrison + training + wounded + pow
  Conscription draws from: unemployed first, then employed (with GDP penalty)
  Training pipeline: recruit → 5 turns basic → active_military
  Quality levels: GREEN (0-2 turns combat), REGULAR (3-10), VETERAN (10-30), ELITE (30+)

═══════════════════════════════════════════════════════════════════════════════
MILITARY LOGISTICS

  Supply Chain: FARM/REFINERY/MIL_FACTORY → Depot → Front Line
  Throughput limited by: INFRASTRUCTURE level × (1 - bombing_damage)
  Each military unit consumes per turn:
    - Food: 1 unit per 1000 troops
    - Fuel: 0.5 per 1000 troops (more for mechanized)
    - Ammo: 2 per 1000 troops in combat, 0.2 at rest
    - Equipment: 0.1 per 1000 troops (wear and tear)

  If supply < demand:
    - Ammo shortage: -50% combat effectiveness
    - Fuel shortage: -30% movement, mechanized units immobilized
    - Food shortage: attrition + morale collapse
    - Equipment shortage: increasing breakdown rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════ #
# Factory Types                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class FactoryType(Enum):
    POWER_PLANT   = 0
    CIVIL_FACTORY = 1
    MIL_FACTORY   = 2
    DOCKYARD      = 3
    AIRFIELD      = 4
    STEEL_MILL    = 5
    REFINERY      = 6
    FARM          = 7
    HOSPITAL      = 8
    INFRASTRUCTURE = 9

N_FACTORY_TYPES = 10

# Workers required per factory level
WORKERS_PER_LEVEL = {
    FactoryType.POWER_PLANT:   500,
    FactoryType.CIVIL_FACTORY: 800,
    FactoryType.MIL_FACTORY:   1000,
    FactoryType.DOCKYARD:      1200,
    FactoryType.AIRFIELD:      600,
    FactoryType.STEEL_MILL:    900,
    FactoryType.REFINERY:      700,
    FactoryType.FARM:          400,
    FactoryType.HOSPITAL:      300,
    FactoryType.INFRASTRUCTURE: 600,
}

# GDP value per unit of output
GDP_VALUE = {
    FactoryType.POWER_PLANT:   0.8,
    FactoryType.CIVIL_FACTORY: 1.5,
    FactoryType.MIL_FACTORY:   2.0,
    FactoryType.DOCKYARD:      2.5,
    FactoryType.AIRFIELD:      1.8,
    FactoryType.STEEL_MILL:    1.2,
    FactoryType.REFINERY:      1.6,
    FactoryType.FARM:          0.6,
    FactoryType.HOSPITAL:      0.5,
    FactoryType.INFRASTRUCTURE: 0.4,
}

# Construction time (turns to build 1 level)
BUILD_TIME = {
    FactoryType.POWER_PLANT:   8,
    FactoryType.CIVIL_FACTORY: 5,
    FactoryType.MIL_FACTORY:   6,
    FactoryType.DOCKYARD:      10,
    FactoryType.AIRFIELD:      4,
    FactoryType.STEEL_MILL:    7,
    FactoryType.REFINERY:      8,
    FactoryType.FARM:          3,
    FactoryType.HOSPITAL:      4,
    FactoryType.INFRASTRUCTURE: 6,
}

# Repair rate per turn (damage reduction)
REPAIR_RATE = 0.02  # 2% per turn — bombing is fast, repair is slow


# ═══════════════════════════════════════════════════════════════════════════ #
# Factory State                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class Factory:
    """A single factory in a cluster."""
    factory_type: FactoryType
    level: int = 1                # 0-20 capacity
    damage: float = 0.0           # 0-1 bombing damage
    workers_assigned: int = 0
    efficiency: float = 0.7       # 0-1 tech + management
    building_progress: float = 0.0  # 0-1 progress toward next level
    is_building: bool = False

    @property
    def required_workers(self) -> int:
        return self.level * WORKERS_PER_LEVEL[self.factory_type]

    @property
    def staffing_ratio(self) -> float:
        if self.required_workers == 0:
            return 1.0
        return min(1.0, self.workers_assigned / max(self.required_workers, 1))

    @property
    def effective_output(self) -> float:
        """Output considering damage, staffing, and efficiency."""
        return self.level * (1.0 - self.damage) * self.staffing_ratio * self.efficiency

    @property
    def gdp_contribution(self) -> float:
        return self.effective_output * GDP_VALUE[self.factory_type]


# ═══════════════════════════════════════════════════════════════════════════ #
# Population Pool                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class PopulationPool:
    """Population state for one cluster."""
    total_population: int = 100000
    working_age_ratio: float = 0.60   # fraction of total that's working age

    # Pool breakdown (within working age)
    employed_civilian: int = 30000
    unemployed: int = 15000
    active_military: int = 5000
    reserves: int = 2000
    garrison: int = 3000
    in_training: int = 0
    wounded: int = 0
    prisoners_of_war: int = 0

    # Training pipeline
    training_turns_remaining: int = 0  # turns until current batch graduates
    training_batch_size: int = 0
    training_quality: float = 0.5      # 0=green, 1=well-trained

    # Demographics
    birth_rate: float = 0.001          # per turn population growth
    death_rate: float = 0.0005         # per turn natural death
    war_casualties_this_turn: int = 0
    civilian_casualties_this_turn: int = 0

    @property
    def working_age(self) -> int:
        return int(self.total_population * self.working_age_ratio)

    @property
    def total_military(self) -> int:
        return self.active_military + self.reserves + self.garrison + self.in_training

    @property
    def military_ratio(self) -> float:
        """Fraction of working age in military."""
        wa = self.working_age
        return self.total_military / max(wa, 1)

    @property
    def unemployment_rate(self) -> float:
        wa = self.working_age
        return self.unemployed / max(wa, 1)

    @property
    def available_for_conscription(self) -> int:
        """People who can be drafted: unemployed + some employed."""
        return self.unemployed

    @property
    def available_for_work(self) -> int:
        """People who can be assigned to factories."""
        return self.unemployed


# ═══════════════════════════════════════════════════════════════════════════ #
# Cluster Economy                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ClusterEconomy:
    """Complete economic state for one cluster."""
    cluster_id: int
    factories: List[Factory] = field(default_factory=list)
    population: PopulationPool = field(default_factory=PopulationPool)

    # Supply depot
    food_stockpile: float = 100.0      # food units
    fuel_stockpile: float = 50.0       # fuel units
    ammo_stockpile: float = 50.0       # ammunition units
    equipment_stockpile: float = 30.0  # military equipment units
    steel_stockpile: float = 40.0      # processed steel

    # Production queues
    ship_production_progress: float = 0.0
    aircraft_production_progress: float = 0.0

    @property
    def power_output(self) -> float:
        """Total power generation."""
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.POWER_PLANT)

    @property
    def power_demand(self) -> float:
        """Total power demand from all non-farm factories."""
        return sum(f.level * 0.3 for f in self.factories if f.factory_type not in (FactoryType.FARM, FactoryType.POWER_PLANT))

    @property
    def power_ratio(self) -> float:
        """Power supply / demand. <1.0 means brownouts."""
        demand = self.power_demand
        if demand <= 0:
            return 1.0
        return min(1.5, self.power_output / max(demand, 0.1))

    @property
    def gdp(self) -> float:
        """Cluster GDP = sum of all factory GDP contributions × power ratio."""
        pr = min(1.0, self.power_ratio)
        return sum(f.gdp_contribution for f in self.factories) * pr

    @property
    def military_production(self) -> float:
        """Military factory output (for equipment production)."""
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.MIL_FACTORY) * min(1.0, self.power_ratio)

    @property
    def food_production(self) -> float:
        """Farm output (food doesn't need power)."""
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.FARM)

    @property
    def fuel_production(self) -> float:
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.REFINERY) * min(1.0, self.power_ratio)

    @property
    def steel_production(self) -> float:
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.STEEL_MILL) * min(1.0, self.power_ratio)

    @property
    def hospital_capacity(self) -> float:
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.HOSPITAL) * min(1.0, self.power_ratio)

    @property
    def logistics_capacity(self) -> float:
        """Infrastructure determines how much supply can flow."""
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.INFRASTRUCTURE) * min(1.0, self.power_ratio)

    @property
    def dockyard_output(self) -> float:
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.DOCKYARD) * min(1.0, self.power_ratio)

    @property
    def airfield_output(self) -> float:
        return sum(f.effective_output for f in self.factories if f.factory_type == FactoryType.AIRFIELD) * min(1.0, self.power_ratio)

    def factory_of_type(self, ft: FactoryType) -> Optional[Factory]:
        for f in self.factories:
            if f.factory_type == ft:
                return f
        return None


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction Economy                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class FactionEconomy:
    """Faction-level economic aggregates and policies."""
    faction_id: int
    war_economy_ratio: float = 0.5    # 0=all civilian, 1=total war
    tax_rate: float = 0.3             # GDP extraction for government
    debt: float = 0.0                 # accumulated war debt
    inflation: float = 0.02           # price inflation rate
    trade_balance: float = 0.0
    total_gdp: float = 0.0           # calculated each turn
    gdp_per_capita: float = 0.0
    war_weariness: float = 0.0       # 0-1, grows with casualties + time


# ═══════════════════════════════════════════════════════════════════════════ #
# Economy World                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class EconomyWorld:
    """Complete economy state for all clusters and factions."""
    clusters: List[ClusterEconomy] = field(default_factory=list)
    factions: Dict[int, FactionEconomy] = field(default_factory=dict)
    turn: int = 0


# ═══════════════════════════════════════════════════════════════════════════ #
# Economy Step                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_economy_v2(
    world: EconomyWorld,
    cluster_owners: Dict[int, int],
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[EconomyWorld, Dict[str, Any]]:
    """Advance the economy by one turn."""
    feedback: Dict[str, Any] = {}

    for ce in world.clusters:
        owner = cluster_owners.get(ce.cluster_id)
        if owner is None:
            continue

        pop = ce.population
        pr = min(1.0, ce.power_ratio)

        # ── 1. Production ─────────────────────────────────────────────── #
        ce.food_stockpile += ce.food_production * dt
        ce.fuel_stockpile += ce.fuel_production * dt
        ce.steel_stockpile += ce.steel_production * dt
        ce.ammo_stockpile += ce.military_production * 0.5 * dt
        ce.equipment_stockpile += ce.military_production * 0.3 * dt

        # ── 2. Consumption ────────────────────────────────────────────── #
        # Food: everyone eats
        food_demand = pop.total_population / 50000.0 * dt
        ce.food_stockpile = max(0, ce.food_stockpile - food_demand)

        # Fuel: military + transport
        fuel_demand = (pop.active_military + pop.garrison) / 20000.0 * dt
        ce.fuel_stockpile = max(0, ce.fuel_stockpile - fuel_demand)

        # Ammo: military in combat uses more (handled externally)
        ammo_base = (pop.active_military + pop.garrison) / 50000.0 * dt
        ce.ammo_stockpile = max(0, ce.ammo_stockpile - ammo_base)

        # Equipment wear
        equip_wear = pop.total_military / 100000.0 * dt
        ce.equipment_stockpile = max(0, ce.equipment_stockpile - equip_wear)

        # ── 3. Population dynamics ────────────────────────────────────── #
        # Natural growth
        births = int(pop.total_population * pop.birth_rate * dt)
        deaths = int(pop.total_population * pop.death_rate * dt)
        pop.total_population += births - deaths
        pop.unemployed += max(0, int(births * pop.working_age_ratio) - deaths)

        # Starvation check
        if ce.food_stockpile <= 0:
            starve = int(pop.total_population * 0.001 * dt)
            pop.total_population = max(1000, pop.total_population - starve)
            pop.civilian_casualties_this_turn += starve

        # Wounded → healed (depends on hospital capacity)
        if pop.wounded > 0:
            heal_rate = min(pop.wounded, int(ce.hospital_capacity * 100 * dt))
            pop.wounded -= heal_rate
            pop.reserves += heal_rate  # healed soldiers return to reserves

        # Training pipeline
        if pop.in_training > 0 and pop.training_turns_remaining > 0:
            pop.training_turns_remaining -= 1
            if pop.training_turns_remaining <= 0:
                # Graduates!
                pop.active_military += pop.in_training
                pop.in_training = 0

        # ── 4. Factory auto-repair ────────────────────────────────────── #
        for f in ce.factories:
            if f.damage > 0:
                f.damage = max(0, f.damage - REPAIR_RATE * dt)
            # Building progress
            if f.is_building:
                build_speed = 1.0 / BUILD_TIME[f.factory_type]
                f.building_progress += build_speed * pr * dt
                if f.building_progress >= 1.0:
                    f.level += 1
                    f.building_progress = 0.0
                    f.is_building = False

        # ── 5. Auto-assign workers ────────────────────────────────────── #
        total_demand = sum(f.required_workers for f in ce.factories)
        available = pop.employed_civilian + pop.unemployed
        if total_demand > 0 and available > 0:
            for f in ce.factories:
                share = f.required_workers / max(total_demand, 1)
                f.workers_assigned = int(available * share)

    # ── 6. Faction-level GDP calculation ──────────────────────────────── #
    for fid, fe in world.factions.items():
        total_gdp = 0.0
        total_pop = 0
        for ce in world.clusters:
            if cluster_owners.get(ce.cluster_id) == fid:
                total_gdp += ce.gdp
                total_pop += ce.population.total_population
        fe.total_gdp = total_gdp
        fe.gdp_per_capita = total_gdp / max(total_pop / 10000, 1)
        # War weariness grows
        fe.war_weariness = min(1.0, fe.war_weariness + 0.001 * dt)
        # Inflation from war economy
        fe.inflation = 0.02 + fe.war_economy_ratio * 0.05
        # Debt from deficit spending
        fe.debt += fe.war_economy_ratio * total_gdp * 0.01 * dt

    world.turn += 1
    return world, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

# Default factory profiles per terrain/role
FACTORY_PROFILES = {
    "capital": {  # London, Paris
        FactoryType.POWER_PLANT: 6, FactoryType.CIVIL_FACTORY: 5,
        FactoryType.MIL_FACTORY: 3, FactoryType.STEEL_MILL: 2,
        FactoryType.FARM: 2, FactoryType.HOSPITAL: 4,
        FactoryType.INFRASTRUCTURE: 5, FactoryType.REFINERY: 2,
    },
    "industrial": {  # Birmingham, Rouen, Glasgow, Cardiff
        FactoryType.POWER_PLANT: 5, FactoryType.CIVIL_FACTORY: 2,
        FactoryType.MIL_FACTORY: 5, FactoryType.STEEL_MILL: 4,
        FactoryType.FARM: 2, FactoryType.HOSPITAL: 2,
        FactoryType.INFRASTRUCTURE: 3, FactoryType.REFINERY: 3,
    },
    "naval_base": {  # Portsmouth, Plymouth, Dunkirk, Antwerp
        FactoryType.POWER_PLANT: 4, FactoryType.CIVIL_FACTORY: 2,
        FactoryType.MIL_FACTORY: 3, FactoryType.DOCKYARD: 5,
        FactoryType.STEEL_MILL: 3, FactoryType.FARM: 2,
        FactoryType.HOSPITAL: 2, FactoryType.INFRASTRUCTURE: 4,
    },
    "airbase": {  # Norwich, Canterbury
        FactoryType.POWER_PLANT: 3, FactoryType.CIVIL_FACTORY: 2,
        FactoryType.MIL_FACTORY: 2, FactoryType.AIRFIELD: 5,
        FactoryType.FARM: 3, FactoryType.HOSPITAL: 2,
        FactoryType.INFRASTRUCTURE: 3,
    },
    "port_city": {  # Liverpool, Southampton, Le Havre
        FactoryType.POWER_PLANT: 4, FactoryType.CIVIL_FACTORY: 3,
        FactoryType.DOCKYARD: 3, FactoryType.REFINERY: 3,
        FactoryType.FARM: 2, FactoryType.HOSPITAL: 2,
        FactoryType.INFRASTRUCTURE: 4,
    },
    "agricultural": {  # Norfolk, rural areas
        FactoryType.POWER_PLANT: 2, FactoryType.CIVIL_FACTORY: 2,
        FactoryType.FARM: 6, FactoryType.HOSPITAL: 2,
        FactoryType.INFRASTRUCTURE: 3,
    },
    "garrison": {  # Dover, Calais, frontline
        FactoryType.POWER_PLANT: 3, FactoryType.MIL_FACTORY: 3,
        FactoryType.FARM: 2, FactoryType.HOSPITAL: 3,
        FactoryType.INFRASTRUCTURE: 4,
    },
    "sub_base": {  # Brest, Cherbourg
        FactoryType.POWER_PLANT: 3, FactoryType.DOCKYARD: 4,
        FactoryType.MIL_FACTORY: 2, FactoryType.REFINERY: 3,
        FactoryType.FARM: 2, FactoryType.HOSPITAL: 2,
        FactoryType.INFRASTRUCTURE: 3,
    },
    "logistics_hub": {  # Amiens, Orleans, Leeds
        FactoryType.POWER_PLANT: 4, FactoryType.CIVIL_FACTORY: 3,
        FactoryType.FARM: 3, FactoryType.HOSPITAL: 2,
        FactoryType.INFRASTRUCTURE: 6, FactoryType.REFINERY: 2,
    },
    "default": {
        FactoryType.POWER_PLANT: 3, FactoryType.CIVIL_FACTORY: 3,
        FactoryType.MIL_FACTORY: 2, FactoryType.FARM: 3,
        FactoryType.HOSPITAL: 2, FactoryType.INFRASTRUCTURE: 3,
    },
}

# Cluster role assignments for the 32-sector map
CLUSTER_ROLES = {
    0: "capital",        # London
    1: "garrison",       # Dover
    2: "naval_base",     # Portsmouth
    3: "port_city",      # Southampton
    4: "airbase",        # Canterbury
    5: "garrison",       # Brighton
    6: "industrial",     # Bristol
    7: "naval_base",     # Plymouth
    8: "industrial",     # Cardiff
    9: "industrial",     # Birmingham
    10: "industrial",    # Manchester
    11: "port_city",     # Liverpool
    12: "logistics_hub", # Leeds
    13: "airbase",       # Norwich
    14: "naval_base",    # Edinburgh
    15: "industrial",    # Glasgow
    16: "port_city",     # Dublin
    17: "naval_base",    # Belfast
    18: "garrison",      # Calais
    19: "naval_base",    # Dunkirk
    20: "port_city",     # Le Havre
    21: "sub_base",      # Cherbourg
    22: "logistics_hub", # Amiens
    23: "industrial",    # Rouen
    24: "industrial",    # Lille
    25: "capital",       # Brussels (Benelux Command)
    26: "naval_base",    # Antwerp
    27: "capital",       # Paris
    28: "logistics_hub", # Orleans
    29: "industrial",    # Lyon
    30: "sub_base",      # Brest
    31: "default",       # Bordeaux
}


def _create_factories(role: str, rng: np.random.Generator) -> List[Factory]:
    """Create factories for a cluster based on its role."""
    profile = FACTORY_PROFILES.get(role, FACTORY_PROFILES["default"])
    factories = []
    for ft, level in profile.items():
        factories.append(Factory(
            factory_type=ft,
            level=level,
            damage=0.0,
            efficiency=0.6 + rng.uniform(0, 0.2),
        ))
    return factories


def _create_population(total_pop: int, military_ratio: float, rng: np.random.Generator) -> PopulationPool:
    """Create population pool for a cluster."""
    working_age = int(total_pop * 0.60)
    military = int(working_age * military_ratio)
    employed = int(working_age * 0.45)
    unemployed = working_age - employed - military
    return PopulationPool(
        total_population=total_pop,
        employed_civilian=employed,
        unemployed=max(0, unemployed),
        active_military=int(military * 0.5),
        reserves=int(military * 0.2),
        garrison=int(military * 0.3),
    )


def initialize_economy_v2(
    n_clusters: int,
    cluster_owners: Dict[int, int],
    populations: np.ndarray,  # population scale per cluster
    rng: np.random.Generator,
) -> EconomyWorld:
    """Initialize the V2 economy for all clusters."""
    clusters = []
    for cid in range(n_clusters):
        role = CLUSTER_ROLES.get(cid, "default")
        pop_scale = populations[cid] if cid < len(populations) else 0.5
        total_pop = int(pop_scale * 200000)  # scale to realistic numbers

        owner = cluster_owners.get(cid, 0)
        mil_ratio = 0.12 if cid in (1, 5, 18, 19) else 0.08  # frontline has more military

        factories = _create_factories(role, rng)
        pop_pool = _create_population(total_pop, mil_ratio, rng)

        # Pre-assign workers to factories based on available workforce
        total_demand = sum(f.required_workers for f in factories)
        available_workers = pop_pool.employed_civilian + pop_pool.unemployed
        if total_demand > 0 and available_workers > 0:
            for f in factories:
                share = f.required_workers / max(total_demand, 1)
                f.workers_assigned = int(available_workers * share)

        ce = ClusterEconomy(
            cluster_id=cid,
            factories=factories,
            population=pop_pool,
            food_stockpile=50 + pop_scale * 100,
            fuel_stockpile=30 + pop_scale * 50,
            ammo_stockpile=20 + pop_scale * 40,
            equipment_stockpile=15 + pop_scale * 30,
            steel_stockpile=20 + pop_scale * 40,
        )
        clusters.append(ce)

    factions = {}
    for fid in set(cluster_owners.values()):
        factions[fid] = FactionEconomy(faction_id=fid)

    return EconomyWorld(clusters=clusters, factions=factions)


# ═══════════════════════════════════════════════════════════════════════════ #
# Observation + Summary                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def economy_obs(world: EconomyWorld, faction_id: int, cluster_owners: Dict[int, int]) -> np.ndarray:
    """Observation vector for the economy (for RL or analysis)."""
    obs = []
    for ce in world.clusters:
        if cluster_owners.get(ce.cluster_id) == faction_id:
            obs.extend([
                ce.gdp / 100.0,
                ce.power_ratio,
                ce.food_stockpile / 200.0,
                ce.fuel_stockpile / 100.0,
                ce.ammo_stockpile / 100.0,
                ce.population.unemployment_rate,
                ce.population.military_ratio,
                ce.military_production / 10.0,
            ])
    return np.array(obs, dtype=np.float32)


def economy_summary(world: EconomyWorld, faction_id: int, cluster_owners: Dict[int, int], cluster_names: List[str]) -> str:
    """Generate a concise economy summary for turn reports."""
    fe = world.factions.get(faction_id)
    if not fe:
        return "No economic data."

    lines = []
    lines.append(f"GDP: {fe.total_gdp:.0f} | Per capita: {fe.gdp_per_capita:.1f} | Inflation: {fe.inflation:.1%} | War weariness: {fe.war_weariness:.0%}")

    # Critical shortages
    shortages = []
    for ce in world.clusters:
        if cluster_owners.get(ce.cluster_id) != faction_id:
            continue
        name = cluster_names[ce.cluster_id] if ce.cluster_id < len(cluster_names) else f"C{ce.cluster_id}"
        if ce.food_stockpile < 10:
            shortages.append(f"FOOD in {name}")
        if ce.fuel_stockpile < 10:
            shortages.append(f"FUEL in {name}")
        if ce.ammo_stockpile < 5:
            shortages.append(f"AMMO in {name}")
        if ce.power_ratio < 0.7:
            shortages.append(f"POWER in {name}")

    if shortages:
        lines.append(f"SHORTAGES: {', '.join(shortages[:5])}")
    else:
        lines.append("Supply lines adequate. No critical shortages.")

    # Factories under construction (so LLMs don't re-order)
    building = []
    for ce in world.clusters:
        if cluster_owners.get(ce.cluster_id) != faction_id:
            continue
        name = cluster_names[ce.cluster_id] if ce.cluster_id < len(cluster_names) else f"C{ce.cluster_id}"
        for f in ce.factories:
            if f.is_building:
                building.append(f"{f.factory_type.name} in {name}")
    if building:
        lines.append(f"UNDER CONSTRUCTION (DO NOT re-order): {', '.join(building[:4])}")

    # Top producers
    mil_prod = []
    for ce in world.clusters:
        if cluster_owners.get(ce.cluster_id) == faction_id and ce.military_production > 1:
            name = cluster_names[ce.cluster_id] if ce.cluster_id < len(cluster_names) else f"C{ce.cluster_id}"
            mil_prod.append(f"{name}({ce.military_production:.0f})")
    if mil_prod:
        lines.append(f"Top military producers: {', '.join(mil_prod[:4])}")

    return " | ".join(lines)
