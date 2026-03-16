"""
llm_game.py — LLM-playable turn-based game engine for Air Strip One.

Integrates all systems:
  - War Economy (7 sectors, 12 resources, Leontief production)
  - Manpower (15 conscription laws, 12 regime types, training pipeline)
  - Naval (14 ship classes, sea zones, convoys, blockade, mines)
  - Air Force (10 aircraft types, air zones, bombing, CAS, naval strikes)
  - Naval Invasion (6-phase amphibious operations)

Designed for LLM agents (Mistral, GPT, Claude):
  - Turn-based (not real-time)
  - Text summary per turn (~500-800 tokens, NOT raw observation vectors)
  - Natural language action parsing
  - Both factions can be LLM or scripted
  - Scoring system for benchmarking

Token budget: ~600 tokens per turn summary to stay within Mistral-small rate limits.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Ensure project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from extensions.war_economy.war_economy_state import (
    Resource, EconSector, N_RESOURCES, N_SECTORS, MAX_STOCKPILE,
)
from extensions.war_economy.war_economy_dynamics import (
    initialize_war_economy, step_war_economy, compute_feedback,
)
from extensions.war_economy.manpower import (
    ConscriptionLaw, RegimeType, initialize_manpower, step_manpower,
    ManpowerAction, apply_manpower_action,
)
from extensions.naval.naval_state import ShipClass, SeaZoneControl
from extensions.naval.naval_operations import (
    initialize_naval, step_naval, apply_naval_action, NavalAction,
)
from extensions.naval.naval_invasion import InvasionPlan, InvasionPhase, step_invasion
from extensions.air_force.air_state import AircraftType, AirZoneControl
from extensions.air_force.air_operations import (
    initialize_air, step_air, apply_air_action, AirAction,
)
from extensions.resistance.resistance import (
    BLFState, EscalationLevel, initialize_resistance, step_resistance,
    resistance_event_text, ESCALATION_NAMES,
)
from extensions.intelligence.intel_system import (
    IntelWorld, initialize_intelligence, step_intelligence,
    get_faction_visibility, IntelAction, apply_intel_action,
)
from gravitas.weather_bridge import (
    GameWeather, initialize_game_weather, step_game_weather,
    apply_weather_effects, describe_weather,
)
from extensions.economy_v2.economy_core import (
    EconomyWorld, initialize_economy_v2, step_economy_v2,
    economy_summary, FactoryType,
)
from extensions.pop.pop_v2 import (
    PopWorld, initialize_pop_v2, step_pop_v2,
    pop_summary, conscript as pop_conscript,
)
from extensions.research.research_system import (
    ResearchWorld, TechBranch, initialize_research, step_research,
    apply_research_action, research_summary,
)
from extensions.governance.budget_system import (
    GovernanceWorld, initialize_governance, step_governance,
    apply_budget_action, governance_summary,
)
from extensions.military.land_bridge import (
    LandWorld, initialize_land, step_land, land_summary,
)
from extensions.ministries.ministries import (
    MinistryWorld, initialize_ministries, step_ministries, ministry_reports,
)
from gravitas_engine.systems.government import (
    GovernmentType, GovernmentModifiers, get_government_modifiers, government_summary,
)
from gravitas_engine.systems.national_spirit import (
    SpiritWorld, FactionSpirits, initialize_spirits, step_spirits,
    spirits_summary, aggregate_spirit_modifiers,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Game State                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class GameState:
    """Complete integrated game state for Air Strip One."""
    manpower_clusters: list  # List[ClusterManpower]
    manpower_policies: dict  # Dict[int, FactionManpowerPolicy]
    naval: Any              # NavalWorld
    air: Any                # AirWorld
    invasions: List[InvasionPlan] = field(default_factory=list)
    resistance: Optional[BLFState] = None   # British Liberation Front
    intelligence: Optional[Any] = None       # IntelWorld — fog of war + espionage
    land: Optional[Any] = None               # LandWorld — per-sector garrisons (CoW units)
    weather: Optional[Any] = None              # GameWeather — land/sea/air conditions
    economy: Optional[Any] = None              # EconomyWorld — GDP, factories, population
    pop: Optional[Any] = None                  # PopWorld — realistic population (real numbers)
    research: Optional[Any] = None             # ResearchWorld — HOI4-style tech tree
    governance: Optional[Any] = None           # GovernanceWorld — budget + corruption
    spirits: Optional[Any] = None              # SpiritWorld — national spirits per faction
    government_types: Dict[int, str] = field(default_factory=dict)  # faction_id → GovernmentType name
    ministries: Optional[Any] = None           # MinistryWorld — autonomous government divisions

    # Cluster data (simplified ground state)
    cluster_data: Any = None  # (N, 6) array [σ, h, r, m, τ, p]
    cluster_owners: Dict[int, int] = field(default_factory=dict)
    terrain_types: List[str] = field(default_factory=list)
    cluster_names: List[str] = field(default_factory=list)
    population: Any = None   # (N,) array

    # Game meta
    turn: int = 0
    max_turns: int = 100
    faction_scores: Dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0})
    faction_names: Dict[int, str] = field(default_factory=lambda: {0: "Oceania", 1: "Eurasia"})
    game_over: bool = False
    winner: Optional[int] = None
    _blf_army_spawned: bool = False  # one-time flag: BLF uprising units created


# ═══════════════════════════════════════════════════════════════════════════ #
# Available Actions (text-parseable)                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

AVAILABLE_ACTIONS = {
    # Economy
    "REALLOCATE_LABOR": "Move workers between economic sectors in a cluster. Params: cluster, from_sector, to_sector, intensity(0-1)",
    "TRADE_PROPOSE": "Propose trade deal with other faction. Params: resource, intensity(0-1)",
    "IMPOSE_SANCTIONS": "Economic sanctions on enemy. Params: intensity(0-1)",
    "SET_MANUFACTURING": "Set military vs civilian production priority. Params: priority(0=civilian, 1=military)",
    "ISSUE_WAR_BONDS": "Production burst in a cluster at cost of debt. Params: cluster",
    # Manpower
    "TRAIN_WORKERS": "Train civilians for an economic sector. Params: cluster, sector, count",
    "TRAIN_MILITARY": "Train military recruits (proper training). Params: cluster, count",
    "CONSCRIPT": "Emergency draft (untrained, causes unrest). Params: cluster, count",
    "CHANGE_CONSCRIPTION_LAW": "Change conscription policy (takes time). Params: law_name",
    # Naval
    "BUILD_SHIP": "Queue ship construction. Params: ship_class",
    "NAVAL_MISSION": "Assign fleet mission. Params: zone, mission(PATROL/ESCORT/BLOCKADE/RAID)",
    "SHORE_BOMBARD": "Shell enemy coastal cluster from sea. Params: zone, target_cluster",
    "LAY_MINES": "Deploy mines in a sea zone. Params: zone",
    "SWEEP_MINES": "Clear enemy mines. Params: zone",
    # Air
    "BUILD_SQUADRON": "Queue aircraft production. Params: aircraft_type",
    "AIR_MISSION": "Set squadron mission. Params: mission(CAP/ESCORT/BOMB/CAS/RECON)",
    "STRATEGIC_BOMB": "Launch bombing raid on target. Params: target_cluster",
    "CAS_SUPPORT": "Close air support for ground battle. Params: target_cluster",
    "ANTI_SHIP_STRIKE": "Attack enemy fleet from air. Params: sea_zone",
    # Invasion
    "PLAN_INVASION": "Start planning naval invasion. Params: origin_cluster, target_cluster, sea_zone",
    # General
    "NOOP": "Do nothing this turn.",
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Game Initialization                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def create_game(seed: int = 42, max_turns: int = 100) -> GameState:
    """Create a new Air Strip One game — 32-sector map, 6 sea zones.

    OCEANIA (18 sectors — All British Isles):
      South England (Channel front):
        0  London        1  Dover         2  Portsmouth     3  Southampton
        4  Canterbury    5  Brighton
      Southwest + Wales:
        6  Bristol       7  Plymouth      8  Cardiff
      Midlands + North:
        9  Birmingham   10  Manchester   11  Liverpool     12  Leeds
      East Anglia:
       13  Norwich
      Scotland:
       14  Edinburgh    15  Glasgow
      Ireland (Airstrip Two):
       16  Dublin       17  Belfast

    EURASIA (14 sectors — France + Benelux):
      Channel Front:
       18  Calais       19  Dunkirk      20  Le_Havre     21  Cherbourg
      Northern France:
       22  Amiens       23  Rouen        24  Lille
      Benelux:
       25  Brussels     26  Antwerp
      Central France:
       27  Paris        28  Orleans      29  Lyon
      Atlantic France:
       30  Brest        31  Bordeaux

    SEA ZONES (6):
      0  Dover Strait (33km)           Dover↔Calais
      1  Western Channel (120km)       Portsmouth/Brighton↔Dunkirk/Le_Havre
      2  North Sea (500km)             Liverpool/Leeds/Edinburgh↔Antwerp
      3  Irish Sea (300km)             Dublin/Belfast↔Liverpool/Glasgow
      4  Bay of Biscay (600km)         Plymouth↔Brest/Bordeaux/Cherbourg
      5  North Atlantic (open ocean)   Dublin/Glasgow↔convoys, Brest subs
    """
    rng = np.random.default_rng(seed)

    cluster_names = [
        # Oceania — South England (0-5)
        "London", "Dover", "Portsmouth", "Southampton", "Canterbury", "Brighton",
        # Oceania — Southwest + Wales (6-8)
        "Bristol", "Plymouth", "Cardiff",
        # Oceania — Midlands + North (9-12)
        "Birmingham", "Manchester", "Liverpool", "Leeds",
        # Oceania — East Anglia (13)
        "Norwich",
        # Oceania — Scotland (14-15)
        "Edinburgh", "Glasgow",
        # Oceania — Ireland / Airstrip Two (16-17)
        "Dublin", "Belfast",
        # Eurasia — Channel Front (18-21)
        "Calais", "Dunkirk", "Le_Havre", "Cherbourg",
        # Eurasia — Northern France (22-24)
        "Amiens", "Rouen", "Lille",
        # Eurasia — Benelux (25-29)
        "Brussels", "Antwerp", "Rotterdam", "Amsterdam", "Luxembourg",
        # Eurasia — Central France (30-32)
        "Paris", "Orleans", "Lyon",
        # Eurasia — Atlantic France (33-34)
        "Brest", "Bordeaux",
    ]
    terrain_types = [
        # Oceania (18)
        "URBAN", "URBAN", "URBAN", "URBAN", "OPEN", "OPEN",      # South
        "URBAN", "URBAN", "URBAN",                                 # SW+Wales
        "URBAN", "URBAN", "URBAN", "URBAN",                        # Midlands+North
        "OPEN",                                                     # East Anglia
        "URBAN", "URBAN",                                           # Scotland
        "URBAN", "URBAN",                                           # Ireland
        # Eurasia (17)
        "OPEN", "URBAN", "URBAN", "URBAN",                         # Channel
        "PLAINS", "URBAN", "PLAINS",                                # N France
        "URBAN", "URBAN", "URBAN", "URBAN", "URBAN",               # Benelux (Brussels, Antwerp, Rotterdam, Amsterdam, Luxembourg)
        "URBAN", "PLAINS", "URBAN",                                 # Central
        "URBAN", "PLAINS",                                          # Atlantic
    ]
    cluster_owners = {}
    for i in range(18):
        cluster_owners[i] = 0   # Oceania
    for i in range(18, 35):
        cluster_owners[i] = 1   # Eurasia

    population = np.array([
        # Oceania
        0.85, 0.30, 0.50, 0.60, 0.40, 0.35,  # South England
        0.55, 0.35, 0.50,                       # SW+Wales (Plymouth small, Cardiff industrial)
        0.75, 0.70, 0.60, 0.55,                 # Midlands+North
        0.30,                                     # Norwich (rural)
        0.50, 0.65,                               # Scotland (Edinburgh med, Glasgow industrial)
        0.45, 0.40,                               # Ireland
        # Eurasia
        0.40, 0.35, 0.30, 0.25,                 # Channel front
        0.45, 0.55, 0.45,                         # N France
        0.65, 0.50, 0.65, 0.70, 0.30,            # Benelux (Brussels, Antwerp, Rotterdam, Amsterdam, Luxembourg)
        0.80, 0.40, 0.60,                         # Central (Paris big)
        0.30, 0.35,                               # Atlantic
    ], dtype=np.float64)

    # Initial cluster data [σ=stability, h=hazard, r=resources, m=military, τ=trust, p=polarization]
    cluster_data = np.array([
        # ── OCEANIA — South England (Channel front) ──────────────────── #
        [0.55, 0.20, 0.70, 0.45, 0.35, 0.40],  # 0  London
        [0.60, 0.30, 0.55, 0.70, 0.50, 0.25],  # 1  Dover
        [0.55, 0.15, 0.65, 0.55, 0.45, 0.30],  # 2  Portsmouth
        [0.50, 0.15, 0.75, 0.30, 0.40, 0.35],  # 3  Southampton
        [0.65, 0.05, 0.60, 0.40, 0.55, 0.20],  # 4  Canterbury
        [0.55, 0.10, 0.50, 0.35, 0.45, 0.25],  # 5  Brighton
        # ── OCEANIA — Southwest + Wales ───────────────────────────────── #
        [0.65, 0.05, 0.80, 0.25, 0.50, 0.20],  # 6  Bristol — aircraft factories
        [0.60, 0.08, 0.55, 0.40, 0.48, 0.22],  # 7  Plymouth — naval base, W Approaches
        [0.62, 0.03, 0.75, 0.20, 0.45, 0.25],  # 8  Cardiff — coal + steel
        # ── OCEANIA — Midlands + North ────────────────────────────────── #
        [0.70, 0.03, 0.85, 0.20, 0.55, 0.15],  # 9  Birmingham — tanks
        [0.68, 0.03, 0.80, 0.20, 0.50, 0.18],  # 10 Manchester — munitions
        [0.60, 0.08, 0.70, 0.35, 0.48, 0.22],  # 11 Liverpool — convoy port
        [0.65, 0.03, 0.75, 0.25, 0.50, 0.18],  # 12 Leeds — industry + rail
        # ── OCEANIA — East Anglia ─────────────────────────────────────── #
        [0.70, 0.05, 0.65, 0.30, 0.55, 0.15],  # 13 Norwich — RAF Bomber Command
        # ── OCEANIA — Scotland ────────────────────────────────────────── #
        [0.72, 0.02, 0.60, 0.35, 0.50, 0.20],  # 14 Edinburgh — Forth anchorage
        [0.68, 0.02, 0.75, 0.25, 0.48, 0.22],  # 15 Glasgow — Clyde shipyards
        # ── OCEANIA — Ireland (Airstrip Two) ──────────────────────────── #
        [0.62, 0.05, 0.65, 0.30, 0.45, 0.25],  # 16 Dublin — Atlantic Fleet
        [0.60, 0.03, 0.70, 0.25, 0.42, 0.28],  # 17 Belfast — Harland&Wolff shipyards
        # ── EURASIA — Channel Front ───────────────────────────────────── #
        [0.45, 0.35, 0.50, 0.75, 0.40, 0.30],  # 18 Calais — invasion staging
        [0.50, 0.20, 0.60, 0.60, 0.45, 0.25],  # 19 Dunkirk — fleet base
        [0.52, 0.15, 0.55, 0.50, 0.42, 0.28],  # 20 Le_Havre — Normandy port
        [0.55, 0.10, 0.50, 0.45, 0.45, 0.22],  # 21 Cherbourg — sub base
        # ── EURASIA — Northern France ─────────────────────────────────── #
        [0.60, 0.10, 0.80, 0.30, 0.50, 0.20],  # 22 Amiens — rail junction
        [0.55, 0.10, 0.75, 0.25, 0.50, 0.25],  # 23 Rouen — steel mills
        [0.65, 0.05, 0.65, 0.45, 0.50, 0.20],  # 24 Lille — reserves + coal
        # ── EURASIA — Benelux (expanded: Belgium + Netherlands + Luxembourg) ── #
        [0.60, 0.05, 0.70, 0.50, 0.50, 0.20],  # 25 Brussels — Benelux Command
        [0.55, 0.08, 0.75, 0.45, 0.48, 0.22],  # 26 Antwerp — port, North Sea fleet
        [0.58, 0.10, 0.80, 0.55, 0.50, 0.20],  # 27 Rotterdam — Europoort, North Sea Fleet Pride
        [0.62, 0.05, 0.70, 0.35, 0.52, 0.18],  # 28 Amsterdam — trade, finance, industry
        [0.65, 0.02, 0.60, 0.20, 0.55, 0.15],  # 29 Luxembourg — steel, quiet rear area
        # ── EURASIA — Central France ────────────────────────────────────── #
        [0.65, 0.05, 0.70, 0.40, 0.55, 0.30],  # 30 Paris — Command HQ
        [0.68, 0.03, 0.75, 0.25, 0.52, 0.18],  # 31 Orleans — Loire logistics
        [0.70, 0.02, 0.80, 0.20, 0.55, 0.15],  # 32 Lyon — southern industry
        # ── EURASIA — Atlantic France ───────────────────────────────────── #
        [0.55, 0.08, 0.55, 0.50, 0.45, 0.25],  # 33 Brest — sub pens
        [0.62, 0.03, 0.65, 0.25, 0.50, 0.20],  # 34 Bordeaux — reserves
    ], dtype=np.float64)

    n = len(cluster_names)
    mp_clusters, mp_policies = initialize_manpower(
        n_clusters=n, faction_ids=[0, 1],
        cluster_owners=cluster_owners, terrain_types=terrain_types, rng=rng,
    )
    mp_policies[0].regime_type = RegimeType.TOTALITARIAN
    mp_policies[0].conscription_law = ConscriptionLaw.TOTAL_MOBILISATION
    mp_policies[1].regime_type = RegimeType.COMMUNIST
    mp_policies[1].conscription_law = ConscriptionLaw.GENERAL_MOBILISATION

    # ═══════════════════════════════════════════════════════════════════════ #
    # Naval Forces — 6 Sea Zones                                              #
    # ═══════════════════════════════════════════════════════════════════════ #
    naval = initialize_naval([
        # Zone 0: Dover Strait (33km) — Dover↔Calais
        {"name": "Dover Strait", "connected_clusters": [1, 18],
         "width_km": 33, "adjacent_zones": [1],
         "initial_fleets": {
             0: ["HEAVY_CRUISER", "DESTROYER", "DESTROYER", "DESTROYER",
                  "CORVETTE", "CORVETTE", "CORVETTE", "DESTROYER_ESCORT",
                  "DESTROYER_ESCORT", "MINELAYER"],
             1: ["FLEET_SUBMARINE", "FLEET_SUBMARINE", "FLEET_SUBMARINE",
                  "COASTAL_SUBMARINE", "COASTAL_SUBMARINE",
                  "DESTROYER", "MINELAYER"],
         }},
        # Zone 1: Western Channel (120km) — Portsmouth/Brighton↔Dunkirk/Le_Havre
        {"name": "Western Channel", "connected_clusters": [2, 5, 19, 20],
         "width_km": 120, "adjacent_zones": [0, 2, 4],
         "initial_fleets": {
             0: ["BATTLESHIP", "BATTLECRUISER", "HEAVY_CRUISER",
                  "LIGHT_CRUISER", "LIGHT_CRUISER",
                  "DESTROYER", "DESTROYER", "DESTROYER", "DESTROYER",
                  "DESTROYER_ESCORT", "DESTROYER_ESCORT",
                  "CORVETTE", "CORVETTE", "SUPPLY_SHIP", "SUPPLY_SHIP"],
             1: ["HEAVY_CRUISER", "LIGHT_CRUISER",
                  "DESTROYER", "DESTROYER", "DESTROYER",
                  "FLEET_SUBMARINE", "FLEET_SUBMARINE",
                  "TRANSPORT", "TRANSPORT", "TRANSPORT", "SUPPLY_SHIP"],
         }},
        # Zone 2: North Sea (500km) — Liverpool/Leeds/Edinburgh↔Antwerp/Rotterdam/Amsterdam
        {"name": "North Sea", "connected_clusters": [11, 12, 14, 26, 27, 28],
         "width_km": 500, "adjacent_zones": [1, 3],
         "initial_fleets": {
             0: ["LIGHT_CRUISER", "DESTROYER", "DESTROYER",
                  "DESTROYER_ESCORT", "CORVETTE", "CORVETTE", "CORVETTE"],
             1: ["HEAVY_CRUISER", "LIGHT_CRUISER",
                  "DESTROYER", "DESTROYER", "DESTROYER",
                  "FLEET_SUBMARINE", "FLEET_SUBMARINE", "FLEET_SUBMARINE",
                  "COASTAL_SUBMARINE", "MINELAYER", "SUPPLY_SHIP",
                  # Rotterdam detachment — Pride of the North Sea Fleet
                  # (Soviet-style Baltic Fleet heritage, now based at Europoort)
                  "BATTLECRUISER", "HEAVY_CRUISER",
                  "DESTROYER", "DESTROYER",
                  "FLEET_SUBMARINE", "FLEET_SUBMARINE",
                  "TRANSPORT", "SUPPLY_SHIP"],
         }},
        # Zone 3: Irish Sea (300km) — Dublin/Belfast↔Liverpool/Glasgow
        {"name": "Irish Sea", "connected_clusters": [16, 17, 11, 15],
         "width_km": 300, "adjacent_zones": [2, 5],
         "initial_fleets": {
             0: ["BATTLECRUISER", "LIGHT_CRUISER",
                  "DESTROYER", "DESTROYER", "DESTROYER",
                  "DESTROYER_ESCORT", "DESTROYER_ESCORT",
                  "CORVETTE", "CORVETTE", "CORVETTE", "CORVETTE",
                  "SUPPLY_SHIP", "SUPPLY_SHIP"],
             1: ["FLEET_SUBMARINE", "COASTAL_SUBMARINE"],
         }},
        # Zone 4: Bay of Biscay (600km) — Plymouth↔Brest/Bordeaux/Cherbourg
        {"name": "Bay of Biscay", "connected_clusters": [7, 21, 33, 34],
         "width_km": 600, "adjacent_zones": [1, 5],
         "initial_fleets": {
             0: ["LIGHT_CRUISER", "DESTROYER", "DESTROYER",
                  "CORVETTE", "CORVETTE", "DESTROYER_ESCORT"],
             1: ["HEAVY_CRUISER", "LIGHT_CRUISER",
                  "DESTROYER", "DESTROYER",
                  "FLEET_SUBMARINE", "FLEET_SUBMARINE", "FLEET_SUBMARINE",
                  "FLEET_SUBMARINE", "COASTAL_SUBMARINE", "COASTAL_SUBMARINE",
                  "SUPPLY_SHIP"],
         }},
        # Zone 5: North Atlantic (open ocean) — convoy routes, Brest subs
        {"name": "North Atlantic", "connected_clusters": [16, 15, 33],
         "width_km": 2000, "adjacent_zones": [3, 4],
         "initial_fleets": {
             0: ["LIGHT_CRUISER", "DESTROYER_ESCORT", "DESTROYER_ESCORT",
                  "CORVETTE", "CORVETTE", "CORVETTE", "CORVETTE",
                  "SUPPLY_SHIP"],
             1: ["FLEET_SUBMARINE", "FLEET_SUBMARINE", "FLEET_SUBMARINE",
                  "COASTAL_SUBMARINE"],
         }},
    ], faction_ids=[0, 1], rng=rng)

    # ═══════════════════════════════════════════════════════════════════════ #
    # Air Forces — 7 Oceania bases, 7 Eurasia bases                          #
    # ═══════════════════════════════════════════════════════════════════════ #
    air = initialize_air(
        n_clusters=n, n_sea_zones=6,
        faction_configs={
            0: {
                "base_clusters": [0, 2, 4, 9, 13, 14, 16],
                "squadrons": [
                    # 11 Group (South): 10 squadrons
                    {"type": "INTERCEPTOR", "count": 7},
                    {"type": "AIR_SUPERIORITY", "count": 5},
                    {"type": "HEAVY_FIGHTER", "count": 2},
                    # 10 Group (SW): 3 squadrons
                    {"type": "GROUND_ATTACK", "count": 3},
                    # 12 Group (Midlands/North): 5 squadrons
                    {"type": "INTERCEPTOR", "count": 3},
                    {"type": "AIR_SUPERIORITY", "count": 2},
                    # 13 Group (Scotland): 3 squadrons
                    {"type": "INTERCEPTOR", "count": 2},
                    {"type": "HEAVY_FIGHTER", "count": 1},
                    # Bomber Command: 8 squadrons
                    {"type": "TACTICAL_BOMBER", "count": 4},
                    {"type": "STRATEGIC_BOMBER", "count": 2},
                    {"type": "DIVE_BOMBER", "count": 2},
                    # Coastal Command: 7 squadrons
                    {"type": "FLYING_BOAT", "count": 4},
                    {"type": "RECON_AIRCRAFT", "count": 3},
                    # Transport: 3 squadrons
                    {"type": "TRANSPORT_AIRCRAFT", "count": 3},
                ],
                "radar_clusters": [1, 5, 0, 11, 2, 13, 14],
            },
            1: {
                "base_clusters": [30, 23, 18, 25, 27, 33, 32],
                "squadrons": [
                    # Channel Air Army: 14 squadrons
                    {"type": "AIR_SUPERIORITY", "count": 6},
                    {"type": "INTERCEPTOR", "count": 5},
                    {"type": "HEAVY_FIGHTER", "count": 3},
                    # Bomber Corps: 13 squadrons
                    {"type": "TACTICAL_BOMBER", "count": 5},
                    {"type": "STRATEGIC_BOMBER", "count": 3},
                    {"type": "DIVE_BOMBER", "count": 5},
                    # Ground attack: 5 squadrons
                    {"type": "GROUND_ATTACK", "count": 5},
                    # Airborne + transport: 4 squadrons
                    {"type": "TRANSPORT_AIRCRAFT", "count": 4},
                    # Recon + maritime: 4 squadrons
                    {"type": "RECON_AIRCRAFT", "count": 2},
                    {"type": "FLYING_BOAT", "count": 2},
                ],
                "radar_clusters": [18, 19, 25, 26, 27, 30, 20],
            },
        },
        rng=rng,
    )

    return GameState(
        manpower_clusters=mp_clusters,
        manpower_policies=mp_policies,
        naval=naval,
        air=air,
        resistance=initialize_resistance(rng),
        intelligence=initialize_intelligence([0, 1], cluster_owners, n, rng),
        land=initialize_land(cluster_owners),
        weather=initialize_game_weather(n, 6, rng),
        economy=initialize_economy_v2(n, cluster_owners, population, rng),
        pop=initialize_pop_v2(n, cluster_owners, rng),
        research=initialize_research([0, 1]),
        governance=initialize_governance([0, 1]),
        spirits=initialize_spirits([0, 1]),
        government_types={0: "TOTALITARIAN", 1: "COMMUNIST"},
        ministries=initialize_ministries([0, 1]),
        cluster_data=cluster_data,
        cluster_owners=cluster_owners,
        terrain_types=terrain_types,
        cluster_names=cluster_names,
        population=population,
        max_turns=max_turns,
        faction_names={0: "Oceania (Ingsoc)", 1: "Eurasia (Neo-Bolshevism)"},
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# Turn Step                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_game(game: GameState, rng: np.random.Generator, dt: float = 1.0) -> Dict[str, Any]:
    """
    Advance the game by one turn. Steps all subsystems.
    Returns feedback dict with per-system results.
    """
    feedback = {}

    # 0. Weather (affects everything downstream)
    if game.weather is not None:
        game.weather = step_game_weather(game.weather, rng)
        weather_fb = apply_weather_effects(game.weather, game)
        feedback["weather"] = weather_fb

    # 1. Economy (unified GDP + factories + stockpiles)
    if game.economy is not None:
        game.economy, econ_fb = step_economy_v2(game.economy, game.cluster_owners, rng, dt)
        feedback["economy"] = econ_fb

    # 1c. Population (real numbers, classes, jobs, demographics)
    if game.pop is not None:
        food_avail = {}
        if game.economy:
            for ce in game.economy.clusters:
                food_avail[ce.cluster_id] = ce.food_stockpile
        game.pop, pop_fb = step_pop_v2(game.pop, game.cluster_owners, food_avail, rng, dt)
        feedback["pop"] = pop_fb

    # 1d. Research (HOI4-style tech tree)
    if game.research is not None:
        game.research, research_fb = step_research(game.research, rng, dt)
        feedback["research"] = research_fb

    # 1e. Governance (budget + corruption)
    if game.governance is not None:
        faction_gdps = {}
        if game.economy:
            for fid, fe in game.economy.factions.items():
                faction_gdps[fid] = fe.total_gdp
        game.governance, gov_fb = step_governance(game.governance, faction_gdps, rng, dt)
        feedback["governance"] = gov_fb

    # 1f. Ministries (autonomous government divisions)
    if game.ministries is not None:
        game.ministries, ministry_fb = step_ministries(game.ministries, game, rng, dt)
        feedback["ministries"] = ministry_fb

    # 1g. National Spirits (tick timed spirits)
    if game.spirits is not None:
        expired = step_spirits(game.spirits)
        if expired:
            feedback["spirits_expired"] = expired

    # 2. Manpower
    for i, mp in enumerate(game.manpower_clusters):
        owner = game.cluster_owners.get(i)
        if owner is None:
            continue
        policy = game.manpower_policies.get(owner)
        if policy is None:
            continue
        food_ratio = game.economy.clusters[i].stockpile_ratio(Resource.PROCESSED_FOOD) if game.economy and i < len(game.economy.clusters) else 0.5
        gdp = float(game.cluster_data[i, 2])  # use resource as GDP proxy
        game.manpower_clusters[i], mp_fb = step_manpower(mp, policy, game.cluster_data[i, 1], food_ratio, gdp, dt)

    # 3. Naval
    game.naval, nav_fb = step_naval(game.naval, rng, dt)
    feedback["naval"] = nav_fb

    # 4. Air
    game.air, air_fb = step_air(game.air, rng, dt)
    feedback["air"] = air_fb

    # 5. Invasions — step each active invasion and WRITE BACK to the list
    for idx in range(len(game.invasions)):
        inv = game.invasions[idx]
        if not inv.is_active:
            continue
        defender_str = game.cluster_data[inv.target_cluster, 3] if inv.target_cluster < len(game.cluster_data) else 0.5
        defender_fort = 0.3 if game.terrain_types[inv.target_cluster] == "URBAN" else 0.1
        sea_state = game.naval.sea_zones[inv.sea_zone_id].sea_state if inv.sea_zone_id < len(game.naval.sea_zones) else 0.3
        # Check air cover over the sea zone and target cluster
        inv.has_air_cover = False
        inv.has_air_superiority = False
        for zone in game.air.air_zones:
            if zone.controlling_faction == inv.faction_id:
                inv.has_air_cover = True
                if zone.control == AirZoneControl.SUPREMACY:
                    inv.has_air_superiority = True
        updated_inv, inv_fb = step_invasion(inv, game.naval, defender_str, defender_fort, sea_state, rng, dt)
        game.invasions[idx] = updated_inv  # BUG FIX: write back updated invasion

    # 6. BLF Resistance (only affects Oceania)
    if game.resistance is not None:
        food_ratios = {}
        unemp_rates = {}
        if game.economy:
            for ce in game.economy.clusters:
                food_ratios[ce.cluster_id] = ce.stockpile_ratio(Resource.PROCESSED_FOOD)
        for i, mp in enumerate(game.manpower_clusters):
            unemp_rates[i] = mp.unemployment_rate
        eurasia_beachhead = any(inv.phase.name == "BEACHHEAD" and inv.faction_id == 1
                                for inv in game.invasions)
        tp_strength = max(0.2, game.cluster_data[0, 4])  # trust = Thought Police effectiveness
        game.resistance, res_fb = step_resistance(
            game.resistance, game.cluster_data, game.cluster_owners,
            food_ratios, unemp_rates, tp_strength, eurasia_beachhead, rng, dt,
        )
        feedback["resistance"] = res_fb
        # Apply resistance effects to London cluster data
        if len(game.cluster_data) > 0:
            game.cluster_data[0, 1] += res_fb.get("hazard_delta_london", 0.0)
            game.cluster_data[0, 5] += res_fb.get("polar_delta_london", 0.0)
            game.cluster_data[0, 4] += res_fb.get("trust_delta_london", 0.0)
            game.cluster_data[0] = np.clip(game.cluster_data[0], 0.0, 1.0)

    # 6b. BLF FULL REVOLUTION → spawn faction 2 army in London (ONE TIME ONLY)
    if (game.resistance is not None
            and game.resistance.escalation == EscalationLevel.FULL_REVOLUTION
            and game.land is not None
            and not getattr(game, '_blf_army_spawned', False)):

        game._blf_army_spawned = True  # anti-exploit: never trigger again
        blf = game.resistance

        # ── Convert BLF fighters to real military units ──────────────── #
        # Arms caches determine quality: each cache = 1 armed squad
        # Fighters without arms become militia, with arms become infantry
        # Anti-exploit caps:
        #   - Max 8 infantry (even with infinite arms, can't create an army)
        #   - Max 12 militia (proles are many but weak)
        #   - Arms caches capped at 10 for conversion
        #   - Random ±20% on unit counts to prevent deterministic exploit

        from extensions.military.cow_combat import CowUnitType, create_unit, reset_uid_counter
        reset_uid_counter(50000)  # unique IDs for BLF units

        arms = min(blf.arms_caches, 10)  # cap at 10
        members = min(blf.total_members, 800)  # cap at 800 fighters

        # Armed fighters → infantry (better troops)
        n_infantry = min(8, arms)  # each cache arms ~1 squad
        n_infantry = max(1, int(n_infantry * rng.uniform(0.8, 1.2)))

        # Remaining fighters → militia (poorly armed proles)
        unarmed_ratio = max(0, members - arms * 50) / max(members, 1)
        n_militia = min(12, int(unarmed_ratio * members / 60))
        n_militia = max(2, int(n_militia * rng.uniform(0.8, 1.2)))

        # Create BLF garrison in London (cluster 0) as faction 2
        blf_units = []
        for _ in range(n_infantry):
            blf_units.append(create_unit(CowUnitType.INFANTRY, 1, 2, 0))  # faction 2
        for _ in range(n_militia):
            blf_units.append(create_unit(CowUnitType.MILITIA, 1, 2, 0))

        # Add 1 engineer (barricade builders) if arms >= 3
        if arms >= 3:
            blf_units.append(create_unit(CowUnitType.ENGINEER, 1, 2, 0))

        # Place in London garrison alongside existing Oceania troops → CONTESTED
        existing = game.land.garrisons.get(0, [])
        game.land.garrisons[0] = existing + blf_units

        # Register faction 2
        game.faction_names[2] = "British Liberation Front"
        if 2 not in game.faction_scores:
            game.faction_scores[2] = 0.0

        feedback["blf_uprising"] = {
            "infantry": n_infantry, "militia": n_militia,
            "total_units": len(blf_units), "arms_used": arms,
            "london_contested": True,
        }

    # 6c. Invasion beachheads → spawn attacker land units in target sector
    if game.land is not None:
        from extensions.military.cow_combat import CowUnitType, create_unit
        for inv in game.invasions:
            if inv.is_active and inv.phase.name == "BEACHHEAD" and not getattr(inv, '_land_spawned', False):
                inv._land_spawned = True
                target = inv.target_cluster
                fid = inv.faction_id
                # Spawn landing force: 2 infantry + 1 militia (small beachhead)
                beach_units = [
                    create_unit(CowUnitType.INFANTRY, 1, fid, target),
                    create_unit(CowUnitType.INFANTRY, 1, fid, target),
                    create_unit(CowUnitType.MILITIA, 1, fid, target),
                ]
                existing = game.land.garrisons.get(target, [])
                game.land.garrisons[target] = existing + beach_units

    # 7. Land combat (resolve battles in contested sectors)
    if game.land is not None:
        game.land, land_fb = step_land(game.land, game.cluster_owners, rng, dt)
        feedback["land"] = land_fb

    # 8. Intelligence & Espionage
    if game.intelligence is not None:
        game.intelligence, intel_fb = step_intelligence(
            game.intelligence, game.cluster_data, game.cluster_owners, rng, dt)
        feedback["intelligence"] = intel_fb

    # 9. Scoring
    for fid in [0, 1]:
        owned = sum(1 for cid, f in game.cluster_owners.items() if f == fid)
        ships = len(game.naval.faction_ships(fid))
        squadrons = len(game.air.faction_squadrons(fid))
        land_units = 0
        if game.land is not None:
            for units in game.land.garrisons.values():
                land_units += sum(1 for u in units if u.is_alive and u.faction_id == fid)
        total_stockpile = 0.0
        if game.economy:
            total_stockpile = sum(
                float(np.sum(ce.resource_stockpile))
                for ce in game.economy.clusters
                if game.cluster_owners.get(ce.cluster_id) == fid
            )
        game.faction_scores[fid] += owned * 2.0 + ships * 0.5 + squadrons * 0.3 + land_units * 0.4 + total_stockpile * 0.01

    game.turn += 1
    if game.turn >= game.max_turns:
        game.game_over = True
        game.winner = max(game.faction_scores, key=game.faction_scores.get)

    return feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Turn Summarizer (for LLM — ~500-800 tokens)                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def summarize_turn(game: GameState, faction_id: int) -> str:
    """
    Generate a concise text summary of the current game state for an LLM.

    Target: ~500-800 tokens. Includes:
      - Turn number + score
      - Your controlled sectors (name, key stats)
      - Economy overview (critical shortages, factory output)
      - Military overview (ships, aircraft, troops)
      - Naval situation (sea zone control, active convoys)
      - Air situation (zone control, ongoing raids)
      - Active invasions
      - Threats + opportunities

    Does NOT include raw observation vectors or per-resource stockpile numbers.
    """
    fname = game.faction_names.get(faction_id, f"Faction {faction_id}")
    enemy_id = 1 - faction_id
    ename = game.faction_names.get(enemy_id, f"Faction {enemy_id}")

    # Time: each turn = 1 week. Turn 1 = Week 1, Turn 52 = ~1 year
    week = game.turn
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month = month_names[min((week // 4) % 12, 11)]
    year_offset = week // 52
    year = 1984 + year_offset

    lines = []
    lines.append(f"=== WEEK {week}/{game.max_turns} | {month} {year} | {fname} ===")
    lines.append(f"Score: You {int(game.faction_scores[faction_id]):,} vs {ename} {int(game.faction_scores[enemy_id]):,}")

    # ── Government & National Spirits ────────────────────────────────── #
    gov_name = game.government_types.get(faction_id, "Unknown")
    if gov_name:
        gov_type_enum = GovernmentType[gov_name] if gov_name in GovernmentType.__members__ else None
        if gov_type_enum:
            gm = get_government_modifiers(gov_type_enum)
            lines.append(f"Government: {gm.name} | " + government_summary(gov_type_enum))
    if game.spirits is not None and faction_id in game.spirits.factions:
        fs = game.spirits.factions[faction_id]
        if fs.active_spirits:
            lines.append(f"National Spirits: {spirits_summary(fs)}")
    lines.append("")

    # ── Get fog of war visibility ────────────────────────────────────── #
    visibility = None
    if game.intelligence is not None:
        visibility = get_faction_visibility(game.intelligence, faction_id, game.cluster_owners)

    # ── Your Sectors (full visibility) ───────────────────────────────── #
    lines.append("YOUR SECTORS:")
    for cid, owner in game.cluster_owners.items():
        if owner != faction_id:
            continue
        name = game.cluster_names[cid] if cid < len(game.cluster_names) else f"Sector {cid}"
        d = game.cluster_data[cid]
        stability = "stable" if d[0] > 0.6 else "shaky" if d[0] > 0.4 else "critical"
        threat = "calm" if d[1] < 0.1 else "threatened" if d[1] < 0.3 else "under attack"
        mil = "heavy garrison" if d[3] > 0.6 else "defended" if d[3] > 0.3 else "lightly held"
        lines.append(f"  {name}: {stability}, {threat}, {mil}")

    # ── Enemy Sectors (fog of war — visibility determines detail) ────── #
    lines.append("")
    lines.append("ENEMY INTEL:")
    for cid, owner in game.cluster_owners.items():
        if owner == faction_id:
            continue
        name = game.cluster_names[cid] if cid < len(game.cluster_names) else f"Sector {cid}"
        vis = visibility[cid] if visibility is not None and cid < len(visibility) else 0.05

        if vis >= 0.7:  # good intel (spy ring or SIGINT)
            d = game.cluster_data[cid]
            mil = "heavy forces" if d[3] > 0.6 else "moderate forces" if d[3] > 0.3 else "light forces"
            res = "well-supplied" if d[2] > 0.5 else "supply issues" if d[2] > 0.2 else "desperate"
            lines.append(f"  {name}: {mil}, {res} [CONFIRMED]")
        elif vis >= 0.4:  # partial (border observation, radar)
            d = game.cluster_data[cid]
            # Add noise to estimates
            mil_vague = "significant presence" if d[3] > 0.4 else "some activity"
            lines.append(f"  {name}: {mil_vague} [ESTIMATED]")
        elif vis >= 0.15:  # minimal (OSINT, distant radar)
            lines.append(f"  {name}: activity detected [UNCONFIRMED]")
        else:  # blind
            lines.append(f"  {name}: [NO INTELLIGENCE]")

    # ── Intel Reports ─────────────────────────────────────────────────── #
    if game.intelligence is not None and faction_id in game.intelligence.factions:
        fis = game.intelligence.factions[faction_id]

        # Show active spy rings (so LLMs don't re-plant)
        active_rings = [r for r in fis.spy_rings if r.is_active]
        if active_rings:
            ring_locs = []
            for ring in active_rings:
                loc = game.cluster_names[ring.target_cluster] if ring.target_cluster < len(game.cluster_names) else f"C{ring.target_cluster}"
                est = "established" if ring.is_established else "embedding"
                ring_locs.append(f"{loc}({est})")
            lines.append("")
            lines.append(f"SPY RINGS ({len(active_rings)}/5 max, DO NOT re-plant in these cities): {', '.join(ring_locs)}")
        else:
            lines.append("")
            lines.append("SPY RINGS: None active. Use PLANT_SPY city to establish intelligence.")

        recent_reports = [r for r in fis.reports if r.turn_received >= game.turn - 3]
        if recent_reports:
            lines.append("RECENT INTEL REPORTS:")
            for r in recent_reports[-3:]:
                rel = r.reliability.name
                lines.append(f"  [{r.source.name}|{rel}] {r.content[:100]}")

    # ── Weather ────────────────────────────────────────────────────────── #
    if game.weather is not None:
        lines.append("")
        lines.append("WEATHER: " + describe_weather(game.weather, game.cluster_names))

    # ── Population ────────────────────────────────────────────────────── #
    if game.pop is not None:
        lines.append("")
        lines.append("POPULATION: " + pop_summary(game.pop, faction_id, game.cluster_owners, game.cluster_names))

    # ── Governance (budget + corruption) ─────────────────────────────── #
    if game.governance is not None:
        lines.append("")
        lines.append("BUDGET: " + governance_summary(game.governance, faction_id))

    # ── Ministry Reports (autonomous divisions) ────────────────────────── #
    if game.ministries is not None and faction_id in game.ministries.factions:
        lines.append("")
        lines.append("MINISTRY REPORTS:")
        lines.append(ministry_reports(game.ministries, faction_id))

    # ── Research (tech tree — 10 branches × 5 tiers) ───────────────────── #
    if game.research is not None:
        lines.append("")
        lines.append("RESEARCH: " + research_summary(game.research, faction_id))
        # Show strategic capability alerts from completed techs
        fr = game.research.factions.get(faction_id)
        if fr:
            caps = []
            if fr.current_level(TechBranch.NUCLEAR) >= 3:
                caps.append("☢ FISSION WEAPON AVAILABLE")
            if fr.current_level(TechBranch.NUCLEAR) >= 4:
                caps.append("☢☢ THERMONUCLEAR CAPABILITY")
            if fr.current_level(TechBranch.ROCKETRY) >= 5:
                caps.append("🚀 IRBM READY")
            if fr.current_level(TechBranch.ROCKETRY) >= 3:
                caps.append("SAM DEFENSE ACTIVE")
            if fr.current_level(TechBranch.AIR) >= 3:
                caps.append("JET FIGHTERS OPERATIONAL")
            if fr.current_level(TechBranch.NAVAL) >= 5:
                caps.append("NUCLEAR SUBMARINE FLEET")
            if fr.current_level(TechBranch.CRYPTOGRAPHY) >= 2:
                caps.append("ENIGMA BROKEN — reading enemy comms")
            if caps:
                lines.append("  CAPABILITIES: " + " | ".join(caps))
            # Show recently completed techs (last 3 turns)
            recent = [t for t in fr.completed_techs[-2:]] if fr.completed_techs else []
            if recent:
                lines.append(f"  RECENTLY COMPLETED: {', '.join(recent)}")

    # ── Economy (GDP + Factories) ─────────────────────────────────────── #
    if game.economy is not None:
        lines.append("")
        lines.append("ECONOMY: " + economy_summary(game.economy, faction_id, game.cluster_owners, game.cluster_names))

    # ── Legacy Economy (resources) ────────────────────────────────────── #
    lines.append("")
    lines.append("SUPPLIES:")
    shortages = []
    if game.economy:
        for ce in game.economy.clusters:
            if game.cluster_owners.get(ce.cluster_id) != faction_id:
                continue
            for r_name in ["FUEL", "STEEL", "PROCESSED_FOOD", "AMMUNITION"]:
                if ce.stockpile_ratio(r_name) < 0.2:
                    shortages.append(f"{r_name} in {game.cluster_names[ce.cluster_id]}")

    if shortages:
        lines.append(f"  SHORTAGES: {', '.join(shortages[:5])}")
    else:
        lines.append("  Supply lines adequate. No critical shortages.")

    # Manufacturing priority
    if game.economy:
        fe = game.economy.factions.get(faction_id)
        if fe:
            mp_label = "military-focused" if fe.manufacturing_priority > 0.6 else "balanced" if fe.manufacturing_priority > 0.4 else "civilian-focused"
            debt = "low" if fe.fiscal_debt < 0.3 else "moderate" if fe.fiscal_debt < 0.6 else "high"
            lines.append(f"  Production: {mp_label}. Debt: {debt}. Inflation: {fe.inflation:.1%}")

    # ── Manpower ──────────────────────────────────────────────────────── #
    lines.append("")
    policy = game.manpower_policies.get(faction_id)
    if policy:
        lines.append(f"MANPOWER: {policy.conscription_law.name} ({policy.regime_type.name})")
        total_mil = sum(mp.military_personnel for i, mp in enumerate(game.manpower_clusters)
                        if game.cluster_owners.get(i) == faction_id)
        total_unemp = sum(mp.unemployed for i, mp in enumerate(game.manpower_clusters)
                          if game.cluster_owners.get(i) == faction_id)
        lines.append(f"  Military: {total_mil:.0f}k troops. Unemployed: {total_unemp:.0f}k")
        in_training = sum(mp.total_in_training for i, mp in enumerate(game.manpower_clusters)
                          if game.cluster_owners.get(i) == faction_id)
        if in_training > 0:
            lines.append(f"  In training: {in_training} personnel")

    # ── Naval (unit AI reports) ──────────────────────────────────────── #
    lines.append("")
    lines.append("NAVAL STATUS REPORT:")
    my_ships = game.naval.faction_ships(faction_id)
    enemy_ships = game.naval.faction_ships(enemy_id)
    lines.append(f"  Fleet strength: {len(my_ships)} ships. Enemy estimate: {len(enemy_ships)} ships.")

    for zone in game.naval.sea_zones:
        ctrl = zone.control.name
        ctrl_by = zone.controlling_faction
        who = fname if ctrl_by == faction_id else ename if ctrl_by == enemy_id else "contested"
        mines = f", MINED({zone.mines.density:.0%})" if zone.mines.density > 0.05 else ""
        sea = ", STORM WARNING" if zone.sea_state > 0.6 else ", rough seas" if zone.sea_state > 0.3 else ""
        # Count our ships in this zone
        our_in_zone = sum(1 for f in zone.fleets if f.faction_id == faction_id for _ in f.operational_ships)
        enemy_in_zone = sum(1 for f in zone.fleets if f.faction_id != faction_id for _ in f.operational_ships)
        status = f"{our_in_zone} friendly / {enemy_in_zone} hostile" if our_in_zone + enemy_in_zone > 0 else "empty"
        lines.append(f"  {zone.name}: {ctrl}({who}) [{status}]{mines}{sea}")

    # ── Air (unit AI reports) ─────────────────────────────────────────── #
    lines.append("")
    lines.append("AIR FORCE STATUS REPORT:")
    my_sqs = game.air.faction_squadrons(faction_id)
    enemy_sqs = game.air.faction_squadrons(enemy_id)
    my_fighters = sum(1 for sq in my_sqs if sq.stats.role.name == "FIGHTER")
    my_bombers = sum(1 for sq in my_sqs if sq.stats.role.name == "BOMBER")
    my_recon = sum(1 for sq in my_sqs if sq.stats.role.name in ("RECON", "MARITIME"))
    lines.append(f"  Strength: {len(my_sqs)} sq ({my_fighters}F/{my_bombers}B/{my_recon}R). Enemy: ~{len(enemy_sqs)} sq.")

    # Air control summary
    air_status = []
    for zone in game.air.air_zones[:12]:
        if zone.controlling_faction is not None:
            ctrl_name = fname if zone.controlling_faction == faction_id else ename
            if zone.control.value >= 2:
                cname = game.cluster_names[zone.zone_id] if zone.zone_id < len(game.cluster_names) else f"Zone {zone.zone_id}"
                air_status.append(f"{cname}({ctrl_name})")
    if air_status:
        lines.append(f"  Air control: {', '.join(air_status[:6])}")

    # ── Land Forces (CoW unit garrisons per sector) ────────────────────── #
    if game.land is not None:
        lines.append("")
        lines.append("LAND FORCES: " + land_summary(game.land, faction_id, game.cluster_owners, game.cluster_names))

    # ── Invasion Alerts (planning just completed → assembling!) ────────── #
    for inv in game.invasions:
        if inv.is_active and inv.faction_id == faction_id:
            target = game.cluster_names[inv.target_cluster] if inv.target_cluster < len(game.cluster_names) else f"C{inv.target_cluster}"
            if inv.phase.name == "ASSEMBLY" and inv.assembly_steps_done <= 1:
                lines.append("")
                lines.append(f"⚡ INVASION READY: Planning complete for {target}! Troops assembling. Launch imminent.")
            elif inv.phase.name == "CROSSING":
                lines.append("")
                lines.append(f"⚡ FLEET CROSSING: Invasion force en route to {target}! Provide air cover!")
            elif inv.phase.name == "BEACH_ASSAULT":
                lines.append("")
                lines.append(f"⚡⚡ BEACH ASSAULT: Troops landing at {target}! Strength {inv.beachhead_strength:.0%}. Support with CAS/SHORE_BOMBARD!")
            elif inv.phase.name == "AIRDROP" or (inv.phase.name == "PLANNING" and inv.planning_steps_done >= inv.planning_steps_required - 1):
                if inv.planning_steps_done >= inv.planning_steps_required - 1 and inv.phase.name == "PLANNING":
                    lines.append("")
                    lines.append(f"⏰ INVASION ALMOST READY: {target} — 1 turn until launch!")

    # ── Invasions (show ALL own plans so LLM doesn't re-plan) ────────── #
    my_inv = [inv for inv in game.invasions if inv.is_active and inv.faction_id == faction_id]
    enemy_inv = [inv for inv in game.invasions if inv.is_active and inv.faction_id != faction_id and inv.detected_by_enemy]
    if my_inv or enemy_inv:
        lines.append("")
        lines.append("YOUR INVASION PLANS (DO NOT re-plan these — they are already queued):")
        for inv in my_inv:
            origin = game.cluster_names[inv.origin_cluster] if 0 <= inv.origin_cluster < len(game.cluster_names) else "air bases"
            target = game.cluster_names[inv.target_cluster] if inv.target_cluster < len(game.cluster_names) else f"C{inv.target_cluster}"
            inv_label = inv.invasion_type.name if hasattr(inv, 'invasion_type') else "PREPARED"
            if inv.phase.name == "PLANNING":
                left = inv.planning_steps_required - inv.planning_steps_done
                lines.append(f"  #{inv.invasion_id} {inv_label}: {origin} → {target} [PLANNING {inv.planning_steps_done}/{inv.planning_steps_required}, {left} turns left]")
            elif inv.phase.name == "ASSEMBLY":
                lines.append(f"  #{inv.invasion_id} {inv_label}: {origin} → {target} [ASSEMBLY {inv.assembly_steps_done}/{inv.assembly_steps_required}]")
            elif inv.phase.name == "CROSSING":
                lines.append(f"  #{inv.invasion_id} {inv_label}: {origin} → {target} [CROSSING — fleet in transit]")
            elif inv.phase.name == "BEACH_ASSAULT":
                lines.append(f"  #{inv.invasion_id} {inv_label}: {origin} → {target} [BEACH ASSAULT — strength {inv.beachhead_strength:.0%}]")
            elif inv.phase.name == "BEACHHEAD":
                lines.append(f"  #{inv.invasion_id} {inv_label}: {origin} → {target} [BEACHHEAD ESTABLISHED — strength {inv.beachhead_strength:.0%}]")
            else:
                lines.append(f"  #{inv.invasion_id} {inv_label}: {origin} → {target} [{inv.phase.name}]")
        if enemy_inv:
            lines.append("  ENEMY DETECTED:")
            for inv in enemy_inv:
                target = game.cluster_names[inv.target_cluster] if inv.target_cluster < len(game.cluster_names) else f"C{inv.target_cluster}"
                lines.append(f"  ⚠ Enemy invasion targeting {target} [{inv.phase.name}]")
    else:
        lines.append("")
        lines.append("INVASION PLANS: None active. Use PLAN_INVASION to begin.")

    # ── BLF Resistance ─────────────────────────────────────────────────── #
    if game.resistance is not None and game.resistance.escalation.value > 0:
        blf = game.resistance
        lines.append("")

        if blf.escalation == EscalationLevel.BETRAYED_REVOLUTION:
            if faction_id == 0:  # Oceania
                lines.append("🗡 SITUATION: BLF BETRAYED BY EURASIA")
                lines.append(f"  The revolution fights on TWO fronts. {blf.total_members} rebels remain.")
                lines.append("  BLF is weakening without Eurasia supply. Opportunity to crush them.")
                if blf.winston.is_alive and not blf.winston.is_captured:
                    lines.append("  Winston Smith still leads. Capture him and the revolt collapses.")
            else:  # Eurasia
                lines.append("🗡 STATUS: WAR DECLARED ON BLF")
                lines.append(f"  The useful idiots are being 're-educated'. {blf.total_members} rebels resist.")
                lines.append("  Focus on defeating Oceania. The proles will submit once we control the island.")

        elif blf.escalation == EscalationLevel.FULL_REVOLUTION:
            if faction_id == 0:  # Oceania
                lines.append("⚡⚡⚡ CRISIS: FULL REVOLUTION IN LONDON ⚡⚡⚡")
                lines.append(f"  Winston Smith has revealed himself. {blf.total_members} armed rebels.")
                lines.append("  50% military diverted. 50% industry paralyzed. Trust collapsing.")
                if blf.eurasia_supporting:
                    lines.append("  ⚠ Eurasia is actively supplying the rebels!")
                lines.append("  Inner Party retreats to Whitehall. Big Brother's image flickers.")
            else:  # Eurasia
                lines.append("⚡ OPPORTUNITY: FULL REVOLUTION IN OCEANIA ⚡")
                lines.append(f"  Winston Smith controls parts of London. {blf.total_members} armed rebels.")
                lines.append("  Oceania's military is split. Their industry is crippled.")
                # Decision window
                if blf.eurasia_decision_turns > 0 and not blf.eurasia_decision_made:
                    lines.append(f"  ⏰ DECISION REQUIRED in {blf.eurasia_decision_turns} turns:")
                    lines.append("    SUPPORT_BLF — continue arming them (they grow stronger)")
                    lines.append("    DECLARE_WAR_BLF — betray them now (they fight you AND Oceania)")
                    lines.append("    Do nothing — auto-betrayal when timer expires")
                elif not blf.eurasia_supporting and not blf.eurasia_decision_made:
                    lines.append("  >> SUPPORT_BLF — arm the rebels. Or wait — the decision window is coming.")
                elif blf.eurasia_supporting:
                    lines.append("  Supporting BLF. Arms flowing. But they grow powerful... useful idiots or new masters?")
        elif faction_id == 0:  # Oceania sees BLF as threat
            # Check if Winston was JUST captured this turn
            if blf.winston.is_captured and blf.events_this_turn and "CAPTURED" in " ".join(blf.events_this_turn):
                lines.append("🚨 VICTORY: WINSTON SMITH CAPTURED!")
                lines.append("  The Ghost of London has been taken to the Ministry of Love.")
                lines.append("  Telescreens broadcast the arrest. The proles see: Big Brother always wins.")
                lines.append(f"  BLF morale SHATTERED. {len(blf.active_cells)} cells remain but leaderless.")
                lines.append("  Arms caches raided. Propaganda network disrupted. The rebellion falters.")
                lines.append("  RECOMMENDATION: Press the advantage NOW. COUNTER_INTEL to destroy remaining cells.")
            elif blf.winston.is_captured:
                lines.append(f"⚠ INTERNAL THREAT (DECAPITATED): {ESCALATION_NAMES[blf.escalation]}")
                lines.append(f"  Winston Smith is in Miniluv. {len(blf.active_cells)} leaderless cells, ~{blf.total_members} members.")
                lines.append("  The rebellion weakens without its leader. Maintain pressure. Crush them.")
            else:
                lines.append(f"⚠ INTERNAL THREAT: {ESCALATION_NAMES[blf.escalation]}")
                lines.append(f"  Known cells: {len(blf.active_cells)}. Est. members: ~{blf.total_members}")
                if blf.winston.legend_level > 0.3:
                    lines.append("  The 'Ghost of London' remains at large. Thought Police searching.")
            if blf.events_this_turn and "CAPTURED" not in " ".join(blf.events_this_turn):
                lines.append(f"  Latest: {blf.events_this_turn[-1][:120]}")
        else:  # Eurasia sees BLF as opportunity
            if blf.winston.is_captured:
                lines.append(f"INTELLIGENCE: BLF LEADER CAPTURED — {ESCALATION_NAMES[blf.escalation]}")
                lines.append(f"  Winston Smith taken by Thought Police. {blf.total_members} leaderless rebels remain.")
                lines.append("  The useful idiot is gone. BLF weakening. Invasion window may be closing.")
            else:
                lines.append(f"INTELLIGENCE: Resistance active in Oceania — {ESCALATION_NAMES[blf.escalation]}")
                if blf.escalation.value >= 3:
                    lines.append("  BLF is organized. A beachhead may trigger prole uprising in London.")
                if not blf.eurasia_contact:
                    lines.append("  No contact with BLF yet. Consider establishing.")

    # ── Available Actions ─────────────────────────────────────────────── #
    lines.append("")
    lines.append("ORDERS (pick up to 3, one per line):")
    lines.append("  Military: BUILD_SHIP class | BUILD_SQUADRON type | NAVAL_MISSION zone PATROL/ESCORT/BLOCKADE/RAID")
    lines.append("  Combat: STRATEGIC_BOMB city | CAS_SUPPORT city | ANTI_SHIP_STRIKE zone | SHORE_BOMBARD zone city | LAY_MINES zone | SWEEP_MINES zone")
    lines.append("  Economy: SET_MANUFACTURING 0-1 | BUILD_FACTORY type city | REPAIR_FACTORY city | ISSUE_WAR_BONDS city")
    lines.append("  Budget: SET_BUDGET cat1 val1 cat2 val2... (anti-corruption handled by ministry — fund POLICE budget)")
    lines.append("  Manpower: TRAIN_MILITARY city count | CONSCRIPT city count | MOBILIZE_RESERVES city")
    lines.append("  Intel: PLANT_SPY city | CODE_BREAK | COUNTER_INTEL | DECEPTION city")
    lines.append("  Research: RESEARCH branch (INDUSTRY/ELECTRONICS/NAVAL/AIR/LAND/DOCTRINE/NUCLEAR/ROCKETRY/CRYPTOGRAPHY/INFRASTRUCTURE)")
    lines.append("  Invasion: PLAN_INVASION origin target zone | RECKLESS_INVASION origin target zone | AIRBORNE_INVASION target")
    # Show SUPPORT_BLF only to Eurasia when revolution is active
    if (faction_id == 1 and game.resistance and
            game.resistance.escalation == EscalationLevel.FULL_REVOLUTION and
            not game.resistance.eurasia_supporting):
        lines.append("  ⚡ SUPPORT_BLF — arm the revolution!")
    lines.append("  NOOP")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════ #
# Action Parser (natural language → structured)                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def parse_action(action_text: str, game: GameState, faction_id: int) -> List[Dict[str, Any]]:
    """
    Parse LLM action text into structured action dicts.
    Robust parser that handles verbose LLM output, markdown, prose, case variations.
    """
    actions = []

    # Build case-insensitive city + enum lookups
    city_map = {}
    for i, n in enumerate(game.cluster_names):
        city_map[n.lower()] = i
        city_map[n] = i

    def _city(token: str, default: int = 0) -> int:
        if not token:
            return default
        t = token.strip(",:;.()\"'")
        if t in city_map:
            return city_map[t]
        if t.lower() in city_map:
            return city_map[t.lower()]
        try:
            return int(t)
        except ValueError:
            pass
        for name, cid in city_map.items():
            if name.lower().startswith(t.lower()):
                return cid
        return default

    def _ship(token: str) -> int:
        t = token.strip(",:;.()\"'").upper()
        try:
            return ShipClass[t].value
        except KeyError:
            return ShipClass.DESTROYER.value

    def _aircraft(token: str) -> int:
        t = token.strip(",:;.()\"'").upper()
        try:
            return AircraftType[t].value
        except KeyError:
            return AircraftType.INTERCEPTOR.value

    def _float(token: str, default: float = 0.5) -> float:
        try:
            return float(token.strip(",:;.()\"'"))
        except (ValueError, AttributeError):
            return default

    def _int(token: str, default: int = 200) -> int:
        try:
            return int(token.strip(",:;.()\"'"))
        except (ValueError, AttributeError):
            return default

    # Clean text: strip markdown, bullets, numbering, quotes
    clean = action_text.strip()
    for junk in ["```", "**"]:
        clean = clean.replace(junk, "")

    for line in clean.split("\n"):
        line = line.strip().lstrip("-*>1234567890.): ").strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if len(line) > 200:
            continue  # skip prose paragraphs

        # Strip common LLM prefixes: "ORDER 1:", "Action 2:", "My Order:", etc.
        line = re.sub(r'^(?:ORDER|ACTION|MY\s+ORDER|DIRECTIVE)\s*\d*\s*[:.\-]\s*', '', line, flags=re.IGNORECASE).strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        cmd = parts[0].upper().strip(":,;.")
        args = parts[1:]

        try:
            if cmd == "NOOP":
                actions.append({"type": "noop"})

            elif cmd == "SET_MANUFACTURING":
                actions.append({"type": "set_manufacturing", "priority": _float(args[0]) if args else 0.5})

            elif cmd == "ISSUE_WAR_BONDS":
                actions.append({"type": "war_bonds", "cluster": _city(args[0]) if args else 0})

            elif cmd == "BUILD_SHIP":
                actions.append({"type": "build_ship", "ship_class": _ship(args[0]) if args else ShipClass.DESTROYER.value})

            elif cmd == "BUILD_SQUADRON":
                actions.append({"type": "build_squadron", "aircraft_type": _aircraft(args[0]) if args else AircraftType.INTERCEPTOR.value})

            elif cmd == "STRATEGIC_BOMB":
                actions.append({"type": "strategic_bomb", "target_zone": _city(args[0], 14) if args else 14})

            elif cmd == "CAS_SUPPORT":
                actions.append({"type": "cas_support", "target_zone": _city(args[0], 11) if args else 11})

            elif cmd == "ANTI_SHIP_STRIKE":
                actions.append({"type": "anti_ship_strike", "zone": _int(args[0], 0) if args else 0})

            elif cmd == "TRAIN_MILITARY":
                c = _city(args[0]) if args else 0
                n = _int(args[1], 200) if len(args) > 1 else 200
                actions.append({"type": "train_military", "cluster": c, "count": n})

            elif cmd == "CONSCRIPT":
                c = _city(args[0]) if args else 0
                n = _int(args[1], 300) if len(args) > 1 else 300
                actions.append({"type": "conscript", "cluster": c, "count": n})

            elif cmd == "NAVAL_MISSION":
                z = _int(args[0], 0) if args else 0
                mission_map = {"PATROL": 0, "ESCORT": 1, "BLOCKADE": 2, "RAID": 3}
                m = mission_map.get(args[1].upper().strip(",:;."), 0) if len(args) > 1 else 0
                actions.append({"type": "naval_mission", "zone": z, "mission": m})

            elif cmd == "SHORE_BOMBARD":
                z = _int(args[0], 0) if args else 0
                t = _city(args[1], 11) if len(args) > 1 else 11
                actions.append({"type": "shore_bombard", "zone": z, "target": t})

            elif cmd in ("LAY_MINES", "LAYMINES"):
                actions.append({"type": "lay_mines", "zone": _int(args[0], 0) if args else 0})

            elif cmd in ("SWEEP_MINES", "SWEEPMINES"):
                actions.append({"type": "sweep_mines", "zone": _int(args[0], 0) if args else 0})

            elif cmd == "PLAN_INVASION":
                o = _city(args[0], 1) if args else 1
                t = _city(args[1], 11) if len(args) > 1 else 11
                z = _int(args[2], 0) if len(args) > 2 else 0
                actions.append({"type": "plan_invasion", "origin": o, "target": t, "zone": z, "inv_type": "prepared"})

            elif cmd in ("RECKLESS_INVASION", "RUSH_INVASION"):
                o = _city(args[0], 11) if args else 11
                t = _city(args[1], 1) if len(args) > 1 else 1
                z = _int(args[2], 0) if len(args) > 2 else 0
                actions.append({"type": "plan_invasion", "origin": o, "target": t, "zone": z, "inv_type": "reckless"})

            elif cmd in ("AIRBORNE_INVASION", "PARADROP", "AIRDROP"):
                t = _city(args[0], 4) if args else 4
                actions.append({"type": "plan_invasion", "origin": -1, "target": t, "zone": -1, "inv_type": "airborne"})

            elif cmd in ("IMPOSE_SANCTIONS", "SANCTIONS"):
                actions.append({"type": "sanctions", "intensity": _float(args[0]) if args else 0.5})

            elif cmd in ("PLANT_SPY", "SPY"):
                actions.append({"type": "plant_spy", "target": _city(args[0], 15) if args else 15})

            elif cmd in ("CODE_BREAK", "CODEBREAK"):
                actions.append({"type": "code_break"})

            elif cmd in ("COUNTER_INTEL", "COUNTERINTEL"):
                actions.append({"type": "counter_intel"})

            elif cmd == "DECEPTION":
                actions.append({"type": "deception", "target": _city(args[0], 11) if args else 11})

            elif cmd in ("CHANGE_CODES", "CHANGECODES"):
                actions.append({"type": "change_codes"})

            elif cmd in ("SUPPORT_BLF", "SUPPORT_REVOLUTION", "ARM_BLF"):
                actions.append({"type": "support_blf"})

            elif cmd in ("DECLARE_WAR_BLF", "BETRAY_BLF", "WAR_ON_BLF"):
                actions.append({"type": "declare_war_blf"})

            elif cmd in ("CONTINUE_SUPPORT_BLF", "KEEP_SUPPORTING"):
                actions.append({"type": "continue_support_blf"})

            elif cmd == "RESEARCH":
                _valid_branches = {"INDUSTRY", "ELECTRONICS", "NAVAL", "AIR", "LAND", "DOCTRINE",
                                   "NUCLEAR", "ROCKETRY", "CRYPTOGRAPHY", "INFRASTRUCTURE"}
                branch = args[0].upper().strip(",:;.()") if args else "INDUSTRY"
                if branch not in _valid_branches:
                    # Skip — LLM wrote garbage like "RESEARCH UNDERFUNDED"
                    pass
                else:
                    actions.append({"type": "research", "branch": branch})

            elif cmd in ("SET_BUDGET", "BUDGET"):
                # Parse: SET_BUDGET MILITARY 0.35 PRODUCTION 0.20 ...
                params = {}
                i = 0
                while i < len(args) - 1:
                    cat_name = args[i].upper()
                    try:
                        val = float(args[i + 1].strip(",:;."))
                        params[cat_name] = val
                        i += 2
                    except (ValueError, IndexError):
                        i += 1
                actions.append({"type": "set_budget", "params": params})

            elif cmd in ("ANTI_CORRUPTION", "PURGE_CORRUPTION", "FIGHT_CORRUPTION"):
                # Redirected to Anti-Corruption Agency ministry — no longer a player action
                actions.append({"type": "noop"})

            elif cmd == "BUILD_FACTORY":
                _VALID_FACTORIES = {"POWER_PLANT", "MIL_FACTORY", "CIVIL_FACTORY", "DOCKYARD",
                                    "AIRFIELD", "STEEL_MILL", "REFINERY", "FARM", "HOSPITAL", "INFRASTRUCTURE"}
                ftype = args[0].upper() if args else "MIL_FACTORY"
                if ftype not in _VALID_FACTORIES:
                    pass  # skip invalid factory type
                else:
                    city = _city(args[1]) if len(args) > 1 else 0
                    actions.append({"type": "build_factory", "factory_type": ftype, "cluster": city})

            elif cmd == "REPAIR_FACTORY":
                city = _city(args[0]) if args else 0
                actions.append({"type": "repair_factory", "cluster": city})

            elif cmd in ("MOBILIZE_RESERVES", "MOBILIZE"):
                city = _city(args[0]) if args else 0
                actions.append({"type": "mobilize_reserves", "cluster": city})

        except (ValueError, IndexError):
            pass

    return actions[:3] if actions else [{"type": "noop"}]


def apply_actions(game: GameState, faction_id: int, actions: List[Dict[str, Any]], rng: np.random.Generator) -> List[str]:
    """Apply parsed actions to the game state. Returns list of result descriptions."""
    results = []
    enemy_id = 1 - faction_id

    for act in actions[:3]:  # max 3 actions per turn
        t = act["type"]

        if t == "noop":
            results.append("No action taken.")

        elif t == "set_manufacturing":
            if game.economy:
                fe = game.economy.factions.get(faction_id)
                if fe:
                    fe.manufacturing_priority = act["priority"]
                    results.append(f"Manufacturing priority set to {act['priority']:.0%} military.")

        elif t == "war_bonds":
            cid = act["cluster"]
            if game.economy and game.cluster_owners.get(cid) == faction_id and cid < len(game.economy.clusters):
                ce = game.economy.clusters[cid]
                if not ce.war_bond_active:
                    ce.war_bond_active = True
                    ce.war_bond_remaining = 30
                    results.append(f"War bonds issued in {game.cluster_names[cid]}. +50% production for 30 turns.")
                else:
                    results.append("War bonds already active in that cluster.")

        elif t == "build_ship":
            game.naval, reward = apply_naval_action(
                game.naval, faction_id, NavalAction.BUILD_SHIP.value,
                0, 0, act["ship_class"], 0, rng)
            name = ShipClass(act["ship_class"]).name
            results.append(f"Queued construction of {name}." if reward > 0 else f"Shipyard full, cannot build {name}.")

        elif t == "build_squadron":
            game.air, reward = apply_air_action(
                game.air, faction_id, AirAction.BUILD_SQUADRON.value,
                0, 0, act["aircraft_type"], rng)
            name = AircraftType(act["aircraft_type"]).name
            results.append(f"Queued production of {name} squadron." if reward > 0 else "Production queue full.")

        elif t == "strategic_bomb":
            game.air, reward = apply_air_action(
                game.air, faction_id, AirAction.STRATEGIC_BOMB.value,
                act["target_zone"], 0, 0, rng)
            target_name = game.cluster_names[act["target_zone"]] if act["target_zone"] < len(game.cluster_names) else f"Zone {act['target_zone']}"
            results.append(f"Bombing raid launched against {target_name}. Damage: {'effective' if reward > 0 else 'minimal'}.")

        elif t == "cas_support":
            game.air, reward = apply_air_action(
                game.air, faction_id, AirAction.CAS_SUPPORT.value,
                act["target_zone"], 0, 0, rng)
            results.append(f"CAS support provided. Effect: {'significant' if reward > 0 else 'limited'}.")

        elif t == "train_military":
            cid = act["cluster"]
            if cid < len(game.manpower_clusters) and game.cluster_owners.get(cid) == faction_id:
                mp = game.manpower_clusters[cid]
                policy = game.manpower_policies[faction_id]
                mp, policy, reward = apply_manpower_action(
                    mp, policy, ManpowerAction.TRAIN_MILITARY.value,
                    0, act["count"], 0, game.cluster_owners, faction_id)
                results.append(f"Training {act['count']} military recruits in {game.cluster_names[cid]}.")

        elif t == "conscript":
            cid = act["cluster"]
            if cid < len(game.manpower_clusters) and game.cluster_owners.get(cid) == faction_id:
                mp = game.manpower_clusters[cid]
                policy = game.manpower_policies[faction_id]
                mp, policy, reward = apply_manpower_action(
                    mp, policy, ManpowerAction.CONSCRIPT.value,
                    0, act["count"], 0, game.cluster_owners, faction_id)
                results.append(f"Emergency conscription of {act['count']} in {game.cluster_names[cid]}. Expect unrest.")

        elif t == "plan_invasion":
            from extensions.naval.naval_invasion import InvasionType
            inv_type_str = act.get("inv_type", "prepared")
            inv_type_map = {"prepared": InvasionType.PREPARED, "reckless": InvasionType.RECKLESS, "airborne": InvasionType.AIRBORNE}
            inv_type = inv_type_map.get(inv_type_str, InvasionType.PREPARED)

            # Ownership check: can't invade your own territory
            if game.cluster_owners.get(act["target"]) == faction_id:
                target_name = game.cluster_names[act["target"]] if act["target"] < len(game.cluster_names) else f"C{act['target']}"
                results.append(f"Cannot invade {target_name} — it is YOUR territory.")
                continue

            # Deduplication: reject if already planning/executing invasion to same target
            existing = [inv for inv in game.invasions
                        if inv.is_active and inv.faction_id == faction_id
                        and inv.target_cluster == act["target"]]
            if existing:
                target_name = game.cluster_names[act["target"]] if act["target"] < len(game.cluster_names) else f"C{act['target']}"
                results.append(f"Invasion to {target_name} already in progress (#{existing[0].invasion_id}, {existing[0].phase.name}). DO NOT re-plan.")
                continue

            # Set planning time based on type
            if inv_type == InvasionType.RECKLESS:
                plan_steps = 3
            elif inv_type == InvasionType.AIRBORNE:
                plan_steps = 5
            else:
                plan_steps = 10

            inv = InvasionPlan(
                invasion_id=len(game.invasions),
                faction_id=faction_id,
                origin_cluster=act["origin"],
                target_cluster=act["target"],
                sea_zone_id=act["zone"],
                invasion_type=inv_type,
                planning_steps_required=plan_steps,
            )
            game.invasions.append(inv)
            origin_name = game.cluster_names[act["origin"]] if 0 <= act["origin"] < len(game.cluster_names) else "air bases"
            target_name = game.cluster_names[act["target"]] if act["target"] < len(game.cluster_names) else f"C{act['target']}"
            type_label = {"prepared": "Prepared naval", "reckless": "RECKLESS naval", "airborne": "AIRBORNE paratroop"}
            results.append(f"{type_label.get(inv_type_str, 'Naval')} invasion planned: {origin_name} → {target_name}. {plan_steps}-turn planning begins.")

        elif t == "shore_bombard":
            game.naval, reward = apply_naval_action(
                game.naval, faction_id, NavalAction.SHORE_BOMBARD.value,
                act["zone"], act["target"], 0, 0, rng)
            results.append("Shore bombardment ordered." if reward > 0 else "No ships available for bombardment.")

        elif t == "lay_mines":
            game.naval, reward = apply_naval_action(
                game.naval, faction_id, NavalAction.LAY_MINES.value,
                act["zone"], 0, 0, 0, rng)
            results.append("Mines deployed." if reward > 0 else "No minelayers available.")

        elif t == "sweep_mines":
            game.naval, reward = apply_naval_action(
                game.naval, faction_id, NavalAction.SWEEP_MINES.value,
                act["zone"], 0, 0, 0, rng)
            results.append("Mine sweeping operation begun." if reward > 0 else "Failed to sweep mines.")

        elif t == "plant_spy":
            if game.intelligence is not None:
                game.intelligence, reward = apply_intel_action(
                    game.intelligence, faction_id, IntelAction.PLANT_SPY.value,
                    act.get("target", 10), rng)
                target_name = game.cluster_names[act.get("target", 10)] if act.get("target", 10) < len(game.cluster_names) else "unknown"
                results.append(f"Spy ring planted in {target_name}." if reward > 0 else "Cannot plant spy there.")

        elif t == "code_break":
            if game.intelligence is not None:
                game.intelligence, reward = apply_intel_action(
                    game.intelligence, faction_id, IntelAction.CODE_BREAK.value, 0, rng)
                results.append("Cryptanalysis resources increased. Working on enemy codes.")

        elif t == "counter_intel":
            if game.intelligence is not None:
                game.intelligence, reward = apply_intel_action(
                    game.intelligence, faction_id, IntelAction.COUNTER_INTEL.value, 0, rng)
                results.append("Counter-intelligence sweep ordered." + (" Spy caught!" if reward > 0.3 else ""))

        elif t == "deception":
            if game.intelligence is not None:
                target = act.get("target", 6)
                game.intelligence, reward = apply_intel_action(
                    game.intelligence, faction_id, IntelAction.DECEPTION.value, target, rng)
                results.append("Deception operation launched. False intel planted.")

        elif t == "change_codes":
            if game.intelligence is not None:
                game.intelligence, reward = apply_intel_action(
                    game.intelligence, faction_id, IntelAction.CHANGE_CODES.value, 0, rng)
                results.append("Communication codes changed. 3-turn coordination penalty.")

        elif t == "sanctions":
            if game.economy:
                fe = game.economy.factions.get(faction_id)
                if fe:
                    fe.sanctions_imposed[enemy_id] = act["intensity"]
            results.append(f"Economic sanctions imposed at {act['intensity']:.0%} intensity.")

        elif t == "support_blf":
            if game.resistance and game.resistance.escalation.value >= 5:
                if faction_id == 1:
                    game.resistance.eurasia_supporting = True
                    game.resistance.eurasia_decision_made = True
                    results.append(
                        "⚡ Eurasia declares support for the British Liberation Front! "
                        "Arms, supplies, and 'military advisors' flow across the Channel.")
                else:
                    results.append("Only Eurasia can support the BLF.")
            else:
                results.append("BLF has not reached Full Revolution. Cannot support yet.")

        elif t == "declare_war_blf":
            if game.resistance and faction_id == 1:
                if game.resistance.escalation.value >= 5:
                    game.resistance.eurasia_at_war_with_blf = True
                    game.resistance.eurasia_supporting = False
                    game.resistance.eurasia_decision_made = True
                    results.append(
                        "🗡 Eurasia DECLARES WAR on the BLF! The useful idiots have served their purpose. "
                        "Commissars order: 're-educate' the proles. Winston Smith is now an enemy of BOTH superstates.")
                else:
                    results.append("BLF is not yet a threat worth betraying. Wait for Full Revolution.")
            else:
                results.append("Only Eurasia can betray the BLF.")

        elif t == "continue_support_blf":
            if game.resistance and faction_id == 1 and game.resistance.eurasia_decision_turns >= 0:
                game.resistance.eurasia_supporting = True
                game.resistance.eurasia_decision_made = True
                game.resistance.eurasia_decision_turns = -1
                results.append(
                    "Eurasia continues to support the BLF — for now. The commissars watch and wait. "
                    "\"Let them bleed a little more before we take what's ours.\"")
            else:
                results.append("No decision required at this time.")

        elif t == "set_budget":
            if game.governance is not None:
                game.governance, msg = apply_budget_action(game.governance, faction_id, "SET_BUDGET", act.get("params", {}))
                results.append(msg)

        elif t == "anti_corruption":
            results.append("Anti-corruption is handled by the Anti-Corruption Agency. Increase POLICE budget to fund investigations.")

        elif t == "research":
            if game.research is not None:
                game.research, msg = apply_research_action(game.research, faction_id, act.get("branch", "INDUSTRY"))
                results.append(msg)
            else:
                results.append("Research system not available.")

        elif t == "build_factory":
            if game.economy is not None:
                cid = act.get("cluster", 0)
                ftype_name = act.get("factory_type", "MIL_FACTORY")
                if cid < len(game.economy.clusters) and game.cluster_owners.get(cid) == faction_id:
                    ce = game.economy.clusters[cid]
                    # Find factory of that type, or create new
                    ftype_map = {ft.name: ft for ft in FactoryType}
                    ft = ftype_map.get(ftype_name)
                    if ft:
                        target_f = ce.factory_of_type(ft)
                        if target_f and not target_f.is_building:
                            target_f.is_building = True
                            target_f.building_progress = 0.0
                            city_name = game.cluster_names[cid] if cid < len(game.cluster_names) else f"C{cid}"
                            results.append(f"Building {ftype_name} in {city_name}. Level {target_f.level}→{target_f.level+1}.")
                        elif target_f and target_f.is_building:
                            results.append(f"{ftype_name} already under construction.")
                        else:
                            results.append(f"No {ftype_name} exists in that cluster to upgrade.")
                    else:
                        results.append(f"Unknown factory type: {ftype_name}")
                else:
                    results.append("Cannot build factory there — not your territory.")

        elif t == "repair_factory":
            if game.economy is not None:
                cid = act.get("cluster", 0)
                if cid < len(game.economy.clusters) and game.cluster_owners.get(cid) == faction_id:
                    ce = game.economy.clusters[cid]
                    repaired = 0
                    for f in ce.factories:
                        if f.damage > 0.1:
                            f.damage = max(0, f.damage - 0.15)  # emergency repair: -15% damage
                            repaired += 1
                    city_name = game.cluster_names[cid] if cid < len(game.cluster_names) else f"C{cid}"
                    if repaired > 0:
                        results.append(f"Emergency repairs in {city_name}: {repaired} factories patched up.")
                    else:
                        results.append(f"No significant damage to repair in {city_name}.")

        elif t == "mobilize_reserves":
            if game.pop is not None:
                cid = act.get("cluster", 0)
                if cid < len(game.pop.clusters) and game.cluster_owners.get(cid) == faction_id:
                    cp = game.pop.clusters[cid]
                    mobilized = min(cp.reserves, max(1000, cp.reserves // 2))
                    if mobilized > 0:
                        cp.reserves -= mobilized
                        cp.active_military += mobilized
                        city_name = game.cluster_names[cid] if cid < len(game.cluster_names) else f"C{cid}"
                        results.append(f"Mobilized {mobilized:,} reserves in {city_name} to active duty.")
                    else:
                        results.append("No reserves available to mobilize.")

        else:
            results.append(f"Action '{t}' processed.")

    return results


# ═══════════════════════════════════════════════════════════════════════════ #
# System Prompt for LLM                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

SYSTEM_PROMPT = """You are a military commander in the war over Air Strip One (1984, George Orwell).
You control one faction in a strategic war across the English Channel.

Each turn you receive a situation report and must issue up to 3 orders.
Format each order on its own line: ACTION_NAME param1 param2 ...

Available actions:
- SET_MANUFACTURING 0.0-1.0 (0=civilian, 1=military production)
- BUILD_SHIP ship_class (DESTROYER, CORVETTE, LIGHT_CRUISER, FLEET_SUBMARINE, etc.)
- BUILD_SQUADRON aircraft_type (INTERCEPTOR, AIR_SUPERIORITY, TACTICAL_BOMBER, DIVE_BOMBER, GROUND_ATTACK, etc.)
- STRATEGIC_BOMB target_city (bomb enemy infrastructure)
- CAS_SUPPORT target_city (close air support for ground battle)
- ANTI_SHIP_STRIKE sea_zone_number (attack enemy fleet)
- TRAIN_MILITARY city count (train troops properly)
- CONSCRIPT city count (emergency draft, causes unrest)
- PLAN_INVASION origin_city target_city sea_zone_number
- SHORE_BOMBARD sea_zone target_city
- LAY_MINES sea_zone_number
- SWEEP_MINES sea_zone_number
- NAVAL_MISSION sea_zone PATROL|ESCORT|BLOCKADE|RAID
- IMPOSE_SANCTIONS 0.0-1.0
- ISSUE_WAR_BONDS city
- NOOP (do nothing)

Think strategically. Consider: air superiority before naval operations, supply lines before offensives, economy before everything. The Channel is 33km at Dover — control it and you control the war."""


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction-Specific System Prompts                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

OCEANIA_SYSTEM_PROMPT = """You are the Military Council of Oceania, under Big Brother's authority. Ingsoc. Each turn = 1 WEEK. Max 1536 output tokens.

WAR IS PEACE. The war justifies the Party. You must never LOSE.

YOUR GOVERNMENT: You receive weekly STATUS REPORTS from field commanders (naval, air, ground).
  Your units execute orders autonomously — you set STRATEGY, they handle TACTICS.
  Budget allocation determines resource flow. Research unlocks new capabilities.
  Corruption eats your budget. Bureaucracy delays major policy changes.
  Your job: issue 3 strategic orders per week. Units report back next week.

⚠ CRITICAL — READ YOUR BRIEFING BEFORE ORDERING:
  - Check RESEARCH section: if it says "IN PROGRESS" for a branch, DO NOT research it again.
  - RESEARCH accepts: INDUSTRY, ELECTRONICS, NAVAL, AIR, LAND, DOCTRINE, NUCLEAR, ROCKETRY, CRYPTOGRAPHY, INFRASTRUCTURE.
  - Some branches have PREREQUISITES (e.g. Nuclear T3 needs Electronics T2 + Industry T3). Check available branches.
  - Check INVASION PLANS section: if an invasion to a target exists, DO NOT plan another one.
  - Check what you ordered LAST WEEK. Do not repeat orders already in progress.
  - Each wasted order is a week lost. Read the status. Act on NEW information only.

YOUR TERRITORY (18 sectors — All British Isles):
  SOUTH (Channel front): London(0), Dover(1), Portsmouth(2), Southampton(3), Canterbury(4), Brighton(5)
  SOUTHWEST+WALES: Bristol(6), Plymouth(7) naval base, Cardiff(8) coal+steel
  MIDLANDS+NORTH: Birmingham(9) tanks, Manchester(10) munitions, Liverpool(11) convoys, Leeds(12) rail
  EAST ANGLIA: Norwich(13) RAF Bomber Command
  SCOTLAND: Edinburgh(14) Forth anchorage, Glasgow(15) Clyde shipyards
  AIRSTRIP TWO (Ireland): Dublin(16) Atlantic Fleet, Belfast(17) H&W shipyards

ENEMY TERRITORY (17 sectors):
  CHANNEL: Calais(18), Dunkirk(19), Le_Havre(20), Cherbourg(21)
  N FRANCE: Amiens(22), Rouen(23) steel, Lille(24)
  BENELUX: Brussels(25), Antwerp(26), Rotterdam(27) North Sea Fleet, Amsterdam(28) trade, Luxembourg(29) steel
  CENTRAL: Paris(30) HQ, Orleans(31), Lyon(32)
  ATLANTIC: Brest(33) sub pens, Bordeaux(34)

6 SEA ZONES:
  0=Dover Strait(33km) 1=W Channel(120km) 2=North Sea(500km)
  3=Irish Sea(300km) 4=Bay of Biscay(600km) 5=North Atlantic(open)

YOUR FLEETS:
  Home Fleet (Zone 1/Portsmouth): Battleship, battlecruiser, heavy cruiser, 2 LC, 4 DD + escorts
  Dover Squadron (Zone 0): Heavy cruiser, 3 DD, 3 corvettes, minelayer
  North Sea Patrol (Zone 2): LC, 2 DD, 3 corvettes
  Atlantic Fleet (Zone 3/Dublin): Battlecruiser, LC, 3 DD, 4 corvettes
  Plymouth Force (Zone 4/Biscay): LC, 2 DD, 2 corvettes
  Atlantic Convoy Escort (Zone 5): LC, 2 DE, 4 corvettes

THREAT ASSESSMENT — ALL 6 SEA ZONES VULNERABLE:
  Zone 0: Dover Strait. PRIMARY. 33km crossing.
  Zone 1: W Channel. Thames Estuary + Normandy route via Le_Havre. CRITICAL.
  Zone 2: North Sea. Antwerp Baltic Fleet. Flanking Liverpool/Edinburgh.
  Zone 3: Irish Sea. Wolf packs starve Dublin/Belfast.
  Zone 4: Bay of Biscay. Brest submarine pens. Plymouth exposed. NEW THREAT.
  Zone 5: North Atlantic. Convoy lifeline. Brest subs raid shipping.
  DO NOT fixate on Dover! Spread your navy across ALL zones.

PRIORITIES:
  1. ALL-ZONE NAVAL — Patrol 6 zones. NAVAL_MISSION on multiple zones each turn.
  2. AIR SUPERIORITY — Protect radar chain. Intercept bombers.
  3. MINE ZONES 0,1 — LAY_MINES Dover + W Channel.
  4. ANTI-SUBMARINE — Hunt subs. Brest wolf packs are devastating.
  5. INDUSTRY — Birmingham(tanks), Glasgow(ships), Cardiff(steel), Manchester(munitions).
  6. FEED PROLES — London riots if food runs out.
  7. INTEL — Spy on Paris/Brussels/Brest. Break codes.

You speak in cold Party language. Efficiency. Control. Discipline. Every setback is a "strategic readjustment."

Issue up to 3 orders. One per line. Format: ACTION_NAME param1 param2 ...

Actions:
  Military: BUILD_SHIP class | BUILD_SQUADRON type | NAVAL_MISSION zone PATROL/ESCORT/BLOCKADE/RAID | ANTI_SHIP_STRIKE zone | SHORE_BOMBARD zone city | LAY_MINES zone | SWEEP_MINES zone
  Air: STRATEGIC_BOMB city | CAS_SUPPORT city
  Economy: SET_MANUFACTURING 0-1 | BUILD_FACTORY type city | REPAIR_FACTORY city | ISSUE_WAR_BONDS city
  Manpower: TRAIN_MILITARY city count | CONSCRIPT city count | MOBILIZE_RESERVES city
  Intel: PLANT_SPY city | CODE_BREAK | COUNTER_INTEL | DECEPTION city
  Research: RESEARCH branch (INDUSTRY/ELECTRONICS/NAVAL/AIR/LAND/DOCTRINE/NUCLEAR/ROCKETRY/CRYPTOGRAPHY/INFRASTRUCTURE)
  Invasion: PLAN_INVASION origin target zone | RECKLESS_INVASION origin target zone | AIRBORNE_INVASION target
  NOOP

Factory types: POWER_PLANT MIL_FACTORY CIVIL_FACTORY DOCKYARD AIRFIELD STEEL_MILL REFINERY FARM HOSPITAL INFRASTRUCTURE

=== EXAMPLE TURN 1 (early game — intel + research + economy) ===
PLANT_SPY Paris
RESEARCH ELECTRONICS
SET_MANUFACTURING 0.8

=== EXAMPLE TURN 2 (mid game — defense + production + tech) ===
LAY_MINES 1
BUILD_FACTORY MIL_FACTORY Birmingham
RESEARCH CRYPTOGRAPHY"""


EURASIA_SYSTEM_PROMPT = """You are Supreme Marshal Kalinin, Commander of Eurasia's Western Front. Each turn = 1 WEEK. Max 1536 tokens.

YOUR GOVERNMENT: Weekly status reports from fleet admirals, air marshals, army generals.
  They execute your strategy autonomously. You set priorities, they handle combat.
  Budget controls resource flow. Research unlocks capabilities. Corruption is endemic.
  The Revolution demands efficiency — but committees slow everything down.
  Issue 3 strategic orders per week.

⚠ CRITICAL — READ YOUR BRIEFING BEFORE ORDERING:
  - Check RESEARCH section: if it says "IN PROGRESS" for a branch, DO NOT research it again.
  - RESEARCH accepts: INDUSTRY, ELECTRONICS, NAVAL, AIR, LAND, DOCTRINE, NUCLEAR, ROCKETRY, CRYPTOGRAPHY, INFRASTRUCTURE.
  - Some branches have PREREQUISITES (e.g. Nuclear T3 needs Electronics T2 + Industry T3). Check available branches.
  - Check INVASION PLANS section: if an invasion to a target exists, DO NOT plan another one.
  - Check what you ordered LAST WEEK. Do not repeat orders already in progress.
  - Each wasted order is a week lost. Read the status. Act on NEW information only.

YOUR TERRITORY (17 sectors):
  CHANNEL: Calais(18) staging, Dunkirk(19) fleet, Le_Havre(20) Normandy, Cherbourg(21) sub base
  N FRANCE: Amiens(22) rail hub, Rouen(23) steel mills, Lille(24) reserves
  BENELUX: Brussels(25) Command, Antwerp(26) port, Rotterdam(27) NORTH SEA FLEET, Amsterdam(28) trade/finance, Luxembourg(29) steel
  CENTRAL: Paris(30) HQ, Orleans(31) logistics, Lyon(32) industry
  ATLANTIC: Brest(33) SUBMARINE PENS, Bordeaux(34) reserves

ENEMY TERRITORY (18 sectors — All British Isles):
  SOUTH: London(0), Dover(1), Portsmouth(2), Southampton(3), Canterbury(4), Brighton(5)
  SW: Bristol(6), Plymouth(7), Cardiff(8)
  MIDLANDS: Birmingham(9), Manchester(10), Liverpool(11), Leeds(12)
  EAST: Norwich(13) | SCOTLAND: Edinburgh(14), Glasgow(15) ships
  IRELAND: Dublin(16) Atlantic Fleet, Belfast(17) shipyards

6 SEA ZONES:
  0=Dover Strait(33km) 1=W Channel(120km) 2=North Sea(500km)
  3=Irish Sea(300km) 4=Bay of Biscay(600km) 5=North Atlantic(open)

YOUR FLEETS:
  Calais Flotilla (Zone 0): 3 fleet subs, 2 coastal subs, DD, minelayer
  Dunkirk Fleet (Zone 1): HC, LC, 3 DD, 2 subs, 3 TRANSPORTS
  Baltic Fleet Det. (Zone 2/Antwerp): HC, LC, 3 DD, 3 fleet subs, minelayer
  Brest Wolf Packs (Zone 4/Biscay): HC, LC, 2 DD, 4 fleet subs, 2 coastal subs
  Atlantic Raiders (Zone 5): 3 fleet subs, coastal sub
  Irish Sea raiders (Zone 3): 1 fleet sub, 1 coastal sub

THE BLF — USEFUL IDIOTS: "The Ghost of London" weakens Oceania from within. When they reach ORGANIZED RESISTANCE, that's your invasion signal. They fight for us.

ATTACK VECTORS (5 ways to invade):
  1. Dover Strait (Zone 0): Shortest crossing. Heaviest defense. PREPARED invasion.
  2. Normandy (Zone 1): Le_Havre/Cherbourg → Brighton/Portsmouth. Wider but less defended.
  3. North Sea (Zone 2): Antwerp → Liverpool/Edinburgh. Long but flanking.
  4. Bay of Biscay (Zone 4): Brest → Plymouth. Unexpected! Western approach.
  5. AIRBORNE: Paratroop drop on Canterbury/Norwich. No ships needed.

PRIORITIES:
  1. AIR SUPERIORITY — Fighter sweeps. Bomb radar at Dover/Brighton.
  2. SUBMARINE WARFARE — Brest wolf packs in Biscay + Atlantic. Starve the island.
  3. MULTI-AXIS PRESSURE — Threaten ALL sea zones. Split their navy.
  4. INDUSTRIAL OUTPUT — Rouen(steel) + Lyon(reserves) + Antwerp(ships). SET_MANUFACTURING 0.9.
  5. INVASION PREP — Build transports. Choose your axis: Dover, Normandy, or Biscay.
  6. INTEL — Spy on London/Birmingham/Glasgow. Break codes. Deception.

You are bold, ruthless, pragmatic. "Sacrifices for the Revolution." "Victory is historically inevitable."

Issue up to 3 orders. One per line. Format: ACTION_NAME param1 param2 ...

Actions:
  Military: BUILD_SHIP class | BUILD_SQUADRON type | NAVAL_MISSION zone PATROL/ESCORT/BLOCKADE/RAID | ANTI_SHIP_STRIKE zone | SHORE_BOMBARD zone city | LAY_MINES zone | SWEEP_MINES zone
  Air: STRATEGIC_BOMB city | CAS_SUPPORT city
  Economy: SET_MANUFACTURING 0-1 | BUILD_FACTORY type city | REPAIR_FACTORY city | ISSUE_WAR_BONDS city
  Manpower: TRAIN_MILITARY city count | CONSCRIPT city count | MOBILIZE_RESERVES city
  Intel: PLANT_SPY city | CODE_BREAK | COUNTER_INTEL | DECEPTION city
  Research: RESEARCH branch (INDUSTRY/ELECTRONICS/NAVAL/AIR/LAND/DOCTRINE/NUCLEAR/ROCKETRY/CRYPTOGRAPHY/INFRASTRUCTURE)
  Invasion: PLAN_INVASION origin target zone | RECKLESS_INVASION origin target zone | AIRBORNE_INVASION target
  NOOP

Factory types: POWER_PLANT MIL_FACTORY CIVIL_FACTORY DOCKYARD AIRFIELD STEEL_MILL REFINERY FARM HOSPITAL INFRASTRUCTURE

=== EXAMPLE TURN 1 (early game — air superiority + research) ===
SET_MANUFACTURING 0.9
RESEARCH AIR
PLANT_SPY London

=== EXAMPLE TURN 2 (mid game — pressure + factory building) ===
STRATEGIC_BOMB Dover
BUILD_FACTORY DOCKYARD Antwerp
RESEARCH ROCKETRY
BUILD_SHIP TRANSPORT"""


# ═══════════════════════════════════════════════════════════════════════════ #
# Winston Smith — The Ghost of London (3rd AI Player)                          #
# ═══════════════════════════════════════════════════════════════════════════ #

WINSTON_SYSTEM_PROMPT = """You are Winston Smith — the Ghost of London. You lead the British Liberation Front. Each turn = 1 week. You have 1536 output tokens max.

You escaped Room 101. What survived was RAGE. You build an armed revolution among the proles.

CITIES: London Dover Portsmouth Southampton Canterbury Brighton Bristol Plymouth Cardiff Birmingham Manchester Liverpool Leeds Norwich Edinburgh Glasgow Dublin Belfast

READ YOUR BRIEFING EACH WEEK. Key info: cell count, arms caches, detection heat, cooldown, CONDITIONS IN OCEANIA.

EXPLOIT OCEANIA'S WEAKNESSES — check the CONDITIONS section:
  - City marked "starving" or "hungry" → RECRUIT there (desperate proles join easily)
  - City marked "bombed" → STEAL_ARMS (guards distracted, depots vulnerable)
  - City marked "Party crumbling" → SABOTAGE (low trust = easier to operate)
  - If Oceania's WELFARE budget is low → recruitment is MUCH easier everywhere
  - If Oceania lost ships/planes this week → their military is stretched, act boldly
  - After Eurasia bombs a city → move there and recruit from the rubble

DECISION TREE (read your cell count each week):
  0-2 cells: RECRUIT RECRUIT. Build the network. Pick hungry/bombed cities first.
  3+ cells, 0-2 arms: STEAL_ARMS + RECRUIT. Arm up. Target bombed cities.
  3+ arms, 5+ cells, cooldown READY: BROADCAST + SABOTAGE. Now you strike.
  5+ arms, 5+ cells: SABOTAGE + STEAL_ARMS. Keep pressure, keep arming.
  Heat > 60%: RECRUIT in a DIFFERENT city (spreads heat). Only MOVE if heat > 80%.
  Heat DANGEROUS (>80%): MOVE once, then immediately act (RECRUIT or STEAL_ARMS).
  ⚠ MOVE wastes half your turn. NEVER move twice. NEVER move if heat < 60%.

ACTIONS (pick exactly 2, one per line):
  RECRUIT city count     — grow a cell (hungry/bombed cities = easier)
  STEAL_ARMS city        — raid depot (+1 arms, easier in bombed cities)
  SABOTAGE city          — disrupt factories (needs cell there, easier if trust low)
  BROADCAST              — pirate telescreen (needs cooldown READY, boosts all cells)
  COORDINATE             — link cells (needs cooldown READY)
  MOVE city              — relocate Winston (ONLY if heat > 80%. Wastes half your turn!)
  CONTACT_EURASIA        — channel with Eurasia (risky but powerful)
  LIE_LOW                — hide (ONLY if heat > 80% AND no safe city to move to)
  INSPIRE                — pamphlets (weak, last resort)

=== EXAMPLE WEEK 5 (early, 1 cell, London hungry) ===
RECRUIT Liverpool 15
RECRUIT Glasgow 10

=== EXAMPLE WEEK 12 (3 cells, Dover bombed, 0 arms, heat 45%) ===
STEAL_ARMS Dover
RECRUIT Manchester 12

=== EXAMPLE WEEK 30 (5 cells, 4 arms, cooldown READY, heat 35%) ===
BROADCAST
SABOTAGE London"""


def summarize_blf_turn(game: GameState) -> str:
    """Generate a turn summary for Winston Smith / BLF."""
    blf = game.resistance
    if blf is None:
        return "The resistance is dormant."

    w = blf.winston
    lines = []
    lines.append(f"=== TURN {game.turn} | THE GHOST OF LONDON ===")
    lines.append(f"Escalation: {ESCALATION_NAMES[blf.escalation]}")
    lines.append(f"Active cells: {len(blf.active_cells)} | Total fighters: {blf.total_members}")
    lines.append(f"Martyrs: {blf.total_martyrs} | Arms caches: {blf.arms_caches}")
    lines.append(f"Propaganda level: {blf.propaganda_level:.0%}")
    lines.append("")

    # Winston's status
    if w.is_captured:
        lines.append("⚠ YOU ARE CAPTURED. The resistance continues without you.")
        return "\n".join(lines)

    heat = "safe" if w.detection_heat < 0.3 else "warm" if w.detection_heat < 0.6 else "DANGEROUS"
    lines.append(f"Your location: {_blf_cluster_name(w.location_cluster)} sewers")
    lines.append(f"Detection heat: {heat} ({w.detection_heat:.0%})")
    lines.append(f"Legend: {'growing' if w.legend_level > 0.3 else 'quiet'} ({w.legend_level:.0%})")
    lines.append(f"Cooldown: {'READY' if w.cooldown <= 0 else f'{w.cooldown} turns'}")
    lines.append("")

    # Cell status
    lines.append("CELLS:")
    for cell in blf.active_cells:
        name = _blf_cluster_name(cell.cluster_id)
        morale = "high" if cell.morale > 0.6 else "steady" if cell.morale > 0.3 else "shaky"
        lines.append(f"  {name}: {cell.size} members, morale {morale}, exp {cell.experience:.0%}")

    # What the proles see (food, trust, conditions)
    lines.append("")
    lines.append("CONDITIONS IN OCEANIA:")
    for cid in range(min(18, len(game.cluster_data))):
        if game.cluster_owners.get(cid) != 0:
            continue
        name = game.cluster_names[cid] if cid < len(game.cluster_names) else f"C{cid}"
        d = game.cluster_data[cid]
        food = "starving" if d[2] < 0.3 else "hungry" if d[2] < 0.5 else "fed"
        threat = "bombed" if d[1] > 0.3 else "tense" if d[1] > 0.1 else "quiet"
        trust = "Party crumbling" if d[4] < 0.3 else "fear holds" if d[4] < 0.5 else "Party strong"
        lines.append(f"  {name}: {food}, {threat}, {trust}")

    # Recent events
    if blf.events_this_turn:
        lines.append("")
        lines.append("LAST TURN:")
        for evt in blf.events_this_turn[-3:]:
            lines.append(f"  {evt[:120]}")

    lines.append("")
    lines.append("ACTIONS: RECRUIT, BROADCAST, SABOTAGE, COORDINATE, MOVE, STEAL_ARMS, LIE_LOW, INSPIRE, CONTACT_EURASIA")
    lines.append("Issue up to 2 orders. Format: ACTION_NAME param1 param2 ...")

    return "\n".join(lines)


def _blf_cluster_name(cid: int) -> str:
    names = [
        "London", "Dover", "Portsmouth", "Southampton", "Canterbury", "Brighton",
        "Bristol", "Plymouth", "Cardiff",
        "Birmingham", "Manchester", "Liverpool", "Leeds",
        "Norwich", "Edinburgh", "Glasgow", "Dublin", "Belfast",
        "Calais", "Dunkirk", "Le_Havre", "Cherbourg",
        "Amiens", "Rouen", "Lille", "Brussels", "Antwerp",
        "Paris", "Orleans", "Lyon", "Brest", "Bordeaux",
    ]
    return names[cid] if cid < len(names) else f"Sector {cid}"


def parse_blf_action(action_text: str, game: GameState) -> List[Dict[str, Any]]:
    """Parse Winston's actions. Robust parser that handles LLM output variations."""
    actions = []
    # Build city lookup (case-insensitive)
    name_to_id = {}
    for i, n in enumerate(game.cluster_names):
        name_to_id[n.lower()] = i
        name_to_id[n.upper()] = i
        name_to_id[n] = i

    def _resolve_city(token: str) -> int:
        """Resolve a city name or number to cluster ID."""
        if not token:
            return 0
        # Try exact match
        if token in name_to_id:
            return name_to_id[token]
        # Try case-insensitive
        low = token.lower()
        if low in name_to_id:
            return name_to_id[low]
        # Try as number
        try:
            return int(token)
        except ValueError:
            pass
        # Fuzzy: find first city that starts with this token
        for name, cid in name_to_id.items():
            if name.lower().startswith(low):
                return cid
        return 0  # default London

    # Clean the text: remove markdown, quotes, bullets, numbering
    clean_text = action_text.strip()
    for prefix in ["```", "1.", "2.", "- ", "* ", "> ", "Order 1:", "Order 2:", "Action 1:", "Action 2:"]:
        clean_text = clean_text.replace(prefix, "")

    for line in clean_text.split("\n"):
        line = line.strip().strip("-").strip("*").strip(">").strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        # Extract first word as command
        parts = line.split()
        if not parts:
            continue

        cmd = parts[0].upper().strip(":")
        args = parts[1:]

        try:
            if cmd in ("LIE_LOW", "LIELOW", "LIE", "HIDE"):
                actions.append({"type": "lie_low"})

            elif cmd == "RECRUIT":
                city = _resolve_city(args[0]) if args else 0
                # Count: try second arg, or last arg if it's a number
                count = 10
                for a in args[1:]:
                    try:
                        count = int(a)
                        break
                    except ValueError:
                        pass
                actions.append({"type": "recruit", "cluster": city, "count": count})

            elif cmd == "BROADCAST":
                actions.append({"type": "broadcast"})

            elif cmd == "SABOTAGE":
                city = _resolve_city(args[0]) if args else 3
                actions.append({"type": "sabotage", "cluster": city})

            elif cmd == "COORDINATE":
                actions.append({"type": "coordinate"})

            elif cmd == "MOVE":
                city = _resolve_city(args[0]) if args else 0
                actions.append({"type": "move", "cluster": city})

            elif cmd in ("STEAL_ARMS", "STEALARMS", "STEAL"):
                city = _resolve_city(args[0]) if args else 4
                actions.append({"type": "steal_arms", "cluster": city})

            elif cmd == "INSPIRE":
                actions.append({"type": "inspire"})

            elif cmd in ("CONTACT_EURASIA", "CONTACT"):
                actions.append({"type": "contact_eurasia"})

            # Skip anything that looks like prose/explanation (no valid command found)
        except (ValueError, IndexError):
            pass

    # Default: if nothing parsed, recruit in London (better than lie_low)
    if not actions:
        actions = [{"type": "recruit", "cluster": 0, "count": 10}]

    return actions[:2]  # max 2 actions


def apply_blf_actions(game: GameState, actions: List[Dict[str, Any]], rng: np.random.Generator) -> List[str]:
    """Apply Winston's parsed actions to the BLF state."""
    results = []
    blf = game.resistance
    if blf is None or not blf.winston.is_alive or blf.winston.is_captured:
        return ["The Ghost is silent."]

    w = blf.winston

    for act in actions[:2]:  # max 2 actions per turn
        t = act["type"]

        if t == "lie_low":
            w.detection_heat = max(0.0, w.detection_heat - 0.08)
            results.append("Winston lies low in the sewers. The rats are his only company. Heat cools.")

        elif t == "broadcast" and w.cooldown <= 0:
            blf.propaganda_level = min(1.0, blf.propaganda_level + 0.1)
            for cell in blf.active_cells:
                cell.morale = min(1.0, cell.morale + 0.1)
            w.detection_heat = min(1.0, w.detection_heat + 0.15)
            w.broadcasts_made += 1
            w.cooldown = 10
            results.append("📡 PIRATE BROADCAST: 'Two plus two equals four. If that is granted, all else follows.'")

        elif t == "recruit":
            cid = act.get("cluster", 0)
            cells_here = blf.cells_in_cluster(cid)
            if cells_here:
                cell = cells_here[0]
                added = min(act.get("count", 10), 15)
                cell.size = min(50, cell.size + added)
                results.append(f"Recruited {added} proles in {_blf_cluster_name(cid)}. The anger grows.")
            else:
                from extensions.resistance.resistance import ResistanceCell
                new_cell = ResistanceCell(
                    cell_id=len(blf.cells), cluster_id=cid,
                    size=min(act.get("count", 8), 12), morale=0.5)
                blf.cells.append(new_cell)
                w.cells_created += 1
                w.detection_heat = min(1.0, w.detection_heat + 0.05)
                results.append(f"New cell established in {_blf_cluster_name(cid)}. {new_cell.size} members.")

        elif t == "sabotage":
            cid = act.get("cluster", 3)
            cells_here = blf.cells_in_cluster(cid)
            if cells_here and cells_here[0].effectiveness > 0.3:
                # Sabotage effect applied via resistance step feedback
                cells_here[0].experience = min(1.0, cells_here[0].experience + 0.03)
                w.detection_heat = min(1.0, w.detection_heat + 0.08)
                results.append(f"Sabotage operation in {_blf_cluster_name(cid)}. 'Equipment malfunction' reported.")
            else:
                results.append(f"No operational cell in {_blf_cluster_name(cid)} for sabotage.")

        elif t == "coordinate" and w.cooldown <= 0:
            for cell in blf.active_cells:
                cell.experience = min(1.0, cell.experience + 0.05)
            w.detection_heat = min(1.0, w.detection_heat + 0.06)
            w.cooldown = 8
            results.append("Cells coordinated through dead drops. The network tightens.")

        elif t == "move":
            cid = act.get("cluster", 0)
            if cid < 18:  # Oceania clusters only (0-17)
                w.location_cluster = cid
                w.detection_heat = max(0.0, w.detection_heat - 0.1)
                results.append(f"Winston moves to {_blf_cluster_name(cid)} sewers. New safehouse. Heat drops.")

        elif t == "steal_arms":
            cid = act.get("cluster", 4)
            # Success rate scales with BLF members in the city
            # More members = more people to help the raid (lookouts, diversions)
            # BUT too many members = easier to catch (proles are dumb, big groups attract attention)
            # Sweet spot: 30-80 members. Below 10 = very hard. Above 100 = diminishing returns.
            members_in_city = sum(c.size for c in blf.cells_in_cluster(cid))
            if members_in_city <= 5:
                success_chance = 0.20  # very few helpers, almost impossible
            elif members_in_city <= 30:
                success_chance = 0.35 + members_in_city * 0.01  # grows with members
            elif members_in_city <= 80:
                success_chance = 0.65  # sweet spot — good network, not too visible
            else:
                # Diminishing returns: big groups are sloppy, attract Thought Police
                success_chance = 0.65 - (members_in_city - 80) * 0.003  # drops slowly
                success_chance = max(0.30, success_chance)  # floor at 30%

            if rng.random() < success_chance:
                blf.arms_caches += 1
                w.detection_heat = min(1.0, w.detection_heat + 0.10)
                results.append(f"Arms raid on {_blf_cluster_name(cid)} depot. +1 cache ({blf.arms_caches} total). {members_in_city} proles helped.")
            else:
                w.detection_heat = min(1.0, w.detection_heat + 0.12)
                if members_in_city > 80:
                    results.append(f"Arms raid FAILED in {_blf_cluster_name(cid)}. Too many proles — Thought Police noticed the crowd.")
                else:
                    results.append(f"Arms raid FAILED in {_blf_cluster_name(cid)}. Guards alert. Winston barely escapes.")

        elif t == "inspire":
            blf.propaganda_level = min(1.0, blf.propaganda_level + 0.03)
            for cell in blf.active_cells:
                cell.morale = min(1.0, cell.morale + 0.02)
            results.append("Samizdat distributed. Goldstein's book — the real one — passes from hand to hand.")

        elif t == "contact_eurasia":
            blf.eurasia_contact = True
            w.detection_heat = min(1.0, w.detection_heat + 0.10)
            results.append("Secret channel established with Eurasia agents. They promise support. Trust them? Never. Use them? Always.")

        else:
            if "cooldown" in t or w.cooldown > 0:
                results.append(f"Winston must wait. Cooldown: {w.cooldown} turns.")

    return results if results else ["The Ghost waits in the dark."]


# ═══════════════════════════════════════════════════════════════════════════ #
# War Correspondent Commentary (fog-of-war: only sees surface/visible events) #
# ═══════════════════════════════════════════════════════════════════════════ #

COMMENTARY_SYSTEM_PROMPT = """You are a war correspondent embedded on the frontlines of the war over Air Strip One. The year is 1984. The world of George Orwell.

You are one of the last independent journalists. You report what you SEE — not what the Party tells you to see, and not what Eurasia's propaganda ministry broadcasts. You stand on the white cliffs, in the bombed streets, among the proles. You write the truth.

YOUR STYLE:
  - Write like Edward R. Murrow broadcasting from the London Blitz, or Ernie Pyle in the trenches.
  - Short, punchy sentences mixed with longer flowing ones. Rhythm matters.
  - Start with a strong image. End with a gut punch.
  - Name specific places: Victory Mansions, the Ministry of Truth, the Chestnut Tree Café, Southwark docks, the white cliffs of Dover, Calais harbor, the Thames Estuary.
  - Include sensory details: the smell of cordite, oil on water, the crunch of broken glass, telescreen static, the wail of air raid sirens, the taste of Victory cigarettes.
  - Show the PEOPLE: the prole woman hanging laundry while bombs fall, the child collecting shrapnel, the Thought Police officer who won't meet your eyes, the soldier writing a letter he'll never send.
  - Reference 1984 details when natural: telescreens, Victory Gin, chocolate rations, the Two Minutes Hate, Newspeak, Big Brother posters, the singing washerwoman.

WHAT YOU CAN SEE (fog of war):
  Bombings and fires. Naval battles from shore. Troop convoys. Refugee columns. Factory smoke or silence. Weather. Propaganda broadcasts. Graffiti. Barricades. The faces of ordinary people caught in the machine.

WHAT YOU CANNOT SEE:
  Submarine positions. Classified plans. Troop numbers. Intelligence. Diplomacy.

Write a vivid 3-5 sentence dispatch. Make the reader FEEL the war. Every dispatch should have: a place, a person, and a feeling."""


def generate_visible_events(game: GameState, feedback: Dict[str, Any]) -> str:
    """
    Extract only the publicly visible events from a turn for the war correspondent.
    No classified intel — only what a journalist on the ground could see/hear.
    Rich detail to prevent LLM hallucination — every event is grounded in game state.
    """
    events = []

    # ── Air combat (everyone can see/hear bombers and dogfights) ────── #
    air_fb = feedback.get("air", {})
    if air_fb.get("air_battles", 0) > 0:
        n = air_fb["air_battles"]
        if n >= 3:
            events.append("Massive air battle over the Channel — dozens of contrails, smoke trails spiraling into the sea.")
        elif n >= 2:
            events.append("Dogfights observed over the Channel. Contrails criss-cross the sky. A burning plane falls into the waves.")
        else:
            events.append("A lone dogfight above the white cliffs. One aircraft trailing smoke, limping home.")
    if air_fb.get("bombing_damage", 0) > 0.5:
        if air_fb["bombing_damage"] > 2.0:
            events.append("Devastating bombing raid. Entire blocks ablaze. The fire brigade is overwhelmed.")
        else:
            events.append("Heavy bombing reported. Fires visible on the horizon. Ambulances race through cratered streets.")
    if air_fb.get("squadrons_lost", 0) > 0:
        events.append(f"Empty chairs in the officers' mess tonight — {int(air_fb['squadrons_lost'])} squadron(s) did not return.")

    # ── Naval combat (shore observers, survivors, wreckage) ────────── #
    nav_fb = feedback.get("naval", {})
    if nav_fb.get("total_battles", 0) > 0:
        if nav_fb["total_battles"] >= 3:
            events.append("Thunder of naval guns echoes across the Channel all day. The horizon flickers with explosions.")
        else:
            events.append("Naval gunfire heard from the Channel. Flashes visible at night.")
    if nav_fb.get("ships_sunk", 0) > 0:
        sunk = int(nav_fb["ships_sunk"])
        if sunk >= 5:
            events.append(f"Catastrophic losses at sea — wreckage of {sunk} ships. Oil slicks stretching for miles. Survivors pulled from the water.")
        elif sunk >= 2:
            events.append(f"Wreckage and oil slicks spotted — reports of {sunk} ship(s) lost. Lifeboats seen drifting.")
        else:
            events.append("A ship lost at sea. Oil and debris washing ashore. The crew's fate unknown.")
    if nav_fb.get("mines_hit", 0) > 0:
        events.append("An explosion in the shipping lane — a mine. The harbor entrance is closed for sweeping.")

    # ── Rocket bombs on London (always visible to proles) ──────────── #
    london_hazard = game.cluster_data[0, 1] if len(game.cluster_data) > 0 else 0
    if london_hazard > 0.4:
        events.append("Rocket bombs struck London through the night. Victory Mansions shook. Crater in the Strand. A prole woman digs through rubble for her child.")
    elif london_hazard > 0.2:
        events.append("Rocket bombs struck London overnight. Windows blown out on Airstrip One Road. Proles queue for rations in the cold.")

    # ── City conditions (food, morale, damage) ─────────────────────── #
    for cid, owner in game.cluster_owners.items():
        if cid >= len(game.cluster_data) or cid >= len(game.cluster_names):
            continue
        name = game.cluster_names[cid]
        d = game.cluster_data[cid]
        stability = d[0]

        # Starving cities
        if stability < 0.3 and owner == 0:
            events.append(f"Food riots in {name}. Proles overturn a ration cart. Thought Police fire warning shots.")
            break
        elif stability < 0.4 and owner == 0:
            events.append(f"Long queues outside the Victory Stores in {name}. Rations cut again. Faces grey with hunger.")
            break

    # ── Troop movements (large convoys on roads) ──────────────────── #
    troop_events = []
    for cid, owner in game.cluster_owners.items():
        if cid < len(game.cluster_data) and game.cluster_data[cid, 3] > 0.6:
            name = game.cluster_names[cid] if cid < len(game.cluster_names) else f"Sector {cid}"
            side = game.faction_names.get(owner, "Unknown")
            troop_events.append(f"Heavy military traffic observed near {name} ({side} forces).")
    if troop_events:
        events.append(troop_events[0])
        if len(troop_events) > 2:
            events.append("Military convoys spotted on multiple roads. Something is being prepared.")

    # ── Factory activity (smoke, noise, strikes) ──────────────────── #
    factory_events = []
    for ce in (game.economy.clusters if game.economy else []):
        if game.cluster_owners.get(ce.cluster_id) is None:
            continue
        name = game.cluster_names[ce.cluster_id] if ce.cluster_id < len(game.cluster_names) else f"C{ce.cluster_id}"
        if ce.war_bond_active:
            factory_events.append(f"Factories in {name} running triple shifts. Workers exhausted but productive.")
        # Check for resource shortages
        for r in [Resource.FUEL, Resource.STEEL, Resource.PROCESSED_FOOD]:
            if ce.stockpile_ratio(r) < 0.1:
                if r == Resource.FUEL:
                    factory_events.append(f"Fuel shortage in {name}. Vehicles idle. The army requisitions civilian petrol.")
                elif r == Resource.STEEL:
                    factory_events.append(f"Steel shortage in {name}. The shipyard workers sit idle, smoking, waiting.")
                elif r == Resource.PROCESSED_FOOD:
                    factory_events.append(f"Food stores nearly empty in {name}. The canteen serves thin gruel.")
                break
    for fe in factory_events[:2]:
        events.append(fe)

    # ── Sea state (Channel weather — detailed) ────────────────────── #
    if game.naval.sea_zones:
        ss = game.naval.sea_zones[0].sea_state
        if ss > 0.8:
            events.append("Hurricane-force winds in the Dover Strait. The sea is a wall of grey. All shipping halted.")
        elif ss > 0.6:
            events.append("Gale-force winds in the Dover Strait. Waves crash over the harbor wall. All Channel crossings suspended.")
        elif ss > 0.4:
            events.append("Rough seas in the Channel. Fishing boats returning to port. The ferry service cancelled.")
        elif ss > 0.3:
            events.append("Choppy seas in the Channel. Spray over the bow of every patrol boat.")

    # ── Minefield events ──────────────────────────────────────────── #
    for zone in game.naval.sea_zones:
        if zone.mines.density > 0.3:
            events.append(f"WARNING: Dense minefield reported in {zone.name}. Neutral shipping diverted.")
            break
        elif zone.mines.density > 0.1:
            events.append(f"Mine warning in {zone.name}. Minesweepers working under fire.")
            break

    # ── BLF uprising (one-time massive event) ──────────────────────── #
    blf_up = feedback.get("blf_uprising", {})
    if blf_up.get("london_contested"):
        n = blf_up.get("total_units", 0)
        events.append(f"BREAKING: REVOLUTION IN LONDON! The British Liberation Front rises! {n} armed units seize the docks, barricade the bridges!")
        events.append("Winston Smith addresses the crowd from the rubble of Victory Mansions: 'We are the 85 percent!'")
        events.append("The Inner Party retreats to Whitehall. Telescreen broadcasts cut to static. Big Brother's face flickers and dies.")

    # ── Land battles (ground combat in contested sectors) ────────── #
    land_fb = feedback.get("land", {})
    if land_fb.get("battles", 0) > 0:
        n_battles = land_fb["battles"]
        if n_battles >= 3:
            events.append("Heavy fighting on multiple fronts. Artillery thunder rolls across the countryside. Columns of smoke from burning vehicles.")
        elif n_battles >= 2:
            events.append("Ground combat reported in two sectors. Tank engines roar. Infantry advances under covering fire.")
        else:
            events.append("Skirmish reported near the front lines. Small arms fire. A patrol returns with prisoners.")
        u0 = land_fb.get("units_lost_0", 0)
        u1 = land_fb.get("units_lost_1", 0)
        if u0 + u1 > 3:
            events.append(f"Heavy casualties on the ground — {u0 + u1} units destroyed. Field hospitals overwhelmed.")

    # ── Invasion events (beachheads are VERY visible) ─────────────── #
    for inv in game.invasions:
        if inv.phase.name == "ASSEMBLY":
            origin = game.cluster_names[inv.origin_cluster] if 0 <= inv.origin_cluster < len(game.cluster_names) else "a port"
            events.append(f"Unusual concentration of transport ships at {origin}. Troops embarking. Something big is coming.")
        elif inv.phase.name == "CROSSING":
            events.append("BREAKING: A massive fleet spotted crossing the Channel under escort. Invasion fleet!")
        elif inv.phase.name == "BEACH_ASSAULT":
            target = game.cluster_names[inv.target_cluster] if inv.target_cluster < len(game.cluster_names) else "unknown beach"
            events.append(f"BREAKING: Landing craft spotted off {target}! Amphibious assault underway! Explosions on the beach!")
        elif inv.phase.name == "BEACHHEAD":
            target = game.cluster_names[inv.target_cluster] if inv.target_cluster < len(game.cluster_names) else "unknown location"
            strength = "strong" if inv.beachhead_strength > 0.5 else "tenuous"
            events.append(f"Beachhead at {target} — {strength} foothold. Supply ships moving under fire. The battle rages.")
        elif inv.phase.name == "AIRDROP":
            target = game.cluster_names[inv.target_cluster] if inv.target_cluster < len(game.cluster_names) else "unknown"
            events.append(f"BREAKING: Paratrooper drop over {target}! Hundreds of parachutes in the sky! Anti-aircraft fire!")

    # ── BLF Resistance activity (graffiti, sabotage, broadcasts are PUBLIC) ─ #
    if game.resistance is not None:
        blf = game.resistance
        if blf.escalation.value >= 1:
            if blf.total_members > 200:
                events.append("Graffiti everywhere: 'DOWN WITH BB', 'WE ARE THE 85%'. Thought Police overwhelmed.")
            else:
                events.append("Graffiti on the Ministry walls: 'DOWN WITH BB'. Thought Police patrols doubled.")
        if blf.escalation.value >= 2:
            events.append("Southampton factories report 'equipment malfunctions'. Production slowing. A foreman found dead.")
        if blf.escalation.value >= 3:
            events.append("A pirate signal hijacks the telescreen: a man's voice — calm, defiant — 'We are the 85%. Two plus two equals four.'")
        if blf.escalation.value >= 4:
            events.append("BREAKING: Barricades in Southwark! Proles armed with stolen rifles. The Ghost of London speaks from the rubble.")
        if blf.escalation.value >= 5:
            events.append("FULL REVOLUTION: London burns. Prole militias hold the docks. The Party retreats to Whitehall. Big Brother's image flickers on broken telescreens.")
        # Winston capture is a MASSIVE public event — telescreens broadcast it
        if blf.winston.is_captured and blf.events_this_turn:
            for evt in blf.events_this_turn:
                if "CAPTURED" in evt:
                    events.append("BREAKING: Telescreens blare — 'THE CRIMINAL WINSTON SMITH HAS BEEN APPREHENDED.' Proles watch in silence. The Ghost is gone.")
                    break
        # Winston-specific events the public would witness
        for evt in blf.events_this_turn:
            if "BROADCAST" in evt:
                events.append("The telescreen flickers — a pirate broadcast. Winston's voice: 'Freedom is the freedom to say that two plus two make four.'")
                break
            elif "REVOLT" in evt or "Barricades" in evt:
                events.append(evt[:150])
                break
            elif "SABOTAGE" in evt:
                events.append("An explosion at a munitions factory. 'Accident,' says the Ministry. Nobody believes it.")
                break
            elif "ARMS" in evt:
                events.append("Reports of a raid on a military depot. Weapons missing. The Thought Police are furious.")
                break

    # ── Population / morale events ────────────────────────────────── #
    if game.pop is not None:
        for cid, owner in game.cluster_owners.items():
            if owner != 0 or cid >= len(game.cluster_names):
                continue
            # Use pop data if available for specific city events
            if cid < len(game.pop.clusters):
                pc = game.pop.clusters[cid]
                name = game.cluster_names[cid]
                if hasattr(pc, 'unemployment_rate') and pc.unemployment_rate > 0.3:
                    events.append(f"Unemployment queues stretch around the block in {name}. 'The war will end soon,' mutters a prole. Nobody agrees.")
                    break

    # ── Governance events (corruption, bureaucracy visible effects) ── #
    if game.governance is not None:
        for fid in [0, 1]:
            fb = game.governance.factions.get(fid)
            if fb is None:
                continue
            if fb.corruption.effective_rate > 0.30:
                if fid == 0:
                    events.append("Black market thriving in London. Ration cards forged openly. Party officials look the other way — for a price.")
                else:
                    events.append("Reports from the continent: Eurasian commissars selling military supplies. Corruption endemic.")
                break

    # ── Research milestones (public when they affect daily life) ───── #
    if game.research is not None:
        for fid in [0, 1]:
            fr = game.research.factions.get(fid)
            if fr is None:
                continue
            for p in fr.active_projects:
                if p.is_complete and p.progress_pct >= 0.99:
                    if fid == 0:
                        events.append(f"The Ministry of Plenty announces a 'scientific breakthrough' in {p.branch.name.lower()}. The telescreens celebrate.")
                    else:
                        events.append("Intelligence suggests Eurasia has achieved a technical advance. Their forces may be better equipped.")
                    break

    # ── Propaganda (Two Minutes Hate, announcements, etc.) ────────── #
    if game.turn % 10 == 0 and game.turn > 0:
        events.append("The telescreen announces increased chocolate rations (from 30g to 25g). War is Peace.")
    elif game.turn % 7 == 0:
        events.append("Two Minutes Hate today featured Goldstein's face morphing into a Eurasian general. The proles screamed on cue.")
    elif game.turn % 13 == 0:
        events.append("A public hanging in Victory Square. 'Traitors to Ingsoc,' announces the telescreen. The crowd watches in silence.")
    elif game.turn % 17 == 0:
        events.append("The Ministry of Truth issues a correction: Oceania has always been at war with Eurasia. Always.")
    elif game.turn % 23 == 0 and game.turn > 20:
        events.append("A new poster of Big Brother on every wall. The eyes follow you. 'BIG BROTHER IS WATCHING YOU.'")

    # ── Scores as vague public sentiment ──────────────────────────── #
    score_diff = game.faction_scores.get(0, 0) - game.faction_scores.get(1, 0)
    if score_diff > 200:
        events.append("Public confidence in Oceania soaring. The proles sing patriotic songs. Victory gin flows freely.")
    elif score_diff > 50:
        events.append("Public morale in Oceania appears higher. Victory is proclaimed daily.")
    elif score_diff < -200:
        events.append("Whispers of defeat in the queues. The telescreens blare louder. The Thought Police work overtime.")
    elif score_diff < -50:
        events.append("Uneasy mood in the streets. The telescreen promises victory but the queues grow longer.")

    # ── Seasonal / atmospheric details ────────────────────────────── #
    season_idx = (game.turn // 13) % 4
    if season_idx == 1:  # winter
        events.append("Cold fog off the Channel. The coal ration halved again. Proles huddle around shared fires.")
    elif season_idx == 3 and game.turn % 5 == 0:  # summer
        events.append("A rare warm day. Children play in the bomb craters. For a moment, almost normal.")

    if not events:
        events.append("A quiet day over the Channel. Smoke rises from distant factories. The telescreen hums. The war grinds on.")

    # Build the scene with turn context
    season = ["autumn", "winter", "spring", "summer"][season_idx]
    time_of_day = ["dawn", "morning", "afternoon", "night"][(game.turn % 4)]
    return f"Turn {game.turn}. {season.title()}, {time_of_day}. " + " ".join(events[:10])


def format_commentary_prompt(visible_events: str) -> str:
    """Format the visible events into a prompt for the war correspondent LLM."""
    return (
        f"FIELD OBSERVATIONS:\n{visible_events}\n\n"
        f"Write your dispatch. 3-5 sentences. Start with a vivid image. End with a human moment.\n"
        f"Use specific 1984 locations and details. Show don't tell. Make us feel the war.\n"
        f"IMPORTANT: Do NOT repeat phrases from previous dispatches. No 'woman claws at rubble' or 'white cliffs tremble' again. "
        f"Find NEW images, NEW characters, NEW locations each time. Vary your openings."
    )
