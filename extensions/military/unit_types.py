"""
unit_types.py — Expanded military unit types and combat system.

This module provides:
  1. Comprehensive unit type definitions (inspired by Hearts of Iron 4)
  2. Damage types and combat matrix
  3. Support unit types and attachments
  4. Advanced combat calculations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Tuple, List, Optional, Any
import numpy as np

# ─────────────────────────────────────────────────────────────────────────── #
# Damage Types                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class DamageType(Enum):
    """Types of damage that units can deal and receive."""
    UNARMORED = auto()      # Effective against infantry, unarmored targets
    LIGHT_PENETRATION = auto()  # Effective against light armor, armored cars
    HEAVY_PENETRATION = auto()  # Effective against heavy armor, tanks
    AA_DAMAGE = auto()      # Anti-air damage
    ARTILLERY_DAMAGE = auto()  # Explosive/area damage
    NAVAL_DAMAGE = auto()    # Anti-ship damage (for future expansion)
    # Expanded damage types used by physics engine
    SMALL_ARMS = auto()       # Rifles, MGs — anti-personnel
    AP_KINETIC = auto()       # Solid-shot AP rounds
    AP_COMPOSITE = auto()     # APCR / APDS rounds
    HEAT = auto()             # Shaped-charge warheads
    HE = auto()               # High-explosive shells
    HE_FRAG = auto()          # HE-Fragmentation (anti-personnel optimised)
    INCENDIARY = auto()       # Fire / thermite / napalm
    ROCKET_HE = auto()        # Rocket-launched HE (Katyusha, Nebelwerfer)
    NAVAL_SHELL = auto()      # Large-calibre naval guns
    BOMB_GP = auto()          # General-purpose aerial bombs
    BOMB_AP = auto()          # Armour-piercing aerial bombs
    MINE_AT = auto()          # Anti-tank mines
    MINE_AP = auto()          # Anti-personnel mines
    DEMOLITION = auto()       # Satchel charges / engineer demo
    AA_AUTOCANNON = auto()    # Rapid-fire AA cannon (20–40 mm)


# ─────────────────────────────────────────────────────────────────────────── #
# Armor Classes                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class ArmorClass(Enum):
    """Protection classes for the damage-effectiveness matrix."""
    UNARMORED = auto()      # Infantry, trucks, towed guns
    LIGHT_ARMOR = auto()    # Armored cars, halftracks, recon vehicles
    MEDIUM_ARMOR = auto()   # Medium tanks, SPGs
    HEAVY_ARMOR = auto()    # Heavy tanks, assault guns
    FORTIFIED = auto()      # Bunkers, fortifications
    NAVAL_ARMOR = auto()    # Ships (future)


# ─────────────────────────────────────────────────────────────────────────── #
# Expanded Unit Types                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

class MilitaryUnitType(Enum):
    """Comprehensive military unit types."""
    # Infantry variants
    MILITIA = auto()            # Lightly armed irregulars
    INFANTRY = auto()           # Standard infantry
    MOUNTAIN_INFANTRY = auto()  # Specialized for mountain combat
    MARINES = auto()            # Amphibious assault specialists
    PARATROOPERS = auto()       # Airborne infantry

    # Motorized/Mechanized
    MOTORIZED_INFANTRY = auto() # Truck-mounted infantry
    MECHANIZED_INFANTRY = auto() # APC-mounted infantry
    MOTORIZED_ARTILLERY = auto() # Mobile artillery

    # Armor variants
    ARMOR = auto()              # Generic armor (backward compatibility)
    LIGHT_TANK = auto()         # Fast, lightly armored
    MEDIUM_TANK = auto()        # Balanced tank
    HEAVY_TANK = auto()         # Slow, heavily armored
    SUPER_HEAVY_TANK = auto()   # Experimental heavy tanks
    ARMORED_CAR = auto()        # Fast reconnaissance
    TANK_DESTROYER = auto()     # Anti-tank specialist

    # Artillery variants
    ARTILLERY = auto()          # Generic artillery (backward compatibility)
    ROCKET_ARTILLERY = auto()   # Multiple rocket launchers
    ANTI_AIR = auto()           # Anti-aircraft artillery
    ANTI_TANK = auto()          # Tow/tank destroyer artillery

    # Air units
    AIR = auto()                # Generic air (backward compatibility)
    FIGHTER = auto()            # Air superiority fighter
    CAS = auto()                # Close air support
    STRATEGIC_BOMBER = auto()   # Long-range bomber
    NAVAL_BOMBER = auto()       # Anti-ship aircraft
    TRANSPORT = auto()          # Troop transport

    # Naval units (for future expansion)
    DESTROYER = auto()
    CRUISER = auto()
    BATTLESHIP = auto()
    SUBMARINE = auto()
    CARRIER = auto()

    # Special forces
    SPECIAL_FORCES = auto()     # Generic special forces
    COMMANDOS = auto()          # Raiding specialists
    RANGERS = auto()            # Light infantry specialists

    # Support/logistics
    LOGISTICS = auto()          # Generic logistics
    ENGINEER = auto()           # Combat engineers
    MAINTENANCE = auto()        # Repair units
    MEDICAL = auto()            # Field hospitals
    SIGNALS = auto()            # Communications

    # Doctrine-specific
    STORM_TROOPERS = auto()     # Elite assault infantry
    SELF_PROPELLED_ARTILLERY = auto()  # Mobile artillery

    # ── Politically-relevant unit types ─────────────────────────────────── #
    # These units exert direct influence on cluster politics/society.
    BORDER_GUARD = auto()          # Secures borders; slows migration loss, small trust +
    GENDARMERIE = auto()           # Paramilitary police; directly suppresses hazard
    CIVIL_AFFAIRS = auto()         # Rebuilds institutions; trust ↑, polarization ↓
    INTELLIGENCE_UNIT = auto()     # Reduces media_bias noise; improves observation accuracy
    POLITICAL_COMMISSAR = auto()   # Enforces ideology; morale ↑ but polarization ↑
    PROPAGANDA_CORPS = auto()      # Amplifies Stance.PROPAGANDA; shifts media_bias
    PEACEKEEPING_FORCE = auto()    # Neutral hazard suppressor; trust ↑, own morale ↓
    SECRET_POLICE = auto()         # Represses dissent; hazard ↓ (short-term), trust ↓, polar ↑
    RECONSTRUCTION_TEAM = auto()   # Rebuilds resource base; resource ↑, stability ↑ (slow)

# ─────────────────────────────────────────────────────────────────────────── #
# Unit Role Categories                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class UnitRole(Enum):
    """High-level categorization of unit roles."""
    INFANTRY = auto()
    ARMOR = auto()
    ARTILLERY = auto()
    AIR = auto()
    NAVAL = auto()
    SPECIAL_FORCES = auto()
    SUPPORT = auto()
    LOGISTICS = auto()
    POLITICAL_CONTROL = auto()   # Units whose primary effect is on cluster politics

def get_unit_role(unit_type: MilitaryUnitType) -> UnitRole:
    """Map unit type to its primary role."""
    role_mapping = {
        # Infantry
        MilitaryUnitType.MILITIA: UnitRole.INFANTRY,
        MilitaryUnitType.INFANTRY: UnitRole.INFANTRY,
        MilitaryUnitType.MOUNTAIN_INFANTRY: UnitRole.INFANTRY,
        MilitaryUnitType.MARINES: UnitRole.INFANTRY,
        MilitaryUnitType.PARATROOPERS: UnitRole.INFANTRY,
        MilitaryUnitType.MOTORIZED_INFANTRY: UnitRole.INFANTRY,
        MilitaryUnitType.MECHANIZED_INFANTRY: UnitRole.INFANTRY,
        MilitaryUnitType.STORM_TROOPERS: UnitRole.INFANTRY,
        MilitaryUnitType.SPECIAL_FORCES: UnitRole.SPECIAL_FORCES,
        MilitaryUnitType.COMMANDOS: UnitRole.SPECIAL_FORCES,
        MilitaryUnitType.RANGERS: UnitRole.SPECIAL_FORCES,

        # Armor
        MilitaryUnitType.ARMOR: UnitRole.ARMOR,
        MilitaryUnitType.LIGHT_TANK: UnitRole.ARMOR,
        MilitaryUnitType.MEDIUM_TANK: UnitRole.ARMOR,
        MilitaryUnitType.HEAVY_TANK: UnitRole.ARMOR,
        MilitaryUnitType.SUPER_HEAVY_TANK: UnitRole.ARMOR,
        MilitaryUnitType.ARMORED_CAR: UnitRole.ARMOR,
        MilitaryUnitType.TANK_DESTROYER: UnitRole.ARMOR,

        # Artillery
        MilitaryUnitType.ARTILLERY: UnitRole.ARTILLERY,
        MilitaryUnitType.ROCKET_ARTILLERY: UnitRole.ARTILLERY,
        MilitaryUnitType.MOTORIZED_ARTILLERY: UnitRole.ARTILLERY,
        MilitaryUnitType.ANTI_AIR: UnitRole.ARTILLERY,
        MilitaryUnitType.ANTI_TANK: UnitRole.ARTILLERY,
        MilitaryUnitType.SELF_PROPELLED_ARTILLERY: UnitRole.ARTILLERY,

        # Air
        MilitaryUnitType.AIR: UnitRole.AIR,
        MilitaryUnitType.FIGHTER: UnitRole.AIR,
        MilitaryUnitType.CAS: UnitRole.AIR,
        MilitaryUnitType.STRATEGIC_BOMBER: UnitRole.AIR,
        MilitaryUnitType.NAVAL_BOMBER: UnitRole.AIR,
        MilitaryUnitType.TRANSPORT: UnitRole.AIR,

        # Naval
        MilitaryUnitType.DESTROYER: UnitRole.NAVAL,
        MilitaryUnitType.CRUISER: UnitRole.NAVAL,
        MilitaryUnitType.BATTLESHIP: UnitRole.NAVAL,
        MilitaryUnitType.SUBMARINE: UnitRole.NAVAL,
        MilitaryUnitType.CARRIER: UnitRole.NAVAL,

        # Support/Logistics
        MilitaryUnitType.LOGISTICS: UnitRole.LOGISTICS,
        MilitaryUnitType.ENGINEER: UnitRole.SUPPORT,
        MilitaryUnitType.MAINTENANCE: UnitRole.SUPPORT,
        MilitaryUnitType.MEDICAL: UnitRole.SUPPORT,
        MilitaryUnitType.SIGNALS: UnitRole.SUPPORT,

        # Political control units
        MilitaryUnitType.BORDER_GUARD:        UnitRole.POLITICAL_CONTROL,
        MilitaryUnitType.GENDARMERIE:         UnitRole.POLITICAL_CONTROL,
        MilitaryUnitType.CIVIL_AFFAIRS:       UnitRole.POLITICAL_CONTROL,
        MilitaryUnitType.INTELLIGENCE_UNIT:   UnitRole.POLITICAL_CONTROL,
        MilitaryUnitType.POLITICAL_COMMISSAR: UnitRole.POLITICAL_CONTROL,
        MilitaryUnitType.PROPAGANDA_CORPS:    UnitRole.POLITICAL_CONTROL,
        MilitaryUnitType.PEACEKEEPING_FORCE:  UnitRole.POLITICAL_CONTROL,
        MilitaryUnitType.SECRET_POLICE:       UnitRole.POLITICAL_CONTROL,
        MilitaryUnitType.RECONSTRUCTION_TEAM: UnitRole.POLITICAL_CONTROL,
    }

    return role_mapping.get(unit_type, UnitRole.INFANTRY)


# ─────────────────────────────────────────────────────────────────────────── #
# Political Unit Effect Registry                                               #
# ─────────────────────────────────────────────────────────────────────────── #

# Per-step political deltas applied by ONE unit at full strength.
# Format: {unit_type: {field: delta_per_unit_per_step}}
# Fields: sigma, hazard, resource, military, trust, polar, media_bias, population
# Positive delta = increase; negative = decrease
POLITICAL_EFFECTS: Dict[MilitaryUnitType, Dict[str, float]] = {
    MilitaryUnitType.BORDER_GUARD: {
        "trust":      +0.004,
        "polar":      -0.002,
        "population": +0.003,   # slows emigration (net)
    },
    MilitaryUnitType.GENDARMERIE: {
        "hazard":     -0.04,
        "trust":      -0.005,   # heavy-handed policing erodes trust
        "polar":      +0.008,
    },
    MilitaryUnitType.CIVIL_AFFAIRS: {
        "sigma":      +0.006,
        "trust":      +0.010,
        "polar":      -0.008,
        "resource":   +0.002,
    },
    MilitaryUnitType.INTELLIGENCE_UNIT: {
        "media_bias": -0.015,   # reduces absolute bias magnitude
        "trust":      -0.002,   # surveillance erodes civil trust
    },
    MilitaryUnitType.POLITICAL_COMMISSAR: {
        "trust":      +0.008,   # perceived legitimacy boost
        "polar":      +0.012,   # enforced ideology deepens divisions
        "sigma":      +0.003,
    },
    MilitaryUnitType.PROPAGANDA_CORPS: {
        "media_bias": +0.020,   # amplifies bias (direction: toward state narrative)
        "trust":      +0.005,
        "polar":      +0.006,
    },
    MilitaryUnitType.PEACEKEEPING_FORCE: {
        "hazard":     -0.035,
        "trust":      +0.012,
        "polar":      -0.010,
        "sigma":      +0.004,
    },
    MilitaryUnitType.SECRET_POLICE: {
        "hazard":     -0.050,   # brutal short-term suppression
        "trust":      -0.018,
        "polar":      +0.020,
        "sigma":      -0.005,   # structural instability from repression
    },
    MilitaryUnitType.RECONSTRUCTION_TEAM: {
        "resource":   +0.008,
        "sigma":      +0.005,
        "trust":      +0.006,
        "population": +0.004,
    },
}

# ─────────────────────────────────────────────────────────────────────────── #
# Combat Matrix                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class CombatMatrix:
    """
    Combat effectiveness matrix defining how unit types perform against each other.

    Values represent damage multipliers (1.0 = normal damage).
    """
    # Damage multipliers: attacker -> defender -> multiplier
    matrix: Dict[MilitaryUnitType, Dict[MilitaryUnitType, float]]

    @classmethod
    def default_matrix(cls) -> 'CombatMatrix':
        """Create default combat matrix with realistic values."""
        matrix = {}

        # Default multiplier (1.0 for most matchups)
        default_multiplier = 1.0

        # Initialize all unit types
        all_unit_types = list(MilitaryUnitType)

        for attacker in all_unit_types:
            matrix[attacker] = {}
            for defender in all_unit_types:
                matrix[attacker][defender] = default_multiplier

        # Apply specific combat relationships

        # Infantry vs Infantry (base case)
        matrix[MilitaryUnitType.INFANTRY][MilitaryUnitType.INFANTRY] = 1.0
        matrix[MilitaryUnitType.MILITIA][MilitaryUnitType.INFANTRY] = 0.7  # Militia less effective

        # Armor advantages
        for armor_type in [MilitaryUnitType.ARMOR, MilitaryUnitType.LIGHT_TANK,
                          MilitaryUnitType.MEDIUM_TANK, MilitaryUnitType.HEAVY_TANK,
                          MilitaryUnitType.SUPER_HEAVY_TANK]:
            # Armor vs Infantry (good)
            matrix[armor_type][MilitaryUnitType.INFANTRY] = 1.8
            matrix[armor_type][MilitaryUnitType.MILITIA] = 2.0

            # Armor vs other armor (varies by weight)
            matrix[armor_type][MilitaryUnitType.ARMOR] = 1.2
            matrix[armor_type][MilitaryUnitType.LIGHT_TANK] = 1.5
            matrix[armor_type][MilitaryUnitType.MEDIUM_TANK] = 1.2
            matrix[armor_type][MilitaryUnitType.HEAVY_TANK] = 0.8
            matrix[armor_type][MilitaryUnitType.SUPER_HEAVY_TANK] = 0.5

        # Heavy tank specific advantages
        matrix[MilitaryUnitType.HEAVY_TANK][MilitaryUnitType.MEDIUM_TANK] = 1.6
        matrix[MilitaryUnitType.SUPER_HEAVY_TANK][MilitaryUnitType.HEAVY_TANK] = 1.8

        # Anti-tank advantages
        for at_type in [MilitaryUnitType.ANTI_TANK, MilitaryUnitType.TANK_DESTROYER]:
            matrix[at_type][MilitaryUnitType.ARMOR] = 2.5
            matrix[at_type][MilitaryUnitType.LIGHT_TANK] = 3.0
            matrix[at_type][MilitaryUnitType.MEDIUM_TANK] = 2.2
            matrix[at_type][MilitaryUnitType.HEAVY_TANK] = 1.8
            matrix[at_type][MilitaryUnitType.SUPER_HEAVY_TANK] = 1.5

        # Artillery advantages
        for artillery_type in [MilitaryUnitType.ARTILLERY, MilitaryUnitType.ROCKET_ARTILLERY,
                              MilitaryUnitType.MOTORIZED_ARTILLERY, MilitaryUnitType.SELF_PROPELLED_ARTILLERY]:
            matrix[artillery_type][MilitaryUnitType.INFANTRY] = 1.6
            matrix[artillery_type][MilitaryUnitType.MILITIA] = 1.8
            matrix[artillery_type][MilitaryUnitType.ARTILLERY] = 1.2  # Counter-battery

        # Air advantages
        matrix[MilitaryUnitType.FIGHTER][MilitaryUnitType.CAS] = 1.8
        matrix[MilitaryUnitType.FIGHTER][MilitaryUnitType.STRATEGIC_BOMBER] = 1.5
        matrix[MilitaryUnitType.CAS][MilitaryUnitType.INFANTRY] = 2.0
        matrix[MilitaryUnitType.CAS][MilitaryUnitType.ARMOR] = 1.2
        matrix[MilitaryUnitType.STRATEGIC_BOMBER][MilitaryUnitType.INFANTRY] = 1.5
        matrix[MilitaryUnitType.STRATEGIC_BOMBER][MilitaryUnitType.ARTILLERY] = 1.8

        # Anti-air advantages
        matrix[MilitaryUnitType.ANTI_AIR][MilitaryUnitType.FIGHTER] = 1.8
        matrix[MilitaryUnitType.ANTI_AIR][MilitaryUnitType.CAS] = 2.0
        matrix[MilitaryUnitType.ANTI_AIR][MilitaryUnitType.STRATEGIC_BOMBER] = 1.5

        # Special forces advantages
        matrix[MilitaryUnitType.SPECIAL_FORCES][MilitaryUnitType.ARTILLERY] = 1.8
        matrix[MilitaryUnitType.COMMANDOS][MilitaryUnitType.ARTILLERY] = 2.0
        matrix[MilitaryUnitType.RANGERS][MilitaryUnitType.INFANTRY] = 1.3

        return cls(matrix)

    def get_damage_multiplier(
        self,
        attacker_type: MilitaryUnitType,
        defender_type: MilitaryUnitType
    ) -> float:
        """Get damage multiplier for attacker vs defender."""
        return self.matrix.get(attacker_type, {}).get(defender_type, 1.0)

# ─────────────────────────────────────────────────────────────────────────── #
# Unit Parameters with Expanded Types                                           #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class ExpandedUnitParams:
    """
    Extended unit parameters with comprehensive unit type support.
    """
    # Base unit stats (can be overridden per unit type)
    base_combat: float = 1.0
    base_hp: float = 100.0
    base_speed: float = 1.0
    base_supply_cost: float = 1.0

    # Unit-specific parameters
    unit_params: Dict[MilitaryUnitType, Dict[str, float]] = None

    def __post_init__(self):
        if self.unit_params is None:
            object.__setattr__(self, 'unit_params', self._create_default_params())

    def _create_default_params(self) -> Dict[MilitaryUnitType, Dict[str, float]]:
        """Create default parameters for all unit types."""
        params = {}

        # Infantry variants
        params[MilitaryUnitType.MILITIA] = {
            'combat': 0.6, 'hp': 80, 'speed': 0.9, 'supply_cost': 0.7,
            'soft_attack': 0.8, 'hard_attack': 0.1, 'air_attack': 0.0,
            'defense': 0.7, 'breakthrough': 0.2, 'air_defense': 0.0
        }

        params[MilitaryUnitType.INFANTRY] = {
            'combat': 1.0, 'hp': 100, 'speed': 1.0, 'supply_cost': 0.8,
            'soft_attack': 1.0, 'hard_attack': 0.2, 'air_attack': 0.0,
            'defense': 1.0, 'breakthrough': 0.3, 'air_defense': 0.0
        }

        params[MilitaryUnitType.MOUNTAIN_INFANTRY] = {
            'combat': 0.9, 'hp': 90, 'speed': 0.8, 'supply_cost': 0.9,
            'soft_attack': 1.2, 'hard_attack': 0.1, 'air_attack': 0.0,
            'defense': 1.3, 'breakthrough': 0.2, 'air_defense': 0.0
        }

        # Armor variants
        params[MilitaryUnitType.LIGHT_TANK] = {
            'combat': 1.2, 'hp': 80, 'speed': 1.8, 'supply_cost': 1.1,
            'soft_attack': 1.5, 'hard_attack': 0.8, 'air_attack': 0.0,
            'defense': 0.8, 'breakthrough': 1.5, 'air_defense': 0.0,
            'armor': 0.5, 'piercing': 0.8
        }

        params[MilitaryUnitType.MEDIUM_TANK] = {
            'combat': 1.5, 'hp': 120, 'speed': 1.2, 'supply_cost': 1.3,
            'soft_attack': 1.8, 'hard_attack': 1.2, 'air_attack': 0.0,
            'defense': 1.2, 'breakthrough': 1.8, 'air_defense': 0.0,
            'armor': 0.8, 'piercing': 1.2
        }

        params[MilitaryUnitType.HEAVY_TANK] = {
            'combat': 2.0, 'hp': 180, 'speed': 0.8, 'supply_cost': 1.6,
            'soft_attack': 2.0, 'hard_attack': 1.8, 'air_attack': 0.0,
            'defense': 1.8, 'breakthrough': 2.2, 'air_defense': 0.0,
            'armor': 1.5, 'piercing': 1.5
        }

        # Artillery variants
        params[MilitaryUnitType.ARTILLERY] = {
            'combat': 0.8, 'hp': 60, 'speed': 0.5, 'supply_cost': 1.0,
            'soft_attack': 1.8, 'hard_attack': 0.5, 'air_attack': 0.0,
            'defense': 0.5, 'breakthrough': 0.1, 'air_defense': 0.0
        }

        params[MilitaryUnitType.ANTI_AIR] = {
            'combat': 0.4, 'hp': 50, 'speed': 0.6, 'supply_cost': 0.8,
            'soft_attack': 0.3, 'hard_attack': 0.2, 'air_attack': 2.5,
            'defense': 0.3, 'breakthrough': 0.1, 'air_defense': 0.0
        }

        # Air units
        params[MilitaryUnitType.FIGHTER] = {
            'combat': 1.5, 'hp': 80, 'speed': 3.0, 'supply_cost': 1.4,
            'soft_attack': 0.2, 'hard_attack': 0.2, 'air_attack': 2.0,
            'defense': 0.3, 'breakthrough': 0.1, 'air_defense': 1.8
        }

        params[MilitaryUnitType.CAS] = {
            'combat': 1.8, 'hp': 90, 'speed': 2.5, 'supply_cost': 1.5,
            'soft_attack': 2.5, 'hard_attack': 1.2, 'air_attack': 0.5,
            'defense': 0.2, 'breakthrough': 0.1, 'air_defense': 0.3
        }

        # ── Politically-relevant unit type params ────────────────────────── #
        # Low combat power; their value is in political POLITICAL_EFFECTS.
        params[MilitaryUnitType.BORDER_GUARD] = {
            'combat': 0.4, 'hp': 40, 'speed': 0.7, 'supply_cost': 0.4,
            'soft_attack': 0.3, 'hard_attack': 0.1, 'defense': 0.5,
        }
        params[MilitaryUnitType.GENDARMERIE] = {
            'combat': 0.6, 'hp': 50, 'speed': 0.8, 'supply_cost': 0.5,
            'soft_attack': 0.5, 'hard_attack': 0.1, 'defense': 0.6,
        }
        params[MilitaryUnitType.CIVIL_AFFAIRS] = {
            'combat': 0.1, 'hp': 30, 'speed': 0.6, 'supply_cost': 0.3,
            'soft_attack': 0.0, 'hard_attack': 0.0, 'defense': 0.2,
        }
        params[MilitaryUnitType.INTELLIGENCE_UNIT] = {
            'combat': 0.2, 'hp': 25, 'speed': 1.0, 'supply_cost': 0.3,
            'soft_attack': 0.1, 'hard_attack': 0.0, 'defense': 0.2,
        }
        params[MilitaryUnitType.POLITICAL_COMMISSAR] = {
            'combat': 0.2, 'hp': 30, 'speed': 0.8, 'supply_cost': 0.3,
            'soft_attack': 0.1, 'hard_attack': 0.0, 'defense': 0.3,
        }
        params[MilitaryUnitType.PROPAGANDA_CORPS] = {
            'combat': 0.1, 'hp': 25, 'speed': 0.8, 'supply_cost': 0.3,
            'soft_attack': 0.0, 'hard_attack': 0.0, 'defense': 0.1,
        }
        params[MilitaryUnitType.PEACEKEEPING_FORCE] = {
            'combat': 0.5, 'hp': 55, 'speed': 0.9, 'supply_cost': 0.5,
            'soft_attack': 0.2, 'hard_attack': 0.1, 'defense': 0.7,
        }
        params[MilitaryUnitType.SECRET_POLICE] = {
            'combat': 0.5, 'hp': 40, 'speed': 1.0, 'supply_cost': 0.4,
            'soft_attack': 0.4, 'hard_attack': 0.1, 'defense': 0.4,
        }
        params[MilitaryUnitType.RECONSTRUCTION_TEAM] = {
            'combat': 0.0, 'hp': 20, 'speed': 0.5, 'supply_cost': 0.6,
            'soft_attack': 0.0, 'hard_attack': 0.0, 'defense': 0.1,
        }

        return params

    def get_param(self, unit_type: MilitaryUnitType, param_name: str, default: float = 1.0) -> float:
        """Get parameter value for unit type."""
        unit_params = self.unit_params.get(unit_type, {})
        return unit_params.get(param_name, default)

    def get_combat_power(self, unit_type: MilitaryUnitType) -> float:
        """Get combat power for unit type."""
        return self.get_param(unit_type, 'combat', self.base_combat)

    def get_max_hp(self, unit_type: MilitaryUnitType) -> float:
        """Get max HP for unit type."""
        return self.get_param(unit_type, 'hp', self.base_hp)

    def get_speed(self, unit_type: MilitaryUnitType) -> float:
        """Get movement speed for unit type."""
        return self.get_param(unit_type, 'speed', self.base_speed)

    def get_supply_cost(self, unit_type: MilitaryUnitType) -> float:
        """Get supply cost for unit type."""
        return self.get_param(unit_type, 'supply_cost', self.base_supply_cost)

# ─────────────────────────────────────────────────────────────────────────── #
# Support Company System                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class SupportCompanyType(Enum):
    """Types of support companies that can be attached to units."""
    ARTILLERY_COMPANY = auto()      # Artillery support
    ANTI_AIR_COMPANY = auto()       # Anti-aircraft support
    ANTI_TANK_COMPANY = auto()      # Anti-tank support
    ENGINEER_COMPANY = auto()       # Combat engineer support
    RECON_COMPANY = auto()          # Reconnaissance support
    MAINTENANCE_COMPANY = auto()    # Repair/maintenance support
    SIGNALS_COMPANY = auto()        # Communications support
    MEDICAL_COMPANY = auto()        # Medical support

@dataclass(frozen=True)
class SupportCompany:
    """A support company that can be attached to a main unit."""
    company_type: SupportCompanyType
    size: int = 1  # Number of support units
    effectiveness: float = 1.0

    def get_bonuses(self) -> Dict[str, float]:
        """Get combat bonuses provided by this support company."""
        bonuses = {}

        if self.company_type == SupportCompanyType.ARTILLERY_COMPANY:
            bonuses['soft_attack'] = 0.2 * self.size * self.effectiveness
            bonuses['defense'] = 0.1 * self.size * self.effectiveness

        elif self.company_type == SupportCompanyType.ANTI_AIR_COMPANY:
            bonuses['air_defense'] = 0.3 * self.size * self.effectiveness

        elif self.company_type == SupportCompanyType.ANTI_TANK_COMPANY:
            bonuses['hard_attack'] = 0.3 * self.size * self.effectiveness
            bonuses['piercing'] = 0.2 * self.size * self.effectiveness

        elif self.company_type == SupportCompanyType.ENGINEER_COMPANY:
            bonuses['defense'] = 0.15 * self.size * self.effectiveness
            bonuses['breakthrough'] = 0.1 * self.size * self.effectiveness

        elif self.company_type == SupportCompanyType.RECON_COMPANY:
            bonuses['soft_attack'] = 0.1 * self.size * self.effectiveness
            bonuses['speed'] = 0.05 * self.size * self.effectiveness

        elif self.company_type == SupportCompanyType.MAINTENANCE_COMPANY:
            bonuses['hp_regen'] = 0.02 * self.size * self.effectiveness

        elif self.company_type == SupportCompanyType.SIGNALS_COMPANY:
            bonuses['combat_effectiveness'] = 0.03 * self.size * self.effectiveness

        elif self.company_type == SupportCompanyType.MEDICAL_COMPANY:
            bonuses['hp_regen'] = 0.03 * self.size * self.effectiveness
            bonuses['morale_regen'] = 0.02 * self.size * self.effectiveness

        return bonuses

# ─────────────────────────────────────────────────────────────────────────── #
# Enhanced Military Unit                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class EnhancedMilitaryUnit:
    """
    Enhanced military unit with support companies and detailed combat stats.
    """
    unit_id: int
    unit_type: MilitaryUnitType
    cluster_id: int
    hit_points: float
    combat_effectiveness: float
    supply_level: float
    experience: float
    morale: float
    objective_id: Optional[int] = None
    support_companies: Tuple[SupportCompany, ...] = ()
    terrain_bonus: float = 1.0
    entrenchment: float = 0.0

    @property
    def is_alive(self) -> bool:
        """Check if unit has any hit points remaining."""
        return self.hit_points > 0.01

    @property
    def combat_power(self) -> float:
        """Current combat power considering all factors."""
        base_power = self.hit_points * self.combat_effectiveness

        # Apply support company bonuses
        support_bonuses = self._calculate_support_bonuses()
        combat_multiplier = 1.0 + support_bonuses.get('combat_effectiveness', 0.0)

        return base_power * combat_multiplier * self.terrain_bonus * (1.0 + self.entrenchment * 0.5)

    def _calculate_support_bonuses(self) -> Dict[str, float]:
        """Calculate cumulative bonuses from support companies."""
        bonuses = {}

        for company in self.support_companies:
            company_bonuses = company.get_bonuses()
            for bonus_type, value in company_bonuses.items():
                bonuses[bonus_type] = bonuses.get(bonus_type, 0.0) + value

        return bonuses

    def get_attack_values(self, params: ExpandedUnitParams) -> Dict[str, float]:
        """Get attack values for different damage types."""
        base_stats = params.unit_params.get(self.unit_type, {})
        support_bonuses = self._calculate_support_bonuses()

        attack_values = {
            'soft_attack': base_stats.get('soft_attack', 1.0) + support_bonuses.get('soft_attack', 0.0),
            'hard_attack': base_stats.get('hard_attack', 0.5) + support_bonuses.get('hard_attack', 0.0),
            'air_attack': base_stats.get('air_attack', 0.0) + support_bonuses.get('air_attack', 0.0),
        }

        return attack_values

    def get_defense_values(self, params: ExpandedUnitParams) -> Dict[str, float]:
        """Get defense values for different damage types."""
        base_stats = params.unit_params.get(self.unit_type, {})
        support_bonuses = self._calculate_support_bonuses()

        defense_values = {
            'defense': base_stats.get('defense', 1.0) + support_bonuses.get('defense', 0.0),
            'breakthrough': base_stats.get('breakthrough', 0.3) + support_bonuses.get('breakthrough', 0.0),
            'air_defense': base_stats.get('air_defense', 0.0) + support_bonuses.get('air_defense', 0.0),
            'armor': base_stats.get('armor', 0.0) + support_bonuses.get('armor', 0.0),
        }

        return defense_values

    def copy_with(
        self,
        cluster_id: Optional[int] = None,
        hit_points: Optional[float] = None,
        combat_effectiveness: Optional[float] = None,
        supply_level: Optional[float] = None,
        experience: Optional[float] = None,
        morale: Optional[float] = None,
        objective_id: Optional[int] = None,
        support_companies: Optional[Tuple[SupportCompany, ...]] = None,
        terrain_bonus: Optional[float] = None,
        entrenchment: Optional[float] = None,
    ) -> 'EnhancedMilitaryUnit':
        """Create a modified copy of this unit."""
        return EnhancedMilitaryUnit(
            unit_id=self.unit_id,
            unit_type=self.unit_type,
            cluster_id=cluster_id if cluster_id is not None else self.cluster_id,
            hit_points=hit_points if hit_points is not None else self.hit_points,
            combat_effectiveness=combat_effectiveness if combat_effectiveness is not None else self.combat_effectiveness,
            supply_level=supply_level if supply_level is not None else self.supply_level,
            experience=experience if experience is not None else self.experience,
            morale=morale if morale is not None else self.morale,
            objective_id=objective_id if objective_id is not None else self.objective_id,
            support_companies=support_companies if support_companies is not None else self.support_companies,
            terrain_bonus=terrain_bonus if terrain_bonus is not None else self.terrain_bonus,
            entrenchment=entrenchment if entrenchment is not None else self.entrenchment,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert unit to dictionary for serialization."""
        return {
            'unit_id': self.unit_id,
            'unit_type': self.unit_type.name,
            'cluster_id': self.cluster_id,
            'hit_points': self.hit_points,
            'combat_effectiveness': self.combat_effectiveness,
            'supply_level': self.supply_level,
            'experience': self.experience,
            'morale': self.morale,
            'objective_id': self.objective_id,
            'support_companies': [c.company_type.name for c in self.support_companies],
            'terrain_bonus': self.terrain_bonus,
            'entrenchment': self.entrenchment,
        }

# ─────────────────────────────────────────────────────────────────────────── #
# Combat Calculation Functions                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def calculate_damage(
    attacker: EnhancedMilitaryUnit,
    defender: EnhancedMilitaryUnit,
    combat_matrix: CombatMatrix,
    params: ExpandedUnitParams,
    terrain_factor: float = 1.0,
    is_surprise_attack: bool = False
) -> Tuple[float, float]:
    """
    Calculate damage between two units using enhanced combat system.

    Args:
        attacker: Attacking unit
        defender: Defending unit
        combat_matrix: Combat effectiveness matrix
        params: Unit parameters
        terrain_factor: Terrain advantage for defender
        is_surprise_attack: Whether attack is surprise (affects damage)

    Returns:
        Tuple of (attacker_damage, defender_damage)
    """
    # Get attack and defense values
    attacker_stats = attacker.get_attack_values(params)
    defender_stats = defender.get_defense_values(params)

    # Get combat matrix multiplier
    matrix_multiplier = combat_matrix.get_damage_multiplier(attacker.unit_type, defender.unit_type)

    # Calculate base damage
    attacker_power = attacker.combat_power
    defender_power = defender.combat_power * terrain_factor

    # Determine damage types
    if attacker.unit_type in [MilitaryUnitType.FIGHTER, MilitaryUnitType.ANTI_AIR]:
        # Air combat
        attacker_damage_type = DamageType.AA_DAMAGE
        defender_damage_type = DamageType.AA_DAMAGE
    elif attacker.unit_type in [MilitaryUnitType.ARTILLERY, MilitaryUnitType.ROCKET_ARTILLERY,
                              MilitaryUnitType.MOTORIZED_ARTILLERY, MilitaryUnitType.SELF_PROPELLED_ARTILLERY]:
        # Artillery combat
        attacker_damage_type = DamageType.ARTILLERY_DAMAGE
        defender_damage_type = DamageType.ARTILLERY_DAMAGE
    elif get_unit_role(attacker.unit_type) == UnitRole.ARMOR:
        # Armor combat
        attacker_damage_type = DamageType.HEAVY_PENETRATION
        defender_damage_type = DamageType.HEAVY_PENETRATION
    else:
        # Infantry/soft combat
        attacker_damage_type = DamageType.UNARMORED
        defender_damage_type = DamageType.UNARMORED

    # Calculate damage with all factors
    base_attacker_damage = (defender_power / (attacker_power + defender_power)) * attacker_power * 0.2
    base_defender_damage = (attacker_power / (attacker_power + defender_power)) * defender_power * 0.2

    # Apply combat matrix
    base_attacker_damage *= matrix_multiplier
    base_defender_damage *= combat_matrix.get_damage_multiplier(defender.unit_type, attacker.unit_type)

    # Apply surprise attack bonus
    if is_surprise_attack:
        base_attacker_damage *= 1.5
        base_defender_damage *= 0.7

    # Apply random variation
    attacker_damage = base_attacker_damage * (0.9 + 0.2 * np.random.random())
    defender_damage = base_defender_damage * (0.9 + 0.2 * np.random.random())

    return attacker_damage, defender_damage

def resolve_enhanced_combat(
    attacker: EnhancedMilitaryUnit,
    defender: EnhancedMilitaryUnit,
    combat_matrix: CombatMatrix,
    params: ExpandedUnitParams,
    terrain_advantage: float = 1.0,
) -> Tuple[EnhancedMilitaryUnit, EnhancedMilitaryUnit]:
    """
    Resolve combat between two enhanced units.

    Args:
        attacker: Attacking unit
        defender: Defending unit
        combat_matrix: Combat effectiveness matrix
        params: Unit parameters
        terrain_advantage: Terrain advantage for defender

    Returns:
        Tuple of (updated_attacker, updated_defender) after combat
    """
    # Calculate damage
    attacker_damage, defender_damage = calculate_damage(
        attacker, defender, combat_matrix, params, terrain_advantage
    )

    # Apply damage with some randomness
    attacker_hp_loss = min(attacker.hit_points * 0.9, attacker_damage)
    defender_hp_loss = min(defender.hit_points * 0.9, defender_damage)

    # Update combat effectiveness (degrades with combat)
    effectiveness_decay = params.get_param(attacker.unit_type, 'combat_effectiveness_decay', 0.05)
    attacker_effectiveness = max(0.1, attacker.combat_effectiveness - effectiveness_decay)
    defender_effectiveness = max(0.1, defender.combat_effectiveness - effectiveness_decay)

    # Update units
    updated_attacker = attacker.copy_with(
        hit_points=attacker.hit_points - attacker_hp_loss,
        combat_effectiveness=attacker_effectiveness,
        experience=min(10.0, attacker.experience + 0.1),
        morale=max(0.1, attacker.morale - 0.05)
    )

    updated_defender = defender.copy_with(
        hit_points=defender.hit_points - defender_hp_loss,
        combat_effectiveness=defender_effectiveness,
        experience=min(10.0, defender.experience + 0.1),
        morale=max(0.1, defender.morale - 0.05)
    )

    return updated_attacker, updated_defender