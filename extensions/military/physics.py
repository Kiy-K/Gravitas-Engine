"""
physics.py — Universal Physical Simulation Engine for Military Operations.

Map-agnostic physics system that models terrain, weather, supply logistics,
combat physics, and movement for any theater of operations.  Each map defines
its terrain and climate via YAML configuration — no code changes needed.

Key systems:
  1. Terrain        — 27 terrain types, composable features, soil, elevation
  2. Weather        — Config-driven climate with 17 conditions, seasonal cycles
  3. Supply         — 6 supply categories, route networks, interdiction
  4. Combat         — Range, penetration, fortification, indirect fire, air
  5. Movement       — Unit-type × terrain mobility matrix
  6. Equipment      — Reliability curves by temperature and faction
  7. Configuration  — YAML-based map definitions (see load_map_physics)

Usage:
    config = load_map_physics("gravitas/scenarios/moscow_terrain.yaml")
    states = initialize_physics(config)
    states = step_all_physics(states, step=1, rng=rng)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────── #
# Fast scalar clamp — avoids numpy.clip overhead on single floats             #
# ─────────────────────────────────────────────────────────────────────────── #

def _sc(v: float, lo: float, hi: float) -> float:
    """Scalar clamp: max(lo, min(v, hi)). 50-100x faster than np.clip on scalars."""
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ─────────────────────────────────────────────────────────────────────────── #
# Enums                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class TerrainType(Enum):
    """Primary terrain classification — covers all major warfare theaters."""
    OPEN          = auto()  # Plains, grassland, fields
    FOREST        = auto()  # Temperate woodland
    DENSE_FOREST  = auto()  # Thick / old-growth forest
    JUNGLE        = auto()  # Tropical dense vegetation
    URBAN         = auto()  # City — mixed buildings
    URBAN_DENSE   = auto()  # Dense city center, narrow streets
    RUINS         = auto()  # Bombed-out urban area
    MOUNTAIN      = auto()  # Alpine / high altitude (>2000 m)
    HILLS         = auto()  # Rolling hills, ridgelines
    MARSH         = auto()  # Wetland, swamp, bog
    DESERT_SANDY  = auto()  # Sand dunes, soft ground
    DESERT_ROCKY  = auto()  # Rocky desert, wadi
    TUNDRA        = auto()  # Arctic / subarctic permafrost
    RIVER         = auto()  # Major river crossing
    LAKE          = auto()  # Lake obstacle
    COASTAL       = auto()  # Beach / shore with amphibious aspects
    ROAD          = auto()  # Major road / highway corridor
    BRIDGE        = auto()  # Bridge crossing (destroyable)
    FORTIFIED     = auto()  # Prepared defensive line
    AIRFIELD      = auto()  # Air base with runway
    PORT          = auto()  # Naval port / harbor
    RAIL_YARD     = auto()  # Major rail junction
    FARMLAND      = auto()  # Agricultural land — open but soft ground
    STEPPE        = auto()  # Vast flat grassland
    TAIGA         = auto()  # Boreal / coniferous forest
    BOCAGE        = auto()  # Hedgerow country (Normandy-style)
    VOLCANIC      = auto()  # Volcanic terrain (Pacific islands)


class SoilType(Enum):
    """Ground composition — affects digging, traction, and drainage."""
    CLAY          = auto()  # Holds water, mud-prone
    SAND          = auto()  # Soft, vehicles sink
    LOAM          = auto()  # Mixed, reasonable traction
    ROCK          = auto()  # Hard, very slow entrenchment
    PERMAFROST    = auto()  # Frozen solid year-round
    PEAT          = auto()  # Boggy, very soft
    GRAVEL        = auto()  # Loose stones, fair traction
    MUD           = auto()  # Saturated ground (seasonal)


class WeatherCondition(Enum):
    """Atmospheric conditions affecting all operations."""
    CLEAR         = auto()  # Clear skies, good visibility
    OVERCAST      = auto()  # Cloud cover, reduced air support
    RAIN          = auto()  # Light rain
    HEAVY_RAIN    = auto()  # Downpour, flooding risk
    THUNDERSTORM  = auto()  # Severe storms, lightning
    DRIZZLE       = auto()  # Light persistent moisture
    SNOW          = auto()  # Moderate snowfall
    HEAVY_SNOW    = auto()  # Heavy snowfall, rapid accumulation
    BLIZZARD      = auto()  # Whiteout conditions
    ICE_STORM     = auto()  # Freezing rain
    FOG           = auto()  # Thick fog, very low visibility
    MIST          = auto()  # Light haze
    MUD           = auto()  # Rasputitsa (mud season)
    DUST_STORM    = auto()  # Desert sandstorm / dust
    HEAT_WAVE     = auto()  # Extreme heat (>40 °C)
    DEEP_FROST    = auto()  # Extreme cold (<-25 °C)
    HAIL          = auto()  # Hailstorm


class Season(Enum):
    """Season of year — drives base weather patterns."""
    SPRING = auto()
    SUMMER = auto()
    AUTUMN = auto()
    WINTER = auto()


class TimeOfDay(Enum):
    """Time of day — affects visibility and operation tempo."""
    DAWN  = auto()   # 0500-0700
    DAY   = auto()   # 0700-1700
    DUSK  = auto()   # 1700-1900
    NIGHT = auto()   # 1900-0500


# ─────────────────────────────────────────────────────────────────────────── #
# Terrain Property Tables                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _full_terrain_table(default: float, overrides: Dict[TerrainType, float]) -> Dict[TerrainType, float]:
    """Build a complete table for every TerrainType with a default fallback."""
    return {t: overrides.get(t, default) for t in TerrainType}


# Movement cost multiplier (1.0 = baseline open ground)
MOVEMENT_COST: Dict[TerrainType, float] = _full_terrain_table(1.0, {
    TerrainType.OPEN:          1.0,
    TerrainType.FOREST:        1.6,
    TerrainType.DENSE_FOREST:  2.2,
    TerrainType.JUNGLE:        3.0,
    TerrainType.URBAN:         2.0,
    TerrainType.URBAN_DENSE:   2.5,
    TerrainType.RUINS:         2.8,
    TerrainType.MOUNTAIN:      3.5,
    TerrainType.HILLS:         1.8,
    TerrainType.MARSH:         2.2,
    TerrainType.DESERT_SANDY:  1.8,
    TerrainType.DESERT_ROCKY:  1.4,
    TerrainType.TUNDRA:        1.6,
    TerrainType.RIVER:         2.5,
    TerrainType.LAKE:          99.0,
    TerrainType.COASTAL:       1.5,
    TerrainType.ROAD:          0.6,
    TerrainType.BRIDGE:        0.5,
    TerrainType.FORTIFIED:     3.0,
    TerrainType.AIRFIELD:      0.8,
    TerrainType.PORT:          1.2,
    TerrainType.RAIL_YARD:     0.7,
    TerrainType.FARMLAND:      1.2,
    TerrainType.STEPPE:        0.9,
    TerrainType.TAIGA:         2.0,
    TerrainType.BOCAGE:        2.4,
    TerrainType.VOLCANIC:      2.6,
})

# Defense bonus (additive)
DEFENSE_BONUS: Dict[TerrainType, float] = _full_terrain_table(0.0, {
    TerrainType.FOREST:        0.20,
    TerrainType.DENSE_FOREST:  0.30,
    TerrainType.JUNGLE:        0.35,
    TerrainType.URBAN:         0.40,
    TerrainType.URBAN_DENSE:   0.50,
    TerrainType.RUINS:         0.45,
    TerrainType.MOUNTAIN:      0.50,
    TerrainType.HILLS:         0.30,
    TerrainType.MARSH:         0.10,
    TerrainType.DESERT_ROCKY:  0.15,
    TerrainType.TUNDRA:        0.05,
    TerrainType.RIVER:         0.15,
    TerrainType.COASTAL:       0.05,
    TerrainType.ROAD:         -0.05,
    TerrainType.BRIDGE:       -0.10,
    TerrainType.FORTIFIED:     0.55,
    TerrainType.AIRFIELD:     -0.05,
    TerrainType.PORT:          0.10,
    TerrainType.RAIL_YARD:     0.05,
    TerrainType.FARMLAND:      0.05,
    TerrainType.TAIGA:         0.25,
    TerrainType.BOCAGE:        0.35,
    TerrainType.VOLCANIC:      0.20,
})

# Concealment / cover factor [0, 1]
COVER_FACTOR: Dict[TerrainType, float] = _full_terrain_table(0.1, {
    TerrainType.OPEN:          0.05,
    TerrainType.FOREST:        0.50,
    TerrainType.DENSE_FOREST:  0.70,
    TerrainType.JUNGLE:        0.80,
    TerrainType.URBAN:         0.60,
    TerrainType.URBAN_DENSE:   0.75,
    TerrainType.RUINS:         0.65,
    TerrainType.MOUNTAIN:      0.40,
    TerrainType.HILLS:         0.30,
    TerrainType.MARSH:         0.15,
    TerrainType.DESERT_SANDY:  0.05,
    TerrainType.DESERT_ROCKY:  0.20,
    TerrainType.TUNDRA:        0.05,
    TerrainType.RIVER:         0.10,
    TerrainType.LAKE:          0.00,
    TerrainType.COASTAL:       0.10,
    TerrainType.ROAD:          0.05,
    TerrainType.BRIDGE:        0.05,
    TerrainType.FORTIFIED:     0.70,
    TerrainType.AIRFIELD:      0.10,
    TerrainType.PORT:          0.30,
    TerrainType.RAIL_YARD:     0.20,
    TerrainType.FARMLAND:      0.10,
    TerrainType.STEPPE:        0.05,
    TerrainType.TAIGA:         0.55,
    TerrainType.BOCAGE:        0.60,
    TerrainType.VOLCANIC:      0.30,
})

# Visibility range modifier (multiplier on base visibility)
VISIBILITY_MODIFIER: Dict[TerrainType, float] = _full_terrain_table(1.0, {
    TerrainType.FOREST:        0.50,
    TerrainType.DENSE_FOREST:  0.30,
    TerrainType.JUNGLE:        0.20,
    TerrainType.URBAN:         0.40,
    TerrainType.URBAN_DENSE:   0.30,
    TerrainType.RUINS:         0.35,
    TerrainType.MOUNTAIN:      1.20,
    TerrainType.HILLS:         0.80,
    TerrainType.MARSH:         0.70,
    TerrainType.FORTIFIED:     0.60,
    TerrainType.PORT:          0.80,
    TerrainType.RAIL_YARD:     0.80,
    TerrainType.FARMLAND:      0.90,
    TerrainType.STEPPE:        1.10,
    TerrainType.TAIGA:         0.40,
    TerrainType.BOCAGE:        0.30,
    TerrainType.VOLCANIC:      0.70,
})

# Entrenchment dig-rate multiplier by soil (<1 = harder)
ENTRENCHMENT_RATE: Dict[SoilType, float] = {
    SoilType.CLAY:       0.80,
    SoilType.SAND:       1.20,
    SoilType.LOAM:       1.00,
    SoilType.ROCK:       0.20,
    SoilType.PERMAFROST: 0.10,
    SoilType.PEAT:       0.60,
    SoilType.GRAVEL:     0.70,
    SoilType.MUD:        0.30,
}

# Vehicle traction multiplier by soil
VEHICLE_TRACTION: Dict[SoilType, float] = {
    SoilType.CLAY:       0.60,
    SoilType.SAND:       0.40,
    SoilType.LOAM:       0.80,
    SoilType.ROCK:       0.90,
    SoilType.PERMAFROST: 0.85,
    SoilType.PEAT:       0.30,
    SoilType.GRAVEL:     0.75,
    SoilType.MUD:        0.20,
}

# ─────────────────────────────────────────────────────────────────────────── #
# Terrain Features & State                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class TerrainFeatures:
    """Composable infrastructure / geographic features for a sector."""
    has_river: bool = False
    has_road: bool = False
    has_rail: bool = False
    has_bridge: bool = False
    has_airfield: bool = False
    has_port: bool = False
    vegetation_density: float = 0.0   # 0-1
    water_table_depth_m: float = 5.0  # shallow → harder to entrench


@dataclass(frozen=True)
class TerrainState:
    """Complete physical terrain state for a sector."""
    sector_id: int
    terrain_type: TerrainType
    elevation_m: float
    cover_factor: float           # 0-1, concealment
    soil: SoilType = SoilType.LOAM
    features: TerrainFeatures = field(default_factory=TerrainFeatures)
    fortification: float = 0.0    # 0-1, built by defenders over time
    entrenchment: float = 0.0     # 0-1, dug-in bonus
    mines_density: float = 0.0    # 0-1, anti-vehicle mines
    obstacle_density: float = 0.0 # 0-1, wire / tank traps / abatis
    bridge_intact: bool = True    # bridge can be blown

    # Backward-compat aliases
    @property
    def has_river(self) -> bool:
        return self.features.has_river

    @property
    def has_rail(self) -> bool:
        return self.features.has_rail

    @property
    def movement_cost(self) -> float:
        """Total movement cost: terrain + features + fortification + obstacles."""
        base = MOVEMENT_COST.get(self.terrain_type, 1.0)
        river = 0.5 if self.features.has_river else 0.0
        road = -0.3 if self.features.has_road else 0.0
        fort = self.fortification * 0.5
        mines = self.mines_density * 1.0
        obstacles = self.obstacle_density * 0.8
        soil_drag = max(0, 1.0 - VEHICLE_TRACTION.get(self.soil, 0.8)) * 0.5
        return max(0.3, base + river + road + fort + mines + obstacles + soil_drag)

    @property
    def defense_bonus(self) -> float:
        """Total defense bonus from terrain + fortification + entrenchment."""
        terrain = DEFENSE_BONUS.get(self.terrain_type, 0.0)
        return terrain + self.fortification * 0.3 + self.entrenchment * 0.2

    @property
    def dig_rate(self) -> float:
        """Entrenchment build rate per step, affected by soil."""
        return 0.002 * ENTRENCHMENT_RATE.get(self.soil, 1.0)

    def copy_with(self, **kwargs) -> 'TerrainState':
        d = {}
        for f in self.__dataclass_fields__.values():
            d[f.name] = getattr(self, f.name)
        d.update(kwargs)
        return TerrainState(**d)


def compute_elevation_advantage(attacker_elevation: float, defender_elevation: float) -> float:
    """
    Elevation advantage: attacking downhill gives a bonus.
    Returns multiplier in [0.85, 1.15].
    """
    delta = attacker_elevation - defender_elevation
    advantage = 1.0 + _sc(delta / 2000.0, -0.15, 0.15)
    return float(advantage)


# ─────────────────────────────────────────────────────────────────────────── #
# Weather Physics — Generic, Config-Driven Climate Model                      #
# ─────────────────────────────────────────────────────────────────────────── #

# Default temperature curve (Moscow 1941, Oct-Dec) — overridden by YAML config
DEFAULT_TEMP_CURVE_C: Dict[int, float] = {
    0:   5.0,    # Early October
    50:  2.0,    # Mid October
    100: -3.0,   # Late October
    150: -8.0,   # Early November
    200: -15.0,  # Mid November
    250: -22.0,  # Late November
    300: -30.0,  # Early December (historic low)
    350: -25.0,  # Mid December
    400: -20.0,  # Late December
}

# Backward-compat alias
MOSCOW_TEMP_CURVE_C = DEFAULT_TEMP_CURVE_C

# Weather condition effects on operations
WEATHER_MOVEMENT_PENALTY: Dict[WeatherCondition, float] = {
    WeatherCondition.CLEAR:        0.00,
    WeatherCondition.OVERCAST:     0.00,
    WeatherCondition.RAIN:         0.05,
    WeatherCondition.HEAVY_RAIN:   0.15,
    WeatherCondition.THUNDERSTORM: 0.20,
    WeatherCondition.DRIZZLE:      0.02,
    WeatherCondition.SNOW:         0.10,
    WeatherCondition.HEAVY_SNOW:   0.20,
    WeatherCondition.BLIZZARD:     0.35,
    WeatherCondition.ICE_STORM:    0.25,
    WeatherCondition.FOG:          0.10,
    WeatherCondition.MIST:         0.03,
    WeatherCondition.MUD:          0.30,
    WeatherCondition.DUST_STORM:   0.25,
    WeatherCondition.HEAT_WAVE:    0.10,
    WeatherCondition.DEEP_FROST:   0.15,
    WeatherCondition.HAIL:         0.10,
}

WEATHER_VISIBILITY_FACTOR: Dict[WeatherCondition, float] = {
    WeatherCondition.CLEAR:        1.00,
    WeatherCondition.OVERCAST:     0.80,
    WeatherCondition.RAIN:         0.60,
    WeatherCondition.HEAVY_RAIN:   0.40,
    WeatherCondition.THUNDERSTORM: 0.30,
    WeatherCondition.DRIZZLE:      0.70,
    WeatherCondition.SNOW:         0.50,
    WeatherCondition.HEAVY_SNOW:   0.30,
    WeatherCondition.BLIZZARD:     0.10,
    WeatherCondition.ICE_STORM:    0.35,
    WeatherCondition.FOG:          0.15,
    WeatherCondition.MIST:         0.50,
    WeatherCondition.MUD:          0.80,
    WeatherCondition.DUST_STORM:   0.15,
    WeatherCondition.HEAT_WAVE:    0.85,
    WeatherCondition.DEEP_FROST:   0.70,
    WeatherCondition.HAIL:         0.40,
}

WEATHER_AIR_SUPPORT_FACTOR: Dict[WeatherCondition, float] = {
    WeatherCondition.CLEAR:        1.00,
    WeatherCondition.OVERCAST:     0.50,
    WeatherCondition.RAIN:         0.30,
    WeatherCondition.HEAVY_RAIN:   0.10,
    WeatherCondition.THUNDERSTORM: 0.00,
    WeatherCondition.DRIZZLE:      0.40,
    WeatherCondition.SNOW:         0.20,
    WeatherCondition.HEAVY_SNOW:   0.05,
    WeatherCondition.BLIZZARD:     0.00,
    WeatherCondition.ICE_STORM:    0.00,
    WeatherCondition.FOG:          0.05,
    WeatherCondition.MIST:         0.30,
    WeatherCondition.MUD:          0.70,
    WeatherCondition.DUST_STORM:   0.05,
    WeatherCondition.HEAT_WAVE:    0.90,
    WeatherCondition.DEEP_FROST:   0.40,
    WeatherCondition.HAIL:         0.10,
}


@dataclass
class WeatherState:
    """Current weather conditions for a sector or region."""
    temperature_c: float = 0.0
    condition: WeatherCondition = WeatherCondition.CLEAR
    snow_depth_cm: float = 0.0
    wind_speed_ms: float = 5.0
    visibility_km: float = 10.0
    ground_frozen: bool = False
    mud_factor: float = 0.0       # 0-1, rasputitsa severity
    humidity_pct: float = 50.0    # 0-100, relative humidity
    sand_depth_cm: float = 0.0    # desert sand accumulation
    heat_index_c: float = 0.0     # heat stress (tropical / desert)
    time_of_day: TimeOfDay = TimeOfDay.DAY
    season: Season = Season.AUTUMN

    @property
    def is_extreme_cold(self) -> bool:
        return self.temperature_c < -25.0

    @property
    def is_extreme_heat(self) -> bool:
        return self.temperature_c > 40.0

    @property
    def attrition_rate(self) -> float:
        """Per-step attrition from weather exposure (cold, heat, storms)."""
        rate = 0.0
        # Cold attrition
        if self.temperature_c < 0:
            cold_severity = abs(self.temperature_c) / 40.0
            rate += 0.001 * (cold_severity ** 2)
        # Heat attrition
        if self.temperature_c > 35:
            heat_severity = (self.temperature_c - 35) / 20.0
            rate += 0.0005 * (heat_severity ** 2)
        # Storm multiplier
        if self.condition in (WeatherCondition.BLIZZARD, WeatherCondition.THUNDERSTORM,
                              WeatherCondition.ICE_STORM, WeatherCondition.DUST_STORM):
            rate *= 2.0
        return _sc(rate, 0.0, 0.015)

    def equipment_reliability(self, winterized: bool = False) -> float:
        """
        Equipment reliability [0, 1].

        winterized=False → German-style (fails in cold).
        winterized=True  → Soviet/adapted (handles cold better).
        Extreme heat also degrades equipment.
        """
        rel = 1.0
        if self.temperature_c < -5:
            cold_pen = (-5.0 - self.temperature_c) / 35.0  # 0→1 over -5..-40
            if winterized:
                rel -= cold_pen * 0.3  # mild degradation
            else:
                rel -= cold_pen * 0.6  # severe degradation
        if self.temperature_c > 40:
            rel -= (self.temperature_c - 40) / 30.0 * 0.3
        # Sand/dust clogs machinery
        if self.condition in (WeatherCondition.DUST_STORM,):
            rel -= 0.15
        return _sc(rel, 0.15, 1.0)

    @property
    def movement_penalty(self) -> float:
        """Movement speed penalty from weather [0, 1]. 0 = none, 1 = immobilized."""
        penalty = WEATHER_MOVEMENT_PENALTY.get(self.condition, 0.0)
        penalty += min(self.snow_depth_cm / 100.0, 0.3)
        penalty += self.mud_factor * 0.4
        penalty += min(self.sand_depth_cm / 50.0, 0.2)
        if self.visibility_km < 2.0:
            penalty += 0.1
        if self.time_of_day == TimeOfDay.NIGHT:
            penalty += 0.15
        elif self.time_of_day in (TimeOfDay.DAWN, TimeOfDay.DUSK):
            penalty += 0.05
        return _sc(penalty, 0.0, 0.95)

    @property
    def air_support_factor(self) -> float:
        """Air support effectiveness [0, 1]. 0 = grounded."""
        base = WEATHER_AIR_SUPPORT_FACTOR.get(self.condition, 0.5)
        if self.time_of_day == TimeOfDay.NIGHT:
            base *= 0.3
        return _sc(base, 0.0, 1.0)

    @property
    def visibility_factor(self) -> float:
        """Combined visibility factor [0, 1]."""
        base = WEATHER_VISIBILITY_FACTOR.get(self.condition, 1.0)
        if self.time_of_day == TimeOfDay.NIGHT:
            base *= 0.25
        elif self.time_of_day in (TimeOfDay.DAWN, TimeOfDay.DUSK):
            base *= 0.6
        return _sc(base, 0.05, 1.0)


def interpolate_temperature(
    step: int,
    temp_curve: Optional[Dict[int, float]] = None,
) -> float:
    """Interpolate temperature from a step→°C curve (config-driven)."""
    curve = temp_curve or DEFAULT_TEMP_CURVE_C
    keys = sorted(curve.keys())
    t = _sc(step, keys[0], keys[-1])
    for i in range(len(keys) - 1):
        if keys[i] <= t <= keys[i + 1]:
            frac = (t - keys[i]) / (keys[i + 1] - keys[i])
            return curve[keys[i]] * (1 - frac) + curve[keys[i + 1]] * frac
    return curve[keys[-1]]


def _classify_weather(
    temp: float, snow: float, wind: float, mud: float,
    humidity: float, rng: np.random.Generator,
    climate_type: str = "continental",
) -> WeatherCondition:
    """Determine weather condition from physical variables."""
    # Extreme conditions take priority
    if temp < -25 and wind > 10:
        return WeatherCondition.BLIZZARD
    if temp < -15:
        return WeatherCondition.DEEP_FROST
    if temp > 45:
        return WeatherCondition.HEAT_WAVE
    if climate_type in ("desert", "arid") and wind > 15 and humidity < 20:
        return WeatherCondition.DUST_STORM

    # Snow / ice
    if temp < 0 and wind > 12 and snow > 5:
        return WeatherCondition.BLIZZARD if rng.random() < 0.3 else WeatherCondition.HEAVY_SNOW
    if temp < -2 and humidity > 60:
        return WeatherCondition.SNOW if rng.random() < 0.5 else WeatherCondition.HEAVY_SNOW
    if temp < 0 and humidity > 80:
        return WeatherCondition.ICE_STORM if rng.random() < 0.1 else WeatherCondition.SNOW

    # Mud season
    if mud > 0.3:
        return WeatherCondition.MUD

    # Rain
    if humidity > 85 and temp > 0:
        if wind > 12:
            return WeatherCondition.THUNDERSTORM if rng.random() < 0.2 else WeatherCondition.HEAVY_RAIN
        return WeatherCondition.HEAVY_RAIN if rng.random() < 0.3 else WeatherCondition.RAIN
    if humidity > 70 and temp > 0:
        return WeatherCondition.RAIN if rng.random() < 0.3 else WeatherCondition.DRIZZLE

    # Fog
    if humidity > 90 and wind < 3 and temp > -5:
        return WeatherCondition.FOG
    if humidity > 75 and wind < 5 and temp > -5:
        return WeatherCondition.MIST if rng.random() < 0.2 else WeatherCondition.OVERCAST

    # Hail (rare, spring/summer convective)
    if temp > 10 and humidity > 70 and wind > 10 and rng.random() < 0.03:
        return WeatherCondition.HAIL

    # Overcast
    if humidity > 60:
        return WeatherCondition.OVERCAST if rng.random() < 0.4 else WeatherCondition.CLEAR

    return WeatherCondition.CLEAR


def step_weather(
    weather: WeatherState,
    step: int,
    rng: np.random.Generator,
    temp_curve: Optional[Dict[int, float]] = None,
    climate_type: str = "continental",
    base_humidity: float = 60.0,
    base_wind_ms: float = 5.0,
    steps_per_day: int = 4,
) -> WeatherState:
    """
    Advance weather by one step using config-driven physical model.

    Parameters:
      temp_curve    — step→°C mapping (loaded from YAML)
      climate_type  — "continental", "desert", "tropical", "arctic", "maritime"
      base_humidity — average humidity for the region (0-100)
      base_wind_ms  — average wind speed
      steps_per_day — how many simulation steps per in-game day
    """
    temp = interpolate_temperature(step, temp_curve)
    temp += rng.normal(0, 1.5)  # daily noise

    # Snow accumulation / melting
    snow = weather.snow_depth_cm
    if temp < -2:
        snow += rng.uniform(0.1, 0.8)
    elif temp > 0:
        snow = max(0, snow - rng.uniform(0.5, 2.0))

    # Sand accumulation (desert climates)
    sand = weather.sand_depth_cm
    if climate_type in ("desert", "arid"):
        sand += rng.uniform(0, 0.3)
        if rng.random() < 0.1:
            sand = max(0, sand - rng.uniform(1, 5))  # wind shift clears some
    else:
        sand = 0.0

    # Mud factor
    mud = 0.0
    if -2 < temp < 5 and not weather.ground_frozen:
        mud = rng.uniform(0.3, 0.8)
    if climate_type == "tropical" and temp > 20:
        mud = max(mud, rng.uniform(0.1, 0.5))  # monsoon mud

    # Humidity
    humidity = base_humidity + rng.normal(0, 10)
    if climate_type == "desert":
        humidity = max(5, humidity - 30)
    elif climate_type == "tropical":
        humidity = min(100, humidity + 20)
    humidity = _sc(humidity, 0, 100)

    # Wind
    wind = max(0, base_wind_ms + rng.normal(0, 2))

    # Visibility (base 10 km, modified by conditions)
    vis = 10.0
    if snow > 20:
        vis -= 3.0
    if wind > 15:
        vis -= 2.0
    if sand > 5:
        vis -= 3.0
    vis = max(0.5, vis + rng.normal(0, 1))

    # Time of day cycling
    day_step = step % steps_per_day
    if day_step == 0:
        tod = TimeOfDay.DAWN
    elif day_step < steps_per_day - 1:
        tod = TimeOfDay.DAY
    elif day_step == steps_per_day - 1:
        tod = TimeOfDay.DUSK
    else:
        tod = TimeOfDay.NIGHT

    # Season from step (approximate: 100 steps per season)
    season_idx = (step // 100) % 4
    season = [Season.AUTUMN, Season.WINTER, Season.SPRING, Season.SUMMER][season_idx]

    # Classify condition
    condition = _classify_weather(temp, snow, wind, mud, humidity, rng, climate_type)

    # Heat index for tropical / desert
    heat_idx = 0.0
    if temp > 30:
        heat_idx = temp + (humidity / 100.0) * 10.0

    return WeatherState(
        temperature_c=float(temp),
        condition=condition,
        snow_depth_cm=_sc(snow, 0, 200),
        wind_speed_ms=_sc(wind, 0, 40),
        visibility_km=_sc(vis, 0.2, 20),
        ground_frozen=temp < -5,
        mud_factor=_sc(mud, 0, 1),
        humidity_pct=humidity,
        sand_depth_cm=_sc(sand, 0, 100),
        heat_index_c=float(heat_idx),
        time_of_day=tod,
        season=season,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Supply Logistics Physics                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class SupplyState:
    """Per-sector supply logistics state — 6 categories."""
    fuel_tons: float = 100.0         # Fuel reserves (petrol/diesel)
    ammo_tons: float = 100.0         # Ammunition reserves
    food_tons: float = 100.0         # Food / rations
    medical_tons: float = 20.0       # Medical supplies
    spare_parts_tons: float = 30.0   # Vehicle / weapon spare parts
    winter_gear_tons: float = 10.0   # Cold-weather clothing & equipment
    # Delivery infrastructure
    rail_capacity_tpd: float = 50.0  # Tons per day rail delivery
    road_capacity_tpd: float = 20.0  # Tons per day road delivery
    air_capacity_tpd: float = 0.0    # Tons per day air supply (if airfield)
    water_capacity_tpd: float = 0.0  # Tons per day river/sea transport
    depot_max_tons: float = 500.0    # Maximum storage capacity
    # Degradation
    spoilage_rate: float = 0.001     # Per-step food/medical spoilage

    @property
    def total_supply(self) -> float:
        return (self.fuel_tons + self.ammo_tons + self.food_tons +
                self.medical_tons + self.spare_parts_tons + self.winter_gear_tons)

    @property
    def supply_ratio(self) -> float:
        """Overall supply health [0, 1]."""
        return _sc(self.total_supply / max(self.depot_max_tons, 1), 0.0, 1.0)

    @property
    def is_critical(self) -> bool:
        """Any supply category critically low."""
        return (self.fuel_tons < 10 or self.ammo_tons < 10 or
                self.food_tons < 10 or self.medical_tons < 2)

    @property
    def fuel_ratio(self) -> float:
        return _sc(self.fuel_tons / 100.0, 0.0, 1.0)

    @property
    def ammo_ratio(self) -> float:
        return _sc(self.ammo_tons / 100.0, 0.0, 1.0)

    def copy_with(self, **kwargs) -> 'SupplyState':
        d = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        d.update(kwargs)
        return SupplyState(**d)


# ── Consumption rates per unit type per step ──────────────────────────────

FUEL_CONSUMPTION: Dict[str, float] = {
    "INFANTRY":             0.10,
    "ARMOR":                1.50,
    "ARTILLERY":            0.30,
    "LOGISTICS":            0.50,
    "MOTORIZED_INFANTRY":   0.80,
    "MILITIA":              0.05,
    "ANTI_TANK":            0.10,
    "ANTI_AIR":             0.20,
    "CAVALRY":              0.02,
    "ENGINEER":             0.40,
    "RECON":                0.60,
    "AIR_WING":             2.00,
    "NAVAL":                1.80,
    "SPECIAL_FORCES":       0.30,
    "PARTISAN":             0.02,
    "HEADQUARTERS":         0.20,
    "default":              0.20,
}

AMMO_CONSUMPTION_PER_COMBAT: Dict[str, float] = {
    "INFANTRY":             2.0,
    "ARMOR":                5.0,
    "ARTILLERY":            8.0,
    "ANTI_TANK":            3.0,
    "ANTI_AIR":             4.0,
    "MOTORIZED_INFANTRY":   2.5,
    "MILITIA":              1.5,
    "CAVALRY":              1.0,
    "ENGINEER":             1.5,
    "RECON":                1.0,
    "AIR_WING":             6.0,
    "NAVAL":                7.0,
    "SPECIAL_FORCES":       2.0,
    "PARTISAN":             0.5,
    "HEADQUARTERS":         0.0,
    "default":              2.0,
}

SPARE_PARTS_CONSUMPTION: Dict[str, float] = {
    "ARMOR":                0.50,
    "MOTORIZED_INFANTRY":   0.30,
    "ARTILLERY":            0.20,
    "LOGISTICS":            0.30,
    "AIR_WING":             0.80,
    "NAVAL":                0.60,
    "ANTI_AIR":             0.15,
    "RECON":                0.20,
    "default":              0.05,
}


def compute_supply_consumption(
    supply: SupplyState,
    unit_counts: Dict[str, int],
    in_combat: bool,
    weather: WeatherState,
) -> SupplyState:
    """
    Compute supply consumption for one step.

    Physical model:
    - Fuel: base × count × cold multiplier
    - Ammo: combat only, proportional to count
    - Food: per-unit, +50% in extreme cold/heat
    - Medical: proportional to weather attrition + combat
    - Spare parts: proportional to mechanized units + weather wear
    - Winter gear: consumed by cold exposure
    - Spoilage: food/medical degrade over time
    """
    total_units = sum(unit_counts.values())
    if total_units == 0:
        # Still apply spoilage
        return SupplyState(
            fuel_tons=supply.fuel_tons,
            ammo_tons=supply.ammo_tons,
            food_tons=float(max(0, supply.food_tons * (1 - supply.spoilage_rate))),
            medical_tons=float(max(0, supply.medical_tons * (1 - supply.spoilage_rate))),
            spare_parts_tons=supply.spare_parts_tons,
            winter_gear_tons=supply.winter_gear_tons,
            rail_capacity_tpd=supply.rail_capacity_tpd,
            road_capacity_tpd=supply.road_capacity_tpd,
            air_capacity_tpd=supply.air_capacity_tpd,
            water_capacity_tpd=supply.water_capacity_tpd,
            depot_max_tons=supply.depot_max_tons,
            spoilage_rate=supply.spoilage_rate,
        )

    # Fuel consumption
    fuel_used = 0.0
    cold_mult = 1.0 + max(0, -weather.temperature_c) / 30.0
    heat_mult = 1.0 + max(0, weather.temperature_c - 35) / 20.0  # AC / cooling
    for utype, count in unit_counts.items():
        rate = FUEL_CONSUMPTION.get(utype, FUEL_CONSUMPTION["default"])
        fuel_used += rate * count * cold_mult * heat_mult

    # Ammo consumption
    ammo_used = 0.0
    if in_combat:
        for utype, count in unit_counts.items():
            rate = AMMO_CONSUMPTION_PER_COMBAT.get(utype, AMMO_CONSUMPTION_PER_COMBAT["default"])
            ammo_used += rate * count

    # Food
    food_stress = 1.0
    if weather.is_extreme_cold or weather.is_extreme_heat:
        food_stress = 1.5
    food_used = 0.5 * total_units * food_stress

    # Medical
    combat_casualties = 0.5 * total_units if in_combat else 0.0
    medical_used = (weather.attrition_rate * total_units * 5.0) + combat_casualties * 0.3

    # Spare parts — wear from weather + combat
    parts_used = 0.0
    wear_mult = 1.0 + weather.mud_factor * 0.5 + (weather.sand_depth_cm / 50.0) * 0.3
    for utype, count in unit_counts.items():
        rate = SPARE_PARTS_CONSUMPTION.get(utype, SPARE_PARTS_CONSUMPTION["default"])
        parts_used += rate * count * wear_mult
    if in_combat:
        parts_used *= 1.5

    # Winter gear — consumed by cold
    wg_used = 0.0
    if weather.temperature_c < 0:
        wg_used = total_units * 0.01 * abs(weather.temperature_c) / 20.0

    # Spoilage
    food_spoil = supply.food_tons * supply.spoilage_rate
    med_spoil = supply.medical_tons * supply.spoilage_rate * 0.5

    return SupplyState(
        fuel_tons=float(max(0, supply.fuel_tons - fuel_used)),
        ammo_tons=float(max(0, supply.ammo_tons - ammo_used)),
        food_tons=float(max(0, supply.food_tons - food_used - food_spoil)),
        medical_tons=float(max(0, supply.medical_tons - medical_used - med_spoil)),
        spare_parts_tons=float(max(0, supply.spare_parts_tons - parts_used)),
        winter_gear_tons=float(max(0, supply.winter_gear_tons - wg_used)),
        rail_capacity_tpd=supply.rail_capacity_tpd,
        road_capacity_tpd=supply.road_capacity_tpd,
        air_capacity_tpd=supply.air_capacity_tpd,
        water_capacity_tpd=supply.water_capacity_tpd,
        depot_max_tons=supply.depot_max_tons,
        spoilage_rate=supply.spoilage_rate,
    )


def compute_supply_delivery(
    supply: SupplyState,
    has_rail: bool,
    distance_to_depot_km: float,
    weather: WeatherState,
    sabotage_factor: float = 0.0,
    has_airfield: bool = False,
    has_port: bool = False,
) -> SupplyState:
    """
    Compute supply delivery for one step.

    Transport modes:
    - Rail: high capacity, vulnerable to sabotage / frozen switches
    - Road: medium capacity, affected by weather
    - Air:  low capacity, unaffected by ground conditions but needs airfield
    - Water: high capacity where available, weather-dependent
    Distance decay: 1 / (1 + d/100)
    """
    dist_factor = 1.0 / (1.0 + distance_to_depot_km / 100.0)

    # Rail delivery
    rail_del = 0.0
    if has_rail:
        rail_eff = supply.rail_capacity_tpd * dist_factor * (1.0 - sabotage_factor)
        if weather.ground_frozen:
            rail_eff *= 0.7
        rail_del = max(0, rail_eff)

    # Road delivery
    road_eff = supply.road_capacity_tpd * dist_factor
    road_eff *= (1.0 - weather.movement_penalty)
    road_del = max(0, road_eff)

    # Air delivery
    air_del = 0.0
    if has_airfield and supply.air_capacity_tpd > 0:
        air_eff = supply.air_capacity_tpd * weather.air_support_factor
        air_del = max(0, air_eff)

    # Water delivery
    water_del = 0.0
    if has_port and supply.water_capacity_tpd > 0:
        water_eff = supply.water_capacity_tpd * dist_factor
        if weather.ground_frozen:
            water_eff *= 0.2  # ice blocks waterways
        if weather.condition in (WeatherCondition.THUNDERSTORM, WeatherCondition.HEAVY_RAIN):
            water_eff *= 0.5
        water_del = max(0, water_eff)

    total = rail_del + road_del + air_del + water_del

    # Priority distribution: ammo 30%, fuel 30%, food 20%, medical 8%, parts 8%, wgear 4%
    p = [0.30, 0.30, 0.20, 0.08, 0.08, 0.04]
    d = [total * pi for pi in p]
    cap = supply.depot_max_tons

    return SupplyState(
        fuel_tons=float(min(cap, supply.fuel_tons + d[1])),
        ammo_tons=float(min(cap, supply.ammo_tons + d[0])),
        food_tons=float(min(cap, supply.food_tons + d[2])),
        medical_tons=float(min(cap / 4, supply.medical_tons + d[3])),
        spare_parts_tons=float(min(cap / 3, supply.spare_parts_tons + d[4])),
        winter_gear_tons=float(min(cap / 5, supply.winter_gear_tons + d[5])),
        rail_capacity_tpd=supply.rail_capacity_tpd,
        road_capacity_tpd=supply.road_capacity_tpd,
        air_capacity_tpd=supply.air_capacity_tpd,
        water_capacity_tpd=supply.water_capacity_tpd,
        depot_max_tons=supply.depot_max_tons,
        spoilage_rate=supply.spoilage_rate,
    )


# ─────────────────────────────────────────────────────────────────────────── #
# Combat Physics — Expanded Damage Model                                     #
# ─────────────────────────────────────────────────────────────────────────── #

# Import damage / armor enums from unit_types (canonical definitions live there)
from .unit_types import DamageType, ArmorClass

# ── Damage Effectiveness Matrix ───────────────────────────────────────────── #
# DamageType → ArmorClass → float multiplier.
# Key design principles:
#   • Heavy AP (kinetic/composite) OVERPENETRATES unarmored targets → low mult
#   • HE / HE-FRAG devastate unarmored, useless vs heavy armor
#   • HEAT ignores thickness (shaped charge jet), good vs all armor classes
#   • Mines only affect ground units (0.0 vs FORTIFIED / NAVAL)

DAMAGE_EFFECTIVENESS: Dict[DamageType, Dict[ArmorClass, float]] = {
    DamageType.SMALL_ARMS: {
        ArmorClass.UNARMORED:   1.00,
        ArmorClass.LIGHT_ARMOR: 0.30,
        ArmorClass.MEDIUM_ARMOR:0.05,
        ArmorClass.HEAVY_ARMOR: 0.02,
        ArmorClass.FORTIFIED:   0.10,
        ArmorClass.NAVAL_ARMOR: 0.01,
    },
    DamageType.AP_KINETIC: {
        ArmorClass.UNARMORED:   0.40,   # Overpenetration — round passes clean through
        ArmorClass.LIGHT_ARMOR: 0.70,
        ArmorClass.MEDIUM_ARMOR:1.00,   # Designed for this
        ArmorClass.HEAVY_ARMOR: 0.60,   # Struggles vs thick plate
        ArmorClass.FORTIFIED:   0.30,   # Concrete absorbs kinetic
        ArmorClass.NAVAL_ARMOR: 0.40,
    },
    DamageType.AP_COMPOSITE: {
        ArmorClass.UNARMORED:   0.35,   # Even worse overpenetration
        ArmorClass.LIGHT_ARMOR: 0.65,
        ArmorClass.MEDIUM_ARMOR:1.10,   # Excellent
        ArmorClass.HEAVY_ARMOR: 0.90,   # Better than standard AP
        ArmorClass.FORTIFIED:   0.35,
        ArmorClass.NAVAL_ARMOR: 0.50,
    },
    DamageType.HEAT: {
        ArmorClass.UNARMORED:   0.50,   # Blast limited, jet overpens soft targets
        ArmorClass.LIGHT_ARMOR: 0.80,
        ArmorClass.MEDIUM_ARMOR:1.10,   # Shaped charge ignores thickness
        ArmorClass.HEAVY_ARMOR: 1.00,   # Effective even on heavy
        ArmorClass.FORTIFIED:   0.70,   # Good for bunker busting
        ArmorClass.NAVAL_ARMOR: 0.60,
    },
    DamageType.HE: {
        ArmorClass.UNARMORED:   1.30,   # Devastating blast + fragments
        ArmorClass.LIGHT_ARMOR: 0.90,
        ArmorClass.MEDIUM_ARMOR:0.40,
        ArmorClass.HEAVY_ARMOR: 0.15,
        ArmorClass.FORTIFIED:   0.50,   # Can damage over time
        ArmorClass.NAVAL_ARMOR: 0.20,
    },
    DamageType.HE_FRAG: {
        ArmorClass.UNARMORED:   1.40,   # Optimized anti-personnel
        ArmorClass.LIGHT_ARMOR: 0.70,
        ArmorClass.MEDIUM_ARMOR:0.20,
        ArmorClass.HEAVY_ARMOR: 0.05,
        ArmorClass.FORTIFIED:   0.30,   # Fragments enter openings
        ArmorClass.NAVAL_ARMOR: 0.10,
    },
    DamageType.INCENDIARY: {
        ArmorClass.UNARMORED:   1.20,   # Fire devastating to troops
        ArmorClass.LIGHT_ARMOR: 0.80,
        ArmorClass.MEDIUM_ARMOR:0.30,
        ArmorClass.HEAVY_ARMOR: 0.10,
        ArmorClass.FORTIFIED:   0.90,   # Fire clears bunkers
        ArmorClass.NAVAL_ARMOR: 0.20,
    },
    DamageType.ROCKET_HE: {
        ArmorClass.UNARMORED:   1.20,   # Area saturation
        ArmorClass.LIGHT_ARMOR: 0.80,
        ArmorClass.MEDIUM_ARMOR:0.50,
        ArmorClass.HEAVY_ARMOR: 0.20,
        ArmorClass.FORTIFIED:   0.60,
        ArmorClass.NAVAL_ARMOR: 0.15,
    },
    DamageType.NAVAL_SHELL: {
        ArmorClass.UNARMORED:   1.50,   # Massive HE effect
        ArmorClass.LIGHT_ARMOR: 1.30,
        ArmorClass.MEDIUM_ARMOR:1.00,
        ArmorClass.HEAVY_ARMOR: 0.70,
        ArmorClass.FORTIFIED:   0.90,   # Heavy shells crack bunkers
        ArmorClass.NAVAL_ARMOR: 0.80,
    },
    DamageType.BOMB_GP: {
        ArmorClass.UNARMORED:   1.40,
        ArmorClass.LIGHT_ARMOR: 1.00,
        ArmorClass.MEDIUM_ARMOR:0.60,
        ArmorClass.HEAVY_ARMOR: 0.30,
        ArmorClass.FORTIFIED:   0.70,
        ArmorClass.NAVAL_ARMOR: 0.40,
    },
    DamageType.BOMB_AP: {
        ArmorClass.UNARMORED:   0.60,
        ArmorClass.LIGHT_ARMOR: 0.80,
        ArmorClass.MEDIUM_ARMOR:1.00,
        ArmorClass.HEAVY_ARMOR: 0.90,
        ArmorClass.FORTIFIED:   0.80,
        ArmorClass.NAVAL_ARMOR: 1.00,
    },
    DamageType.MINE_AT: {
        ArmorClass.UNARMORED:   0.30,   # Not anti-personnel
        ArmorClass.LIGHT_ARMOR: 1.00,
        ArmorClass.MEDIUM_ARMOR:1.10,
        ArmorClass.HEAVY_ARMOR: 0.80,
        ArmorClass.FORTIFIED:   0.00,   # Mines don't affect bunkers
        ArmorClass.NAVAL_ARMOR: 0.00,
    },
    DamageType.MINE_AP: {
        ArmorClass.UNARMORED:   1.30,   # Designed for personnel
        ArmorClass.LIGHT_ARMOR: 0.20,
        ArmorClass.MEDIUM_ARMOR:0.05,
        ArmorClass.HEAVY_ARMOR: 0.02,
        ArmorClass.FORTIFIED:   0.00,
        ArmorClass.NAVAL_ARMOR: 0.00,
    },
    DamageType.DEMOLITION: {
        ArmorClass.UNARMORED:   1.00,
        ArmorClass.LIGHT_ARMOR: 0.90,
        ArmorClass.MEDIUM_ARMOR:0.80,
        ArmorClass.HEAVY_ARMOR: 0.50,
        ArmorClass.FORTIFIED:   1.30,   # Designed for breaching
        ArmorClass.NAVAL_ARMOR: 0.30,
    },
    DamageType.AA_AUTOCANNON: {
        ArmorClass.UNARMORED:   1.10,   # Rapid-fire HE/AP mix
        ArmorClass.LIGHT_ARMOR: 0.80,
        ArmorClass.MEDIUM_ARMOR:0.25,
        ArmorClass.HEAVY_ARMOR: 0.05,
        ArmorClass.FORTIFIED:   0.15,
        ArmorClass.NAVAL_ARMOR: 0.10,
    },
}


def get_damage_multiplier(damage_type: DamageType, armor_class: ArmorClass) -> float:
    """Look up damage effectiveness multiplier for a damage type vs armor class."""
    dt_map = DAMAGE_EFFECTIVENESS.get(damage_type)
    if dt_map is None:
        return 1.0
    return dt_map.get(armor_class, 1.0)


# ── Unit Weapon Profiles ──────────────────────────────────────────────────── #

@dataclass(frozen=True)
class UnitWeaponProfile:
    """Complete weapon and protection profile for a unit type."""
    primary_damage: DamageType
    secondary_damage: Optional[DamageType]
    armor_class: ArmorClass
    base_dps: float           # base damage output per step (normalised 0-1)
    range_km: float           # effective engagement range
    armor_mm: float           # equivalent armor thickness (mm RHA)
    penetration_mm: float     # penetration capability (mm RHA at 500 m)


# Profiles keyed by unit-type string (matches UNIT_CONSUMPTION_RATES keys)
UNIT_PROFILES: Dict[str, UnitWeaponProfile] = {
    "INFANTRY":           UnitWeaponProfile(
        DamageType.SMALL_ARMS, DamageType.HE_FRAG,
        ArmorClass.UNARMORED,  0.50, 0.5,   0.0,  20.0),
    "ARMOR":              UnitWeaponProfile(
        DamageType.AP_KINETIC, DamageType.HE,
        ArmorClass.HEAVY_ARMOR,0.90, 2.0,  80.0,  75.0),
    "ARTILLERY":          UnitWeaponProfile(
        DamageType.HE,         DamageType.HE_FRAG,
        ArmorClass.UNARMORED,  0.80, 15.0,  0.0,  60.0),
    "ANTI_TANK":          UnitWeaponProfile(
        DamageType.AP_COMPOSITE, DamageType.HEAT,
        ArmorClass.UNARMORED,  0.70, 1.5,   0.0, 100.0),
    "ANTI_AIR":           UnitWeaponProfile(
        DamageType.AA_AUTOCANNON, None,
        ArmorClass.LIGHT_ARMOR,0.60, 5.0,   5.0,  15.0),
    "MOTORIZED_INFANTRY": UnitWeaponProfile(
        DamageType.SMALL_ARMS, DamageType.HE_FRAG,
        ArmorClass.LIGHT_ARMOR,0.55, 1.0,  10.0,  25.0),
    "MILITIA":            UnitWeaponProfile(
        DamageType.SMALL_ARMS, None,
        ArmorClass.UNARMORED,  0.30, 0.3,   0.0,  10.0),
    "LOGISTICS":          UnitWeaponProfile(
        DamageType.SMALL_ARMS, None,
        ArmorClass.UNARMORED,  0.10, 0.0,   0.0,   5.0),
    "CAVALRY":            UnitWeaponProfile(
        DamageType.SMALL_ARMS, DamageType.HE_FRAG,
        ArmorClass.UNARMORED,  0.45, 0.8,   0.0,  15.0),
    "ENGINEER":           UnitWeaponProfile(
        DamageType.DEMOLITION, DamageType.SMALL_ARMS,
        ArmorClass.UNARMORED,  0.40, 0.3,   0.0,  30.0),
    "RECON":              UnitWeaponProfile(
        DamageType.SMALL_ARMS, None,
        ArmorClass.LIGHT_ARMOR,0.35, 1.5,   8.0,  20.0),
    "AIR_WING":           UnitWeaponProfile(
        DamageType.BOMB_GP,    DamageType.BOMB_AP,
        ArmorClass.LIGHT_ARMOR,0.85, 50.0,  5.0,  70.0),
    "NAVAL":              UnitWeaponProfile(
        DamageType.NAVAL_SHELL, DamageType.HE,
        ArmorClass.NAVAL_ARMOR,0.90, 20.0, 50.0,  90.0),
    "SPECIAL_FORCES":     UnitWeaponProfile(
        DamageType.SMALL_ARMS, DamageType.DEMOLITION,
        ArmorClass.UNARMORED,  0.60, 1.0,   0.0,  40.0),
    "PARTISAN":           UnitWeaponProfile(
        DamageType.SMALL_ARMS, DamageType.MINE_AP,
        ArmorClass.UNARMORED,  0.25, 0.3,   0.0,  10.0),
    "HEADQUARTERS":       UnitWeaponProfile(
        DamageType.SMALL_ARMS, None,
        ArmorClass.UNARMORED,  0.05, 0.0,   0.0,   5.0),
    # ── Expanded armor sub-types ───────────────────────────────────────── #
    "LIGHT_TANK":         UnitWeaponProfile(
        DamageType.AP_KINETIC, DamageType.HE,
        ArmorClass.LIGHT_ARMOR,0.70, 1.8,  25.0,  55.0),
    "MEDIUM_TANK":        UnitWeaponProfile(
        DamageType.AP_KINETIC, DamageType.HE,
        ArmorClass.MEDIUM_ARMOR,0.85, 2.0, 60.0,  70.0),
    "HEAVY_TANK":         UnitWeaponProfile(
        DamageType.AP_COMPOSITE, DamageType.HE,
        ArmorClass.HEAVY_ARMOR,1.00, 2.0, 120.0, 100.0),
    "SUPER_HEAVY_TANK":   UnitWeaponProfile(
        DamageType.AP_COMPOSITE, DamageType.HE,
        ArmorClass.HEAVY_ARMOR,1.10, 2.5, 200.0, 130.0),
    "ARMORED_CAR":        UnitWeaponProfile(
        DamageType.SMALL_ARMS, DamageType.AP_KINETIC,
        ArmorClass.LIGHT_ARMOR,0.40, 1.2,  12.0,  30.0),
    "TANK_DESTROYER":     UnitWeaponProfile(
        DamageType.AP_COMPOSITE, DamageType.HEAT,
        ArmorClass.MEDIUM_ARMOR,0.80, 2.0,  50.0, 110.0),
    "MECHANIZED_INFANTRY":UnitWeaponProfile(
        DamageType.SMALL_ARMS, DamageType.HE_FRAG,
        ArmorClass.LIGHT_ARMOR,0.60, 0.8,  15.0,  25.0),
    "ROCKET_ARTILLERY":   UnitWeaponProfile(
        DamageType.ROCKET_HE,  None,
        ArmorClass.UNARMORED,  0.85, 12.0,  0.0,  40.0),
    # ── Air sub-types ──────────────────────────────────────────────────── #
    "FIGHTER":            UnitWeaponProfile(
        DamageType.AA_AUTOCANNON, DamageType.BOMB_GP,
        ArmorClass.LIGHT_ARMOR,0.75, 40.0,  5.0,  20.0),
    "CAS":                UnitWeaponProfile(
        DamageType.BOMB_GP,    DamageType.BOMB_AP,
        ArmorClass.LIGHT_ARMOR,0.90, 30.0,  5.0,  60.0),
    "STRATEGIC_BOMBER":   UnitWeaponProfile(
        DamageType.BOMB_GP,    DamageType.INCENDIARY,
        ArmorClass.LIGHT_ARMOR,1.00, 100.0, 5.0,  30.0),
    # ── Naval sub-types ────────────────────────────────────────────────── #
    "DESTROYER":          UnitWeaponProfile(
        DamageType.NAVAL_SHELL, DamageType.HE,
        ArmorClass.LIGHT_ARMOR,0.65, 15.0, 20.0,  50.0),
    "CRUISER":            UnitWeaponProfile(
        DamageType.NAVAL_SHELL, DamageType.HE,
        ArmorClass.MEDIUM_ARMOR,0.80, 25.0, 80.0, 70.0),
    "BATTLESHIP":         UnitWeaponProfile(
        DamageType.NAVAL_SHELL, DamageType.BOMB_AP,
        ArmorClass.NAVAL_ARMOR,1.00, 35.0,200.0, 120.0),
}

# Backward-compat aliases
COMBAT_RANGE_KM: Dict[str, float] = {k: v.range_km for k, v in UNIT_PROFILES.items()}
COMBAT_RANGE_KM["default"] = 0.5
ARMOR_THICKNESS: Dict[str, float] = {k: v.armor_mm for k, v in UNIT_PROFILES.items()}
ARMOR_THICKNESS["default"] = 0.0
PENETRATION: Dict[str, float] = {k: v.penetration_mm for k, v in UNIT_PROFILES.items()}
PENETRATION["default"] = 10.0


# ── Damage Computation ────────────────────────────────────────────────────── #

def compute_damage_factor(
    attacker_type: str,
    defender_type: str,
    use_secondary: bool = False,
) -> float:
    """
    Compute damage factor based on attacker's weapon vs defender's armor class.

    Key behaviour requested:
      • Heavy AP vs UNARMORED → low (overpenetration)
      • Heavy AP vs LIGHT_ARMOR → medium
      • HE vs UNARMORED → devastating
      • HEAT effective against all armor levels

    Returns: multiplier in [0.01, 1.5]
    """
    a_profile = UNIT_PROFILES.get(attacker_type)
    d_profile = UNIT_PROFILES.get(defender_type)
    if a_profile is None or d_profile is None:
        return 1.0

    dmg_type = a_profile.secondary_damage if (use_secondary and a_profile.secondary_damage) else a_profile.primary_damage
    armor_cls = d_profile.armor_class

    return get_damage_multiplier(dmg_type, armor_cls)


def compute_penetration_factor(
    attacker_type: str,
    defender_type: str,
) -> float:
    """
    Combined penetration + damage-type factor [0.05, 2.0].

    Merges the old kinetic penetration check with the new damage-type
    effectiveness matrix so that both mechanisms contribute.
    """
    a_profile = UNIT_PROFILES.get(attacker_type)
    d_profile = UNIT_PROFILES.get(defender_type)
    if a_profile is None or d_profile is None:
        return 1.0

    # 1. Kinetic penetration ratio
    armor = d_profile.armor_mm
    pen = a_profile.penetration_mm
    if armor <= 0:
        pen_ratio = 1.0
    else:
        pen_ratio = _sc(pen / armor, 0.1, 1.5)

    # 2. Damage-type effectiveness
    dmg_eff = get_damage_multiplier(a_profile.primary_damage, d_profile.armor_class)

    # Geometric mean of both factors
    combined = (pen_ratio * dmg_eff) ** 0.5
    return _sc(combined, 0.05, 2.0)


def compute_combat_effectiveness(
    terrain: TerrainState,
    weather: WeatherState,
    supply: SupplyState,
    is_attacker: bool,
    winterized: bool = False,
) -> float:
    """
    Compute overall combat effectiveness multiplier [0.1, 1.5].

    Combines terrain, weather, supply, and equipment readiness.
    Use winterized=True for factions with cold-adapted equipment.
    """
    eff = 1.0

    # Terrain bonus for defender
    if not is_attacker:
        eff *= (1.0 + terrain.defense_bonus)

    # Equipment reliability (faction-agnostic via winterized flag)
    eff *= weather.equipment_reliability(winterized=winterized)

    # Movement penalty reduces attacker effectiveness
    if is_attacker:
        eff *= (1.0 - weather.movement_penalty * 0.5)

    # Supply effects
    if supply.fuel_tons < 20:
        eff *= 0.7
    if supply.ammo_tons < 20:
        eff *= 0.5
    if supply.food_tons < 10:
        eff *= 0.8
    if supply.medical_tons < 5:
        eff *= 0.9
    if supply.spare_parts_tons < 5:
        eff *= 0.85  # Can't repair equipment
    if weather.temperature_c < -10 and supply.winter_gear_tons < 5:
        eff *= 0.7   # Troops freezing

    # Visibility affects attackers more
    if is_attacker and weather.visibility_factor < 0.3:
        eff *= 0.8

    # Night combat penalty (less for trained units — caller can adjust)
    if weather.time_of_day == TimeOfDay.NIGHT:
        eff *= 0.75
    elif weather.time_of_day in (TimeOfDay.DAWN, TimeOfDay.DUSK):
        eff *= 0.9

    return _sc(eff, 0.1, 1.5)


def compute_artillery_effectiveness(
    weather: WeatherState,
    terrain: TerrainState,
    distance_km: float = 5.0,
) -> float:
    """Artillery effectiveness [0, 1.5] considering weather and terrain."""
    eff = 1.0

    max_range = UNIT_PROFILES.get("ARTILLERY", UNIT_PROFILES["INFANTRY"]).range_km
    if distance_km > max_range:
        return 0.0
    eff *= 1.0 - (distance_km / max_range) * 0.3

    if weather.visibility_factor < 0.3:
        eff *= 0.6
    if weather.ground_frozen:
        eff *= 0.85

    # Terrain modifiers for artillery
    arty_terrain_mod: Dict[TerrainType, float] = {
        TerrainType.URBAN:        0.70,
        TerrainType.URBAN_DENSE:  0.60,
        TerrainType.RUINS:        0.65,
        TerrainType.FOREST:       1.20,  # Tree bursts
        TerrainType.DENSE_FOREST: 1.15,
        TerrainType.JUNGLE:       1.10,
        TerrainType.MOUNTAIN:     0.75,
        TerrainType.MARSH:        1.10,  # No cover
        TerrainType.OPEN:         1.15,
        TerrainType.STEPPE:       1.20,
        TerrainType.FORTIFIED:    0.50,  # Bunkers absorb
    }
    eff *= arty_terrain_mod.get(terrain.terrain_type, 1.0)

    return _sc(eff, 0.0, 1.5)


def compute_air_support_effectiveness(
    weather: WeatherState,
    terrain: TerrainState,
    has_air_superiority: bool = False,
) -> float:
    """Air support effectiveness [0, 1.0] considering weather and AA threat."""
    eff = weather.air_support_factor

    # Terrain concealment reduces air strike accuracy
    cover = COVER_FACTOR.get(terrain.terrain_type, 0.1)
    eff *= (1.0 - cover * 0.5)

    if has_air_superiority:
        eff *= 1.3

    return _sc(eff, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Movement / Mobility Matrix                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

# Speed multiplier: unit_type → terrain_type → float (1.0 = base speed)
# Only overrides needed; missing entries default to 1.0
_MOBILITY_OVERRIDES: Dict[str, Dict[TerrainType, float]] = {
    "INFANTRY": {
        TerrainType.MOUNTAIN: 0.6, TerrainType.JUNGLE: 0.5,
        TerrainType.MARSH: 0.5, TerrainType.URBAN_DENSE: 0.7,
    },
    "ARMOR": {
        TerrainType.MOUNTAIN: 0.2, TerrainType.JUNGLE: 0.1,
        TerrainType.MARSH: 0.15, TerrainType.DENSE_FOREST: 0.3,
        TerrainType.FOREST: 0.5, TerrainType.URBAN_DENSE: 0.4,
        TerrainType.BOCAGE: 0.3, TerrainType.DESERT_SANDY: 0.5,
        TerrainType.RIVER: 0.1, TerrainType.RUINS: 0.35,
        TerrainType.ROAD: 1.5, TerrainType.STEPPE: 1.3,
    },
    "ARTILLERY": {
        TerrainType.MOUNTAIN: 0.15, TerrainType.JUNGLE: 0.1,
        TerrainType.MARSH: 0.2, TerrainType.DENSE_FOREST: 0.25,
        TerrainType.ROAD: 1.3, TerrainType.RIVER: 0.1,
    },
    "MOTORIZED_INFANTRY": {
        TerrainType.MOUNTAIN: 0.3, TerrainType.JUNGLE: 0.15,
        TerrainType.MARSH: 0.2, TerrainType.DESERT_SANDY: 0.6,
        TerrainType.ROAD: 1.4, TerrainType.STEPPE: 1.2,
    },
    "CAVALRY": {
        TerrainType.MOUNTAIN: 0.4, TerrainType.MARSH: 0.3,
        TerrainType.FOREST: 0.8, TerrainType.STEPPE: 1.4,
        TerrainType.DESERT_SANDY: 0.7, TerrainType.URBAN: 0.5,
    },
    "ENGINEER": {
        TerrainType.RIVER: 0.8,  # Bridging capability
        TerrainType.MARSH: 0.6, TerrainType.MOUNTAIN: 0.4,
        TerrainType.FORTIFIED: 0.7,  # Breaching
    },
    "RECON": {
        TerrainType.ROAD: 1.5, TerrainType.STEPPE: 1.3,
        TerrainType.MOUNTAIN: 0.5, TerrainType.JUNGLE: 0.3,
    },
    "SPECIAL_FORCES": {
        TerrainType.MOUNTAIN: 0.8, TerrainType.JUNGLE: 0.7,
        TerrainType.FOREST: 0.9, TerrainType.URBAN_DENSE: 0.8,
        TerrainType.MARSH: 0.6,
    },
    "AIR_WING": {t: 1.0 for t in TerrainType},  # Unaffected by ground terrain
    "NAVAL": {
        TerrainType.COASTAL: 1.0, TerrainType.PORT: 1.2,
        TerrainType.RIVER: 0.7, TerrainType.LAKE: 0.8,
    },
    "LOGISTICS": {
        TerrainType.ROAD: 1.5, TerrainType.RAIL_YARD: 1.3,
        TerrainType.MOUNTAIN: 0.2, TerrainType.JUNGLE: 0.15,
        TerrainType.MARSH: 0.2,
    },
}


def get_mobility(unit_type: str, terrain_type: TerrainType) -> float:
    """Get speed multiplier for a unit type on a given terrain."""
    overrides = _MOBILITY_OVERRIDES.get(unit_type, {})
    return overrides.get(terrain_type, 1.0)


def compute_movement_speed(
    unit_type: str,
    terrain: TerrainState,
    weather: WeatherState,
    supply: SupplyState,
) -> float:
    """
    Compute effective movement speed [0, 1] for a unit type.

    Combines terrain mobility, weather penalty, supply (fuel), and soil traction.
    """
    base = get_mobility(unit_type, terrain.terrain_type)
    # Soil traction (mainly affects vehicles)
    is_vehicle = unit_type in ("ARMOR", "MOTORIZED_INFANTRY", "ARTILLERY",
                               "LOGISTICS", "RECON", "ANTI_AIR")
    if is_vehicle:
        base *= VEHICLE_TRACTION.get(terrain.soil, 0.8)
    # Weather penalty
    base *= (1.0 - weather.movement_penalty)
    # No fuel = immobilized vehicles
    if is_vehicle and supply.fuel_tons < 5:
        base *= 0.1
    return _sc(base, 0.0, 2.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Line-of-Sight (LOS) System — Heatmap + Vector Representation              #
# ─────────────────────────────────────────────────────────────────────────── #
#
# Instead of per-unit raycasting we represent LOS as:
#   • positions    (N, 2) — sector centres in km
#   • elevation_m  (N,)   — sector mean elevation
#   • distance_km  (N, N) — pairwise distance (precomputed)
#   • direction_vectors (N, N, 2) — unit direction vectors between sectors
#   • los_quality   (N, N) — visibility quality in [0, 1]
#   • visibility_heatmap (N,) — per-sector aggregate observability
#
# This is fully vectorised (numpy), cheap to update each step, and directly
# consumable as an RL observation or by the combat resolver.

@dataclass
class LOSState:
    """Line-of-sight state for the entire battlefield."""
    positions: NDArray[np.float32]           # (N, 2)  sector centres (km)
    elevation_m: NDArray[np.float32]         # (N,)    mean elevation
    distance_km: NDArray[np.float32]         # (N, N)  pairwise distance
    direction_vectors: NDArray[np.float32]   # (N, N, 2) unit direction vectors
    los_quality: NDArray[np.float32]         # (N, N)  visibility quality [0, 1]
    visibility_heatmap: NDArray[np.float32]  # (N,)    per-sector aggregate


def build_los_state(
    terrain_states: List[TerrainState],
    positions: Optional[NDArray[np.float32]] = None,
) -> LOSState:
    """
    Build initial LOS state from terrain.

    If *positions* is None, sectors are placed on a grid ~30 km apart.
    All heavy lifting uses vectorised numpy — O(N²) but N ≤ ~20 sectors.
    """
    N = len(terrain_states)

    # Auto-generate positions on a square grid if not provided
    if positions is None:
        cols = max(1, int(np.ceil(np.sqrt(N))))
        positions = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            positions[i, 0] = float(i % cols) * 30.0
            positions[i, 1] = float(i // cols) * 30.0
    else:
        positions = np.asarray(positions, dtype=np.float32)

    elevation = np.array(
        [t.elevation_m for t in terrain_states], dtype=np.float32
    )

    # ── Pairwise distance ──────────────────────────────────────────────── #
    dx = positions[:, np.newaxis, 0] - positions[np.newaxis, :, 0]
    dy = positions[:, np.newaxis, 1] - positions[np.newaxis, :, 1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    distance[distance == 0] = 0.001  # avoid /0

    # ── Direction vectors (normalised) ─────────────────────────────────── #
    direction = np.stack([dx, dy], axis=-1)                  # (N, N, 2)
    norms = np.linalg.norm(direction, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    direction = (direction / norms).astype(np.float32)

    # ── Base LOS quality ───────────────────────────────────────────────── #
    # Range factor: quality degrades beyond 10 km, ≈ 0 at 50 km
    range_factor = np.clip(1.0 - distance / 50.0, 0.0, 1.0)

    # Elevation advantage: higher = better observation
    elev_diff = elevation[:, np.newaxis] - elevation[np.newaxis, :]
    elev_factor = np.clip(1.0 + elev_diff / 500.0, 0.5, 1.5)

    # Cover attenuation: target's cover reduces observer's LOS (up to 50%)
    cover = np.array(
        [t.cover_factor for t in terrain_states], dtype=np.float32
    )
    cover_target = 1.0 - cover[np.newaxis, :] * 0.5   # target concealment
    cover_observer = 1.0 - cover[:, np.newaxis] * 0.3  # own cover limits obs

    # Visibility modifier from terrain type
    vis_mod = np.array(
        [VISIBILITY_MODIFIER.get(t.terrain_type, 1.0) for t in terrain_states],
        dtype=np.float32,
    )
    vis_combined = np.sqrt(
        vis_mod[:, np.newaxis] * vis_mod[np.newaxis, :]
    )

    los_quality = range_factor * elev_factor * cover_target * cover_observer * vis_combined
    np.fill_diagonal(los_quality, 1.0)  # perfect self-visibility
    los_quality = np.clip(los_quality, 0.0, 1.0).astype(np.float32)

    # Heatmap: mean outgoing LOS quality per sector
    visibility_heatmap = np.mean(los_quality, axis=1).astype(np.float32)

    return LOSState(
        positions=positions,
        elevation_m=elevation,
        distance_km=distance.astype(np.float32),
        direction_vectors=direction,
        los_quality=los_quality,
        visibility_heatmap=visibility_heatmap,
    )


def update_los_for_weather(
    los: LOSState,
    weather_states: List[WeatherState],
    smoke_sectors: Optional[List[int]] = None,
) -> LOSState:
    """
    Update LOS quality based on current weather + smoke.  Vectorised.

    Call once per physics step to keep the heatmap current.
    """
    N = los.los_quality.shape[0]
    smoke = set(smoke_sectors or [])

    # Per-sector weather visibility factor
    vis_factor = np.array(
        [w.visibility_factor for w in weather_states[:N]], dtype=np.float32
    )

    # Smoke penalty mask
    smoke_mask = np.ones(N, dtype=np.float32)
    for s in smoke:
        if 0 <= s < N:
            smoke_mask[s] = 0.15  # heavy smoke decimates LOS into that sector

    # Weather modifier: geometric mean of observer and target weather
    weather_mod = np.sqrt(
        vis_factor[:, np.newaxis] * vis_factor[np.newaxis, :]
    )

    # Smoke reduces LOS *to* the smoked sector
    smoke_mod = smoke_mask[np.newaxis, :]   # (1, N) broadcast

    # Recompute base quality from distance/elevation (stored implicitly)
    # Instead of re-deriving base, we modulate the stored quality.
    new_los = los.los_quality * weather_mod * smoke_mod
    np.fill_diagonal(new_los, 1.0)
    new_los = np.clip(new_los, 0.0, 1.0).astype(np.float32)

    new_heatmap = np.mean(new_los, axis=1).astype(np.float32)

    return LOSState(
        positions=los.positions,
        elevation_m=los.elevation_m,
        distance_km=los.distance_km,
        direction_vectors=los.direction_vectors,
        los_quality=new_los,
        visibility_heatmap=new_heatmap,
    )


def compute_engagement_los(
    los: LOSState,
    attacker_sector: int,
    defender_sector: int,
) -> Tuple[float, NDArray[np.float32]]:
    """
    Get LOS quality and direction vector for an engagement.

    Returns:
        (quality, direction_vector) where quality ∈ [0, 1] and
        direction is a (2,) unit vector from attacker → defender.
    """
    quality = float(los.los_quality[attacker_sector, defender_sector])
    direction = los.direction_vectors[attacker_sector, defender_sector]
    return quality, direction


# ── LOS observation for RL ─────────────────────────────────────────────── #

LOS_HEATMAP_DIM = 12   # matches max_N default
LOS_TRI_DIM = LOS_HEATMAP_DIM * (LOS_HEATMAP_DIM - 1) // 2   # 66
LOS_OBS_DIM = LOS_HEATMAP_DIM + LOS_TRI_DIM                   # 78


def get_los_obs(los: LOSState, max_N: int = 12) -> NDArray[np.float32]:
    """
    Flatten LOS state to a fixed-size observation vector.

    Layout: [heatmap(max_N) | upper-triangle of quality matrix(max_N*(max_N-1)/2)]
    Total dims: max_N + max_N*(max_N-1)/2  (78 for max_N=12)
    """
    N = min(los.los_quality.shape[0], max_N)

    # Heatmap
    heatmap = np.zeros(max_N, dtype=np.float32)
    heatmap[:N] = los.visibility_heatmap[:N]

    # Upper triangle of quality matrix
    tri_size = max_N * (max_N - 1) // 2
    tri = np.zeros(tri_size, dtype=np.float32)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            if idx < tri_size:
                tri[idx] = los.los_quality[i, j]
                idx += 1

    return np.concatenate([heatmap, tri])


# ─────────────────────────────────────────────────────────────────────────── #
# Battlefield State — Unified per-sector physical state                       #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class SectorPhysicsState:
    """Complete physical state for one battlefield sector."""
    sector_id: int
    terrain: TerrainState
    weather: WeatherState
    supply_axis: SupplyState
    supply_soviet: SupplyState
    step: int = 0

    def to_obs_vector(self) -> NDArray[np.float32]:
        """
        Flatten to observation vector for RL agent (20 dims per sector).

        [terrain_type(1), elevation_norm(1), cover(1), fortification(1),
         temperature_norm(1), snow_norm(1), visibility_norm(1), movement_penalty(1),
         equipment_reliability(1), weather_attrition(1),
         supply_ratio_axis(1), fuel_axis(1), ammo_axis(1),
         supply_ratio_soviet(1), fuel_soviet(1), ammo_soviet(1),
         defense_bonus(1), movement_cost(1), mud(1), mines(1)]
        """
        n_terrain = len(TerrainType)
        return np.array([
            self.terrain.terrain_type.value / max(n_terrain, 1),
            self.terrain.elevation_m / 500.0,
            self.terrain.cover_factor,
            self.terrain.fortification,
            (self.weather.temperature_c + 50) / 100.0,  # normalize [-50,50] → [0,1]
            self.weather.snow_depth_cm / 200.0,
            self.weather.visibility_km / 20.0,
            self.weather.movement_penalty,
            self.weather.equipment_reliability(winterized=False),
            self.weather.attrition_rate * 100,
            self.supply_axis.supply_ratio,
            self.supply_axis.fuel_tons / 100.0,
            self.supply_axis.ammo_tons / 100.0,
            self.supply_soviet.supply_ratio,
            self.supply_soviet.fuel_tons / 100.0,
            self.supply_soviet.ammo_tons / 100.0,
            self.terrain.defense_bonus,
            self.terrain.movement_cost / 4.0,
            self.weather.mud_factor,
            self.terrain.mines_density,
        ], dtype=np.float32)


PHYSICS_OBS_PER_SECTOR = 20


# ─────────────────────────────────────────────────────────────────────────── #
# YAML Configuration — Map-agnostic terrain + climate loader                  #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass
class MapPhysicsConfig:
    """Configuration loaded from YAML for a specific map / theater."""
    name: str = "default"
    n_sectors: int = 9
    # Climate
    climate_type: str = "continental"    # continental, desert, tropical, arctic, maritime
    temperature_curve: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_TEMP_CURVE_C))
    base_humidity: float = 60.0
    base_wind_ms: float = 5.0
    steps_per_day: int = 4
    # Per-sector terrain
    sectors: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # Supply routes (list of dicts with from, to, type, capacity_tpd)
    supply_routes: List[Dict[str, Any]] = field(default_factory=list)
    # Faction config
    factions: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def _resolve_terrain_type(name: str) -> TerrainType:
    """Resolve a terrain type name string to the enum value."""
    name_upper = name.upper().replace(" ", "_").replace("-", "_")
    try:
        return TerrainType[name_upper]
    except KeyError:
        # Fuzzy match
        for t in TerrainType:
            if t.name == name_upper or t.name.startswith(name_upper):
                return t
        return TerrainType.OPEN


def _resolve_soil_type(name: str) -> SoilType:
    """Resolve a soil type name string to the enum value."""
    name_upper = name.upper().replace(" ", "_").replace("-", "_")
    try:
        return SoilType[name_upper]
    except KeyError:
        return SoilType.LOAM


def load_map_physics(yaml_path: Union[str, Path]) -> MapPhysicsConfig:
    """
    Load map physics configuration from a YAML file.

    Expected YAML structure:
    ```yaml
    map_physics:
      name: "Moscow 1941"
      climate:
        type: continental
        temperature_curve: {0: 5.0, 50: 2.0, ...}
        humidity: 60
        wind_ms: 5.0
        steps_per_day: 4
      sectors:
        0:
          name: Moscow
          terrain: URBAN
          elevation_m: 156
          cover: 0.7
          soil: CLAY
          features: {river: false, rail: true, road: true, airfield: true}
          initial_supply:
            axis: {fuel: 5, ammo: 5, food: 5, medical: 1, spare_parts: 2, winter_gear: 0}
            soviet: {fuel: 100, ammo: 100, food: 100, medical: 20, spare_parts: 30, winter_gear: 15}
          fortification: 0.4
        ...
      supply_routes:
        - {from: 0, to: 1, type: rail, capacity_tpd: 50}
        - {from: 0, to: 2, type: road, capacity_tpd: 20}
      factions:
        axis: {winterized: false, depot_sectors: [5, 6, 7, 8]}
        soviet: {winterized: true, depot_sectors: [0, 1, 3]}
    ```
    """
    import yaml

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Map physics config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    mp = raw.get("map_physics", raw)  # Allow top-level or nested

    # Climate
    climate = mp.get("climate", {})
    temp_curve = climate.get("temperature_curve", dict(DEFAULT_TEMP_CURVE_C))
    # Ensure keys are int
    temp_curve = {int(k): float(v) for k, v in temp_curve.items()}

    config = MapPhysicsConfig(
        name=mp.get("name", "default"),
        n_sectors=mp.get("n_sectors", len(mp.get("sectors", {}))),
        climate_type=climate.get("type", "continental"),
        temperature_curve=temp_curve,
        base_humidity=climate.get("humidity", 60.0),
        base_wind_ms=climate.get("wind_ms", 5.0),
        steps_per_day=climate.get("steps_per_day", 4),
        sectors={},
        supply_routes=mp.get("supply_routes", []),
        factions=mp.get("factions", {}),
    )

    # Parse sectors
    for sid_str, sdata in mp.get("sectors", {}).items():
        sid = int(sid_str)
        config.sectors[sid] = {
            "name": sdata.get("name", f"Sector {sid}"),
            "terrain": _resolve_terrain_type(sdata.get("terrain", "OPEN")),
            "elevation_m": float(sdata.get("elevation_m", 100)),
            "cover": float(sdata.get("cover", COVER_FACTOR.get(
                _resolve_terrain_type(sdata.get("terrain", "OPEN")), 0.1))),
            "soil": _resolve_soil_type(sdata.get("soil", "LOAM")),
            "features": sdata.get("features", {}),
            "initial_supply": sdata.get("initial_supply", {}),
            "fortification": float(sdata.get("fortification", 0.0)),
            "mines": float(sdata.get("mines", 0.0)),
            "obstacles": float(sdata.get("obstacles", 0.0)),
        }

    if not config.sectors:
        config.n_sectors = max(config.n_sectors, 1)

    return config


def _build_terrain_from_config(sid: int, sdata: Dict[str, Any]) -> TerrainState:
    """Build a TerrainState from parsed sector config dict."""
    feat_raw = sdata.get("features", {})
    features = TerrainFeatures(
        has_river=feat_raw.get("river", False),
        has_road=feat_raw.get("road", False),
        has_rail=feat_raw.get("rail", False),
        has_bridge=feat_raw.get("bridge", False),
        has_airfield=feat_raw.get("airfield", False),
        has_port=feat_raw.get("port", False),
        vegetation_density=float(feat_raw.get("vegetation_density", 0.0)),
        water_table_depth_m=float(feat_raw.get("water_table_depth_m", 5.0)),
    )
    # Ensure terrain_type is TerrainType enum
    terrain = sdata["terrain"]
    if isinstance(terrain, str):
        terrain = _resolve_terrain_type(terrain)
    return TerrainState(
        sector_id=sid,
        terrain_type=terrain,
        elevation_m=sdata["elevation_m"],
        cover_factor=sdata["cover"],
        soil=sdata.get("soil", SoilType.LOAM),
        features=features,
        fortification=sdata.get("fortification", 0.0),
        mines_density=sdata.get("mines", 0.0),
        obstacle_density=sdata.get("obstacles", 0.0),
    )


def _build_supply_from_config(raw: Dict[str, Any]) -> SupplyState:
    """Build a SupplyState from a supply config dict."""
    return SupplyState(
        fuel_tons=float(raw.get("fuel", 50)),
        ammo_tons=float(raw.get("ammo", 50)),
        food_tons=float(raw.get("food", 50)),
        medical_tons=float(raw.get("medical", 10)),
        spare_parts_tons=float(raw.get("spare_parts", 15)),
        winter_gear_tons=float(raw.get("winter_gear", 5)),
        rail_capacity_tpd=float(raw.get("rail_capacity_tpd", 50)),
        road_capacity_tpd=float(raw.get("road_capacity_tpd", 20)),
        air_capacity_tpd=float(raw.get("air_capacity_tpd", 0)),
        water_capacity_tpd=float(raw.get("water_capacity_tpd", 0)),
        depot_max_tons=float(raw.get("depot_max_tons", 500)),
    )


def initialize_physics(config: MapPhysicsConfig) -> List[SectorPhysicsState]:
    """
    Create initial physics state for all sectors from a MapPhysicsConfig.

    This is the primary entry point for map-agnostic initialization.
    """
    states = []
    for sid in range(config.n_sectors):
        sdata = config.sectors.get(sid)
        if sdata is None:
            # Auto-generate a default sector
            sdata = {
                "name": f"Sector {sid}",
                "terrain": TerrainType.OPEN,
                "elevation_m": 100.0,
                "cover": 0.1,
                "soil": SoilType.LOAM,
                "features": {},
                "initial_supply": {},
                "fortification": 0.0,
            }

        terrain = _build_terrain_from_config(sid, sdata)
        weather = WeatherState()

        # Supply per faction from config
        isup = sdata.get("initial_supply", {})
        supply_axis = _build_supply_from_config(isup.get("axis", {}))
        supply_soviet = _build_supply_from_config(isup.get("soviet", {}))

        states.append(SectorPhysicsState(
            sector_id=sid,
            terrain=terrain,
            weather=weather,
            supply_axis=supply_axis,
            supply_soviet=supply_soviet,
        ))

    return states


# ─────────────────────────────────────────────────────────────────────────── #
# Backward-compat: Moscow defaults (used when no YAML is provided)            #
# ─────────────────────────────────────────────────────────────────────────── #

# Legacy Moscow sector terrain definitions (kept for backward compat)
MOSCOW_SECTOR_TERRAIN: Dict[int, Dict] = {
    0: {"name": "Moscow",          "type": TerrainType.URBAN,  "elevation_m": 156, "cover": 0.7,
        "river": False, "rail_hub": True, "soil": SoilType.CLAY},
    1: {"name": "Yaroslavl",       "type": TerrainType.FOREST, "elevation_m": 98,  "cover": 0.5,
        "river": True,  "rail_hub": True, "soil": SoilType.LOAM},
    2: {"name": "Tula",            "type": TerrainType.HILLS,  "elevation_m": 220, "cover": 0.4,
        "river": False, "rail_hub": True, "soil": SoilType.CLAY},
    3: {"name": "Soviet Reserves", "type": TerrainType.FOREST, "elevation_m": 130, "cover": 0.6,
        "river": False, "rail_hub": False, "soil": SoilType.LOAM},
    4: {"name": "Bryansk",         "type": TerrainType.FOREST, "elevation_m": 190, "cover": 0.5,
        "river": True,  "rail_hub": True, "soil": SoilType.CLAY},
    5: {"name": "Vyazma",          "type": TerrainType.OPEN,   "elevation_m": 200, "cover": 0.2,
        "river": False, "rail_hub": True, "soil": SoilType.LOAM},
    6: {"name": "Smolensk",        "type": TerrainType.URBAN,  "elevation_m": 235, "cover": 0.6,
        "river": True,  "rail_hub": True, "soil": SoilType.CLAY},
    7: {"name": "Kalinin",         "type": TerrainType.MARSH,  "elevation_m": 130, "cover": 0.3,
        "river": True,  "rail_hub": True, "soil": SoilType.PEAT},
    8: {"name": "Klin",            "type": TerrainType.OPEN,   "elevation_m": 160, "cover": 0.2,
        "river": False, "rail_hub": False, "soil": SoilType.LOAM},
}


def initialize_sector_physics(n_sectors: int = 9) -> List[SectorPhysicsState]:
    """
    Backward-compatible initialization using hardcoded Moscow defaults.

    Prefer initialize_physics(config) with YAML for new maps.
    """
    states = []
    for sid in range(n_sectors):
        meta = MOSCOW_SECTOR_TERRAIN.get(sid, MOSCOW_SECTOR_TERRAIN[0])

        features = TerrainFeatures(
            has_river=meta.get("river", False),
            has_rail=meta.get("rail_hub", False),
            has_road=True,  # All Moscow sectors have roads
        )
        terrain = TerrainState(
            sector_id=sid,
            terrain_type=meta["type"],
            elevation_m=meta["elevation_m"],
            cover_factor=meta["cover"],
            soil=meta.get("soil", SoilType.LOAM),
            features=features,
        )

        weather = WeatherState()

        if sid >= 5:  # Axis sectors
            supply_axis = SupplyState(fuel_tons=80, ammo_tons=90, food_tons=70,
                                      medical_tons=15, spare_parts_tons=25, winter_gear_tons=3)
            supply_soviet = SupplyState(fuel_tons=10, ammo_tons=10, food_tons=10,
                                        medical_tons=2, spare_parts_tons=3, winter_gear_tons=1)
        elif sid <= 3:  # Soviet sectors
            supply_axis = SupplyState(fuel_tons=5, ammo_tons=5, food_tons=5,
                                      medical_tons=1, spare_parts_tons=2, winter_gear_tons=0)
            supply_soviet = SupplyState(fuel_tons=100, ammo_tons=100, food_tons=100,
                                        medical_tons=20, spare_parts_tons=30, winter_gear_tons=15)
        else:  # Contested
            supply_axis = SupplyState(fuel_tons=40, ammo_tons=50, food_tons=40,
                                      medical_tons=8, spare_parts_tons=12, winter_gear_tons=2)
            supply_soviet = SupplyState(fuel_tons=50, ammo_tons=50, food_tons=50,
                                        medical_tons=10, spare_parts_tons=15, winter_gear_tons=8)

        states.append(SectorPhysicsState(
            sector_id=sid,
            terrain=terrain,
            weather=weather,
            supply_axis=supply_axis,
            supply_soviet=supply_soviet,
        ))

    return states


# ─────────────────────────────────────────────────────────────────────────── #
# Step Functions                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def step_all_physics(
    states: List[SectorPhysicsState],
    step: int,
    rng: np.random.Generator,
    combat_sectors: Optional[List[int]] = None,
    unit_counts_per_sector: Optional[Dict[int, Dict[str, int]]] = None,
    sabotage_factors: Optional[Dict[int, float]] = None,
    config: Optional[MapPhysicsConfig] = None,
) -> List[SectorPhysicsState]:
    """
    Advance all sector physics by one step.

    If config is provided, uses its climate settings; otherwise uses defaults.
    Returns updated list of SectorPhysicsState.
    """
    combat_sectors = combat_sectors or []
    unit_counts = unit_counts_per_sector or {}
    sabotage = sabotage_factors or {}

    # Climate params from config or defaults
    temp_curve = config.temperature_curve if config else None
    climate_type = config.climate_type if config else "continental"
    base_humidity = config.base_humidity if config else 60.0
    base_wind_ms = config.base_wind_ms if config else 5.0
    steps_per_day = config.steps_per_day if config else 4

    new_states = []

    for state in states:
        sid = state.sector_id

        # 1. Weather update
        new_weather = step_weather(
            state.weather, step, rng,
            temp_curve=temp_curve,
            climate_type=climate_type,
            base_humidity=base_humidity,
            base_wind_ms=base_wind_ms,
            steps_per_day=steps_per_day,
        )

        # 2. Supply consumption
        axis_units = unit_counts.get(sid, {}).get("axis", {})
        soviet_units = unit_counts.get(sid, {}).get("soviet", {})
        in_combat = sid in combat_sectors

        new_supply_axis = compute_supply_consumption(
            state.supply_axis, axis_units, in_combat, new_weather)
        new_supply_soviet = compute_supply_consumption(
            state.supply_soviet, soviet_units, in_combat, new_weather)

        # 3. Supply delivery
        has_rail = state.terrain.features.has_rail
        has_airfield = state.terrain.features.has_airfield
        has_port = state.terrain.features.has_port

        # Distance heuristic: use config supply routes if available, else positional
        n = len(states)
        axis_dist = 50 + sid * 30
        soviet_dist = 20 + max(0, (n - 1 - sid)) * 25
        sab = sabotage.get(sid, 0.0)

        new_supply_axis = compute_supply_delivery(
            new_supply_axis, has_rail, axis_dist, new_weather, sab,
            has_airfield=has_airfield, has_port=has_port)
        new_supply_soviet = compute_supply_delivery(
            new_supply_soviet, has_rail, soviet_dist, new_weather, sab * 0.3,
            has_airfield=has_airfield, has_port=has_port)

        # 4. Terrain evolution
        new_terrain = state.terrain
        if sid not in combat_sectors:
            new_entrench = min(1.0, state.terrain.entrenchment + state.terrain.dig_rate)
            new_terrain = state.terrain.copy_with(entrenchment=new_entrench)
        else:
            new_fort = max(0, state.terrain.fortification - 0.005)
            new_entrench = max(0, state.terrain.entrenchment - 0.01)
            # Combat can reduce obstacles (breaching)
            new_obs = max(0, state.terrain.obstacle_density - 0.003)
            new_terrain = state.terrain.copy_with(
                fortification=new_fort, entrenchment=new_entrench,
                obstacle_density=new_obs)

        new_states.append(SectorPhysicsState(
            sector_id=sid,
            terrain=new_terrain,
            weather=new_weather,
            supply_axis=new_supply_axis,
            supply_soviet=new_supply_soviet,
            step=step,
        ))

    return new_states


def get_physics_obs(states: List[SectorPhysicsState], max_N: int = 12) -> NDArray[np.float32]:
    """
    Flatten all sector physics into a single observation vector.

    Returns: (PHYSICS_OBS_PER_SECTOR * max_N,) float32 array
    """
    obs = np.zeros(PHYSICS_OBS_PER_SECTOR * max_N, dtype=np.float32)
    for state in states:
        if state.sector_id < max_N:
            start = state.sector_id * PHYSICS_OBS_PER_SECTOR
            obs[start:start + PHYSICS_OBS_PER_SECTOR] = state.to_obs_vector()
    return obs
