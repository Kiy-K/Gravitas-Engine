"""
weather_bridge.py — Unified weather system bridging physics.py to the LLM game engine.

Covers land, sea, and air weather in one step. Uses the existing physics.py
primitives (WeatherState, step_weather, WeatherCondition) but adds:
  - Per-sea-zone sea state (wave height, current, storm)
  - Per-air-zone cloud/wind conditions
  - Cross-system effects (weather → naval, air, economy, ground combat)
  - English Channel seasonal model (maritime climate)

The Channel has its own micro-climate:
  - Autumn (turns 0-100): rain, fog, moderate winds, 8-15°C
  - Winter (turns 100-200): storms, gales, cold rain, 0-8°C
  - Spring (turns 200-300): clearing, variable, 5-15°C
  - Summer (turns 300-400): warm, occasional thunderstorms, 15-25°C
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from extensions.military.physics import (
    WeatherState, WeatherCondition, TimeOfDay, Season,
    step_weather, _sc,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# Channel Climate — English Channel / North Sea / Irish Sea                   #
# ═══════════════════════════════════════════════════════════════════════════ #

# Temperature curves for Air Strip One (maritime climate, milder than continental)
CHANNEL_TEMP_CURVE = {
    0: 12.0,     # Early autumn — mild
    50: 8.0,     # Late autumn — cooling
    100: 4.0,    # Early winter — cold
    150: 1.0,    # Deep winter — Channel storms, near freezing
    200: 5.0,    # Early spring — thawing
    250: 10.0,   # Late spring — warming
    300: 18.0,   # Summer — warm
    350: 20.0,   # Peak summer
    400: 15.0,   # Late summer → autumn
    500: 8.0,    # Back to autumn
    600: 4.0,    # Winter again
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Unified Weather State (land + sea + air per region)                         #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class RegionWeather:
    """Weather for a single cluster or zone, covering land/sea/air."""
    # Land conditions (from physics.py WeatherState)
    land: WeatherState = field(default_factory=WeatherState)

    # Sea conditions (affects naval operations)
    sea_state: float = 0.2          # [0,1] 0=calm, 0.3=moderate, 0.6=rough, 0.8+=storm
    wave_height_m: float = 1.0      # meters
    current_knots: float = 0.5      # tidal current
    sea_surface_temp_c: float = 12.0

    # Air conditions (affects air operations)
    cloud_ceiling_m: float = 3000.0  # meters — below 500m = grounded
    cloud_cover: float = 0.3         # [0,1] overcast fraction
    wind_aloft_ms: float = 10.0      # wind at altitude (jet stream effect)
    turbulence: float = 0.1          # [0,1] affects bombing accuracy
    icing_risk: float = 0.0          # [0,1] ice on wings at altitude

    def copy(self) -> "RegionWeather":
        return RegionWeather(
            land=WeatherState(
                temperature_c=self.land.temperature_c,
                condition=self.land.condition,
                snow_depth_cm=self.land.snow_depth_cm,
                wind_speed_ms=self.land.wind_speed_ms,
                visibility_km=self.land.visibility_km,
                ground_frozen=self.land.ground_frozen,
                mud_factor=self.land.mud_factor,
                humidity_pct=self.land.humidity_pct,
                time_of_day=self.land.time_of_day,
                season=self.land.season,
            ),
            sea_state=self.sea_state,
            wave_height_m=self.wave_height_m,
            current_knots=self.current_knots,
            sea_surface_temp_c=self.sea_surface_temp_c,
            cloud_ceiling_m=self.cloud_ceiling_m,
            cloud_cover=self.cloud_cover,
            wind_aloft_ms=self.wind_aloft_ms,
            turbulence=self.turbulence,
            icing_risk=self.icing_risk,
        )


@dataclass
class GameWeather:
    """Complete weather state for the entire theatre."""
    cluster_weather: List[RegionWeather]   # per land cluster
    sea_zone_weather: List[RegionWeather]  # per sea zone
    turn: int = 0

    def get_cluster(self, cid: int) -> RegionWeather:
        if cid < len(self.cluster_weather):
            return self.cluster_weather[cid]
        return RegionWeather()

    def get_sea_zone(self, zid: int) -> RegionWeather:
        if zid < len(self.sea_zone_weather):
            return self.sea_zone_weather[zid]
        return RegionWeather()

    def copy(self) -> "GameWeather":
        return GameWeather(
            cluster_weather=[w.copy() for w in self.cluster_weather],
            sea_zone_weather=[w.copy() for w in self.sea_zone_weather],
            turn=self.turn,
        )


# ═══════════════════════════════════════════════════════════════════════════ #
# Step Weather                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_game_weather(
    weather: GameWeather,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> GameWeather:
    """Advance weather for all clusters and sea zones by one step."""
    turn = weather.turn

    # ── Land weather (per cluster) ────────────────────────────────────── #
    for i, rw in enumerate(weather.cluster_weather):
        new_land = step_weather(
            rw.land, turn, rng,
            temp_curve=CHANNEL_TEMP_CURVE,
            climate_type="maritime",
            base_humidity=70.0,   # England is wet
            base_wind_ms=6.0,
            steps_per_day=4,
        )
        rw.land = new_land

        # Derive air conditions from land weather
        rw.cloud_cover = _sc(new_land.humidity_pct / 100.0 * 0.8 + rng.uniform(-0.1, 0.1), 0.0, 1.0)
        rw.cloud_ceiling_m = max(200, 5000 - new_land.humidity_pct * 40 + rng.uniform(-500, 500))
        rw.wind_aloft_ms = max(0, new_land.wind_speed_ms * 2.0 + rng.uniform(-3, 3))
        rw.turbulence = _sc(new_land.wind_speed_ms / 30.0 + rng.uniform(-0.05, 0.05), 0.0, 1.0)
        rw.icing_risk = _sc((0.0 - new_land.temperature_c) / 20.0 if new_land.temperature_c < 0 else 0.0, 0.0, 1.0)

    # ── Sea weather (per sea zone) ────────────────────────────────────── #
    for i, rw in enumerate(weather.sea_zone_weather):
        # Sea state driven by wind + season + Channel geography
        base_wind = 6.0 + rng.normal(0, 2)
        # Channel storms are worse in winter
        season_factor = 1.0
        season_idx = (turn // 100) % 4
        if season_idx == 1:   # winter
            season_factor = 1.6
            base_wind += 4.0
        elif season_idx == 0:  # autumn
            season_factor = 1.3
            base_wind += 2.0

        wind = max(0, base_wind + rng.normal(0, 3)) * season_factor
        rw.land = step_weather(
            rw.land, turn, rng,
            temp_curve=CHANNEL_TEMP_CURVE,
            climate_type="maritime",
            base_humidity=75.0,
            base_wind_ms=wind,
        )

        # Sea state: driven by wind speed (Beaufort scale approximation)
        # Force 0-3: calm (0-0.2), Force 4-5: moderate (0.2-0.4),
        # Force 6-7: rough (0.4-0.7), Force 8+: storm (0.7+)
        beaufort = wind / 3.0  # rough conversion
        rw.sea_state = _sc(beaufort / 12.0 + rng.uniform(-0.05, 0.05), 0.0, 1.0)
        rw.wave_height_m = max(0.1, beaufort * 0.5 + rng.uniform(-0.5, 0.5))
        rw.current_knots = 0.5 + abs(rng.normal(0, 0.3))

        # Sea surface temperature (Channel: 8-18°C depending on season)
        rw.sea_surface_temp_c = 8.0 + 5.0 * np.sin(2 * np.pi * (turn - 100) / 400) + rng.uniform(-1, 1)

        # Air over sea
        rw.cloud_cover = _sc(rw.land.humidity_pct / 100.0 * 0.9 + rng.uniform(-0.1, 0.1), 0.0, 1.0)
        rw.cloud_ceiling_m = max(100, 4000 - rw.land.humidity_pct * 35 + rng.uniform(-300, 300))
        rw.wind_aloft_ms = wind * 1.5
        rw.turbulence = _sc(wind / 25.0, 0.0, 1.0)

    weather.turn += 1
    return weather


# ═══════════════════════════════════════════════════════════════════════════ #
# Cross-System Effects                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_weather_effects(
    weather: GameWeather,
    game,  # GameState (duck-typed to avoid circular import)
) -> Dict[str, float]:
    """
    Apply weather effects to all game subsystems.

    Returns feedback dict with weather-related events.
    """
    feedback = {"storms": 0, "grounded_air": 0, "frozen_ports": 0}

    # ── Sea zones: update naval sea_state from weather ────────────────── #
    for i, zone in enumerate(game.naval.sea_zones):
        if i < len(weather.sea_zone_weather):
            sw = weather.sea_zone_weather[i]
            zone.sea_state = sw.sea_state
            if sw.sea_state > 0.7:
                feedback["storms"] += 1

    # ── Air zones: update cloud cover from weather ────────────────────── #
    for i, zone in enumerate(game.air.air_zones):
        if i < len(weather.cluster_weather):
            cw = weather.cluster_weather[i]
            zone.cloud_cover = cw.cloud_cover
            if cw.cloud_ceiling_m < 500:
                feedback["grounded_air"] += 1

    # ── Ground: weather affects cluster data ──────────────────────────── #
    for i in range(min(len(weather.cluster_weather), len(game.cluster_data))):
        cw = weather.cluster_weather[i]
        # Mud/snow reduces resource level slightly (logistics disruption)
        if cw.land.mud_factor > 0.3:
            game.cluster_data[i, 2] = max(0.0, game.cluster_data[i, 2] - 0.005)
        if cw.land.snow_depth_cm > 20:
            game.cluster_data[i, 2] = max(0.0, game.cluster_data[i, 2] - 0.003)
        # Extreme cold increases hazard slightly
        if cw.land.temperature_c < -5:
            game.cluster_data[i, 1] = min(1.0, game.cluster_data[i, 1] + 0.002)

    # ── Economy: weather affects production ────────────────────────────── #
    for ce in (game.economy.clusters if game.economy else []):
        cid = ce.cluster_id
        if cid < len(weather.cluster_weather):
            cw = weather.cluster_weather[cid]
            # Storms reduce industrial output
            if cw.land.condition in (WeatherCondition.BLIZZARD, WeatherCondition.THUNDERSTORM,
                                     WeatherCondition.ICE_STORM):
                # Reduce effective capacity temporarily
                ce.infrastructure = max(0.3, ce.infrastructure - 0.005)
            # Good weather slowly repairs
            elif cw.land.condition == WeatherCondition.CLEAR:
                ce.infrastructure = min(1.0, ce.infrastructure + 0.001)

    return feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Weather Description for Turn Summaries                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

def describe_weather(weather: GameWeather, cluster_names: List[str]) -> str:
    """Generate a concise weather report for the turn summary (~50-80 tokens)."""
    lines = []

    # Find the most dramatic weather
    worst_land = None
    worst_sea = None
    worst_idx = 0
    worst_sea_idx = 0

    for i, cw in enumerate(weather.cluster_weather):
        if worst_land is None or cw.land.wind_speed_ms > worst_land.land.wind_speed_ms:
            worst_land = cw
            worst_idx = i

    for i, sw in enumerate(weather.sea_zone_weather):
        if worst_sea is None or sw.sea_state > worst_sea.sea_state:
            worst_sea = sw
            worst_sea_idx = i

    # Land weather summary
    if worst_land:
        w = worst_land.land
        loc = cluster_names[worst_idx] if worst_idx < len(cluster_names) else f"Sector {worst_idx}"
        temp_desc = f"{w.temperature_c:.0f}°C"
        cond = w.condition.name.replace("_", " ").lower()

        if w.condition in (WeatherCondition.BLIZZARD, WeatherCondition.HEAVY_SNOW, WeatherCondition.ICE_STORM):
            lines.append(f"SEVERE WEATHER: {cond} at {loc}, {temp_desc}. Operations impaired.")
        elif w.condition in (WeatherCondition.THUNDERSTORM, WeatherCondition.HEAVY_RAIN):
            lines.append(f"Weather: {cond} at {loc}, {temp_desc}. Visibility poor.")
        elif w.condition == WeatherCondition.FOG:
            lines.append(f"Dense fog across the theatre. Visibility {w.visibility_km:.0f}km.")
        else:
            lines.append(f"Weather: {cond}, {temp_desc}, wind {w.wind_speed_ms:.0f}m/s.")

    # Sea state
    if worst_sea and worst_sea.sea_state > 0.3:
        sea_zone_names = ["Dover Strait", "Western Channel", "North Sea", "Irish Sea"]
        zname = sea_zone_names[worst_sea_idx] if worst_sea_idx < len(sea_zone_names) else f"Zone {worst_sea_idx}"
        if worst_sea.sea_state > 0.7:
            lines.append(f"CHANNEL STORM: {zname} Force 8+. All crossings CANCELLED.")
        elif worst_sea.sea_state > 0.5:
            lines.append(f"Rough seas in {zname}. Wave height {worst_sea.wave_height_m:.1f}m. Crossings hazardous.")
        else:
            lines.append(f"Moderate seas in {zname}. Wave height {worst_sea.wave_height_m:.1f}m.")

    # Air conditions
    grounded = sum(1 for cw in weather.cluster_weather if cw.cloud_ceiling_m < 500)
    if grounded > 3:
        lines.append(f"Low cloud ceiling across {grounded} sectors. Air operations limited.")

    if not lines:
        lines.append("Clear skies. Good conditions for operations.")

    return " ".join(lines)


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_game_weather(
    n_clusters: int,
    n_sea_zones: int,
    rng: np.random.Generator,
    start_turn: int = 0,
) -> GameWeather:
    """Initialize weather for all clusters and sea zones."""
    cluster_weather = []
    for _ in range(n_clusters):
        land = step_weather(
            WeatherState(), start_turn, rng,
            temp_curve=CHANNEL_TEMP_CURVE,
            climate_type="maritime",
            base_humidity=70.0,
            base_wind_ms=6.0,
        )
        cluster_weather.append(RegionWeather(
            land=land,
            cloud_cover=land.humidity_pct / 100.0 * 0.7,
            cloud_ceiling_m=3000.0 + rng.uniform(-500, 500),
        ))

    sea_zone_weather = []
    for _ in range(n_sea_zones):
        land = step_weather(
            WeatherState(), start_turn, rng,
            temp_curve=CHANNEL_TEMP_CURVE,
            climate_type="maritime",
            base_humidity=75.0,
            base_wind_ms=8.0,
        )
        sea_zone_weather.append(RegionWeather(
            land=land,
            sea_state=rng.uniform(0.1, 0.3),
            wave_height_m=rng.uniform(0.5, 2.0),
            cloud_cover=land.humidity_pct / 100.0 * 0.8,
        ))

    return GameWeather(
        cluster_weather=cluster_weather,
        sea_zone_weather=sea_zone_weather,
        turn=start_turn,
    )
