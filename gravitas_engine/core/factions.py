"""
Faction aggregation functions.

All nine macro variables (E, C, F, I, L, M, R, V, Exh) are derived from the
faction state distribution via the following pure functions.  No macro variable
evolves independently except Exhaustion (handled in exhaustion.py).

Aggregation equations (all variables ∈ [0, 1]):

  L  = Σ(P_i · Coh_i)       / Σ(P_i)
  C  = Σ(P_i · Coh_i²)      / Σ(P_i)
  I  = Σ(P_i · Rad_i · (1−Coh_i)) / Σ(P_i)
  M  = Σ(P_i · Mem_i · Rad_i)     / Σ(P_i)
  F  = 1 − exp(−λ_F · Gini(P))
  R  = 1 − L
  E  = L · (1 − F)
  V  = tanh(κ_V · M · (1−Exh) · (1+I))
  Exh — evolved by its own ODE (preserved across aggregation calls)
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from .parameters import SystemParameters
from .state import FactionState, RegimeState, SystemState

# ── Cython fast-path imports ──────────────────────────────────────────────── #
try:
    from ._kernels import compute_gini_c as _compute_gini_c
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


# --------------------------------------------------------------------------- #
# Helper                                                                       #
# --------------------------------------------------------------------------- #


def _safe_weighted_mean(
    weights: NDArray[np.float64],
    values: NDArray[np.float64],
) -> float:
    """Return Σ(w_i · v_i) / Σ(w_i), falling back to 0 when Σ(w_i) = 0."""
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    return float(np.clip(np.sum(weights * values) / total, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Gini coefficient                                                              #
# --------------------------------------------------------------------------- #


def compute_gini(powers: NDArray[np.float64]) -> float:
    """Compute the Gini coefficient of the faction power distribution.

    Formula:
        Gini(P) = (1 / (2 · N · μ)) · Σ_i Σ_j |P_i − P_j|
    where μ = mean(P_i).

    Returns:
        Gini ∈ [0, 1 − 1/N]  (analytically bounded below 1 for N ≥ 2).
    """
    if _USE_CYTHON:
        return _compute_gini_c(np.ascontiguousarray(powers, dtype=np.float64))
    n = len(powers)
    if n < 2:
        return 0.0
    total = float(np.sum(powers))
    if total <= 0.0:
        return 0.0
    # Vectorised absolute difference sum
    diff_sum = float(np.sum(np.abs(powers[:, None] - powers[None, :])))
    gini = diff_sum / (2.0 * n * total)
    return float(np.clip(gini, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Individual aggregation functions                                              #
# --------------------------------------------------------------------------- #


def compute_legitimacy(state: RegimeState) -> float:
    """L = Σ(P_i · Coh_i) / Σ(P_i).

    Power-weighted mean cohesion.  Measures the extent to which the dominant
    factions maintain internal organisational coherence.

    Returns:
        L ∈ [0, 1].
    """
    return _safe_weighted_mean(
        state.get_faction_powers(), state.get_faction_cohesions()
    )


def compute_cohesion(state: RegimeState) -> float:
    """C = Σ(P_i · Coh_i²) / Σ(P_i).

    Power-weighted mean of squared cohesion.  Squaring emphasises high-cohesion
    factions more strongly than L does, making C < L in heterogeneous systems.

    Returns:
        C ∈ [0, 1].
    """
    cohesions = state.get_faction_cohesions()
    return _safe_weighted_mean(state.get_faction_powers(), cohesions ** 2)


def compute_fragmentation(
    state: RegimeState, lambda_frag: float
) -> float:
    """F = 1 − exp(−λ_F · Gini(P)).

    Saturating transform of the Gini coefficient so that F ∈ [0, 1) always.
    As Gini → max (=(N−1)/N), F approaches 1 − exp(−λ_F·(N−1)/N) < 1.

    Args:
        state:       Current regime state.
        lambda_frag: Fragmentation sensitivity (> 0).

    Returns:
        F ∈ [0, 1).
    """
    gini = compute_gini(state.get_faction_powers())
    frag = 1.0 - np.exp(-lambda_frag * gini)
    return float(np.clip(frag, 0.0, 1.0))


def compute_instability(state: RegimeState) -> float:
    """I = Σ(P_i · Rad_i · (1−Coh_i)) / Σ(P_i).

    Power-weighted product of radicalization and lack of cohesion.

    Returns:
        I ∈ [0, 1].
    """
    powers = state.get_faction_powers()
    rads = state.get_faction_radicalizations()
    cohs = state.get_faction_cohesions()
    return _safe_weighted_mean(powers, rads * (1.0 - cohs))


def compute_mobilization(state: RegimeState) -> float:
    """M = Σ(P_i · Mem_i · Rad_i) / Σ(P_i).

    Power-weighted grievance memory activated by radicalization.

    Returns:
        M ∈ [0, 1].
    """
    powers = state.get_faction_powers()
    mems = state.get_faction_memories()
    rads = state.get_faction_radicalizations()
    return _safe_weighted_mean(powers, mems * rads)


def compute_repression(legitimacy: float) -> float:
    """R = 1 − L.

    Proxy: when the regime has low legitimacy it compensates via repression.

    Args:
        legitimacy: Current L value.

    Returns:
        R ∈ [0, 1].
    """
    return float(np.clip(1.0 - legitimacy, 0.0, 1.0))


def compute_elite_alignment(legitimacy: float, fragmentation: float) -> float:
    """E = L · (1 − F).

    Elite factions align with the regime when legitimacy is high **and** the
    power landscape is concentrated (low fragmentation).

    Args:
        legitimacy:    Current L value.
        fragmentation: Current F value.

    Returns:
        E ∈ [0, 1].
    """
    return float(np.clip(legitimacy * (1.0 - fragmentation), 0.0, 1.0))


def compute_volatility(
    mobilization: float,
    instability: float,
    exhaustion: float,
    kappa_v: float,
) -> float:
    """V = tanh(κ_V · M · (1−Exh) · (1+I)).

    Bounded cascade measure.  tanh ensures V < 1 for all finite inputs.
    Exhaustion acts as a brake: as Exh → 1, V → 0 regardless of M and I.

    Stability: tanh(·) ≤ tanh(2·κ_V) < 1 for typical parameter ranges.

    Args:
        mobilization: Current M value.
        instability:  Current I value.
        exhaustion:   Current Exh value.
        kappa_v:      Amplification factor (> 0).

    Returns:
        V ∈ [0, 1).
    """
    argument = kappa_v * mobilization * (1.0 - exhaustion) * (1.0 + instability)
    volatility = float(np.tanh(argument))
    return float(np.clip(volatility, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Full system-state recomputation                                               #
# --------------------------------------------------------------------------- #


def recompute_system_state(
    state: RegimeState,
    params: SystemParameters,
) -> RegimeState:
    """Re-derive all 8 algebraic macro variables from the current faction array.

    Exhaustion is **preserved** from state.system.exhaustion — it is governed
    by its own ODE and not re-derived here.

    Args:
        state:  Current regime state (faction arrays are the source of truth).
        params: System parameters (lambda_frag, kappa_v).

    Returns:
        Updated RegimeState with freshly computed SystemState.
    """
    leg = compute_legitimacy(state)
    coh = compute_cohesion(state)
    frag = compute_fragmentation(state, params.lambda_frag)
    # Phase 3: blend with district-level fragmentation when hierarchical
    if getattr(state, "hierarchical", None) is not None and getattr(params, "use_hierarchy", False):
        from .hierarchical_coupling import global_fragmentation_from_districts
        frag = global_fragmentation_from_districts(state.hierarchical, frag, params)
    inst = compute_instability(state)
    mob = compute_mobilization(state)
    rep = compute_repression(leg)
    elite = compute_elite_alignment(leg, frag)
    exh = state.system.exhaustion  # preserved from previous step
    vol = compute_volatility(mob, inst, exh, params.kappa_v)
    # Phase 3: volatility bump from district unrest variance
    if getattr(state, "hierarchical", None) is not None and getattr(params, "use_hierarchy", False):
        from .hierarchical_coupling import volatility_bump_from_districts
        vol = float(np.clip(vol + volatility_bump_from_districts(state.hierarchical, params), 0.0, 1.0))

    new_system = SystemState(
        legitimacy=leg,
        cohesion=coh,
        fragmentation=frag,
        instability=inst,
        mobilization=mob,
        repression=rep,
        elite_alignment=elite,
        volatility=vol,
        exhaustion=exh,
        state_gdp=state.system.state_gdp,
        pillars=state.system.pillars,
    )
    return state.copy_with_system(new_system)


# --------------------------------------------------------------------------- #
# Faction initialisation factories                                              #
# --------------------------------------------------------------------------- #


def create_balanced_factions(n_factions: int) -> List[FactionState]:
    """Create N equal-share, neutral-moderate factions.

    Args:
        n_factions: Number of factions (2–6).

    Returns:
        List of FactionState with equal power, zero radicalization and memory,
        moderate cohesion (0.5).
    """
    if not 2 <= n_factions <= 6:
        raise ValueError(
            f"n_factions must be in [2, 6], got {n_factions}"
        )
    power_share = 1.0 / n_factions
    return [
        FactionState(
            power=power_share,
            radicalization=0.0,
            cohesion=0.5,
            memory=0.0,
            wealth=0.5,
        )
        for _ in range(n_factions)
    ]


def create_dominant_factions(
    n_factions: int, dominant_idx: int, dominant_power: float = 0.6
) -> List[FactionState]:
    """Create a configuration with one dominant and N-1 minor factions.

    The dominant faction receives ``dominant_power`` share; the remainder is
    split equally among the other factions.

    Args:
        n_factions:     Total number of factions (2–6).
        dominant_idx:   Index of the dominant faction (0-indexed).
        dominant_power: Power share for the dominant faction (0 < x < 1).

    Returns:
        List of FactionState objects summing power to 1.
    """
    if not 2 <= n_factions <= 6:
        raise ValueError(
            f"n_factions must be in [2, 6], got {n_factions}"
        )
    if not 0 <= dominant_idx < n_factions:
        raise ValueError(
            f"dominant_idx must be in [0, {n_factions - 1}], got {dominant_idx}"
        )
    if not 0.0 < dominant_power < 1.0:
        raise ValueError(
            f"dominant_power must be in (0, 1), got {dominant_power}"
        )
    minor_power = (1.0 - dominant_power) / (n_factions - 1)
    factions: List[FactionState] = []
    for idx in range(n_factions):
        p = dominant_power if idx == dominant_idx else minor_power
        factions.append(
            FactionState(power=p, radicalization=0.0, cohesion=0.5, memory=0.0, wealth=0.5)
        )
    return factions