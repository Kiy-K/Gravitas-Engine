"""
Fragmentation dynamics.

System fragmentation is derived from the Gini coefficient of the faction power
distribution via a saturating exponential transform:

    Gini(P) = (1 / (2 · N · Σ P_i)) · Σ_i Σ_j |P_i − P_j|

    F = 1 − exp(−λ_F · Gini(P))

Stability properties:
  - Gini ∈ [0, (N−1)/N] analytically for N ≥ 2.
  - F = 1 − exp(·) maps [0,∞) → [0, 1) — F is strictly bounded below 1.
  - As one faction monopolises all power (theoretical maximum concentration),
      Gini → (N−1)/N  →  F → 1 − exp(−λ_F · (N−1)/N) < 1.
  - As power becomes perfectly uniform, Gini = 0 → F = 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import SystemParameters
from .state import RegimeState

# ── Cython fast-path imports ──────────────────────────────────────────────── #
try:
    from ._kernels import compute_gini_c as _compute_gini_c
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


def compute_gini(powers: NDArray[np.float64]) -> float:
    """Compute Gini coefficient of power shares.

    Formula:
        Gini = Σ_i Σ_j |P_i − P_j| / (2 · N · Σ P_i)

    Args:
        powers: 1-D array of faction power values (all ≥ 0).

    Returns:
        Gini coefficient ∈ [0, (N−1)/N].
    """
    if _USE_CYTHON:
        return _compute_gini_c(np.ascontiguousarray(powers, dtype=np.float64))
    n = len(powers)
    if n < 2:
        return 0.0
    total = float(np.sum(powers))
    if total <= 0.0:
        return 0.0
    diff_sum = float(np.sum(np.abs(powers[:, None] - powers[None, :])))
    return float(np.clip(diff_sum / (2.0 * n * total), 0.0, 1.0))


def compute_fragmentation(
    state: RegimeState, params: SystemParameters
) -> float:
    """F = 1 − exp(−λ_F · Gini(P)).

    Args:
        state:  Current regime state.
        params: System parameters (lambda_frag).

    Returns:
        Fragmentation F ∈ [0, 1).
    """
    gini = compute_gini(state.get_faction_powers())
    frag = 1.0 - float(np.exp(-params.lambda_frag * gini))
    return float(np.clip(frag, 0.0, 1.0))


def fragmentation_upper_bound(n_factions: int, lambda_frag: float) -> float:
    """Return the theoretical upper bound of F for N factions.

    As power concentrates maximally in one faction:
        Gini_max = (N − 1) / N
        F_max    = 1 − exp(−λ_F · (N−1)/N)

    This is always strictly less than 1 for finite λ_F and N ≥ 2.

    Args:
        n_factions:  Number of factions (2–6).
        lambda_frag: Fragmentation sensitivity.

    Returns:
        Theoretical maximum F < 1.
    """
    if n_factions < 2:
        return 0.0
    gini_max = (n_factions - 1) / n_factions
    return float(1.0 - np.exp(-lambda_frag * gini_max))


def fragmentation_from_gini(gini: float, lambda_frag: float) -> float:
    """Apply the saturating exponential transform to a pre-computed Gini value.

    Args:
        gini:        Gini coefficient ∈ [0, 1].
        lambda_frag: Fragmentation sensitivity (> 0).

    Returns:
        Fragmentation F ∈ [0, 1).
    """
    frag = 1.0 - float(np.exp(-lambda_frag * gini))
    return float(np.clip(frag, 0.0, 1.0))