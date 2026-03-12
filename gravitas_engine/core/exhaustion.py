"""
Exhaustion dynamics.

Exhaustion is the only macro variable evolved by its own ODE rather than
derived algebraically.  It accumulates during high volatility / instability
episodes and recovers during calm periods:

    dExh/dt = α_Exh · [V · I · (1 − Exh) − β_Exh · (1 − V) · (1 − I) · Exh]

Stability proof:
  - When Exh = 1:
        dExh/dt = α_Exh · [V·I·0 − β_Exh·(1−V)·(1−I)·1] ≤ 0
    (growth term vanishes; recovery term is non-positive) → cannot exceed 1.
  - When Exh = 0:
        dExh/dt = α_Exh · V · I · 1 ≥ 0 → cannot go below 0.

At high exhaustion the system approaches a frozen state: all faction ODEs are
gated by (1 − Exh) → 0 except memory (which is intentionally ungated).
"""

from __future__ import annotations

import numpy as np

from .parameters import SystemParameters
from .state import RegimeState

# ── Cython fast-path imports ──────────────────────────────────────────────── #
try:
    from ._kernels import compute_exhaustion_deriv_c as _compute_exhaustion_deriv_c
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


def compute_exhaustion_derivative(
    state: RegimeState,
    params: SystemParameters,
) -> float:
    """Compute dExh/dt.

    Equation:
        dExh/dt = α_Exh · [V · I · (1 − Exh) − β_Exh · (1 − V) · (1 − I) · Exh]

    The raw derivative is clipped to [−Exh, 1−Exh] to prevent sub-step
    evaluations from pushing Exh outside [0, 1].

    Args:
        state:  Current regime state.
        params: System parameters (alpha_exh, beta_exh).

    Returns:
        dExh/dt ∈ [−1, 1].
    """
    exh = state.system.exhaustion
    vol = state.system.volatility
    inst = state.system.instability

    # Phase 3: district stress contributes to exhaustion accumulation
    has_hierarchy = (getattr(state, "hierarchical", None) is not None
                     and getattr(params, "use_hierarchy", False))

    if _USE_CYTHON and not has_hierarchy:
        return _compute_exhaustion_deriv_c(
            exh, vol, inst, params.alpha_exh, params.beta_exh,
        )

    accumulation = vol * inst * (1.0 - exh)
    recovery = params.beta_exh * (1.0 - vol) * (1.0 - inst) * exh

    raw = params.alpha_exh * (accumulation - recovery)
    if has_hierarchy:
        from .hierarchical_coupling import exhaustion_increment_from_districts
        raw += exhaustion_increment_from_districts(state.hierarchical, params)
    return float(np.clip(raw, -exh, 1.0 - exh))


def exhaustion_steady_state(
    volatility: float, instability: float, params: SystemParameters
) -> float:
    """Compute the analytical steady-state Exh* for fixed V and I.

    Setting dExh/dt = 0 and solving:
        Exh* = V · I / (V · I + β_Exh · (1 − V) · (1 − I))

    Returns 1.0 when the denominator is zero (permanent crisis).

    Args:
        volatility:  Current V.
        instability: Current I.
        params:      System parameters.

    Returns:
        Steady-state exhaustion ∈ [0, 1].
    """
    numerator = volatility * instability
    denominator = numerator + params.beta_exh * (1.0 - volatility) * (
        1.0 - instability
    )
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def is_frozen(state: RegimeState, threshold: float = 0.99) -> bool:
    """Return True if the system is in an exhaustion-freeze condition.

    At freeze, all faction ODEs are effectively zero (gated by 1−Exh ≈ 0).

    Args:
        state:     Current regime state.
        threshold: Exhaustion level considered frozen (default 0.99).

    Returns:
        Boolean freeze flag.
    """
    return state.system.exhaustion >= threshold