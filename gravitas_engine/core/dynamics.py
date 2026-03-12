"""
Complete system dynamics.

Assembles all faction-level ODE derivatives into a single function.  The
derivative equations are:

  Eq. P  —  dP_i/dt = α_P · (1−Exh) · [E·Coh_i − β_P·Rad_i·(1−Coh_i) − γ_P·F·P_i]

  Eq. R  —  dRad_i/dt = α_R · (1−Exh) · [Mem_i·(1−Coh_i)·(1−R_sys)
                          − β_R·Rad_i·Coh_i − γ_R·Rad_i²]

  Eq. C  —  dCoh_i/dt = α_Coh · (1−Exh) · [(1−F)·(1−Rad_i)
                          − β_Coh·|P_i − P̄| − Coh_i·Rad_i²]

  Eq. Mem—  dMem_i/dt = α_Mem · [I·(1−Coh_i) − β_Mem·Mem_i]
             (NOT gated by Exh — memory persists across exhaustion)

  Eq. Exh—  dExh/dt = α_Exh · [V·I·(1−Exh) − β_Exh·(1−V)·(1−I)·Exh]

All derivatives are clipped to the safe interval [−x_i, 1−x_i] so that
Euler-like sub-step evaluations cannot overshoot [0, 1].
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .exhaustion import compute_exhaustion_derivative
from .memory import compute_memory_derivatives
from .dynamics_extensions import compute_economic_derivatives, compute_topological_derivatives
from .parameters import SystemParameters
from .state import RegimeState

# ── Cython fast-path imports ──────────────────────────────────────────────── #
try:
    from ._kernels import compute_faction_derivs_c as _compute_faction_derivs_c
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


# --------------------------------------------------------------------------- #
# Faction derivative computation                                                #
# --------------------------------------------------------------------------- #


def compute_faction_derivatives(
    state: RegimeState,
    params: SystemParameters,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute (dP/dt, dRad/dt, dCoh/dt, dMem/dt) for all factions.

    Args:
        state:  Current regime state (system variables must be up-to-date).
        params: System parameters.

    Returns:
        Four arrays each of shape (n_factions,):
        dP_dt, dRad_dt, dCoh_dt, dMem_dt.
    """
    powers = state.get_faction_powers()
    rads = state.get_faction_radicalizations()
    cohs = state.get_faction_cohesions()
    mems = state.get_faction_memories()

    exh = state.system.exhaustion
    frag = state.system.fragmentation
    inst = state.system.instability
    rep = state.system.repression
    elite = state.system.elite_alignment

    if _USE_CYTHON:
        aff_mat = None
        if state.affinity_matrix:
            aff_mat = np.array(state.affinity_matrix, dtype=np.float64)
            np.fill_diagonal(aff_mat, 0.0)
        return _compute_faction_derivs_c(
            np.ascontiguousarray(powers), np.ascontiguousarray(rads),
            np.ascontiguousarray(cohs), np.ascontiguousarray(mems),
            exh, frag, inst, rep, elite,
            params.alpha_power, params.beta_power, params.gamma_power,
            params.alpha_rad, params.beta_rad, params.gamma_rad,
            params.alpha_coh, params.beta_coh,
            params.alpha_mem, params.beta_mem,
            aff_mat,
        )

    activity = 1.0 - exh  # All ODEs except Mem are gated by (1-Exh)

    # Topological Affinity modifier
    if state.affinity_matrix:
        aff = np.array(state.affinity_matrix)
        np.fill_diagonal(aff, 0.0)
        aff_p_mod = 0.05 * (aff @ powers)
        aff_r_mod = -0.05 * (aff @ rads)
    else:
        aff_p_mod = 0.0
        aff_r_mod = 0.0

    # ---------------------------------------------------------------------- #
    # Eq. P                                                                    #
    # dP_i/dt = α_P · (1−Exh) · [E·Coh_i − β_P·Rad_i·(1−Coh_i)             #
    #                              − γ_P·F·P_i] + Affinity                       #
    # ---------------------------------------------------------------------- #
    dP_dt_raw = params.alpha_power * activity * (
        elite * cohs
        - params.beta_power * rads * (1.0 - cohs)
        - params.gamma_power * frag * powers
    ) + aff_p_mod
    dP_dt = np.clip(dP_dt_raw, -powers, 1.0 - powers)

    # ---------------------------------------------------------------------- #
    # Eq. R (Radicalization)                                                   #
    # dRad_i/dt = α_R · (1−Exh) · [Mem_i·(1−Coh_i)·(1−R_sys)               #
    #              − β_R·Rad_i·Coh_i − γ_R·Rad_i²] + Affinity                 #
    # Stability: β_R + γ_R > 1 ensures dRad<0 when Rad_i=1 (enforced by     #
    #            SystemParameters.__post_init__).                              #
    # ---------------------------------------------------------------------- #
    dRad_dt_raw = params.alpha_rad * activity * (
        mems * (1.0 - cohs) * (1.0 - rep)
        - params.beta_rad * rads * cohs
        - params.gamma_rad * rads ** 2
    ) + aff_r_mod
    dRad_dt = np.clip(dRad_dt_raw, -rads, 1.0 - rads)

    # ---------------------------------------------------------------------- #
    # Eq. C (Cohesion)                                                         #
    # dCoh_i/dt = α_Coh · (1−Exh) · [(1−F)·(1−Rad_i)                        #
    #              − β_Coh·|P_i − P̄| − Coh_i·Rad_i²]                        #
    # ---------------------------------------------------------------------- #
    mean_power = float(np.mean(powers))
    power_dev = np.abs(powers - mean_power)  # |P_i − P̄| ∈ [0, 1]

    dCoh_dt_raw = params.alpha_coh * activity * (
        (1.0 - frag) * (1.0 - rads)
        - params.beta_coh * power_dev
        - cohs * rads ** 2
    )
    dCoh_dt = np.clip(dCoh_dt_raw, -cohs, 1.0 - cohs)

    # ---------------------------------------------------------------------- #
    # Eq. Mem (delegates to memory module)                                     #
    # ---------------------------------------------------------------------- #
    dMem_dt = compute_memory_derivatives(state, params)

    return dP_dt, dRad_dt, dCoh_dt, dMem_dt


def compute_exhaustion_deriv(
    state: RegimeState, params: SystemParameters
) -> float:
    """Compute dExh/dt by delegating to the exhaustion module.

    Args:
        state:  Current regime state.
        params: System parameters.

    Returns:
        dExh/dt ∈ [−1, 1].
    """
    return compute_exhaustion_derivative(state, params)


def compute_all_derivatives(
    state: RegimeState,
    params: SystemParameters,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    NDArray[np.float64],
    tuple[tuple[float, ...], ...],
]:
    """Compute derivatives for every state variable.

    Returns:
        (dP_dt, dRad_dt, dCoh_dt, dMem_dt, dWealth_dt, dExh_dt, dGDP_dt, dPillars_dt, dAff_dt)
    """
    dP, dRad, dCoh, dMem = compute_faction_derivatives(state, params)
    dExh = compute_exhaustion_deriv(state, params)
    dWealth, dGDP = compute_economic_derivatives(state, params)
    dPillars, dAff = compute_topological_derivatives(state, params)
    return dP, dRad, dCoh, dMem, dWealth, dExh, dGDP, dPillars, dAff