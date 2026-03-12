"""
GravitasDynamics — Transition kernel for GRAVITAS.

Contains all ODE derivative functions and the RK4 integrator.
Every interaction is non-linear. Every variable has delayed consequences.

Key design principles:
  - No variable evolves independently
  - Military effects are split: immediate (in apply_action) vs accumulated (in ODEs)
  - Cascade hazard uses conflict topology c_ij, not proximity A_ij
  - Exhaustion gates all recovery terms (high exhaustion → everything freezes)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .gravitas_params import GravitasParams
from .gravitas_state import ClusterState, GlobalState, GravitasWorld, N_CLUSTER_VARS
from ..systems.diplomacy import alliance_cluster_derivatives

# ── Cython fast-path imports ──────────────────────────────────────────────── #
try:
    from ._kernels import (
        compute_hazard_c as _compute_hazard_c,
        cluster_derivatives_c as _cluster_derivatives_c,
        global_derivatives_c as _global_derivatives_c,
        rk4_substep_c as _rk4_substep_c,
        alliance_cluster_derivatives_c as _alliance_cluster_derivatives_c,
    )
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


# ─────────────────────────────────────────────────────────────────────────── #
# Hazard index (algebraic — re-derived each step, not integrated)             #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_hazard(
    sigma: NDArray[np.float64],
    trust: NDArray[np.float64],
    polar: NDArray[np.float64],
    conflict: NDArray[np.float64],
    sys_pol: float,
    params: GravitasParams,
) -> NDArray[np.float64]:
    """
    hᵢ = γ₁·(1-σᵢ)^κ_h · pᵢ^κ_p
       + γ₂·(1-τᵢ)·Π
       + γ₃·Σⱼ cᵢⱼ·hⱼ·(1-σⱼ)     [cascade; solved iteratively]

    We approximate the cascade term in one forward pass (sufficient for dt=0.01).
    """
    if _USE_CYTHON:
        return _compute_hazard_c(
            np.ascontiguousarray(sigma), np.ascontiguousarray(trust),
            np.ascontiguousarray(polar), np.ascontiguousarray(conflict),
            sys_pol, params.gamma_h1, params.gamma_h2, params.gamma_h3,
            params.kappa_h, params.kappa_p,
        )
    # Local hazard (no cascade yet)
    h_local = (
        params.gamma_h1 * np.power(np.clip(1.0 - sigma, 0.0, 1.0), params.kappa_h)
                        * np.power(polar, params.kappa_p)
        + params.gamma_h2 * (1.0 - trust) * sys_pol
    )
    # One-step cascade approximation
    cascade = params.gamma_h3 * (conflict @ (h_local * (1.0 - sigma)))
    return np.clip(h_local + cascade, 0.0, 5.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Cluster-level ODE derivatives  d(cluster)/dt                                #
# ─────────────────────────────────────────────────────────────────────────── #

def cluster_derivatives(
    arr: NDArray[np.float64],                    # (N, 6): [σ, h, r, m, τ, p]
    hazard: NDArray[np.float64],                 # (N,)  pre-computed
    adjacency: NDArray[np.float64],              # (N, N)
    g: GlobalState,
    params: GravitasParams,
    world: Optional[GravitasWorld] = None,
    alliance: Optional[NDArray[np.float64]] = None,  # (N, N) ∈ [-1,+1]
) -> NDArray[np.float64]:
    """
    Compute d(cluster_array)/dt.  Returns (N, 6).

    Index mapping: 0=σ, 1=h(unused/overwritten), 2=r, 3=m, 4=τ, 5=p
    """
    if _USE_CYTHON and (world is None or not hasattr(world, '_cached')):
        deriv = _cluster_derivatives_c(
            np.ascontiguousarray(arr), np.ascontiguousarray(hazard),
            np.ascontiguousarray(adjacency),
            g.exhaustion, g.polarization, g.fragmentation,
            params.alpha_sigma, params.beta_sigma, params.nu_sigma,
            params.alpha_res, params.hazard_res_cost,
            params.military_decay,
            params.military_tau_cost, params.deprivation_tau_cost, params.tau_decay,
            params.alpha_pol, params.beta_pol,
        )
        if alliance is not None:
            N = arr.shape[0]
            ally = np.ascontiguousarray(alliance[:N, :N])
            deriv += _alliance_cluster_derivatives_c(
                np.ascontiguousarray(arr), ally,
                params.nu_alliance, params.nu_res_alliance, params.alpha_hostility,
            )
        return deriv

    N = arr.shape[0]
    sigma    = arr[:, 0]
    resource = arr[:, 2]
    military = arr[:, 3]
    trust    = arr[:, 4]
    polar    = arr[:, 5]

    E   = g.exhaustion
    Pi  = g.polarization
    Phi = g.fragmentation
    act = 1.0 - E     # activity gate: high exhaustion freezes everything

    # ── dσ/dt: stability ────────────────────────────────────────────────── #
    # Spatial diffusion of stability (cached)
    if world is not None and hasattr(world, '_cached'):
        lap_s = world._cached['adj_sigma'] - sigma * np.sum(adjacency, axis=1)
    else:
        deg = np.sum(adjacency, axis=1)
        lap_s = (adjacency @ sigma) - sigma * deg
    d_sigma = (
        params.alpha_sigma * act * (
            (1.0 - polar) * (1.0 - Phi) - params.beta_sigma * hazard * sigma
        )
        + params.nu_sigma * lap_s
    )

    # ── dr/dt: resource ─────────────────────────────────────────────────── #
    d_resource = (
        params.alpha_res * (1.0 - resource)            # natural recovery
        - params.hazard_res_cost * hazard * resource    # hazard drain
    ) * act

    # ── dm/dt: military presence decay ──────────────────────────────────── #
    d_military = -params.military_decay * military

    # ── dτ/dt: institutional trust ──────────────────────────────────────── #
    d_trust = (
        - params.military_tau_cost * military * (1.0 - sigma)
        - params.deprivation_tau_cost * (1.0 - resource) * hazard
        - params.tau_decay * trust
    ) * act

    # ── dp/dt: local polarization ────────────────────────────────────────── #
    d_polar = (
        params.alpha_pol * Phi * (1.0 - trust)
        - params.beta_pol * trust * (1.0 - polar)       # trust depolarizes
    )

    # ── Assemble & batch-clip to [0,1] bounds ────────────────────────────── #
    deriv = np.empty_like(arr)
    deriv[:, 0] = d_sigma
    deriv[:, 1] = 0.0       # hazard is re-derived algebraically, not integrated
    deriv[:, 2] = d_resource
    deriv[:, 3] = d_military
    deriv[:, 4] = d_trust
    deriv[:, 5] = d_polar
    # Clamp: derivative must not push state outside [0, 1]
    np.clip(deriv, -arr, 1.0 - arr, out=deriv)
    deriv[:, 1] = 0.0  # keep hazard derivative zero

    # ── Alliance-based inter-cluster effects ─────────────────────────────── #
    if alliance is not None:
        N = arr.shape[0]
        ally = alliance[:N, :N]
        deriv += alliance_cluster_derivatives(arr, ally, params)

    return deriv


# ─────────────────────────────────────────────────────────────────────────── #
# Global ODE derivatives  d(global)/dt                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def global_derivatives(
    g_arr: NDArray[np.float64],
    hazard: NDArray[np.float64],
    military: NDArray[np.float64],
    trust: NDArray[np.float64],
    params: GravitasParams,
    military_load: float,
    propaganda_load: float,
) -> NDArray[np.float64]:
    """
    Compute d(global_array)/dt.  Returns (6,): [E, Φ, Π, Ψ, M, T]

    Accepts raw array [E, Φ, Π, Ψ, M, T] — no GlobalState construction needed.
    """
    if _USE_CYTHON:
        return _global_derivatives_c(
            np.ascontiguousarray(g_arr), np.ascontiguousarray(hazard),
            np.ascontiguousarray(military), np.ascontiguousarray(trust),
            military_load, propaganda_load,
            params.alpha_exh, params.beta_exh, params.military_exh_coeff,
            params.alpha_phi, params.beta_phi, params.military_phi_coeff,
            params.alpha_pol, params.beta_pol, params.propaganda_pol_coeff,
            params.psi_recovery, params.psi_propaganda_cost,
            params.alpha_tau, params.tau_decay, params.military_tau_cost,
        )

    E   = float(g_arr[0])
    Phi = float(g_arr[1])
    Pi  = float(g_arr[2])
    Psi = float(g_arr[3])
    M   = float(g_arr[4])
    T   = float(g_arr[5])

    mean_h   = float(np.mean(hazard))
    mean_m   = military_load
    mean_tau = float(np.mean(trust))

    _c = lambda v, lo, hi: max(lo, min(v, hi))  # fast scalar clamp

    # ── dE/dt: exhaustion ────────────────────────────────────────────────── #
    accumulation = Pi * (1.0 - Psi) * (1.0 - E) + params.military_exh_coeff * mean_m * M
    recovery     = params.beta_exh * (1.0 - Pi) * Psi * E
    d_E = _c(params.alpha_exh * (accumulation - recovery), -E, 1.0 - E)

    # ── dΦ/dt: fragmentation probability ─────────────────────────────────── #
    d_Phi = _c(
        params.alpha_phi * mean_h * (1.0 - mean_tau)
        + params.military_phi_coeff * mean_m ** 2 * E
        - params.beta_phi * mean_tau * (1.0 - Phi),
        -Phi, 1.0 - Phi,
    )

    # ── dΠ/dt: systemic polarization ─────────────────────────────────────── #
    d_Pi = _c(
        params.alpha_pol * Phi * (1.0 - T)
        + params.propaganda_pol_coeff * propaganda_load * (1.0 - T)
        - params.beta_pol * T * (1.0 - Pi),
        -Pi, 1.0 - Pi,
    )

    # ── dΨ/dt: information coherence ─────────────────────────────────────── #
    d_Psi = _c(
        params.psi_recovery * (1.0 - Psi)
        - params.psi_propaganda_cost * propaganda_load,
        -Psi, 1.0 - Psi,
    )

    # ── dM/dt: military strength ──────────────────────────────────────────── #
    d_M = _c(0.01 * (1.0 - M) - 0.05 * mean_m * M, -M, 1.0 - M)

    # ── dT/dt: aggregate institutional trust ──────────────────────────────── #
    d_T = _c(
        params.alpha_tau * mean_tau
        - params.tau_decay * T
        - params.military_tau_cost * mean_m * (1.0 - mean_tau),
        -T, 1.0 - T,
    )

    return np.array([d_E, d_Phi, d_Pi, d_Psi, d_M, d_T], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────── #
# RK4 integrator                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

def _eval_derivatives(
    c_arr: NDArray[np.float64],
    g_arr: NDArray[np.float64],
    hazard: NDArray[np.float64],
    adjacency: NDArray[np.float64],
    params: GravitasParams,
    military_load: float,
    propaganda_load: float,
    world: Optional[GravitasWorld] = None,
    alliance: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate (d_cluster, d_global) at a given state point.

    Now takes raw global array [E,Φ,Π,Ψ,M,T] instead of GlobalState.
    """
    trust = c_arr[:, 4]
    g_deriv = global_derivatives(g_arr, hazard, c_arr[:, 3], trust, params,
                                  military_load, propaganda_load)
    # cluster_derivatives still needs a GlobalState-like thing for field access;
    # build a lightweight shim to avoid the full constructor overhead.
    g = _GlobalShim(g_arr)
    c_deriv = cluster_derivatives(c_arr, hazard, adjacency, g, params, world, alliance)
    return c_deriv, g_deriv


class _GlobalShim:
    """Lightweight stand-in for GlobalState to avoid constructor overhead in RK4."""
    __slots__ = ('exhaustion', 'fragmentation', 'polarization',
                 'coherence', 'military_str', 'trust')
    def __init__(self, arr):
        self.exhaustion    = float(arr[0])
        self.fragmentation = float(arr[1])
        self.polarization  = float(arr[2])
        self.coherence     = float(arr[3])
        self.military_str  = float(arr[4])
        self.trust         = float(arr[5])


def rk4_step(
    world: GravitasWorld,
    params: GravitasParams,
    military_load: float,
    propaganda_load: float,
    sigma_noise: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> GravitasWorld:
    """
    Advance world by one RK4 step (dt = params.dt).

    Hazard is re-derived at each sub-step (algebraic, not integrated).
    SDE noise is applied to σ and r after the deterministic RK4.
    """
    dt     = params.dt
    step   = world.global_state.step

    c0     = world.cluster_array()
    g0     = world.global_state.to_array()
    A      = world.adjacency
    C      = world.conflict
    AL     = world.alliance
    N      = c0.shape[0]

    # ─── Cython fast-path: entire RK4 in C (no alliance support yet) ───── #
    if _USE_CYTHON and AL is None:
        c_new, g_new_a = _rk4_substep_c(
            np.ascontiguousarray(c0), np.ascontiguousarray(g0),
            np.ascontiguousarray(A), np.ascontiguousarray(C),
            military_load, propaganda_load, dt, sigma_noise, rng,
            # hazard params
            params.gamma_h1, params.gamma_h2, params.gamma_h3,
            params.kappa_h, params.kappa_p,
            # cluster params
            params.alpha_sigma, params.beta_sigma, params.nu_sigma,
            params.alpha_res, params.hazard_res_cost,
            params.military_decay,
            params.military_tau_cost, params.deprivation_tau_cost, params.tau_decay,
            params.alpha_pol, params.beta_pol,
            # global params
            params.alpha_exh, params.beta_exh, params.military_exh_coeff,
            params.alpha_phi, params.beta_phi, params.military_phi_coeff,
            params.alpha_pol, params.beta_pol,
            params.propaganda_pol_coeff,
            params.psi_recovery, params.psi_propaganda_cost,
            params.alpha_tau, params.tau_decay, params.military_tau_cost,
        )
        new_clusters = [
            ClusterState._from_array_fast(c_new[i]) for i in range(N)
        ]
        new_global = GlobalState.from_array(g_new_a, step=step)
        return world.copy_with_clusters(new_clusters).copy_with_global(new_global)

    # ─── Python fallback (with alliance support) ───────────────────────── #
    half   = dt * 0.5

    # All intermediate global states are raw arrays — no GlobalState construction.
    # Only `polarization` (index 2) is needed for hazard computation.

    # ─── k1 ─────────────────────────────────────────────────────────────── #
    h0 = compute_hazard(c0[:,0], c0[:,4], c0[:,5], C, float(g0[2]), params)
    dc1, dg1 = _eval_derivatives(c0, g0, h0, A, params,
                                  military_load, propaganda_load, world, AL)

    # ─── k2 ─────────────────────────────────────────────────────────────── #
    c2     = np.clip(c0 + half * dc1, 0.0, 1.0)
    c2[:,1]= 0.0
    g2     = np.clip(g0 + half * dg1, 0.0, 1.0)
    h2     = compute_hazard(c2[:,0], c2[:,4], c2[:,5], C, float(g2[2]), params)
    dc2, dg2 = _eval_derivatives(c2, g2, h2, A, params,
                                  military_load, propaganda_load, alliance=AL)

    # ─── k3 ─────────────────────────────────────────────────────────────── #
    c3     = np.clip(c0 + half * dc2, 0.0, 1.0)
    c3[:,1]= 0.0
    g3     = np.clip(g0 + half * dg2, 0.0, 1.0)
    h3     = compute_hazard(c3[:,0], c3[:,4], c3[:,5], C, float(g3[2]), params)
    dc3, dg3 = _eval_derivatives(c3, g3, h3, A, params,
                                  military_load, propaganda_load, alliance=AL)

    # ─── k4 ─────────────────────────────────────────────────────────────── #
    c4     = np.clip(c0 + dt * dc3, 0.0, 1.0)
    c4[:,1]= 0.0
    g4     = np.clip(g0 + dt * dg3, 0.0, 1.0)
    h4     = compute_hazard(c4[:,0], c4[:,4], c4[:,5], C, float(g4[2]), params)
    dc4, dg4 = _eval_derivatives(c4, g4, h4, A, params,
                                  military_load, propaganda_load, alliance=AL)

    # ─── Weighted sum ────────────────────────────────────────────────────── #
    factor  = dt / 6.0
    c_new   = np.clip(c0 + factor * (dc1 + 2*dc2 + 2*dc3 + dc4), 0.0, 1.0)
    g_new_a = np.clip(g0 + factor * (dg1 + 2*dg2 + 2*dg3 + dg4), 0.0, 1.0)

    # ─── SDE noise on σ (stability) and r (resource) ────────────────────── #
    if sigma_noise > 0.0 and rng is not None:
        noise_scale = sigma_noise * np.sqrt(dt)
        c_new[:, 0] += noise_scale * rng.standard_normal(N)
        c_new[:, 2] += noise_scale * 0.5 * rng.standard_normal(N)
        c_new = np.clip(c_new, 0.0, 1.0)

    # ─── Re-derive hazard at new state ──────────────────────────────────── #
    h_new  = compute_hazard(c_new[:,0], c_new[:,4], c_new[:,5], C,
                             float(g_new_a[2]), params)
    c_new[:, 1] = h_new

    # ─── Reconstruct cluster states (fast path — data already clipped) ─── #
    new_clusters = [
        ClusterState._from_array_fast(c_new[i]) for i in range(N)
    ]
    new_global = GlobalState.from_array(g_new_a, step=step)

    return world.copy_with_clusters(new_clusters).copy_with_global(new_global)


