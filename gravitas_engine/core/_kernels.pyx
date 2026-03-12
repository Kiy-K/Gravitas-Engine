# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# cython: language_level=3
"""
Cython-optimized numerical kernels for GRAVITAS Engine.

Replaces the hottest inner-loop functions with typed memoryview
implementations that bypass Python/NumPy per-call overhead.

Optimized functions:
  - compute_hazard_c          (called 5x per RK4 step)
  - cluster_derivatives_c     (called 4x per RK4 step)
  - global_derivatives_c      (called 4x per RK4 step)
  - rk4_substep_c             (complete RK4 integration in C)
  - compute_gini_c            (called in faction aggregation)
  - compute_faction_derivs_c  (called 4x per regime integrator step)
  - compute_memory_derivs_c   (called 4x per regime integrator step)
  - compute_exhaustion_deriv_c(called 4x per regime integrator step)
"""

from libc.math cimport sqrt, exp, tanh, fabs, pow as cpow, log
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as cnp

cnp.import_array()


# ─────────────────────────────────────────────────────────────────────────── #
# Scalar helpers                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

cdef inline double dclamp(double v, double lo, double hi) noexcept nogil:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


cdef inline double dmax(double a, double b) noexcept nogil:
    return a if a > b else b


cdef inline double dmin(double a, double b) noexcept nogil:
    return a if a < b else b


# ─────────────────────────────────────────────────────────────────────────── #
# compute_hazard — vectorised hazard index                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_hazard_c(
    double[::1] sigma,
    double[::1] trust,
    double[::1] polar,
    double[:, ::1] conflict,
    double sys_pol,
    double gamma_h1,
    double gamma_h2,
    double gamma_h3,
    double kappa_h,
    double kappa_p,
):
    """
    hᵢ = γ₁·(1-σᵢ)^κ_h · pᵢ^κ_p + γ₂·(1-τᵢ)·Π + γ₃·Σⱼ cᵢⱼ·hⱼ·(1-σⱼ)
    Returns ndarray of shape (N,).
    """
    cdef Py_ssize_t N = sigma.shape[0]
    cdef Py_ssize_t i, j
    cdef double cascade, h_local_i

    cdef cnp.ndarray[double, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef double[::1] result = out

    # Phase 1: local hazard
    cdef double *h_local = <double *>malloc(N * sizeof(double))
    if h_local == NULL:
        raise MemoryError()

    try:
        for i in range(N):
            h_local[i] = (
                gamma_h1 * cpow(dclamp(1.0 - sigma[i], 0.0, 1.0), kappa_h)
                         * cpow(polar[i], kappa_p)
                + gamma_h2 * (1.0 - trust[i]) * sys_pol
            )

        # Phase 2: one-step cascade approximation
        for i in range(N):
            cascade = 0.0
            for j in range(N):
                cascade += conflict[i, j] * h_local[j] * (1.0 - sigma[j])
            result[i] = dclamp(h_local[i] + gamma_h3 * cascade, 0.0, 5.0)
    finally:
        free(h_local)

    return out


# ─────────────────────────────────────────────────────────────────────────── #
# cluster_derivatives — per-cluster ODE d(cluster)/dt                         #
# ─────────────────────────────────────────────────────────────────────────── #

def cluster_derivatives_c(
    double[:, ::1] arr,          # (N, 6): [σ, h, r, m, τ, p]
    double[::1] hazard,          # (N,)
    double[:, ::1] adjacency,    # (N, N)
    double exhaustion,
    double polarization_sys,
    double fragmentation_sys,
    double alpha_sigma,
    double beta_sigma,
    double nu_sigma,
    double alpha_res,
    double hazard_res_cost,
    double military_decay,
    double military_tau_cost,
    double deprivation_tau_cost,
    double tau_decay,
    double alpha_pol,
    double beta_pol,
):
    """
    Compute d(cluster_array)/dt.  Returns (N, 6) ndarray.
    Index: 0=σ, 1=h(zero), 2=r, 3=m, 4=τ, 5=p
    """
    cdef Py_ssize_t N = arr.shape[0]
    cdef Py_ssize_t i, j
    cdef double act = 1.0 - exhaustion
    cdef double sigma_i, resource_i, military_i, trust_i, polar_i
    cdef double lap_s, deg_i, d_s, d_r, d_m, d_t, d_p
    cdef double lo, hi

    cdef cnp.ndarray[double, ndim=2] out = np.zeros((N, 6), dtype=np.float64)
    cdef double[:, ::1] deriv = out

    for i in range(N):
        sigma_i    = arr[i, 0]
        resource_i = arr[i, 2]
        military_i = arr[i, 3]
        trust_i    = arr[i, 4]
        polar_i    = arr[i, 5]

        # Laplacian for spatial diffusion of stability
        lap_s = 0.0
        deg_i = 0.0
        for j in range(N):
            lap_s += adjacency[i, j] * arr[j, 0]
            deg_i += adjacency[i, j]
        lap_s -= sigma_i * deg_i

        # dσ/dt
        d_s = (
            alpha_sigma * act * (
                (1.0 - polar_i) * (1.0 - fragmentation_sys)
                - beta_sigma * hazard[i] * sigma_i
            )
            + nu_sigma * lap_s
        )

        # dr/dt
        d_r = (
            alpha_res * (1.0 - resource_i)
            - hazard_res_cost * hazard[i] * resource_i
        ) * act

        # dm/dt
        d_m = -military_decay * military_i

        # dτ/dt
        d_t = (
            - military_tau_cost * military_i * (1.0 - sigma_i)
            - deprivation_tau_cost * (1.0 - resource_i) * hazard[i]
            - tau_decay * trust_i
        ) * act

        # dp/dt
        d_p = (
            alpha_pol * fragmentation_sys * (1.0 - trust_i)
            - beta_pol * trust_i * (1.0 - polar_i)
        )

        # Clamp: derivative must not push state outside [0, 1]
        deriv[i, 0] = dclamp(d_s, -sigma_i,    1.0 - sigma_i)
        deriv[i, 1] = 0.0  # hazard re-derived algebraically
        deriv[i, 2] = dclamp(d_r, -resource_i,  1.0 - resource_i)
        deriv[i, 3] = dclamp(d_m, -military_i,  1.0 - military_i)
        deriv[i, 4] = dclamp(d_t, -trust_i,     1.0 - trust_i)
        deriv[i, 5] = dclamp(d_p, -polar_i,     1.0 - polar_i)

    return out


# ─────────────────────────────────────────────────────────────────────────── #
# global_derivatives — system-wide ODE d(global)/dt                           #
# ─────────────────────────────────────────────────────────────────────────── #

def global_derivatives_c(
    double[::1] g_arr,          # (6,): [E, Φ, Π, Ψ, M, T]
    double[::1] hazard,         # (N,)
    double[::1] military,       # (N,)
    double[::1] trust,          # (N,)
    double military_load,
    double propaganda_load,
    # params
    double alpha_exh,
    double beta_exh,
    double military_exh_coeff,
    double alpha_phi,
    double beta_phi,
    double military_phi_coeff,
    double alpha_pol,
    double beta_pol_param,
    double propaganda_pol_coeff,
    double psi_recovery,
    double psi_propaganda_cost,
    double alpha_tau,
    double tau_decay,
    double military_tau_cost,
):
    """
    Compute d(global_array)/dt.  Returns (6,) ndarray: [dE, dΦ, dΠ, dΨ, dM, dT].
    """
    cdef double E   = g_arr[0]
    cdef double Phi = g_arr[1]
    cdef double Pi  = g_arr[2]
    cdef double Psi = g_arr[3]
    cdef double M   = g_arr[4]
    cdef double T   = g_arr[5]

    cdef Py_ssize_t N = hazard.shape[0]
    cdef Py_ssize_t i
    cdef double mean_h = 0.0, mean_tau = 0.0

    for i in range(N):
        mean_h   += hazard[i]
        mean_tau += trust[i]
    if N > 0:
        mean_h   /= N
        mean_tau /= N

    cdef double mean_m = military_load

    # dE/dt: exhaustion
    cdef double accum = Pi * (1.0 - Psi) * (1.0 - E) + military_exh_coeff * mean_m * M
    cdef double recov = beta_exh * (1.0 - Pi) * Psi * E
    cdef double d_E = dclamp(alpha_exh * (accum - recov), -E, 1.0 - E)

    # dΦ/dt: fragmentation probability
    cdef double d_Phi = dclamp(
        alpha_phi * mean_h * (1.0 - mean_tau)
        + military_phi_coeff * mean_m * mean_m * E
        - beta_phi * mean_tau * (1.0 - Phi),
        -Phi, 1.0 - Phi,
    )

    # dΠ/dt: systemic polarization
    cdef double d_Pi = dclamp(
        alpha_pol * Phi * (1.0 - T)
        + propaganda_pol_coeff * propaganda_load * (1.0 - T)
        - beta_pol_param * T * (1.0 - Pi),
        -Pi, 1.0 - Pi,
    )

    # dΨ/dt: information coherence
    cdef double d_Psi = dclamp(
        psi_recovery * (1.0 - Psi)
        - psi_propaganda_cost * propaganda_load,
        -Psi, 1.0 - Psi,
    )

    # dM/dt: military strength
    cdef double d_M = dclamp(0.01 * (1.0 - M) - 0.05 * mean_m * M, -M, 1.0 - M)

    # dT/dt: aggregate institutional trust
    cdef double d_T = dclamp(
        alpha_tau * mean_tau
        - tau_decay * T
        - military_tau_cost * mean_m * (1.0 - mean_tau),
        -T, 1.0 - T,
    )

    cdef cnp.ndarray[double, ndim=1] out = np.empty(6, dtype=np.float64)
    out[0] = d_E
    out[1] = d_Phi
    out[2] = d_Pi
    out[3] = d_Psi
    out[4] = d_M
    out[5] = d_T
    return out


# ─────────────────────────────────────────────────────────────────────────── #
# alliance_cluster_derivatives — diplomacy ODE contributions                  #
# ─────────────────────────────────────────────────────────────────────────── #

def alliance_cluster_derivatives_c(
    double[:, ::1] c_arr,        # (N, 6)
    double[:, ::1] alliance,     # (N, N) ∈ [-1,+1]
    double nu_alliance,
    double nu_res_alliance,
    double alpha_hostility,
):
    """
    Return per-cluster derivative contribution (N, 6) from diplomatic relations.
    """
    cdef Py_ssize_t N = c_arr.shape[0]
    cdef Py_ssize_t i, j
    cdef double pos_sum_sigma, pos_sum_res, neg_sum_trust, neg_polar
    cdef double a_ij, pos_ij, neg_ij

    cdef cnp.ndarray[double, ndim=2] out = np.zeros((N, 6), dtype=np.float64)
    cdef double[:, ::1] deriv = out

    for i in range(N):
        pos_sum_sigma = 0.0
        pos_sum_res   = 0.0
        neg_sum_trust = 0.0
        neg_polar     = 0.0

        for j in range(N):
            a_ij = alliance[i, j]
            if a_ij > 0.0:
                pos_sum_sigma += a_ij * (c_arr[j, 0] - c_arr[i, 0])
                pos_sum_res   += a_ij * (c_arr[j, 2] - c_arr[i, 2])
            elif a_ij < 0.0:
                neg_ij = -a_ij
                neg_sum_trust += neg_ij
                neg_polar     += neg_ij * c_arr[j, 5]

        deriv[i, 0] = nu_alliance * pos_sum_sigma
        deriv[i, 2] = nu_res_alliance * pos_sum_res
        deriv[i, 4] = -alpha_hostility * neg_sum_trust * c_arr[i, 4]
        deriv[i, 5] = alpha_hostility * neg_polar * (1.0 - c_arr[i, 5])

        # Clamp
        for k in range(6):
            deriv[i, k] = dclamp(deriv[i, k], -c_arr[i, k], 1.0 - c_arr[i, k])

    return out


# ─────────────────────────────────────────────────────────────────────────── #
# Full RK4 step for GRAVITAS dynamics — entire integration in C               #
# ─────────────────────────────────────────────────────────────────────────── #

def rk4_substep_c(
    double[:, ::1] c0,           # (N, 6) cluster state
    double[::1] g0,              # (6,) global state
    double[:, ::1] adjacency,    # (N, N)
    double[:, ::1] conflict,     # (N, N)
    double military_load,
    double propaganda_load,
    double dt,
    double sigma_noise,
    object rng,                  # numpy Generator or None
    # hazard params
    double gamma_h1, double gamma_h2, double gamma_h3,
    double kappa_h, double kappa_p,
    # cluster params
    double alpha_sigma, double beta_sigma, double nu_sigma,
    double alpha_res, double hazard_res_cost,
    double military_decay,
    double military_tau_cost_cluster, double deprivation_tau_cost, double tau_decay_cluster,
    double alpha_pol_cluster, double beta_pol_cluster,
    # global params
    double alpha_exh, double beta_exh, double military_exh_coeff,
    double alpha_phi, double beta_phi, double military_phi_coeff,
    double alpha_pol_global, double beta_pol_global,
    double propaganda_pol_coeff,
    double psi_recovery, double psi_propaganda_cost,
    double alpha_tau, double tau_decay_global, double military_tau_cost_global,
):
    """
    Complete RK4 step. Returns (c_new, g_new) as ndarrays.
    Eliminates all Python object construction in the inner loop.
    """
    cdef Py_ssize_t N = c0.shape[0]
    cdef Py_ssize_t i, j, k
    cdef double half = dt * 0.5
    cdef double factor = dt / 6.0

    # Allocate working arrays
    cdef cnp.ndarray[double, ndim=2] c_new_np = np.empty((N, 6), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] g_new_np = np.empty(6, dtype=np.float64)

    # Intermediate arrays for k-stages
    cdef cnp.ndarray[double, ndim=2] c2_np = np.empty((N, 6), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] c3_np = np.empty((N, 6), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] c4_np = np.empty((N, 6), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] g2_np = np.empty(6, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] g3_np = np.empty(6, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] g4_np = np.empty(6, dtype=np.float64)

    # Hazard arrays
    cdef cnp.ndarray[double, ndim=1] h0_np = np.empty(N, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] h2_np = np.empty(N, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] h3_np = np.empty(N, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] h4_np = np.empty(N, dtype=np.float64)

    # Derivative arrays
    cdef cnp.ndarray[double, ndim=2] dc1_np = np.empty((N, 6), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] dc2_np = np.empty((N, 6), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] dc3_np = np.empty((N, 6), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] dc4_np = np.empty((N, 6), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] dg1_np = np.empty(6, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] dg2_np = np.empty(6, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] dg3_np = np.empty(6, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] dg4_np = np.empty(6, dtype=np.float64)

    # Typed memoryviews
    cdef double[:, ::1] c2 = c2_np, c3 = c3_np, c4 = c4_np, c_new = c_new_np
    cdef double[::1] g2 = g2_np, g3 = g3_np, g4 = g4_np, g_new = g_new_np
    cdef double[::1] h0 = h0_np, h2 = h2_np, h3 = h3_np, h4 = h4_np
    cdef double[:, ::1] dc1 = dc1_np, dc2 = dc2_np, dc3 = dc3_np, dc4 = dc4_np
    cdef double[::1] dg1 = dg1_np, dg2 = dg2_np, dg3 = dg3_np, dg4 = dg4_np

    # ─── Inline hazard computation ─────────────────────────────────────── #
    cdef double h_local_val, cascade_val
    cdef double *h_local_buf = <double *>malloc(N * sizeof(double))
    if h_local_buf == NULL:
        raise MemoryError()

    try:
        # ─── k1 ──────────────────────────────────────────────────────── #
        _compute_hazard_inplace(c0, conflict, g0[2], h0,
                                gamma_h1, gamma_h2, gamma_h3, kappa_h, kappa_p,
                                h_local_buf, N)
        _eval_derivs_inplace(c0, g0, h0, adjacency, dc1, dg1,
                             military_load, propaganda_load,
                             alpha_sigma, beta_sigma, nu_sigma,
                             alpha_res, hazard_res_cost, military_decay,
                             military_tau_cost_cluster, deprivation_tau_cost, tau_decay_cluster,
                             alpha_pol_cluster, beta_pol_cluster,
                             alpha_exh, beta_exh, military_exh_coeff,
                             alpha_phi, beta_phi, military_phi_coeff,
                             alpha_pol_global, beta_pol_global, propaganda_pol_coeff,
                             psi_recovery, psi_propaganda_cost,
                             alpha_tau, tau_decay_global, military_tau_cost_global, N)

        # ─── k2 ──────────────────────────────────────────────────────── #
        for i in range(N):
            for k in range(6):
                c2[i, k] = dclamp(c0[i, k] + half * dc1[i, k], 0.0, 1.0)
            c2[i, 1] = 0.0
        for k in range(6):
            g2[k] = dclamp(g0[k] + half * dg1[k], 0.0, 1.0)

        _compute_hazard_inplace(c2, conflict, g2[2], h2,
                                gamma_h1, gamma_h2, gamma_h3, kappa_h, kappa_p,
                                h_local_buf, N)
        _eval_derivs_inplace(c2, g2, h2, adjacency, dc2, dg2,
                             military_load, propaganda_load,
                             alpha_sigma, beta_sigma, nu_sigma,
                             alpha_res, hazard_res_cost, military_decay,
                             military_tau_cost_cluster, deprivation_tau_cost, tau_decay_cluster,
                             alpha_pol_cluster, beta_pol_cluster,
                             alpha_exh, beta_exh, military_exh_coeff,
                             alpha_phi, beta_phi, military_phi_coeff,
                             alpha_pol_global, beta_pol_global, propaganda_pol_coeff,
                             psi_recovery, psi_propaganda_cost,
                             alpha_tau, tau_decay_global, military_tau_cost_global, N)

        # ─── k3 ──────────────────────────────────────────────────────── #
        for i in range(N):
            for k in range(6):
                c3[i, k] = dclamp(c0[i, k] + half * dc2[i, k], 0.0, 1.0)
            c3[i, 1] = 0.0
        for k in range(6):
            g3[k] = dclamp(g0[k] + half * dg2[k], 0.0, 1.0)

        _compute_hazard_inplace(c3, conflict, g3[2], h3,
                                gamma_h1, gamma_h2, gamma_h3, kappa_h, kappa_p,
                                h_local_buf, N)
        _eval_derivs_inplace(c3, g3, h3, adjacency, dc3, dg3,
                             military_load, propaganda_load,
                             alpha_sigma, beta_sigma, nu_sigma,
                             alpha_res, hazard_res_cost, military_decay,
                             military_tau_cost_cluster, deprivation_tau_cost, tau_decay_cluster,
                             alpha_pol_cluster, beta_pol_cluster,
                             alpha_exh, beta_exh, military_exh_coeff,
                             alpha_phi, beta_phi, military_phi_coeff,
                             alpha_pol_global, beta_pol_global, propaganda_pol_coeff,
                             psi_recovery, psi_propaganda_cost,
                             alpha_tau, tau_decay_global, military_tau_cost_global, N)

        # ─── k4 ──────────────────────────────────────────────────────── #
        for i in range(N):
            for k in range(6):
                c4[i, k] = dclamp(c0[i, k] + dt * dc3[i, k], 0.0, 1.0)
            c4[i, 1] = 0.0
        for k in range(6):
            g4[k] = dclamp(g0[k] + dt * dg3[k], 0.0, 1.0)

        _compute_hazard_inplace(c4, conflict, g4[2], h4,
                                gamma_h1, gamma_h2, gamma_h3, kappa_h, kappa_p,
                                h_local_buf, N)
        _eval_derivs_inplace(c4, g4, h4, adjacency, dc4, dg4,
                             military_load, propaganda_load,
                             alpha_sigma, beta_sigma, nu_sigma,
                             alpha_res, hazard_res_cost, military_decay,
                             military_tau_cost_cluster, deprivation_tau_cost, tau_decay_cluster,
                             alpha_pol_cluster, beta_pol_cluster,
                             alpha_exh, beta_exh, military_exh_coeff,
                             alpha_phi, beta_phi, military_phi_coeff,
                             alpha_pol_global, beta_pol_global, propaganda_pol_coeff,
                             psi_recovery, psi_propaganda_cost,
                             alpha_tau, tau_decay_global, military_tau_cost_global, N)

        # ─── Weighted sum ─────────────────────────────────────────────── #
        for i in range(N):
            for k in range(6):
                c_new[i, k] = dclamp(
                    c0[i, k] + factor * (dc1[i, k] + 2.0*dc2[i, k] + 2.0*dc3[i, k] + dc4[i, k]),
                    0.0, 1.0,
                )
        for k in range(6):
            g_new[k] = dclamp(
                g0[k] + factor * (dg1[k] + 2.0*dg2[k] + 2.0*dg3[k] + dg4[k]),
                0.0, 1.0,
            )

    finally:
        free(h_local_buf)

    # ─── SDE noise (must go back to Python for rng) ────────────────────── #
    if sigma_noise > 0.0 and rng is not None:
        noise_scale = sigma_noise * sqrt(dt)
        noise_sigma = rng.standard_normal(N)
        noise_res   = rng.standard_normal(N)
        for i in range(N):
            c_new_np[i, 0] = dclamp(c_new_np[i, 0] + noise_scale * noise_sigma[i], 0.0, 1.0)
            c_new_np[i, 2] = dclamp(c_new_np[i, 2] + noise_scale * 0.5 * noise_res[i], 0.0, 1.0)

    # ─── Re-derive hazard at new state ──────────────────────────────────── #
    cdef cnp.ndarray[double, ndim=1] h_final = np.empty(N, dtype=np.float64)
    cdef double[::1] h_final_v = h_final
    cdef double *h_local_final = <double *>malloc(N * sizeof(double))
    if h_local_final == NULL:
        raise MemoryError()
    try:
        _compute_hazard_inplace(c_new, conflict, g_new[2], h_final_v,
                                gamma_h1, gamma_h2, gamma_h3, kappa_h, kappa_p,
                                h_local_final, N)
    finally:
        free(h_local_final)

    for i in range(N):
        c_new_np[i, 1] = h_final[i]

    return c_new_np, g_new_np


# ─────────────────────────────────────────────────────────────────────────── #
# Internal C-level helpers (nogil where possible)                              #
# ─────────────────────────────────────────────────────────────────────────── #

cdef void _compute_hazard_inplace(
    double[:, ::1] c_arr,
    double[:, ::1] conflict,
    double sys_pol,
    double[::1] out,
    double gamma_h1, double gamma_h2, double gamma_h3,
    double kappa_h, double kappa_p,
    double *h_local_buf,
    Py_ssize_t N,
) noexcept nogil:
    """Compute hazard into out[N] using pre-allocated scratch buffer."""
    cdef Py_ssize_t i, j
    cdef double cascade_val

    for i in range(N):
        h_local_buf[i] = (
            gamma_h1 * cpow(dclamp(1.0 - c_arr[i, 0], 0.0, 1.0), kappa_h)
                     * cpow(c_arr[i, 5], kappa_p)
            + gamma_h2 * (1.0 - c_arr[i, 4]) * sys_pol
        )

    for i in range(N):
        cascade_val = 0.0
        for j in range(N):
            cascade_val += conflict[i, j] * h_local_buf[j] * (1.0 - c_arr[j, 0])
        out[i] = dclamp(h_local_buf[i] + gamma_h3 * cascade_val, 0.0, 5.0)


cdef void _eval_derivs_inplace(
    double[:, ::1] c_arr,
    double[::1] g_arr,
    double[::1] hazard,
    double[:, ::1] adjacency,
    double[:, ::1] dc_out,
    double[::1] dg_out,
    double military_load,
    double propaganda_load,
    # cluster params
    double alpha_sigma, double beta_sigma, double nu_sigma,
    double alpha_res, double hazard_res_cost, double military_decay,
    double military_tau_cost, double deprivation_tau_cost, double tau_decay,
    double alpha_pol, double beta_pol,
    # global params
    double alpha_exh, double beta_exh, double military_exh_coeff,
    double alpha_phi, double beta_phi_param, double military_phi_coeff,
    double alpha_pol_g, double beta_pol_g, double propaganda_pol_coeff,
    double psi_recovery, double psi_propaganda_cost,
    double alpha_tau, double tau_decay_g, double military_tau_cost_g,
    Py_ssize_t N,
) noexcept nogil:
    """Evaluate cluster + global derivatives, writing into dc_out and dg_out."""
    cdef Py_ssize_t i, j
    cdef double E   = g_arr[0]
    cdef double Phi = g_arr[1]
    cdef double Pi  = g_arr[2]
    cdef double Psi = g_arr[3]
    cdef double M   = g_arr[4]
    cdef double T   = g_arr[5]
    cdef double act = 1.0 - E

    cdef double sigma_i, resource_i, military_i, trust_i, polar_i
    cdef double lap_s, deg_i
    cdef double d_s, d_r, d_m, d_t, d_p

    # ── Cluster derivatives ───────────────────────────────────────────── #
    for i in range(N):
        sigma_i    = c_arr[i, 0]
        resource_i = c_arr[i, 2]
        military_i = c_arr[i, 3]
        trust_i    = c_arr[i, 4]
        polar_i    = c_arr[i, 5]

        lap_s = 0.0
        deg_i = 0.0
        for j in range(N):
            lap_s += adjacency[i, j] * c_arr[j, 0]
            deg_i += adjacency[i, j]
        lap_s -= sigma_i * deg_i

        d_s = (alpha_sigma * act * ((1.0 - polar_i) * (1.0 - Phi) - beta_sigma * hazard[i] * sigma_i)
               + nu_sigma * lap_s)
        d_r = (alpha_res * (1.0 - resource_i) - hazard_res_cost * hazard[i] * resource_i) * act
        d_m = -military_decay * military_i
        d_t = (-military_tau_cost * military_i * (1.0 - sigma_i)
               - deprivation_tau_cost * (1.0 - resource_i) * hazard[i]
               - tau_decay * trust_i) * act
        d_p = alpha_pol * Phi * (1.0 - trust_i) - beta_pol * trust_i * (1.0 - polar_i)

        dc_out[i, 0] = dclamp(d_s, -sigma_i,    1.0 - sigma_i)
        dc_out[i, 1] = 0.0
        dc_out[i, 2] = dclamp(d_r, -resource_i,  1.0 - resource_i)
        dc_out[i, 3] = dclamp(d_m, -military_i,  1.0 - military_i)
        dc_out[i, 4] = dclamp(d_t, -trust_i,     1.0 - trust_i)
        dc_out[i, 5] = dclamp(d_p, -polar_i,     1.0 - polar_i)

    # ── Global derivatives ────────────────────────────────────────────── #
    cdef double mean_h = 0.0, mean_tau = 0.0
    for i in range(N):
        mean_h   += hazard[i]
        mean_tau += c_arr[i, 4]
    if N > 0:
        mean_h   /= N
        mean_tau /= N
    cdef double mean_m = military_load

    cdef double accum = Pi * (1.0 - Psi) * (1.0 - E) + military_exh_coeff * mean_m * M
    cdef double recov = beta_exh * (1.0 - Pi) * Psi * E
    dg_out[0] = dclamp(alpha_exh * (accum - recov), -E, 1.0 - E)

    dg_out[1] = dclamp(
        alpha_phi * mean_h * (1.0 - mean_tau) + military_phi_coeff * mean_m * mean_m * E
        - beta_phi_param * mean_tau * (1.0 - Phi), -Phi, 1.0 - Phi)

    dg_out[2] = dclamp(
        alpha_pol_g * Phi * (1.0 - T) + propaganda_pol_coeff * propaganda_load * (1.0 - T)
        - beta_pol_g * T * (1.0 - Pi), -Pi, 1.0 - Pi)

    dg_out[3] = dclamp(
        psi_recovery * (1.0 - Psi) - psi_propaganda_cost * propaganda_load,
        -Psi, 1.0 - Psi)

    dg_out[4] = dclamp(0.01 * (1.0 - M) - 0.05 * mean_m * M, -M, 1.0 - M)

    dg_out[5] = dclamp(
        alpha_tau * mean_tau - tau_decay_g * T - military_tau_cost_g * mean_m * (1.0 - mean_tau),
        -T, 1.0 - T)


# ─────────────────────────────────────────────────────────────────────────── #
# Faction-level derivatives (for the regime integrator)                        #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_faction_derivs_c(
    double[::1] powers,
    double[::1] rads,
    double[::1] cohs,
    double[::1] mems,
    double exhaustion,
    double fragmentation,
    double instability,
    double repression,
    double elite_alignment,
    double alpha_power,
    double beta_power,
    double gamma_power,
    double alpha_rad,
    double beta_rad,
    double gamma_rad,
    double alpha_coh,
    double beta_coh,
    double alpha_mem,
    double beta_mem,
    # Optional: affinity_matrix as flat (N*N) or None
    object affinity_flat = None,
):
    """
    Compute (dP/dt, dRad/dt, dCoh/dt, dMem/dt) for all factions.
    Returns tuple of 4 ndarrays each of shape (n_factions,).
    """
    cdef Py_ssize_t n = powers.shape[0]
    cdef Py_ssize_t i, j
    cdef double act = 1.0 - exhaustion

    cdef cnp.ndarray[double, ndim=1] dP_out   = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] dRad_out = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] dCoh_out = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] dMem_out = np.empty(n, dtype=np.float64)

    cdef double mean_power = 0.0
    for i in range(n):
        mean_power += powers[i]
    if n > 0:
        mean_power /= n

    cdef double aff_p_mod, aff_r_mod
    cdef bint has_aff = affinity_flat is not None
    cdef double[:, ::1] aff_view
    if has_aff:
        aff_view = np.ascontiguousarray(affinity_flat, dtype=np.float64)

    for i in range(n):
        # Affinity modifiers
        aff_p_mod = 0.0
        aff_r_mod = 0.0
        if has_aff:
            for j in range(n):
                if i != j:
                    aff_p_mod += aff_view[i, j] * powers[j]
                    aff_r_mod -= aff_view[i, j] * rads[j]
            aff_p_mod *= 0.05
            aff_r_mod *= 0.05

        # dP/dt
        dP_raw = alpha_power * act * (
            elite_alignment * cohs[i]
            - beta_power * rads[i] * (1.0 - cohs[i])
            - gamma_power * fragmentation * powers[i]
        ) + aff_p_mod
        dP_out[i] = dclamp(dP_raw, -powers[i], 1.0 - powers[i])

        # dRad/dt
        dRad_raw = alpha_rad * act * (
            mems[i] * (1.0 - cohs[i]) * (1.0 - repression)
            - beta_rad * rads[i] * cohs[i]
            - gamma_rad * rads[i] * rads[i]
        ) + aff_r_mod
        dRad_out[i] = dclamp(dRad_raw, -rads[i], 1.0 - rads[i])

        # dCoh/dt
        power_dev = fabs(powers[i] - mean_power)
        dCoh_raw = alpha_coh * act * (
            (1.0 - fragmentation) * (1.0 - rads[i])
            - beta_coh * power_dev
            - cohs[i] * rads[i] * rads[i]
        )
        dCoh_out[i] = dclamp(dCoh_raw, -cohs[i], 1.0 - cohs[i])

        # dMem/dt (NOT gated by exhaustion)
        dMem_raw = alpha_mem * (
            instability * (1.0 - cohs[i]) - beta_mem * mems[i]
        )
        mem_lo = -dclamp(mems[i], 0.0, 1.0)
        mem_hi = 1.0 - dclamp(mems[i], 0.0, 1.0)
        dMem_out[i] = dclamp(dMem_raw, mem_lo, mem_hi)

    return dP_out, dRad_out, dCoh_out, dMem_out


# ─────────────────────────────────────────────────────────────────────────── #
# Gini coefficient                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_gini_c(double[::1] powers):
    """Compute Gini coefficient of power distribution."""
    cdef Py_ssize_t n = powers.shape[0]
    if n < 2:
        return 0.0
    cdef double total = 0.0
    cdef Py_ssize_t i, j
    for i in range(n):
        total += powers[i]
    if total <= 0.0:
        return 0.0
    cdef double diff_sum = 0.0
    for i in range(n):
        for j in range(n):
            diff_sum += fabs(powers[i] - powers[j])
    cdef double gini = diff_sum / (2.0 * n * total)
    return dclamp(gini, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Economic derivatives                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_economic_derivs_c(
    double[::1] powers,
    double[::1] wealths,
    double gdp,
    double instability,
    double volatility,
    double exhaustion,
    double alpha_gdp,
    double beta_gdp,
    double wealth_extraction,
):
    """Compute d(Wealth_i)/dt and d(GDP)/dt. Returns (dWealth, dGDP)."""
    cdef Py_ssize_t n = powers.shape[0]
    cdef Py_ssize_t i

    # GDP derivative
    cdef double growth = alpha_gdp * (1.0 - instability) * (1.0 - volatility) * (1.0 - gdp)
    cdef double drag = beta_gdp * (instability + volatility) * gdp
    cdef double d_gdp = (growth - drag) * (1.0 - exhaustion)

    cdef cnp.ndarray[double, ndim=1] d_wealth_np = np.empty(n, dtype=np.float64)
    for i in range(n):
        d_wealth_np[i] = (
            wealth_extraction * powers[i] * gdp * (1.0 - wealths[i])
            - 0.05 * wealths[i] * (1.0 - powers[i])
        )

    return d_wealth_np, d_gdp


# ─────────────────────────────────────────────────────────────────────────── #
# Topological derivatives (affinity matrix)                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_topological_derivs_c(
    double[::1] rads,
    double[::1] pillars,
    double legitimacy,
    double instability,
    Py_ssize_t n_factions,
    object affinity_matrix,  # None or (n, n) ndarray
):
    """Compute d(Pillars)/dt and affinity drift. Returns (dPillars, dAff_ndarray)."""
    cdef Py_ssize_t n_p = pillars.shape[0]
    cdef Py_ssize_t i, j

    cdef cnp.ndarray[double, ndim=1] d_pillars_np = np.empty(n_p, dtype=np.float64)
    for i in range(n_p):
        d_pillars_np[i] = 0.1 * (legitimacy - pillars[i]) - 0.2 * instability * pillars[i]

    cdef Py_ssize_t n = n_factions
    cdef cnp.ndarray[double, ndim=2] d_aff_np = np.empty((n, n), dtype=np.float64)
    cdef double[:, ::1] d_aff = d_aff_np

    cdef double[:, ::1] current_aff
    cdef double rad_diff

    if affinity_matrix is None:
        for i in range(n):
            for j in range(n):
                if i == j:
                    d_aff[i, j] = 0.0
                else:
                    rad_diff = fabs(rads[i] - rads[j])
                    d_aff[i, j] = 0.1 * (0.5 - rad_diff)
    else:
        current_aff = np.ascontiguousarray(affinity_matrix, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    d_aff[i, j] = 0.0
                else:
                    rad_diff = fabs(rads[i] - rads[j])
                    d_aff[i, j] = 0.1 * (0.5 - rad_diff) - 0.05 * current_aff[i, j]

    return d_pillars_np, d_aff_np


# ─────────────────────────────────────────────────────────────────────────── #
# Exhaustion derivative                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_exhaustion_deriv_c(
    double exhaustion,
    double volatility,
    double instability,
    double alpha_exh,
    double beta_exh,
):
    """Compute dExh/dt. Returns scalar."""
    cdef double accum = volatility * instability * (1.0 - exhaustion)
    cdef double recov = beta_exh * (1.0 - volatility) * (1.0 - instability) * exhaustion
    cdef double raw = alpha_exh * (accum - recov)
    return dclamp(raw, -exhaustion, 1.0 - exhaustion)
