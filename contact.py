"""
contact.py — soft-sphere spring-dashpot-slider contact kernels (Cundall & Strack).

Sign conventions (these are the three places that bit us during development):

  * Normal:  F_n = k_n δ − η_n v_n   with v_n = (v_j − v_i)·n̂, n̂ from i to j.
             v_n < 0 when approaching → −η_n v_n > 0 adds to repulsion.  Clamped F_n ≥ 0.
  * Pair tangential:  F_t = +η_t v_t  (NOT −η_t v_t).  Power on the pair is −F_t v_t,
             so dissipation requires the plus sign.  Capped by Coulomb |F_t| ≤ μ F_n.
  * Wall: penetration rate is −v_x at the left wall and +v_x at the right wall, so
             F = k δ − η v_x  (left),  F = k δ + η v_x  (right),  F = k δ − η v_y  (floor).

Rotation is NOT integrated (tangential dashpot only, no history spring). With the
paper's stiffness the rotational ODE is stiff (I ≈ 2e-11 kg m²) and diverges under
a dashpot-only tangential model; a history spring (k_t δ_t) is required first.
"""
import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def pair_force(rx, ry, rvx, rvy, R, kn, etan, etat, mu_f):
    """
    Force on particle i from particle j.
    rx, ry   : r_j − r_i
    rvx, rvy : v_j − v_i
    Returns (Fx, Fy, in_contact).  Force on j is (−Fx, −Fy).
    """
    d2 = rx * rx + ry * ry
    if d2 >= (2.0 * R) * (2.0 * R) or d2 < 1e-18:
        return 0.0, 0.0, False
    d = np.sqrt(d2)
    nx = rx / d
    ny = ry / d
    delta = 2.0 * R - d

    vn = rvx * nx + rvy * ny
    Fn = kn * delta - etan * vn
    if Fn < 0.0:
        Fn = 0.0                       # no cohesion

    tx = -ny
    ty = nx
    vt = rvx * tx + rvy * ty
    Ft = etat * vt                     # + sign: dissipative on the pair
    cap = mu_f * Fn
    if Ft > cap:
        Ft = cap
    elif Ft < -cap:
        Ft = -cap

    Fx = -Fn * nx + Ft * tx
    Fy = -Fn * ny + Ft * ty
    return Fx, Fy, True


@njit(cache=True, error_model="numpy")
def wall_forces(x, y, vx, vy, R, W, kn, etan, etat, mu_f):
    """
    Left/right/floor walls: infinite-mass flat surfaces with the same contact law.
    Floor is frictional (Coulomb-capped dashpot). Returns (Fx, Fy).
    """
    Fx = 0.0
    Fy = 0.0
    # left wall
    if x < R:
        delta = R - x
        Fn = kn * delta - etan * vx
        if Fn > 0.0:
            Fx += Fn
    # right wall
    if x > W - R:
        delta = x - (W - R)
        Fn = kn * delta + etan * vx
        if Fn > 0.0:
            Fx -= Fn
    # floor (distributor plate)
    if y < R:
        delta = R - y
        Fn = kn * delta - etan * vy
        if Fn > 0.0:
            Fy += Fn
            Ft = -etat * vx
            cap = mu_f * Fn
            if Ft > cap:
                Ft = cap
            elif Ft < -cap:
                Ft = -cap
            Fx += Ft
    return Fx, Fy
