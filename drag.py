"""
drag.py — gas-solid coupling: voidage field, Gidaspow β, per-particle drag, Ergun reference.

Coupling scheme (one-way, unresolved):
  * The gas phase is NOT solved as a momentum equation. A superficial velocity U_g is
    prescribed at the distributor; the interstitial velocity in each cell is u_g = U_g / ε_g.
  * The particle-phase drag term (paper Eq. 3) is F_d = β (u_g − v_p) V_p / ε_s.
  * The fluid-phase reaction term S_p (paper Eq. 2) is not implemented.
  * The voidage field is refreshed every `nfluid` DEM steps (default 1 → fluid : DEM
    ratio 1 : 1; 5 mimics the paper's 5 : 1, which arises from their coupled
    Navier–Stokes solve). β and the drag are evaluated every DEM step from that field.

Voidage uses the 2-D monolayer slab convention: cell volume = Δx Δy d_p, particle
volume = π d_p³ / 6, so a close-packed 2-D layer gives ε_g ≈ 0.46 (physical) rather
than the areal value ≈ 0.2 (outside the range of the correlations).

β is written for the INTERSTITIAL slip velocity, so that ΔP/L = β (u_g − u_s).
Substituting U = ε u into Ergun gives the ε² and ε¹ denominators below — this is the
factor that was missing in the first implementation and made u_mf 60 % too high.
"""
import numpy as np
from numba import njit
from scipy.optimize import brentq


# --------------------------------------------------------------------------- voidage
@njit(cache=True, error_model="numpy")
def voidage(x, y, N, NCX, NCY, dx, dy, Vp, dp, eps_min):
    """Gas volume fraction per fluid cell (slab of thickness d_p)."""
    solid = np.zeros((NCX, NCY))
    for i in range(N):
        cx = int(x[i] / dx)
        cy = int(y[i] / dy)
        if 0 <= cx < NCX and 0 <= cy < NCY:
            solid[cx, cy] += Vp
    eps = 1.0 - solid / (dx * dy * dp)
    for a in range(NCX):
        for b in range(NCY):
            if eps[a, b] < eps_min:
                eps[a, b] = eps_min
            if eps[a, b] > 1.0:
                eps[a, b] = 1.0
    return eps


# --------------------------------------------------------------------------- Gidaspow
@njit(cache=True, error_model="numpy")
def beta_gidaspow(eps, urel, dp, rg, mug, eps_switch):
    """
    Inter-phase momentum exchange coefficient β [kg m^-3 s^-1], interstitial-slip form.
    Dense (ε < eps_switch): Ergun.  Dilute: Wen & Yu.  Discontinuous at the switch.
    """
    es = 1.0 - eps
    au = abs(urel)
    if au < 1e-6:                       # floor the VELOCITY, not Re, so C_d·Re cancels cleanly
        au = 1e-6
    if eps < eps_switch:
        return (150.0 * es * es * mug / (eps * eps * dp * dp)
                + 1.75 * es * rg * au / (eps * dp))
    Re = eps * rg * au * dp / mug
    if Re < 1e-6:
        Re = 1e-6
    if Re <= 1000.0:
        Cd = 24.0 / Re * (1.0 + 0.15 * Re ** 0.687)
    else:
        Cd = 0.44
    return 0.75 * Cd * es * eps * rg * au / dp * eps ** (-2.65)


@njit(cache=True, error_model="numpy")
def apply_drag(x, y, vy, fy, N, Ug, eps, NCX, NCY, dx, dy, dp, rg, mug, Vp, eps_switch):
    """
    Add vertical drag to fy for every particle. Returns the total drag force (N),
    which divided by (W d_p) is the bed pressure drop.
    """
    dtot = 0.0
    for i in range(N):
        cx = int(x[i] / dx)
        cy = int(y[i] / dy)
        if cx < 0:
            cx = 0
        if cx >= NCX:
            cx = NCX - 1
        if cy < 0:
            cy = 0
        if cy >= NCY:
            cy = NCY - 1
        e = eps[cx, cy]
        es = 1.0 - e
        if es < 1e-4:
            es = 1e-4
        ug = Ug / e
        urel = ug - vy[i]
        bt = beta_gidaspow(e, urel, dp, rg, mug, eps_switch)
        Fd = bt * urel * Vp / es
        fy[i] += Fd
        dtot += Fd
    return dtot


# --------------------------------------------------------------------------- Ergun reference (pure Python)
def ergun_dPdz(U, eps, dp, rho_g, mu_g):
    """Ergun pressure gradient [Pa/m] for SUPERFICIAL velocity U."""
    es = 1.0 - eps
    return (150.0 * mu_g * U * es * es / (eps ** 3 * dp ** 2)
            + 1.75 * rho_g * U * U * es / (eps ** 3 * dp))


def ergun_umf(eps, dp, rho_p, rho_g, mu_g, g=9.81):
    """Minimum fluidization velocity from the full Ergun balance (numerical root)."""
    rhs = (1.0 - eps) * (rho_p - rho_g) * g
    return brentq(lambda U: ergun_dPdz(U, eps, dp, rho_g, mu_g) - rhs, 1e-6, 50.0)


def ergun_umf_inertial(eps, dp, rho_p, rho_g, g=9.81):
    """
    Inertial-regime closed form  u_mf = C ε^{3/2},  C = sqrt((ρ_p−ρ_g) g d_p / (1.75 ρ_g)).
    For the SSCP particles C ≈ 4.13 m/s. Overpredicts the full solve by ~8–10 %.
    """
    C = np.sqrt((rho_p - rho_g) * g * dp / (1.75 * rho_g))
    return C * eps ** 1.5


def wen_yu_umf(dp, rho_p, rho_g, mu_g, g=9.81):
    """Wen & Yu (1966) correlation, independent of ε_mf: Re = sqrt(33.7² + 0.0408 Ar) − 33.7."""
    Ar = rho_g * (rho_p - rho_g) * g * dp ** 3 / mu_g ** 2
    Re = np.sqrt(33.7 ** 2 + 0.0408 * Ar) - 33.7
    return Re * mu_g / (rho_g * dp)
