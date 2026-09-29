"""
verify.py — regression checkpoints. Run at the start of every session and after any
change to contact.py / drag.py / sim.py.

  1a  binary collision reproduces the prescribed e_n
  1b  gravity settling: kinetic energy decays monotonically to ~0 (catches sign errors)
  2   frozen lattice at known voidage: summed drag / area matches Ergun ΔP/L
  3   (run after run_ramp.py) fluidized plateau ≈ N m g / (W d_p)   — see analysis.py

Usage:  python verify.py            (all)      python verify.py 1a 2   (subset)
"""
import sys
import numpy as np
from params import Params
from sim import step, kinetic_energy
from bed import build_bed, lattice_layer
from drag import voidage, apply_drag, ergun_dPdz

PASS = "\u2713 PASS"
FAIL = "\u2717 FAIL"


def checkpoint_1a(tol=0.02):
    """Two particles, head-on, no gravity, no drag. Measure e_n = -v_rel_after / v_rel_before."""
    p = Params(W=0.05, H=0.05, g=0.0)
    v0 = 0.5
    x = np.array([0.02, 0.02 + 2 * p.R + 1e-5])
    y = np.array([0.025, 0.025])
    vx = np.array([v0, -v0])
    vy = np.zeros(2)
    args = p.kernel_args(2)
    nsteps = int(3 * p.tc / p.dt)
    step(x, y, vx, vy, 2, 0.0, nsteps, 0, *args)
    e_meas = -(vx[1] - vx[0]) / (-2 * v0)
    ok = abs(e_meas - p.en) < tol
    print(f"[1a] restitution: e_n(input)={p.en:.3f}  e_n(measured)={e_meas:.4f}   {PASS if ok else FAIL}")
    return ok


def checkpoint_1b(t_settle=0.25, ke_tol=1e-5):
    """Settle the small demo bed under gravity only; KE must fall to near zero and not grow."""
    p = Params(W=0.09, H=0.34, nrow=26)
    state = build_bed(p)
    x, y, vx, vy = state
    N = x.size
    args = p.kernel_args(N)
    chunk = 15000
    ke_hist = []
    for k in range(int(t_settle / (chunk * p.dt))):
        step(x, y, vx, vy, N, 0.0, chunk, 0, *args)
        ke_hist.append(kinetic_energy(vx, vy, p.m))
    ke = np.array(ke_hist)
    peak = ke.argmax()
    tail = ke[peak + 1:]
    monotone = np.all(np.diff(tail) <= 0.15 * tail[:-1] + 1e-12)   # allow small rebounds
    ok = ke[-1] < ke_tol and monotone and y.max() < 0.12
    print(f"[1b] settling:   N={N}  KE_final={ke[-1]:.2e} J  h={y.max():.4f} m  "
          f"decay {'monotone' if monotone else 'NON-MONOTONE'}   {PASS if ok else FAIL}")
    return ok


def checkpoint_2(tol=0.03):
    """
    (a) β algebra: β(ε, U/ε)·(U/ε) must equal Ergun ΔP/L(U, ε) on the dense branch.
    (b) cell-aligned frozen lattice: summed drag / (W d_p L) must equal Ergun ΔP/L.
    """
    from drag import beta_gidaspow
    ok_all = True
    print("[2a] beta algebra vs Ergun (dense branch):")
    p = Params()
    for eps in (0.40, 0.46, 0.55, 0.70, 0.79):
        for U in (0.3, 1.0, 2.0):
            u = U / eps
            dPdz_beta = beta_gidaspow(eps, u, p.dp, p.rho_g, p.mu_g, p.eps_switch) * u
            dPdz_erg = ergun_dPdz(U, eps, p.dp, p.rho_g, p.mu_g)
            err = abs(dPdz_beta - dPdz_erg) / dPdz_erg
            ok = err < 1e-9
            ok_all &= ok
        print(f"     eps={eps:.2f}  max rel err over U = {err:.1e}   {PASS if ok else FAIL}")

    print("[2b] summed drag on a cell-aligned lattice vs Ergun:")
    for cell_dp, npc in ((2.0, 2), (3.0, 2)):
        p = Params(W=0.23, H=0.30, cell_dp=cell_dp)
        state, eps_exact, L = lattice_layer(p, npc, ncells_y=4)
        x, y, vx, vy = state
        N = x.size
        eps = voidage(x, y, N, p.NCX, p.NCY, p.dx, p.dy, p.Vp, p.dp, p.eps_min)
        for Ug in (0.5, 1.0):
            fy = np.zeros(N)
            F = apply_drag(x, y, vy, fy, N, Ug, eps, p.NCX, p.NCY, p.dx, p.dy,
                           p.dp, p.rho_g, p.mu_g, p.Vp, p.eps_switch)
            dPdz_sim = F / (p.W * p.dp) / L
            dPdz_erg = ergun_dPdz(Ug, eps_exact, p.dp, p.rho_g, p.mu_g)
            err = abs(dPdz_sim - dPdz_erg) / dPdz_erg
            ok = err < tol
            ok_all &= ok
            print(f"     N={N:4d}  eps={eps_exact:.3f}  Ug={Ug:.1f}  sim={dPdz_sim:9.1f}  "
                  f"Ergun={dPdz_erg:9.1f} Pa/m  err={100*err:4.2f}%  {PASS if ok else FAIL}")
    return ok_all


if __name__ == "__main__":
    which = sys.argv[1:] or ["1a", "1b", "2"]
    results = {}
    if "1a" in which:
        results["1a"] = checkpoint_1a()
    if "1b" in which:
        results["1b"] = checkpoint_1b()
    if "2" in which:
        results["2"] = checkpoint_2()
    print()
    print("ALL PASSED" if all(results.values()) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results.values()) else 1)
