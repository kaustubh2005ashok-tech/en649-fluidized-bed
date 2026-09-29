"""
analysis.py — extract u_mf from a ramp, run Checkpoint 3, make the figures.

    python analysis.py --tag full
    python analysis.py --tag demo

u_mf extraction (paper §4.2: "turning point of the downward curve"):
    on the DESCENDING branch, fit a line to the fixed-bed region (low U_g) and a
    horizontal line to the plateau region (high U_g); u_mf is their intersection.
    The ascending branch is plotted but not used (it carries the overshoot spike).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drag import ergun_umf, ergun_umf_inertial, wen_yu_umf

U_MF_EXPT = 1.05      # paper §4.2 (Table 1 quotes 1.09)
U_MF_PAPER_SIM = 1.0  # paper's CFD-DEM result


def smooth(v, n=11):
    k = np.ones(n) / n
    return np.convolve(v, k, mode="same")


def extract_umf(Ug, dP, dP_theory, plateau_frac=0.9):
    """
    Descending-branch fit. Fixed-bed region follows the Ergun form dP = a U + b U²
    (quadratic because Re_p ~ 200 puts these particles in the inertial regime);
    plateau region is a constant c. u_mf solves a U + b U² = c.
    """
    dPs = smooth(dP)
    plateau = dPs > plateau_frac * dP_theory
    if plateau.sum() < 5:
        return np.nan, None
    U_edge = Ug[plateau].min()                 # lowest U_g still on the plateau
    c = dPs[plateau].mean()
    fixed = (Ug < 0.7 * U_edge) & (Ug > 0.05)
    if fixed.sum() < 5:
        return np.nan, None
    A = np.vstack([Ug[fixed], Ug[fixed] ** 2]).T
    a, b = np.linalg.lstsq(A, dPs[fixed], rcond=None)[0]
    umf = (-a + np.sqrt(a * a + 4 * b * c)) / (2 * b) if b > 0 else c / a
    return umf, dict(a=a, b=b, c=c, U_edge=U_edge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()

    d = np.load(f"out/ramp_{a.tag}.npz")
    Ug, dP, br, h = d["Ug"], d["dP"], d["branch"], d["h"]
    dP_th = float(d["dP_theory"])
    eps_mf = float(d["eps_mf"])
    dp, rho_g = float(d["dp"]), 1.21
    rho_p, mu_g, g = 1131.0, 1.8e-5, float(d["g"])

    up, dn = br == 1, br == -1

    # ---- u_mf from descending branch ----
    umf, fit = extract_umf(Ug[dn], dP[dn], dP_th)
    umf_ergun = ergun_umf(eps_mf, dp, rho_p, rho_g, mu_g, g)
    umf_ergun_i = ergun_umf_inertial(eps_mf, dp, rho_p, rho_g, g)
    umf_wy = wen_yu_umf(dp, rho_p, rho_g, mu_g, g)

    # ---- Checkpoint 3: plateau vs bed weight ----
    plateau_mask = dn & (Ug > 1.15 * umf) if np.isfinite(umf) else dn & (Ug > 0.8 * Ug.max())
    plateau = dP[plateau_mask].mean()
    ok3 = abs(plateau - dP_th) / dP_th < 0.10

    print(f"bed:  N={int(d['N'])}  eps_mf={eps_mf:.4f}  dP_theory={dP_th:.1f} Pa")
    print(f"[3 ] plateau dP = {plateau:.1f} Pa  ({100*plateau/dP_th:.1f}% of theory)   "
          f"{'✓ PASS' if ok3 else '✗ FAIL'}")
    print()
    print(f"u_mf (simulation, descending branch) = {umf:.3f} m/s")
    print(f"u_mf Ergun (full, at eps_mf)          = {umf_ergun:.3f} m/s   dev = {100*(umf/umf_ergun-1):+.1f}%")
    print(f"u_mf Ergun (inertial, C eps^1.5)      = {umf_ergun_i:.3f} m/s")
    print(f"u_mf Wen–Yu correlation               = {umf_wy:.3f} m/s")
    print(f"u_mf paper simulation                 = {U_MF_PAPER_SIM:.2f} m/s   dev = {100*(umf/U_MF_PAPER_SIM-1):+.1f}%")
    print(f"u_mf experiment (NETL SSCP)           = {U_MF_EXPT:.2f} m/s   dev = {100*(umf/U_MF_EXPT-1):+.1f}%")
    print(f"packing-density factor (eps_mf/0.42)^1.5 = {(eps_mf/0.42)**1.5:.3f}")

    # ---- figure 1: fluidization curve ----
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(Ug[up], dP[up], "-", color="#1f6f8b", lw=1.2, alpha=0.6, label="fluidization (ascending)")
    ax.plot(Ug[dn], dP[dn], "-", color="#c1440e", lw=1.6, label="de-fluidization (descending)")
    ax.axhline(dP_th, ls="--", color="k", lw=1, label=r"bed weight / area")
    if fit:
        Uf = np.linspace(0, umf * 1.05, 20)
        ax.plot(Uf, fit["a"] * Uf + fit["b"] * Uf ** 2, ":", color="gray", lw=1)
        ax.axvline(umf, color="gray", ls=":", lw=1)
        ax.annotate(rf"$u_{{mf}}$ = {umf:.2f} m/s", (umf, dP_th * 1.38), xytext=(6, 0),
                    textcoords="offset points", fontsize=9)
    ax.axvline(U_MF_EXPT, color="green", ls="-.", lw=1, label=f"experiment {U_MF_EXPT} m/s")
    ax.set_xlabel(r"superficial gas velocity $U_g$ (m/s)")
    ax.set_ylabel(r"$\Delta P_{bed}$ (Pa)")
    ax.set_title(rf"2-D DEM fluidization curve  (N={int(d['N'])}, $\varepsilon_{{mf}}$={eps_mf:.3f})", fontsize=10)
    ax.set_xlim(0, Ug.max()); ax.set_ylim(0, dP_th * 1.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"out/fluidization_curve_{a.tag}.png", dpi=150); plt.close(fig)

    # ---- figure 2: bed expansion ----
    fig, ax = plt.subplots(figsize=(7, 3.6))
    h0 = float(d["h0"])
    ax.plot(Ug[up], h[up] / h0, "-", color="#1f6f8b", alpha=0.6, label="ascending")
    ax.plot(Ug[dn], h[dn] / h0, "-", color="#c1440e", label="descending")
    ax.axvline(U_MF_EXPT, color="green", ls="-.", lw=1)
    ax.set_xlabel(r"$U_g$ (m/s)"); ax.set_ylabel(r"bed expansion $h/h_0$")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"out/bed_expansion_{a.tag}.png", dpi=150); plt.close(fig)

    # ---- write a results row ----
    with open(f"out/results_{a.tag}.txt", "w") as f:
        f.write(f"tag={a.tag} N={int(d['N'])} en={float(d['en'])} mu={float(d['mu_f'])} seed={int(d['seed'])}\n")
        f.write(f"eps_mf={eps_mf:.4f} dP_theory={dP_th:.1f} plateau={plateau:.1f}\n")
        f.write(f"umf_sim={umf:.4f} umf_ergun={umf_ergun:.4f} umf_expt={U_MF_EXPT}\n")
    print(f"\nsaved out/fluidization_curve_{a.tag}.png, out/bed_expansion_{a.tag}.png, out/results_{a.tag}.txt")


if __name__ == "__main__":
    main()
