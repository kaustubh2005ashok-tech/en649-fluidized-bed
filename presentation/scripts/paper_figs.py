"""
paper_figs.py — figures of Liu & van Wachem (2019), Powder Technol. 343, 145–158, reproduced with this
2-D DEM code in the one-way and quasi-two-way (1-D volume-flux) couplings.

    python paper_figs.py 1 2            # Fig. 1 (drag models), Fig. 2 (ΔP vs u_g)
    python paper_figs.py 3 4 5 ...      # need out/hold_full_U*_{1way,2way}.npz from run_hold.py
    python paper_figs.py all            # everything that has data

PNGs go to figs_paper/. The experiment (NETL SSCP) data are not included, so figures that overlay
measurements show the simulation only (Figs 10–15), and Fig. 15 (error vs experiment) cannot be made.
"""
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("FIG_DIR", "figs_paper")
os.makedirs(OUT, exist_ok=True)

# validated reference palette (light surface); fixed slot order
C = dict(oneway="#2a78d6", twoway="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3de", surf="#fcfcfb")
plt.rcParams.update({
    "figure.facecolor": C["surf"], "axes.facecolor": C["surf"], "savefig.facecolor": C["surf"],
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink2"], "text.color": C["ink"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 1.6,
})

# paper constants (Tables 1–2)
DP, RHO_P, RHO_G, MU_G, G = 3.256e-3, 1131.0, 1.21, 1.8e-5, 9.81
UMF_EXPT, UMF_PAPER = 1.05, 1.00
UG_PAPER = {"2 u$_{mf}$": 2.19, "3 u$_{mf}$": 3.28, "4 u$_{mf}$": 4.38}
BED_W = 0.23


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ Fig. 1: drag models
def _cd(Re):
    Re = np.maximum(Re, 1e-12)
    return np.where(Re <= 1000, 24 / Re * (1 + 0.15 * Re ** 0.687), 0.44)


def beta_models(eg, urel=3.0, umf=1.09, eps_mf=0.44):
    """β (kg/m³/s, Table A1 of the paper) for WY, WYE, DF, SYO, BEE at gas fraction eg, |u−v| = urel."""
    es = 1 - eg
    Re = eg * RHO_G * DP * urel / MU_G                  # particle Reynolds number with ε_g
    pref = MU_G * es / DP ** 2
    wy = pref * 0.75 * _cd(Re) * Re * eg ** -2.65
    wye = np.where(eg < 0.8, pref * (150 * es / eg + 1.75 * Re), wy)
    chi = 3.7 - 0.65 * np.exp(-0.5 * (1.5 - np.log10(Re)) ** 2)
    df = pref * 0.75 * (0.63 + 4.8 / np.sqrt(Re)) ** 2 * Re * eg ** (1 - chi)
    # Syamlal–O'Brien, written as in Table A1: the Reynolds number in the V_rm and C_d terms is Re/ε_g,
    # i.e. without the ε_g factor. c2 is chosen so the model reproduces u_mf at ε_mf (c1 from continuity at 0.85).
    def syo_beta(eg_, u_, c2):
        es_ = 1 - eg_
        Re_ = eg_ * RHO_G * DP * u_ / MU_G
        Rs = Re_ / eg_
        c1 = 0.85 ** (c2 - 1.28)
        A = eg_ ** 4.14
        B = np.where(eg_ <= 0.85, c1 * eg_ ** 1.28, eg_ ** c2)
        Vr = 0.5 * (A - 0.06 * Rs + np.sqrt((0.06 * Rs) ** 2 + 0.12 * Rs * (2 * B - A) + A ** 2))
        # prefactor uses Re (with ε_g) so β carries the ε_g factor, like the WY and WYE rows
        return MU_G * es_ / DP ** 2 * 0.75 * (Re_ / Vr ** 2) * (0.63 + 4.8 * np.sqrt(Vr / Rs)) ** 2

    def syo(eg_, c2):
        return syo_beta(eg_, urel, c2)

    # calibrate c2 on the force balance at minimum fluidization. With the paper's gas equation (ε_g ∇p) and
    # F_d = β (u−v) V_p/ε_s the balance is β u_int = ε_g ε_s (ρp−ρg) g at (ε_mf, u_int = u_mf/ε_mf).
    def mf_resid(c2):
        u = umf / eps_mf
        return float(syo_beta(eps_mf, u, c2)) * u - eps_mf * (1 - eps_mf) * (RHO_P - RHO_G) * G
    lo, hi = 1.0, 6.0
    f_lo = mf_resid(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.sign(mf_resid(mid)) == np.sign(f_lo):
            lo = mid
        else:
            hi = mid
    c2 = 0.5 * (lo + hi)
    syo_b = syo(eg, c2)
    Bb = 0.31 * (1 / eg + 3 * eg * es + 8.4 * Re ** -0.343) / (1 + 10 ** (3 * es) * Re ** (2 * eg - 2.5))
    bee = pref * (180 * es / eg + 18 * eg ** 3 * (1 + 1.5 * np.sqrt(es)) + Bb * Re)
    return dict(WY=wy, WYE=wye, DF=df, SYO=syo_b, BEE=bee), c2


def fig1():
    eg = np.linspace(0.4, 0.9999, 500)
    b, c2 = beta_models(eg)
    es = 1 - eg
    m = RHO_P * np.pi / 6 * DP ** 3
    Vp = np.pi / 6 * DP ** 3
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cols = dict(WY=C["oneway"], WYE=C["twoway"], DF=C["aqua"], SYO=C["yellow"], BEE=C["magenta"])
    for k, v in b.items():
        Fd = v * 3.0 * Vp / es / m                       # F_d = β(u−v) V_p/ε_s, per unit particle mass (m/s²)
        ax.plot(eg, Fd / G, color=cols[k], label=k + (" (Gidaspow; Table A1 as printed)" if k == "WYE" else ""))
    from drag import beta_gidaspow                      # the Gidaspow actually used in this code
    bc = np.array([beta_gidaspow(float(e), 3.0, DP, RHO_G, MU_G, 0.8) for e in eg])
    ax.plot(eg, bc * 3.0 * Vp / es / m / G, color=C["ink"], ls="--", lw=1.2, label="WYE as coded in drag.py")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("gas volume fraction ε$_g$")
    ax.set_ylabel("drag force / m g  (−)")
    ax.set_title("Fig. 1 — drag models at |u$_g$ − v$_p$| = 3 m/s (Table A1)")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(0.41, 0.03 + 0.0, f"SYO constant c$_2$ = {c2:.2f}, set so SYO gives u$_{{mf}}$ = 1.09 m/s at ε$_{{mf}}$ = 0.44",
            color=C["ink2"], fontsize=7.5)
    save(fig, "fig01_drag_models.png")


# ------------------------------------------------------------------ Fig. 2: ΔP vs u_g
def _binned(r, branch, edges):
    m = r["branch"] == branch
    Ug, dP = r["Ug"][m], r["dP"][m]
    idx = np.digitize(Ug, edges) - 1
    c = 0.5 * (edges[1:] + edges[:-1])
    out = np.array([dP[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(c))])
    return c, out


def _ramp_files(kind):
    pat = "out/ramp_full_5to1_2way_s*.npz" if kind == "twoway" else "out/ramp_full_5to1_s*.npz"
    seed7 = "out/ramp_full_5to1_2way.npz" if kind == "twoway" else "out/ramp_full_5to1.npz"
    fs = [seed7] + sorted(glob.glob(pat))
    return [f for f in fs if not f.endswith("_s23.npz")]          # bed 23 excluded (README)


def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    labels = dict(oneway="one-way 5:1", twoway="quasi-two-way 5:1")
    edges = np.arange(0, 2.25, 0.05)
    r7 = {k: np.load(_ramp_files(k)[0]) for k in labels}
    dPth = float(r7["oneway"]["dP_theory"])
    ax = axs[0]
    for k in labels:
        r = r7[k]
        up, dn = r["branch"] == 1, r["branch"] == -1
        ax.plot(r["Ug"][up], r["dP"][up], color=C[k], lw=1.1, label=labels[k] + " ↑")
        ax.plot(r["Ug"][dn], r["dP"][dn], color=C[k], lw=1.1, ls="--", label=labels[k] + " ↓")
    ax.axhline(dPth, color=C["ink2"], lw=0.8, ls=":")
    ax.text(0.05, dPth + 15, f"bed weight / area = {dPth:.0f} Pa", color=C["ink2"], fontsize=8)
    ax.set_title("(a) one bed (seed 7)")
    ax.set_xlabel("fluidizing gas velocity u$_g$ (m/s)")
    ax.set_ylabel("bed pressure drop ΔP$_{bed}$ (Pa)")
    ax.legend(loc="lower right", fontsize=8)
    ax = axs[1]
    for k in labels:
        ups, dns = [], []
        for f in _ramp_files(k):
            r = np.load(f)
            ups.append(_binned(r, 1, edges)[1])
            dns.append(_binned(r, -1, edges)[1])
        c = 0.5 * (edges[1:] + edges[:-1])
        for arr, ls, lab in ((ups, "-", " ↑"), (dns, "--", " ↓")):
            a = np.array(arr)
            mu, sd = np.nanmean(a, 0), np.nanstd(a, 0)
            ax.plot(c, mu, color=C[k], ls=ls, lw=1.4, label=f"{labels[k]}{lab}")
            if ls == "-":
                ax.fill_between(c, mu - sd, mu + sd, color=C[k], alpha=0.15, lw=0)
    ax.axhline(dPth, color=C["ink2"], lw=0.8, ls=":")
    for u, lab in ((UMF_PAPER, "paper 1.00"), (UMF_EXPT, "expt 1.05")):
        ax.axvline(u, color=C["ink2"], lw=0.6, ls=":")
    ax.text(UMF_PAPER - 0.02, 40, "u$_{mf}$ paper 1.00 / expt 1.05", rotation=90, ha="right", color=C["ink2"], fontsize=8)
    n1, n2 = len(_ramp_files("oneway")), len(_ramp_files("twoway"))
    ax.set_title(f"(b) mean ± SD over beds (one-way n={n1}, quasi-two-way n={n2}; bed 23 excluded)")
    ax.set_xlabel("fluidizing gas velocity u$_g$ (m/s)")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Fig. 2 — averaged bed pressure drop vs fluidizing gas velocity (solid ↑, dashed ↓)", y=1.02)
    save(fig, "fig02_dP_vs_ug.png")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2"]
    todo = {"1": fig1, "2": fig2}
    import paper_flow                                   # Figs 3–14, A1–A3 (need the run_hold.py output)
    todo.update(paper_flow.FIGS)
    for w in (list(todo) if want == ["all"] else want):
        if w in todo:
            todo[w]()
        else:
            print("unknown figure", w)
