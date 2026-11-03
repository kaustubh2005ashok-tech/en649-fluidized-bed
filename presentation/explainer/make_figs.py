"""
make_figs.py — figures for the project explainer (explainer.html → EN649_project_explained.pdf).

    python make_figs.py          # writes figs/*.png and prints the numbers quoted in the text

Run data come from en649_2way/out/ (ramp_full_5to1*.npz, hold_full_U*_*.npz); the loaders and the momentum
residual are reused from ../poster/make_poster.py so the explainer and the poster quote the same numbers.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(HERE, "..", "poster"))
sys.path.insert(0, REPO)
import make_poster as mp                                     # noqa: E402  (sets rcParams too)
from params import Params                                    # noqa: E402
from drag import ergun_dPdz, ergun_umf, wen_yu_umf            # noqa: E402

FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
C = mp.C
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 8.5, "lines.linewidth": 1.5})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=250, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2: one collision, our integrator
def collision(p, dt_factor, v0=1.0):
    """Head-on collision of two equal spheres, reduced to one m_eff oscillator, integrated with the same
    symplectic Euler and force law as sim.py / contact.py. Returns t, overlap, force, measured e_n."""
    dt = p.tc / dt_factor
    d, v = 0.0, v0                                            # overlap and its rate (approach positive)
    t, D, F = [0.0], [0.0], [0.0]
    while True:
        f = max(0.0, p.kn * d + p.etan * v)                  # repulsive force, clamped (no cohesion)
        v += -dt * f / p.meff
        d += dt * v
        t.append(t[-1] + dt)
        D.append(d)
        F.append(f)
        if d <= 0 and len(t) > 3:
            return np.array(t), np.array(D), np.array(F), -v / v0


def fig_collision(p):
    t, D, F, e = collision(p, 50)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axs[0].plot(t * 1e6, D * 1e6, color=C["navy"])
    axs[0].set_xlabel("time (µs)")
    axs[0].set_ylabel("overlap δ (µm)")
    axs[0].set_title(f"impact at 1 m/s, Δt = t$_c$/50 = {p.dt * 1e6:.1f} µs")
    axs[0].text(0.5, 0.08, f"t$_c$ = {p.tc * 1e6:.0f} µs, δ$_{{max}}$ = {D.max() * 1e6:.0f} µm "
                f"({D.max() / p.dp * 100:.1f} % of d$_p$)", transform=axs[0].transAxes, ha="center", fontsize=8.5)
    fs = np.array([5, 6, 8, 10, 15, 20, 30, 50, 80, 120])
    es = np.array([collision(p, f)[3] for f in fs])
    axs[1].axhline(p.en, color=C["ink2"], ls=":", lw=1)
    axs[1].text(11, p.en - 0.012, f"input e$_n$ = {p.en}", fontsize=8.5, color=C["ink2"])
    axs[1].semilogx(fs, es, "o-", color=C["twoway"], ms=4)
    for f, lab in ((8.3, "paper\nΔt ≈ t$_c$/8"), (50, "this code\nt$_c$/50")):
        axs[1].axvline(f, color=C["ink2"], lw=0.8, ls="--")
        axs[1].text(f * 1.06, 0.75, lab, fontsize=8, color=C["ink2"])
    axs[1].set_xlabel("steps per contact, t$_c$ / Δt")
    axs[1].set_ylabel("measured e$_n$")
    axs[1].set_title("restitution actually produced by symplectic Euler")
    axs[1].set_ylim(0.72, 0.9)
    axs[1].set_xticks([5, 10, 20, 50, 100])
    axs[1].set_xticklabels(["5", "10", "20", "50", "100"])
    fig.tight_layout()
    save(fig, "fig2_collision.png")
    return dict(e50=e, e8=collision(p, 8.3)[3], dmax=D.max())


# ------------------------------------------------------------------ Fig. 3: ramp ΔP–U_g
def fig_ramp(p, eps_mf=0.441):
    edges = np.arange(0, 2.25, 0.05)
    c = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    U = np.linspace(0, 1.6, 200)
    L = 0.146                                                  # settled height, m
    ax.plot(U, np.minimum(ergun_dPdz(U, eps_mf, p.dp, p.rho_g, p.mu_g) * L, 1e9), color="#9a9a9a", lw=1.1,
            ls="-.", label=f"Ergun, fixed bed at ε = {eps_mf}")
    vals = {}
    for k, lab in (("oneway", "one-way"), ("twoway", "quasi-two-way")):
        for br, ls, arrow in ((1, "-", "up"), (-1, "--", "down")):
            rows = []
            for f in mp._ramp_files(k):
                r = np.load(f)
                msk = r["branch"] == br
                idx = np.digitize(r["Ug"][msk], edges) - 1
                rows.append([r["dP"][msk][idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(c))])
            a = np.array(rows)
            mu, sd = np.nanmean(a, 0), np.nanstd(a, 0)
            vals[(k, br)] = (c, mu)
            ax.plot(c, mu, color=C[k], ls=ls, lw=1.7, label=f"{lab}, {arrow}")
            if br == 1:
                ax.fill_between(c, mu - sd, mu + sd, color=C[k], alpha=0.15, lw=0)
    ax.axhline(mp.WEIGHT, color=C["ink2"], lw=0.9, ls=":")
    ax.text(1.58, mp.WEIGHT - 95, "bed weight 905 Pa", ha="right", fontsize=8.5, color=C["ink2"])
    ax.axvline(0.99, color=C["ink2"], lw=0.9, ls=":")
    ax.text(0.975, 60, "u$_{mf}$ = 0.99 m/s", ha="right", fontsize=8.5)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1300)
    ax.set_xlabel("superficial gas velocity U$_g$ (m/s)")
    ax.set_ylabel("ΔP$_{bed}$ (Pa)")
    ax.legend(loc="upper left", ncol=1, fontsize=8, title="14-bed mean (± SD shaded)", title_fontsize=8)
    save(fig, "fig3_ramp.png")
    c1, mu1 = vals[("twoway", 1)]
    return dict(dP05=float(np.interp(0.5, c1, mu1)))


# ------------------------------------------------------------------ Fig. 4: u_mf per bed (README table)
UMF = {  # seeds 7, 11–22, 24; bed 23 excluded (see README)
    "two-way\n5:1": [1.0021, .9944, .9886, .9919, .9765, .9888, .9803, 1.0061, .9859, .9829, .9792, .9838, 1.0039, .9891],
    "one-way\n1:1": [.9885, 1.0155, .9847, .9927, .9933, .9922, 1.0009, .9965, .9727, .9924, .9962, 1.0067, .9828, 1.0015],
    "one-way\n5:1": [1.0051, .9921, .9764, .9988, .9865, .9934, .9864, .9811, .9906, .9947, .9896, 1.0058, .9963, .9837],
    "one-way\n10:1": [.9895, 1.0150, 1.0044, .9987, .9864, 1.0206, .9833, .9979, .9849, .9917, .9835, .9969, .9851, 1.0060],
    "one-way 1:1\n+ interp.": [1.0386, 1.0087, 1.0364, 1.0489, 1.0472, 1.0346, 1.0556, 1.0541, 1.0187, 1.0224, 1.0310,
                               1.0205, 1.0441, 1.0473],
    "one-way 5:1\n+ interp.": [1.0445, 1.0292, 1.0378, 1.0399, 1.0356, 1.0431, 1.0382, 1.0479, 1.0378, 1.0487, 1.0178,
                               1.0274, 1.0414, 1.0323],
    "one-way 10:1\n+ interp.": [1.0358, 1.0265, 1.0514, 1.0474, 1.0372, 1.0297, 1.0318, 1.0415, 1.0378, 1.0553, 1.0644,
                                1.0290, 1.0462, 1.0371],
}


def fig_umf():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    rng = np.random.default_rng(1)
    for i, (k, v) in enumerate(UMF.items()):
        col = C["twoway"] if i == 0 else (C["oneway"] if i < 4 else "#8a8f98")
        v = np.array(v)
        ax.scatter(i + rng.uniform(-0.12, 0.12, v.size), v, s=12, color=col, alpha=0.75, lw=0)
        ax.errorbar(i + 0.28, v.mean(), yerr=v.std(ddof=1), fmt="D", color=C["ink"], ms=4, capsize=3, lw=1)
    for y, lab in ((1.05, "NETL experiment 1.05"), (1.00, "paper simulation 1.00")):
        ax.axhline(y, color=C["good"], lw=0.9, ls="--")
        ax.text(6.55, y + 0.002, lab, ha="right", fontsize=8, color=C["good"])
    ax.axvline(3.5, color=C["grid"], lw=1)
    ax.text(3.6, 0.958, "extension: cloud-in-cell voidage\n(code not in the repository)", fontsize=7.5,
            color=C["ink2"])
    ax.set_xticks(range(len(UMF)))
    ax.set_xticklabels(UMF.keys(), fontsize=8)
    ax.set_ylabel("u$_{mf}$ (m/s)")
    ax.set_ylim(0.955, 1.075)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_umf_beds.png")


# ------------------------------------------------------------------ Fig. 5: ΔP / weight at 1–4 u_mf
def fig_dP_levels():
    us = ["1.10", "2.19", "3.28", "4.38"]
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for kind, k, lab in (("1way", "oneway", "one-way"), ("2way", "twoway", "quasi-two-way")):
        m, s = [], []
        for u in us:
            r = mp.hold(u, kind)
            d = r["dP"][r["t"] >= 5] / mp.WEIGHT
            m.append(d.mean())
            s.append(d.std())
            out[(u, kind)] = (d.mean() * mp.WEIGHT, d.std() * mp.WEIGHT)
        off = -0.06 if kind == "1way" else 0.06
        axs[0].errorbar(np.arange(1, 5) + off, m, yerr=s, fmt="o", color=C[k], ms=5, capsize=3, label=lab)
    axs[0].plot([2, 3, 4], [1013 / 1075, 996 / 1075, 914 / 1075], "s", mfc="none", color=C["good"], ms=7,
                label="paper (vs its 1075 Pa)")
    axs[0].axhline(1, color=C["ink2"], ls=":", lw=0.9)
    axs[0].set_yscale("log")
    axs[0].set_yticks([0.5, 1, 2, 5, 10, 20])
    axs[0].set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    axs[0].set_ylim(0.5, 20)
    axs[0].set_xticks([1, 2, 3, 4])
    axs[0].set_xticklabels(["1 u$_{mf}$\n1.10", "2 u$_{mf}$\n2.19", "3 u$_{mf}$\n3.28", "4 u$_{mf}$\n4.38 m/s"])
    axs[0].set_ylabel("mean ΔP$_{bed}$ / bed weight")
    axs[0].legend(loc="upper left", fontsize=8)
    axs[0].set_title("time mean, t = 5–15 s (bars: SD)")
    for kind, k, lab in (("1way", "oneway", "one-way"), ("2way", "twoway", "quasi-two-way")):
        r = mp.hold("2.19", kind)
        msk = r["t"] >= 5
        f, P = welch(r["dP"][msk] - r["dP"][msk].mean(), fs=50.0, nperseg=256)
        out[("psd", kind)] = f[np.argmax(P)]
        axs[1].semilogy(f, P, color=C[k], label=f"{lab}, peak {f[np.argmax(P)]:.1f} Hz")
    axs[1].axvspan(2, 5, color="#999", alpha=0.15, lw=0)
    axs[1].text(3.5, 3e-2, "paper\n2–5 Hz", ha="center", fontsize=8, color=C["ink2"])
    axs[1].set_xlim(0, 20)
    axs[1].set_ylim(1e-2, 1e4)
    axs[1].set_xlabel("frequency (Hz)")
    axs[1].set_ylabel("PSD of ΔP$_{bed}$ (Pa²/Hz)")
    axs[1].set_title("spectrum at 2 u$_{mf}$")
    axs[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_dP_levels.png")
    return out


# ------------------------------------------------------------------ Fig. 6: voidage snapshots
def fig_snaps(t_snap=14.5):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 4.0))
    cases = (("2.19", "1way", "2 u$_{mf}$, one-way"), ("2.19", "2way", "2 u$_{mf}$, quasi-two-way"),
             ("4.38", "1way", "4 u$_{mf}$, one-way"), ("4.38", "2way", "4 u$_{mf}$, quasi-two-way"))
    hb = {}
    for ax, (u, kind, lab) in zip(axs, cases):
        r = mp.hold(u, kind)
        k = int(round(t_snap * float(r["sample_hz"]))) - 1
        W, H = float(r["W"]), float(r["H"])
        im = ax.imshow(r["eps"][k].T, origin="lower", extent=(0, W, 0, H), cmap="RdBu", vmin=0.4, vmax=1.0,
                       interpolation="bilinear")
        ax.scatter(r["x"][k], r["y"][k], s=0.15, color="k", alpha=0.35, lw=0)
        ax.set_title(lab, fontsize=9, color=C["oneway"] if kind == "1way" else C["twoway"])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xticks([0, 0.1, 0.2])
        ax.grid(False)
        if ax is not axs[0]:
            ax.set_yticklabels([])
        ys = r["y"][r["t"] >= 5]
        hb[(u, kind)] = (float(np.percentile(ys, 5, axis=1).mean()), float(np.percentile(ys, 95, axis=1).mean()))
    axs[0].set_ylabel("height (m)")
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("gas volume fraction ε$_g$")
    fig.text(0.45, 0.0, f"t = {t_snap} s; dots are particles", ha="center", fontsize=8.5, color=C["ink2"])
    save(fig, "fig6_snapshots.png")
    return hb


# ------------------------------------------------------------------ Fig. 7: momentum residual
def fig_residual():
    fig, axs = plt.subplots(1, 2, figsize=(4.6, 3.6))
    st = {}
    for ax, (kind, lab) in zip(axs, (("1way", "one-way"), ("2way", "quasi-two-way"))):
        R, E, W, H = mp.momentum_residual("2.19", kind)
        bed = E < 0.8
        st[kind] = (float(R[bed].mean()), float(R.max()))
        im = ax.imshow(R.T, origin="lower", extent=(0, W, 0, H), cmap="Reds", vmin=0, vmax=1.5)
        ax.set_title(f"{lab}\nmean in bed {R[bed].mean():.2f}", fontsize=9,
                     color=C["oneway"] if kind == "1way" else C["twoway"])
        ax.set_xticks([0, 0.1, 0.2])
        ax.grid(False)
        if kind == "2way":
            ax.set_yticklabels([])
    axs[0].set_ylabel("height (m)")
    cb = fig.colorbar(im, ax=axs, fraction=0.05, pad=0.03)
    cb.set_label("|R$_{cell}$| / bed weight per volume")
    save(fig, "fig7_residual.png")
    return st


def main():
    p = Params()
    N = 3380
    print("derived:", dict(m=p.m, meff=p.meff, zeta=p.zeta, etan=p.etan, tc=p.tc, dt=p.dt, NCX=p.NCX, NCY=p.NCY,
                           dx=p.dx, dy=p.dy, weight=p.dP_theory(N), bed_mass=N * p.m))
    print("Ar", p.rho_g * (p.rho_p - p.rho_g) * p.g * p.dp ** 3 / p.mu_g ** 2,
          "Ergun umf(0.441)", ergun_umf(0.441, p.dp, p.rho_p, p.rho_g, p.mu_g),
          "Wen-Yu umf", wen_yu_umf(p.dp, p.rho_p, p.rho_g, p.mu_g))
    print("collision:", fig_collision(p))
    print("ramp:", fig_ramp(p))
    fig_umf()
    print("dP levels:", fig_dP_levels())
    print("bed extent (5th, 95th pct of y):", fig_snaps())
    print("residual:", fig_residual())


if __name__ == "__main__":
    main()
