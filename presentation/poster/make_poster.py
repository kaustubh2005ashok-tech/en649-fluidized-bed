"""
make_poster.py — A1 portrait poster: Liu & van Wachem (2019) vs this work (one-way and quasi-two-way 2-D DEM).

    python make_poster.py            # writes poster_A1.pdf and poster_A1_preview.png next to this file

Paper figures are cropped from the article PDF (PAPER below); our figures are drawn from the run outputs in
en649_2way/out/.
"""
import glob
import os
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.signal import welch
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
PAPER = r"C:\Users\kaust\.claude\uploads\e3373015-ba19-40cc-bb5c-edb229fdafb2\a61b4ca5-1-s2.0-S003259101830929X-main.pdf"
PAN = os.path.join(HERE, "panels")
os.makedirs(PAN, exist_ok=True)

C = dict(oneway="#2a78d6", twoway="#eb6834", ink="#0b0b0b", ink2="#52514e", grid="#e4e3de", surf="#ffffff",
         navy="#1d2f5e", band="#e8edf7", warn="#b3261e", warnbg="#fbeceb", good="#1b7a4b", goodbg="#e9f5ee")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14, "legend.frameon": False, "lines.linewidth": 1.8,
})
WEIGHT = 905.0                                       # our bed weight / area, Pa
PAPER_W = 1075.0                                     # paper's theoretical ΔP_bed


def hold(u, kind):
    return np.load(os.path.join(REPO, "out", f"hold_full_U{u}_{kind}.npz"))


def savep(fig, name):
    p = os.path.join(PAN, name)
    fig.savefig(p, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ paper crops
def paper_crops():
    d = pymupdf.open(PAPER)
    out = {}
    for page in d:
        for k, b in enumerate(page.get_image_info(xrefs=True)):
            if b["width"] < 500:
                continue
            img = d.extract_image(b["xref"])
            fn = os.path.join(PAN, f"paper_p{page.number + 1}_{k}.{img['ext']}")
            with open(fn, "wb") as f:
                f.write(img["image"])
            out[(page.number + 1, k)] = fn
    from PIL import Image

    def crop(key, box, name):                       # box as fractions (left, top, right, bottom)
        im = Image.open(out[key])
        w, h = im.size
        im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))).save(os.path.join(PAN, name))
        return os.path.join(PAN, name)

    return dict(
        fig1=out[(2, 1)], fig2=out[(2, 0)],
        fig3=crop((3, 1), (0, 0, 1, 0.31), "paper_fig3_2umf.png"),
        fig4=crop((3, 0), (0, 0, 0.335, 1), "paper_fig4_2umf.png"),
        fig5=crop((4, 0), (0, 0, 1, 0.33), "paper_fig5_2umf.png"),
    )


# ------------------------------------------------------------------ our panels
W_VOL = 1131.0 * (1 - 0.44) * 9.81                   # bed weight per unit volume at ε_mf, N/m³


def momentum_residual(u, kind, t0=5.0, every=5):
    """Time-mean |R| per fluid cell, R = f_cell − ε_cell G_row: the part of the gas–particle drag in a cell that the
    row-uniform pressure gradient G_row = Σ_row f / Σ_row ε (the only one this model implies) does not balance.
    A solved gas momentum equation (the paper) gives each cell its own ∇p, so R = 0 there."""
    import sys
    sys.path.insert(0, REPO)
    from drag import apply_drag, apply_drag_2way
    r = hold(u, kind)
    NCX, NCY, dx, dy, dp, Vp = (int(r["NCX"]), int(r["NCY"]), float(r["dx"]), float(r["dy"]), float(r["dp"]),
                                float(r["Vp"]))
    N, Ug = int(r["N"]), float(r["Ug"])
    ks = np.where(r["t"] >= t0)[0][::every]
    A = np.zeros((NCX, NCY))
    E = np.zeros((NCX, NCY))
    for k in ks:
        x, y, vy = (r[a][k].astype(float) for a in ("x", "y", "vy"))
        eps = r["eps"][k].astype(float)
        fy = np.zeros(N)
        if kind == "2way":
            apply_drag_2way(x, y, vy, fy, N, Ug, eps, r["qs"][k].astype(float), NCX, NCY, dx, dy, dp, 1.21, 1.8e-5,
                            Vp, 0.8)
        else:
            apply_drag(x, y, vy, fy, N, Ug, eps, NCX, NCY, dx, dy, dp, 1.21, 1.8e-5, Vp, 0.8)
        f = np.zeros((NCX, NCY))
        np.add.at(f, (np.clip((x / dx).astype(int), 0, NCX - 1), np.clip((y / dy).astype(int), 0, NCY - 1)), fy)
        f /= dx * dy * dp
        A += np.abs(f - eps * (f.sum(0) / eps.sum(0))[None, :])
        E += eps
    return A / len(ks) / W_VOL, E / len(ks), float(r["W"]), float(r["H"])


def our_momentum():
    # reference side: the equation the paper solves in every cell
    fig = plt.figure(figsize=(5.0, 3.8))
    fig.text(0.02, 0.95, "Gas momentum, solved in every fluid cell:", fontsize=14, fontweight="bold", va="top")
    fig.text(0.04, 0.78, r"$\frac{\partial(\varepsilon_g\rho_g\mathbf{u}_g)}{\partial t}+\nabla\cdot(\varepsilon_g"
             r"\rho_g\mathbf{u}_g\mathbf{u}_g)=$", fontsize=16)
    fig.text(0.12, 0.65, r"$-\varepsilon_g\nabla p+\nabla\cdot(\varepsilon_g\boldsymbol{\tau}_g)"
             r"+\varepsilon_g\rho_g\mathbf{g}+\mathbf{S}_p$", fontsize=16)
    fig.text(0.04, 0.52, r"$\mathbf{S}_p=-\frac{1}{V_{cell}}\sum_{i\in cell}\mathbf{F}_{d,i}$"
             "  (drag on the gas)", fontsize=14)
    fig.text(0.02, 0.40, "Each cell has its own ∇p (x and y), so the\ndrag in every cell is balanced locally:",
             fontsize=13, color=C["ink2"], linespacing=1.35, va="top")
    fig.text(0.5, 0.17, r"$R_{cell}=0$ in every cell", fontsize=20, ha="center", color=C["good"],
             fontweight="bold")
    fig.text(0.5, 0.03, "→ gas turns round dense regions → bubbles", fontsize=13, ha="center", color=C["ink2"])
    p_ref = savep(fig, "ref_momentum.png")

    # this work: residual maps at 2 u_mf, one-way and quasi-two-way, same colour scale
    fig = plt.figure(figsize=(5.0, 3.8))
    fig.text(0.01, 0.985, r"No gas momentum eq.; one $\partial p/\partial y$ per row, $\partial p/\partial x=0$:",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 0.905, r"$R_{cell}=f_{drag,cell}-\varepsilon_{cell}\,\Sigma_{row}f_{drag}\,/\,"
             r"\Sigma_{row}\varepsilon$", fontsize=14, va="top")
    stats = {}
    for j, (kind, lab, col) in enumerate((("1way", "one-way", C["oneway"]), ("2way", "quasi-two-way", C["twoway"]))):
        R, E, W, H = momentum_residual("2.19", kind)
        bed = E < 0.8
        stats[kind] = (R[bed].mean(), R.max())
        ax = fig.add_axes([0.06 + 0.31 * j, 0.06, 0.28, 0.66])
        im = ax.imshow(R.T, origin="lower", extent=(0, W, 0, H), cmap="Reds", vmin=0, vmax=1.5, aspect="equal")
        ax.set_title(f"{lab}\nbed mean {R[bed].mean():.2f}", fontsize=12.5, color=col, fontweight="bold")
        ax.set_xticks([0, 0.1, 0.2])
        ax.tick_params(labelsize=9.5)
        ax.grid(False)
        if j:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, cax=fig.add_axes([0.70, 0.10, 0.03, 0.56]))
    cb.set_label("|R$_{cell}$| / bed weight\nper volume (time mean)", fontsize=11.5)
    cb.ax.tick_params(labelsize=9.5)
    fig.text(0.70, 0.73, "2 u$_{mf}$, 5–15 s", fontsize=11.5, color=C["ink2"])
    print("momentum residual (mean in bed, max):", stats)
    return p_ref, savep(fig, "our_momentum_2umf.png"), stats


def _ramp_files(kind):
    pat = "ramp_full_5to1_2way_s*.npz" if kind == "twoway" else "ramp_full_5to1_s*.npz"
    s7 = "ramp_full_5to1_2way.npz" if kind == "twoway" else "ramp_full_5to1.npz"
    fs = [os.path.join(REPO, "out", s7)] + sorted(glob.glob(os.path.join(REPO, "out", pat)))
    return [f for f in fs if not f.endswith("_s23.npz")]


def our_fig2():
    """Drawn on the paper's exact axes (0–1.4 m/s, 0–1500 Pa, same ticks) so the two panels compare by eye.
    Grey marks are read off the paper's Fig. 2."""
    edges = np.arange(0, 2.25, 0.05)
    c = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.axhline(1030, color="#9a9a9a", lw=1.2, ls="--")
    ax.text(0.02, 1050, "paper plateau ≈ 1030 Pa", fontsize=11.5, color="#6a6a6a")
    ax.plot([0.8], [1130], marker="v", ms=10, color="#9a9a9a", ls="none")
    ax.text(0.8, 1185, "paper overshoot ≈ 1130 Pa", fontsize=11.5, color="#6a6a6a", ha="center")
    for k, lab in (("oneway", "1-way"), ("twoway", "q-2-way")):
        for br, ls, arrow in ((1, "-", "↑"), (-1, "--", "↓")):
            rows = []
            for f in _ramp_files(k):
                r = np.load(f)
                msk = r["branch"] == br
                idx = np.digitize(r["Ug"][msk], edges) - 1
                rows.append([r["dP"][msk][idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(c))])
            a = np.array(rows)
            mu, sd = np.nanmean(a, 0), np.nanstd(a, 0)
            ax.plot(c, mu, color=C[k], ls=ls, lw=2.2, label=f"{lab} {arrow}")
            if br == 1:
                ax.fill_between(c, mu - sd, mu + sd, color=C[k], alpha=0.15, lw=0)
    ax.axhline(WEIGHT, color=C["ink2"], lw=1, ls=":")
    ax.text(0.02, WEIGHT - 75, "our bed weight 905 Pa", fontsize=11.5, color=C["ink2"])
    ax.axvline(0.99, color=C["ink2"], lw=1, ls=":")
    ax.text(0.97, 1380, "u$_{mf}$ = 0.99", fontsize=12, color=C["ink"], ha="right")
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1500)
    ax.set_xticks(np.arange(0, 1.41, 0.25))
    ax.set_yticks(np.arange(0, 1501, 300))
    ax.tick_params(labelsize=11)
    ax.set_xlabel("U$_g$ (m/s)", fontsize=13)
    ax.set_ylabel("bed pressure drop ΔP$_{bed}$ (Pa)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11, ncol=2, columnspacing=0.8, handlelength=1.6,
              title="14-bed mean ± SD", title_fontsize=10.5)
    return savep(fig, "our_fig2.png")


def our_fig3():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for kind, k, lab in (("1way", "oneway", "one-way"), ("2way", "twoway", "quasi-two-way")):
        r = hold("2.19", kind)
        m = r["t"] >= 5
        ax.plot(r["t"][m], r["dP"][m], color=C[k], lw=1.5,
                label=f"{lab}: mean {r['dP'][m].mean():.0f} Pa, SD {r['dP'][m].std():.0f} Pa")
    ax.axhline(WEIGHT, color="#c2185b", ls="--", lw=1.2)
    ax.text(14.9, WEIGHT - 150, "theoretical ΔP$_{bed}$ = 905 Pa", ha="right", color="#c2185b", fontsize=12)
    ax.set_xlim(5, 15)
    ax.set_ylim(0, 2000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ΔP$_{bed}$ (Pa)")
    ax.set_title("U$_g$ = 2.19 m/s (2 u$_{mf}$), same y-scale as the paper", fontsize=12.5)
    ax.legend(loc="upper left", fontsize=11.5)
    return savep(fig, "our_fig3_2umf.png")


def our_fig4():
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for kind, k, lab in (("1way", "oneway", "one-way"), ("2way", "twoway", "quasi-two-way")):
        r = hold("2.19", kind)
        m = r["t"] >= 5
        f, P = welch(r["dP"][m] - r["dP"][m].mean(), fs=50.0, nperseg=256)
        ax.semilogy(f, P, color=C[k], lw=1.8, label=f"{lab}: peak {f[np.argmax(P)]:.1f} Hz")
    ax.axvspan(2, 5, color="#999", alpha=0.12, lw=0)
    ax.text(3.5, 2e-2, "paper's\ndominant band\n2–5 Hz", ha="center", fontsize=11, color=C["ink2"])
    ax.set_xlim(0, 25)
    ax.set_ylim(1e-2, 1e5)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSDF (Pa²/Hz)")
    ax.set_title("U$_g$ = 2.19 m/s (2 u$_{mf}$)", fontsize=12.5)
    ax.legend(loc="upper right", fontsize=11)
    return savep(fig, "our_fig4_2umf.png")


def _snap(ax_p, ax_e, r, k, vmax):
    x, y = r["x"][k], r["y"][k]
    sp = np.hypot(r["vx"][k], r["vy"][k])
    W, H = float(r["W"]), float(r["H"])
    sc = ax_p.scatter(x, y, c=sp, s=2.2, cmap="turbo", vmin=0, vmax=vmax, lw=0)
    ax_e.imshow(r["eps"][k].T, origin="lower", extent=(0, W, 0, H), cmap="RdBu", vmin=0.4, vmax=1.0,
                aspect="equal", interpolation="bilinear")
    for a in (ax_p, ax_e):
        a.set_xlim(0, W)
        a.set_ylim(0, H)
        a.set_aspect("equal")
        a.grid(False)
        a.tick_params(labelsize=10)
    return sc


def our_snapshots(u, kinds, name, t_snap=14.5, vmax=1.5):
    fig = plt.figure(figsize=(10.5, 6.4))
    axs = [fig.add_axes([0.04 + 0.205 * i, 0.08, 0.185, 0.8]) for i in range(2 * len(kinds))]
    for j, (kind, lab) in enumerate(kinds):
        r = hold(u, kind)
        k = int(round(t_snap * float(r["sample_hz"]))) - 1
        sc = _snap(axs[2 * j], axs[2 * j + 1], r, k, vmax)
        axs[2 * j].set_title(f"{lab}\nparticle speed", fontsize=13)
        axs[2 * j + 1].set_title(f"{lab}\nε$_g$", fontsize=13)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.86, 0.12, 0.015, 0.7]))
    cb.set_label("|v$_p$| (m/s)")
    cb2 = fig.colorbar(axs[1].images[0], cax=fig.add_axes([0.94, 0.12, 0.015, 0.7]))
    cb2.set_label("ε$_g$ (−)")
    return savep(fig, name)


def our_problem():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for u, lab, ls in (("3.28", "3 u$_{mf}$", "-"), ("4.38", "4 u$_{mf}$", "--")):
        for kind, k, cl in (("1way", "oneway", "one-way"), ("2way", "twoway", "quasi-two-way")):
            r = hold(u, kind)
            ax.plot(r["t"], r["dP"] / WEIGHT, color=C[k], ls=ls, lw=1.6, label=f"{lab}, {cl}")
    ax.axhspan(0.85, 0.94, color="#2e7d32", alpha=0.25, lw=0)
    ax.text(14.8, 1.6, "paper: 0.85–0.94 × weight", ha="right", fontsize=12, color="#2e7d32")
    ax.axhline(1.0, color=C["ink2"], lw=1, ls=":")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 13.5)
    ax.set_xlabel("time (s)  (U$_g$ ramped over the first 2 s)")
    ax.set_ylabel("ΔP$_{bed}$ / bed weight (−)")
    ax.legend(loc="center right", fontsize=11, ncol=1)
    p1 = savep(fig, "our_problem_dP.png")
    r = hold("4.38", "2way")
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 6.0))
    k = int(round(14.5 * float(r["sample_hz"]))) - 1
    _snap(axs[0], axs[1], r, k, 2.6)
    axs[0].set_title("4 u$_{mf}$ particles", fontsize=12)
    axs[1].set_title("ε$_g$", fontsize=12)
    p2 = savep(fig, "our_problem_snap.png")
    return p1, p2


# ------------------------------------------------------------------ poster layout
IN = 1 / 25.4
PW, PH = 594 * IN, 841 * IN                          # A1 portrait, inches


def box(fig, x, y, w, h, title, color=None, bg="#ffffff"):
    """Rounded section box in figure fractions (x, y = bottom-left). Returns inner content rect."""
    color = color or C["navy"]
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, fc=bg, ec=color, lw=2.2, zorder=-3))
    th = 0.017
    fig.patches.append(FancyBboxPatch((x, y + h - th), w, th, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, fc=color, ec=color, lw=0, zorder=-2))
    fig.text(x + w / 2, y + h - th / 2, title, ha="center", va="center", color="white", fontsize=30,
             fontweight="bold", zorder=2)
    return x + 0.008, y + 0.006, w - 0.016, h - th - 0.012


def para(fig, x, y, lines, width, size=19, gap=1.32, color=None):
    """Write bullet/plain lines top-down from (x, y); wraps to `width` characters. Returns the new y."""
    lh = size / 72 / PH * gap
    for ln in lines:
        bold = ln.startswith("!")
        ln = ln.lstrip("!")
        bullet = ln.startswith("• ")
        body = ln[2:] if bullet else ln
        wrapped = textwrap.wrap(body, width - (2 if bullet else 0)) or [""]
        for i, w in enumerate(wrapped):
            pre = ("• " if i == 0 else "   ") if bullet else ""
            fig.text(x, y, pre + w, fontsize=size, va="top", color=color or C["ink"],
                     fontweight="bold" if bold else "normal")
            y -= lh
        y -= lh * 0.25
    return y


def image(fig, path, x, y, w, h, label=None, label_color=None):
    im = plt.imread(path)
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(im)
    ax.set_axis_off()
    ax.set_zorder(5)
    ax.set_anchor("C")
    if label:
        fig.text(x + w / 2, y + h + 0.002, label, ha="center", va="bottom", fontsize=18, fontweight="bold",
                 color=label_color or C["ink"])
    return ax


def compare_card(fig, x, y, w, h, title, paper_img, our_img, takeaway, ok=True, our_label="This work", tk_h=0.034):
    cx, cy, cw, ch = box(fig, x, y, w, h, title)
    img_h = ch - tk_h - 0.012
    half = (cw - 0.01) / 2
    image(fig, paper_img, cx, cy + tk_h, half, img_h - 0.002, "Liu & van Wachem (2019)", C["ink2"])
    image(fig, our_img, cx + half + 0.01, cy + tk_h, half, img_h - 0.002, our_label, C["navy"])
    col, bg, mark = {True: (C["good"], C["goodbg"], "✔ "), "partial": ("#8a5a00", "#fdf4e3", "◐ "),
                     False: (C["warn"], C["warnbg"], "✘ ")}[ok]
    fig.patches.append(Rectangle((cx, cy), cw, tk_h - 0.004, transform=fig.transFigure, fc=bg, ec=col, lw=1.5,
                                 zorder=-1))
    para(fig, cx + 0.006, cy + tk_h - 0.008, [mark + takeaway], int(cw * 180), size=17, color=col)


def poster(pc, ours):
    fig = plt.figure(figsize=(PW, PH))
    # ---------------- header
    fig.patches.append(Rectangle((0, 0.925), 1, 0.075, transform=fig.transFigure, fc=C["navy"], ec="none"))
    # logo on a white badge (the emblem is blue on transparent)
    lw_ = 0.068
    lh_ = lw_ * PW / PH
    lx, ly = 0.018, 0.9625 - lh_ / 2
    fig.patches.append(FancyBboxPatch((lx - 0.004, ly - 0.004 * PW / PH), lw_ + 0.008, lh_ + 0.008 * PW / PH,
                                      boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                      fc="white", ec="none", zorder=2))
    lax = fig.add_axes([lx, ly, lw_, lh_])
    lax.imshow(plt.imread(os.path.join(HERE, "logo", "iitb_logo.png")))
    lax.set_axis_off()
    lax.set_zorder(5)
    xc = 0.545                                       # text centre, shifted right of the logo
    fig.text(xc, 0.983, "Reproducing a CFD-DEM Study of a Geldart-D Fluidized Bed in Python",
             ha="center", va="center", color="white", fontsize=36, fontweight="bold")
    fig.text(xc, 0.965, "One-Way vs Quasi-Two-Way Gas Coupling, compared with Liu & van Wachem (2019)",
             ha="center", va="center", color="white", fontsize=28, fontweight="bold")
    fig.text(xc, 0.947, "Kaustubh Ashok (24B1522)  ·  EN 649 Course Project", ha="center", va="center",
             color="#dfe6f5", fontsize=24)
    fig.text(xc, 0.933, "Instructor: Prof. Manaswita Bose  ·  Indian Institute of Technology Bombay",
             ha="center", va="center", color="#dfe6f5", fontsize=24)

    M, G = 0.018, 0.012                             # outer margin, gutter
    colw = (1 - 2 * M - 2 * G) / 3
    xs = [M, M + colw + G, M + 2 * (colw + G)]
    top = 0.915

    # ---------------- row 1: aim / methodology / python
    h1 = 0.212
    y1 = top - h1
    cx, cy, cw, ch = box(fig, xs[0], y1, colw, h1, "Aim & Objectives")
    para(fig, cx, cy + ch - 0.004, [
        "!Aim",
        "Reproduce the results of Liu & van Wachem (2019) for the NETL small-scale challenge bed with a DEM "
        "code written from scratch in Python.",
        "!Objectives",
        "• Build a soft-sphere DEM solver (Numba) for 3,380 nylon beads, d_p = 3.256 mm.",
        "• Couple it to the gas (Gidaspow drag) one-way and quasi-two-way.",
        "• Reproduce Figs 1–5: drag models, ΔP–u_g and u_mf, ΔP time series, PSD, snapshots.",
        "• Measure bed-to-bed scatter (14 beds per coupling) and find where the simple gas model fails.",
    ], 47, size=18)

    cx, cy, cw, ch = box(fig, xs[1], y1, colw, h1, "Methodology")
    y = para(fig, cx, cy + ch - 0.004, [
        "!Particles (DEM, paper Table 1)",
        "• Spring–dashpot contacts, k_n = 3654 N/m, e_n = 0.84, μ = 0.35; Δt = t_c/50 = 3.3 μs.",
        "• 2-D monolayer, W = 0.23 m, 24 × 56 fluid cells (3 d_p); fluid update every 5 DEM steps.",
        "!Gas–solid coupling",
        "• Drag per particle: F_d = β (u_g − v_p) V_p / ε_s, β = Gidaspow (Ergun ε < 0.8, Wen–Yu above).",
    ], 47, size=17)
    eq_y = y - 0.001
    fig.text(cx + 0.005, eq_y, r"one-way:   $u_g = U_g\,/\,\varepsilon_g$", fontsize=20, va="top",
             color=C["oneway"])
    fig.text(cx + 0.005, eq_y - 0.011, r"quasi-two-way:   $u_g = (U_g - q_s)\,/\,\varepsilon_g$", fontsize=20,
             va="top", color=C["twoway"])
    fig.text(cx + 0.03, eq_y - 0.022, r"$q_s$ = row-averaged solids volume flux", fontsize=17, va="top",
             color=C["twoway"])
    para(fig, cx, eq_y - 0.034, [
        "• ΔP_bed = Σ F_d / (W d_p); bed weight / area = 905 Pa.",
        "• Ramp U_g at ±0.2 m/s² (→ u_mf); hold 2, 3, 4 u_mf for 15 s, statistics 5–15 s at 50 Hz.",
    ], 47, size=17)

    cx, cy, cw, ch = box(fig, xs[2], y1, colw, h1, "How Python Reproduces It")
    flow = [("params.py", "all Table 1–2 inputs, derived t_c, Δt, grid"),
            ("bed.py · run_settle.py", "lattice fill, gravity settling"),
            ("sim.py  @njit", "cell list → contacts → walls → drag → integrate"),
            ("drag.py  @njit", "voidage, Gidaspow β, one-/two-way drag"),
            ("run_ramp.py · run_hold.py", "ramp for u_mf; 15 s holds, 50 Hz output"),
            ("analysis.py · paper_figs.py", "u_mf fit, PSD (SciPy), figures (Matplotlib)")]
    yy = cy + ch - 0.004
    bh = 0.0205
    for i, (a, b) in enumerate(flow):
        fig.patches.append(FancyBboxPatch((cx + 0.004, yy - bh), cw - 0.008, bh - 0.002,
                                          boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                          fc=C["band"], ec=C["navy"], lw=1.2, zorder=-1))
        fig.text(cx + 0.010, yy - 0.0055, a, fontsize=17, fontweight="bold", va="center",
                 family="monospace", color=C["navy"])
        fig.text(cx + 0.010, yy - 0.0145, b, fontsize=15, va="center", color=C["ink"])
        if i < len(flow) - 1:
            fig.text(cx + cw - 0.02, yy - bh - 0.0004, "▼", fontsize=11, ha="center", va="center", color=C["navy"])
        yy -= bh + 0.0015
    para(fig, cx, yy - 0.002, [
        "• Numba JIT compiles the Python loops: ≈ 2.5 ms per DEM step for 3,380 particles.",
        "• 28 ramp runs (14 beds × 2 couplings) + 6 hold runs, ≈ 3 h each.",
    ], 50, size=16)

    # ---------------- results band
    yb = y1 - 0.024
    fig.patches.append(Rectangle((M, yb), 1 - 2 * M, 0.017, transform=fig.transFigure, fc=C["navy"], ec="none"))
    fig.text(0.5, yb + 0.0085, "RESULTS — reference paper (left) vs this work (right)",
             ha="center", va="center", color="white", fontsize=32, fontweight="bold")

    w2 = (1 - 2 * M - G) / 2
    rh = 0.14
    rh1 = 0.165
    r1 = yb - 0.006 - rh1
    m1, m2 = ours["mom_stats"]["1way"][0], ours["mom_stats"]["2way"][0]
    compare_card(fig, M, r1, w2, rh1, "Gas momentum conservation per fluid cell", ours["mom_ref"], ours["mom"],
                 f"Not conserved. Inside the bed, drag equal to {50 * (m1 + m2):.0f} % of the local bed weight is left "
                 "unbalanced by any pressure gradient (red). The paper balances every cell, so its gas can turn "
                 "round dense regions and form bubbles; ours cannot.", ok=False, tk_h=0.047)
    compare_card(fig, M + w2 + G, r1, w2, rh1, "Fig. 2 — ΔP_bed vs u_g (same axes)", pc["fig2"], ours["fig2"],
                 "u_mf matches (0.99 vs 1.00 paper, 1.05 expt). Missing: the 10 % overshoot at 0.8 m/s, the early "
                 "rise (260 vs ≈ 550 Pa at 0.5 m/s) and the ≈ 100 Pa hysteresis. Lower plateau = lighter 2-D bed "
                 "(905 vs 1075 Pa).", ok="partial", tk_h=0.047)
    r2 = r1 - G - rh
    compare_card(fig, M, r2, w2, rh, "Fig. 3 — ΔP_bed time series at 2 u_mf", pc["fig3"], ours["fig3"],
                 "Mean ΔP = 0.99 × weight (paper 0.94 ×) ✔. Fluctuations (SD ≈ 30–40 Pa) are much smaller "
                 "than the paper's (≈ 500–1800 Pa swings).", ok="partial")
    compare_card(fig, M + w2 + G, r2, w2, rh, "Fig. 4 — Power spectrum of ΔP_bed at 2 u_mf", pc["fig4"],
                 ours["fig4"], "One-way peak 4.5 Hz lies in the paper's 2–5 Hz band; quasi-two-way peak 1.2 Hz.")
    r3 = r2 - G - 0.162
    compare_card(fig, M, r3, w2, 0.162, "Fig. 5 — Snapshot at 2 u_mf, t = 14.5 s", pc["fig5"], ours["fig5"],
                 "Bed fluidizes with a dense base and dilute core, but forms one large central void with solids "
                 "rising at the walls; the paper shows discrete bubbles.", ok="partial")

    # ---------------- problem box (3 & 4 u_mf)
    px = M + w2 + G
    cx, cy, cw, ch = box(fig, px, r3, w2, 0.162, "The Problem at 3 u_mf and 4 u_mf (this work)", color=C["warn"],
                         bg="#fffafa")
    image(fig, ours["prob_dP"], cx, cy + 0.056, cw * 0.66, ch - 0.058)
    image(fig, ours["prob_snap"], cx + cw * 0.67, cy + 0.056, cw * 0.33, ch - 0.058)
    para(fig, cx + 0.004, cy + 0.053, [
        "• ΔP climbs to 6× (3 u_mf) and 12× (4 u_mf) the bed weight; the paper stays at 0.85–0.94×.",
        "• Cause: the gas is solved only in 1-D (per row), so it cannot bypass the bed as bubbles. The whole bed "
        "lifts as a plug and is pressed against the top wall, in both couplings.",
    ], 80, size=17, color=C["ink"])

    # ---------------- bottom row: summary table / conclusions / future work
    hb = r3 - G - M
    yb3 = M
    cx, cy, cw, ch = box(fig, xs[0], yb3, colw, hb, "Summary of Numbers")
    rows = [("Quantity", "Paper", "1-way", "q-2-way"),
            ("u_mf (m/s)", "1.00", "0.991", "0.990"),
            ("ΔP/W at 2 u_mf", "0.94", "0.99", "0.99"),
            ("ΔP/W at 3 u_mf", "0.93", "6.2", "5.8"),
            ("ΔP/W at 4 u_mf", "0.85", "12.2", "12.3"),
            ("PSD peak, 2 u_mf", "2–5 Hz", "4.5 Hz", "1.2 Hz")]
    ty = cy + ch - 0.006
    cxs = [cx + 0.004, cx + cw * 0.47, cx + cw * 0.64, cx + cw * 0.80]
    for i, row in enumerate(rows):
        if i == 0:
            fig.patches.append(Rectangle((cx, ty - 0.012), cw, 0.013, transform=fig.transFigure, fc=C["band"],
                                         ec="none"))
        for j, v in enumerate(row):
            bad = i in (3, 4) and j in (2, 3)
            fig.text(cxs[j], ty - 0.0055, v, fontsize=17, va="center", fontweight="bold" if i == 0 else "normal",
                     color=C["warn"] if bad else C["ink"])
        ty -= 0.0145
    para(fig, cx, ty - 0.001, ["W = bed weight / area (paper 1075 Pa, ours 905 Pa).",
                               "u_mf: mean of 14 beds per coupling."], 60, size=15, color=C["ink2"])

    cx, cy, cw, ch = box(fig, xs[1], yb3, colw, hb, "Conclusions")
    para(fig, cx, cy + ch - 0.004, [
        "• Drag closures, the ΔP–u_g curve and u_mf are reproduced within 1 % of the paper.",
        "• At 2 u_mf the mean ΔP and the one-way ΔP frequency match; bubble shapes and ΔP swings do not yet.",
        "• Quasi-two-way vs one-way changes u_mf by < 0.5 %: solids flux matters little near u_mf.",
        "• Above 2 u_mf a 1-D gas field cannot form bubbles, so the bed plugs: the limit of this coupling.",
    ], 50, size=17)

    cx, cy, cw, ch = box(fig, xs[2], yb3, colw, hb, "Next Steps & References")
    y = para(fig, cx, cy + ch - 0.004, [
        "• Solve the gas pressure in 2-D (porous-media/Darcy–Ergun) so gas can form bubbles at 3–4 u_mf.",
        "• Lethe CFD-DEM (VANS) runs as a full two-way check (5 beds done; drag convention patched).",
        "• Compare with NETL SSCP experiments (Gopalan et al., 2016) once the data are obtained.",
    ], 50, size=17)
    para(fig, cx, y - 0.003, [
        "[1] Liu, D., van Wachem, B. (2019) Powder Technol. 343, 145–158.",
        "[2] Gopalan, B. et al. (2016) Powder Technol. 291, 299–310.",
        "[3] Gidaspow, D. (1994) Multiphase Flow and Fluidization.",
        "[4] Code: github.com/kaustubh2005ashok-tech/en649-fluidized-bed",
    ], 64, size=14, color=C["ink2"])

    out = os.path.join(HERE, "poster_A1.pdf")
    fig.savefig(out, dpi=300)                        # dpi sets the resolution of the embedded figure images
    fig.savefig(os.path.join(HERE, "poster_A1_preview.png"), dpi=100)
    fig.savefig(os.path.join(HERE, "poster_A1_300dpi.png"), dpi=300)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    pc = paper_crops()
    ours = dict(fig2=our_fig2(), fig3=our_fig3(), fig4=our_fig4(),
                fig5=our_snapshots("2.19", (("1way", "one-way"), ("2way", "quasi-two-way")), "our_fig5_2umf.png"))
    ours["prob_dP"], ours["prob_snap"] = our_problem()
    ours["mom_ref"], ours["mom"], ours["mom_stats"] = our_momentum()
    poster(pc, ours)
