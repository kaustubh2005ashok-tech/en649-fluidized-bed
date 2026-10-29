"""
paper_flow.py — Figs 3–14 and A1–A3 of Liu & van Wachem (2019) from the constant-U_g runs of run_hold.py
(out/hold_full_U{2.19,3.28,4.38}_{1way,2way}.npz). Statistics use t > TSTART (paper: 5–15 s).

Differences from the paper that every figure inherits: 2-D monolayer (N = 3380) instead of the 3-D bed
(92,547 particles); a 1-D gas model (vertical gas velocity only), so gas-velocity vectors are vertical and the
gas flow has no lateral component; granular temperature is the 2-D one, θ = (⟨C_xC_x⟩ + ⟨C_yC_y⟩)/2.
NETL SSCP measurements are not available here, so experiment overlays and Fig. 15 are not produced.
"""
import os
import numpy as np
from scipy import ndimage as ndi
from scipy import signal
import paper_figs as F
from paper_figs import C, plt, save, G

TSTART = float(os.environ.get("TSTART", "5"))
HOLD_DIR = os.environ.get("HOLD_DIR", "out")          # folder holding hold_full_U*_{1way,2way}.npz
UGS = [("2 u$_{mf}$", "2.19"), ("3 u$_{mf}$", "3.28"), ("4 u$_{mf}$", "4.38")]
KINDS = [("oneway", "1way", "one-way"), ("twoway", "2way", "quasi-two-way")]
UMF_SIM = 0.99                      # u_mf of this model (quasi-two-way, 14 beds); used in the correlations
YPROBE, CELL = 0.0672, 0.0457       # sampling row of the SSCP data and its cell size (paper §3.3)
BUB_THR = 0.75
_cache = {}


def load(u, kind):
    key = (u, kind)
    if key not in _cache:
        path = f"{HOLD_DIR}/hold_full_U{u}_{kind}.npz"
        _cache[key] = dict(np.load(path)) if os.path.exists(path) else None
    return _cache[key]


def have(*pairs):
    return all(load(u, k) is not None for u, k in pairs)


def win(d):
    return np.where(d["t"] > TSTART)[0]


# ------------------------------------------------------------------ gas velocity and bubbles
def gas_vy(d, k):
    """Vertical gas velocity per fluid cell for frame k (the 1-D model has no lateral component)."""
    eps = d["eps"][k]
    qs = d["qs"][k][None, :] if int(d["two_way"]) == 1 else 0.0
    return (float(d["Ug"]) - qs) / eps


def bubble_map(eps2d, thr=BUB_THR, up=3, min_cells=2.0):
    """
    Binary bubble image of one ε field (upsampled bilinearly ×up). Bubbles are the connected regions with
    ε ≥ thr that do NOT connect to the freeboard (the top row), i.e. bubbles inside the bed.
    Returns (labels on the upsampled grid, number of bubbles, bed area fraction helper dict).
    """
    Z = ndi.zoom(eps2d, up, order=1)
    lab, n = ndi.label(Z >= thr)
    top = set(np.unique(lab[:, -1])) - {0}
    keep = np.zeros(n + 1, bool)
    area_px = np.bincount(lab.ravel(), minlength=n + 1)
    for i in range(1, n + 1):
        if i not in top and area_px[i] >= min_cells * up * up:
            keep[i] = True
    lab = np.where(keep[lab], lab, 0)
    # bed surface per upsampled column: highest pixel that is emulsion (ε < thr) or an enclosed bubble
    inbed = (Z < thr) | (lab > 0)
    ny = Z.shape[1]
    h = np.where(inbed.any(1), ny - 1 - np.argmax(inbed[:, ::-1], axis=1) + 1, 0)
    return lab, Z, h


def bubble_stats(d, k, thr=BUB_THR, up=3):
    """Per-frame bubble fraction, ug_e, ug_b and the list of bubbles (area m², centroid y m)."""
    eps = d["eps"][k]
    lab, Z, h = bubble_map(eps, thr, up)
    dxu, dyu = float(d["dx"]) / up, float(d["dy"]) / up
    bed_area = h.sum() * dxu * dyu
    ids = [i for i in np.unique(lab) if i > 0]
    bub_area = sum((lab == i).sum() for i in ids) * dxu * dyu
    bubs = []
    for i in ids:
        ix, iy = np.nonzero(lab == i)
        bubs.append((len(ix) * dxu * dyu, (iy.mean() + 0.5) * dyu, (ix.mean() + 0.5) * dxu, i))
    # gas velocity in the two phases, sampled at the fluid-cell centres inside the bed
    ug = gas_vy(d, k)
    NCX, NCY = eps.shape
    ci = (np.arange(NCX) * up + up // 2)[:, None]
    cj = (np.arange(NCY) * up + up // 2)[None, :]
    isb = lab[ci, cj] > 0
    inb = (Z[ci, cj] < thr) | isb
    inb &= (cj * 1.0 / up + 0.5) * float(d["dy"]) / float(d["dy"]) <= (h[ci[:, 0]][:, None] / up + 1.0)
    ug_b = ug[isb].mean() if isb.any() else np.nan
    emu = inb & ~isb
    ug_e = ug[emu].mean() if emu.any() else np.nan
    delta = bub_area / bed_area if bed_area > 0 else np.nan
    return delta, ug_e, ug_b, bubs, lab


def bubble_series(d, thr=BUB_THR, up=3, stride=1):
    """Over the stats window: δ, ug_e, ug_b, list of bubble (area, y) and rise velocities (m/s, y)."""
    ks = win(d)[::stride]
    dt = 1.0 / float(d["sample_hz"]) * stride
    deltas, ugE, ugB, sizes, rises = [], [], [], [], []
    prev = None
    for k in ks:
        delta, ue, ub, bubs, lab = bubble_stats(d, k, thr, up)
        deltas.append(delta); ugE.append(ue); ugB.append(ub)
        for a, cy, cx, i in bubs:
            sizes.append((np.sqrt(4 * a / np.pi), cy))            # 2-D equivalent diameter
        if prev is not None:
            plab, pbubs = prev
            for a, cy, cx, i in bubs:                              # match to the previous frame by overlap
                m = lab == i
                cand = plab[m]
                cand = cand[cand > 0]
                if cand.size == 0:
                    continue
                j = np.bincount(cand).argmax()
                ov = (cand == j).sum()
                pa = [b for b in pbubs if b[3] == j]
                if pa and ov >= 0.5 * min(m.sum(), (plab == j).sum()):
                    rises.append(((cy - pa[0][1]) / dt, 0.5 * (cy + pa[0][1])))
        prev = (lab, bubs)
    return (np.nanmean(deltas), np.nanmean(ugE), np.nanmean(ugB), np.array(sizes).reshape(-1, 2),
            np.array(rises).reshape(-1, 2))


# ------------------------------------------------------------------ particle velocity statistics
def cell_series(d, yc, ncell_x=None, cell=CELL, ks=None):
    """Per-frame spatial means and second moments in a row of sampling cells at height yc (eqs 6, 9)."""
    W = float(d["W"])
    nx = ncell_x or max(1, int(round(W / cell)))
    edges = np.linspace(0, W, nx + 1)
    ks = win(d) if ks is None else ks
    vsx = np.full((len(ks), nx), np.nan); vsy = vsx.copy(); cxx = vsx.copy(); cyy = vsx.copy()
    for n, k in enumerate(ks):
        x, y, vx, vy = d["x"][k], d["y"][k], d["vx"][k], d["vy"][k]
        row = np.abs(y - yc) <= cell / 2
        for i in range(nx):
            s = row & (x >= edges[i]) & (x < edges[i + 1])
            if s.sum() >= 2:
                a, b = vx[s], vy[s]
                vsx[n, i], vsy[n, i] = a.mean(), b.mean()
                cxx[n, i], cyy[n, i] = ((a - a.mean()) ** 2).mean(), ((b - b.mean()) ** 2).mean()
    xc = 0.5 * (edges[1:] + edges[:-1])
    return xc, vsx, vsy, cxx, cyy


def profile(d, yc=YPROBE, cell=CELL, ncell_x=None):
    xc, vsx, vsy, cxx, cyy = cell_series(d, yc, ncell_x, cell)
    mean = lambda a: np.nanmean(a, 0)
    rms = lambda a: np.sqrt(np.nanmean((mean(a)[None, :] - a) ** 2, 0))          # eq. (8)
    return dict(x=xc, mean_x=mean(vsx), mean_y=mean(vsy), rms_x=rms(vsx), rms_y=rms(vsy),
                theta=0.5 * (mean(cxx) + mean(cyy)))                              # 2-D granular temperature


def field_maps(d, cell=CELL, ymax=0.40):
    """Time-averaged mean and RMS of the cell-averaged particle velocity on a cell grid up to ymax."""
    W = float(d["W"])
    nx = int(round(W / cell))
    ny = int(round(ymax / cell))
    xs = np.linspace(0, W, nx + 1)
    ys = np.arange(ny + 1) * cell
    ks = win(d)
    VX = np.full((len(ks), nx, ny), np.nan); VY = VX.copy()
    for n, k in enumerate(ks):
        ix = np.minimum((d["x"][k] / (W / nx)).astype(int), nx - 1)
        iy = (d["y"][k] / cell).astype(int)
        ok = (iy >= 0) & (iy < ny)
        cnt = np.zeros((nx, ny)); sx = cnt.copy(); sy = cnt.copy()
        np.add.at(cnt, (ix[ok], iy[ok]), 1)
        np.add.at(sx, (ix[ok], iy[ok]), d["vx"][k][ok])
        np.add.at(sy, (ix[ok], iy[ok]), d["vy"][k][ok])
        m = cnt >= 2
        VX[n][m] = sx[m] / cnt[m]
        VY[n][m] = sy[m] / cnt[m]
    mx, my = np.nanmean(VX, 0), np.nanmean(VY, 0)
    rx = np.sqrt(np.nanmean((VX - mx[None]) ** 2, 0))
    ry = np.sqrt(np.nanmean((VY - my[None]) ** 2, 0))
    return xs, ys, mx, my, rx, ry


# ------------------------------------------------------------------ Fig. 3, 4
def fig3():
    fig, axs = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for ax, (lab, u) in zip(axs, UGS):
        for ck, kd, name in KINDS:
            d = load(u, kd)
            if d is None:
                continue
            m = d["t"] > TSTART
            ax.plot(d["t"], d["dP"], color=C[ck], lw=0.8, label=f"{name}: mean {d['dP'][m].mean():.0f} Pa (t > {TSTART:g} s)")
        d0 = load(u, "1way") or load(u, "2way")
        if d0 is not None:
            ax.axhline(float(d0["dP_theory"]), color=C["ink2"], ls="--", lw=0.9)
            ax.text(0.99, float(d0["dP_theory"]) + 8, f"bed weight / area = {float(d0['dP_theory']):.0f} Pa",
                    ha="right", color=C["ink2"], fontsize=8, transform=ax.get_yaxis_transform())
        ax.axvline(TSTART, color=C["grid"], lw=1)
        ax.set_ylabel("ΔP$_{bed}$ (Pa)")
        ax.set_title(f"{lab} (u$_g$ = {u} m/s)", loc="left", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("Fig. 3 — bed pressure drop time series, u$_g$ = 2, 3, 4 u$_{mf}$ (paper: 1013, 996, 914 Pa vs 1075 Pa)", y=0.995)
    fig.tight_layout()
    save(fig, "fig03_dP_timeseries.png")


def psd(d):
    m = d["t"] > TSTART
    x = d["dP"][m] - d["dP"][m].mean()
    f, P = signal.welch(x, fs=float(d["sample_hz"]), nperseg=min(256, len(x)), noverlap=min(128, len(x) // 2))
    return f, P


def fig4():
    fig, axs = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)
    for ax, (lab, u) in zip(axs, UGS):
        for ck, kd, name in KINDS:
            d = load(u, kd)
            if d is None:
                continue
            f, P = psd(d)
            sel = (f > 0.2) & (f < 12)
            fd = f[sel][np.argmax(P[sel])]
            ax.plot(f, P, color=C[ck], label=f"{name}: dominant {fd:.1f} Hz")
        ax.set_xlim(0, 10)
        ax.set_yscale("log")
        ax.set_ylabel("PSD (Pa$^2$/Hz)")
        ax.set_title(lab, loc="left", fontsize=10)
        ax.legend(fontsize=8)
    axs[-1].set_xlabel("frequency (Hz)")
    fig.suptitle("Fig. 4 — PSD of ΔP$_{bed}$ (paper: dominant frequencies 2–5 Hz)", y=0.995)
    fig.tight_layout()
    save(fig, "fig04_psd.png")


# ------------------------------------------------------------------ Fig. 5, 6
def _frame_at(d, t):
    return int(np.argmin(np.abs(d["t"] - t)))


def fig5(t=10.0):
    for ck, kd, name in KINDS:
        if not have(*[(u, kd) for _, u in UGS]):
            continue
        fig, axs = plt.subplots(1, 6, figsize=(13, 6.4), gridspec_kw=dict(wspace=0.15))
        for j, (lab, u) in enumerate(UGS):
            d = load(u, kd)
            k = _frame_at(d, t)
            W, Hh = float(d["W"]), 0.50
            sp = np.hypot(d["vx"][k], d["vy"][k])
            ax = axs[2 * j]
            sc = ax.scatter(d["x"][k], d["y"][k], c=sp, s=2.0, cmap="Blues", vmin=0, vmax=max(1.5, 0.35 * float(u)), linewidths=0)
            ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.set_aspect("equal")
            ax.set_title(f"{lab}\nparticle speed", fontsize=9)
            ax = axs[2 * j + 1]
            e = d["eps"][k]
            cs = ax.imshow(e.T, origin="lower", extent=[0, W, 0, float(d["H"])], cmap="Blues_r", vmin=0.4, vmax=1.0,
                           aspect="equal", interpolation="bilinear")
            ax.set_ylim(0, Hh); ax.set_yticks([])
            ax.set_title("gas volume\nfraction", fontsize=9)
            for a in (axs[2 * j], axs[2 * j + 1]):
                a.grid(False)
                a.tick_params(labelsize=7)
        cb = fig.colorbar(sc, ax=axs[0::2].tolist(), shrink=0.5, pad=0.01, label="|v$_p$| (m/s)")
        fig.suptitle(f"Fig. 5 — instantaneous snapshots at t = {t:g} s, {name} coupling", y=0.97)
        save(fig, f"fig05_snapshots_{kd}.png")


def fig6(t0=10.0, dtf=0.1, nfr=4):
    for ck, kd, name in KINDS:
        if not have(*[(u, kd) for _, u in UGS]):
            continue
        fig, axs = plt.subplots(3, nfr, figsize=(1.9 * nfr + 0.6, 10.0), gridspec_kw=dict(wspace=0.08, hspace=0.12))
        for i, (lab, u) in enumerate(UGS):
            d = load(u, kd)
            for j in range(nfr):
                ax = axs[i, j]
                k = _frame_at(d, t0 + j * dtf)
                W = float(d["W"])
                ug = gas_vy(d, k)
                NCX, NCY = ug.shape
                xs = (np.arange(NCX) + 0.5) * float(d["dx"])
                ys = (np.arange(NCY) + 0.5) * float(d["dy"])
                im = ax.imshow(ug.T, origin="lower", extent=[0, W, 0, float(d["H"])], cmap="Blues", vmin=0, vmax=3 * float(u),
                               interpolation="nearest", aspect="equal")
                e = d["eps"][k]
                ax.contour(xs, ys, e.T, levels=[0.75, 0.80], colors=C["twoway"], linewidths=1.0)
                Q = ax.quiver(xs[::2], ys[::2], np.zeros_like(ug[::2, ::2].T), ug[::2, ::2].T / (3 * float(u)) * 0.04,
                              angles="xy", scale_units="xy", scale=1, color=C["ink"], width=0.004, headwidth=3)
                ax.set_ylim(0, 0.50); ax.set_xticks([]); ax.set_yticks([])
                ax.grid(False)
                if j == 0:
                    ax.set_ylabel(lab)
                ax.set_title(f"t = {t0 + j * dtf:.1f} s", fontsize=8)
        fig.suptitle(f"Fig. 6 — vertical gas velocity (colour, arrows) and bubble boundary ε$_g$ ∈ [0.75, 0.80] (orange), {name}\n"
                     "1-D gas model: no lateral gas velocity", y=0.93, fontsize=10)
        save(fig, f"fig06_gas_velocity_bubbles_{kd}.png")


# ------------------------------------------------------------------ Fig. 7, 8, 9
_bub = {}


def bub(u, kd, thr=BUB_THR):
    key = (u, kd, thr)
    if key not in _bub:
        _bub[key] = bubble_series(load(u, kd), thr=thr, stride=1 if thr == BUB_THR else 5)
    return _bub[key]


def fig7():
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax2 = ax.twinx() if False else None
    uvals = [float(u) for _, u in UGS]
    for ck, kd, name in KINDS:
        if not have(*[(u, kd) for _, u in UGS]):
            continue
        rows = [bub(u, kd) for _, u in UGS]
        ax.plot(uvals, [r[0] for r in rows], "o-", color=C[ck], label=f"{name}: bubble fraction δ")
    ax.set_xlabel("fluidization velocity u$_g$ (m/s)")
    ax.set_ylabel("bubble fraction δ (−)")
    ax.set_title("Fig. 7 — bubble fraction and gas velocity in the two phases (ε$_g$ = 0.75 boundary)")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig07a_bubble_fraction.png")
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for ck, kd, name in KINDS:
        if not have(*[(u, kd) for _, u in UGS]):
            continue
        rows = [bub(u, kd) for _, u in UGS]
        ax.plot(uvals, [r[1] for r in rows], "s-", color=C[ck], label=f"{name}: emulsion u$_{{g,e}}$")
        ax.plot(uvals, [r[2] for r in rows], "^--", color=C[ck], mfc="none", label=f"{name}: bubble u$_{{g,b}}$")
    ax.set_xlabel("fluidization velocity u$_g$ (m/s)")
    ax.set_ylabel("vertical gas velocity (m/s)")
    ax.set_title("Fig. 7 — gas velocity in emulsion and bubble phases (vertical component)")
    ax.legend(fontsize=8)
    save(fig, "fig07b_gas_velocity_phases.png")


def _binned(arr, edges):
    idx = np.digitize(arr[:, 1], edges) - 1
    c = 0.5 * (edges[1:] + edges[:-1])
    mu = np.array([arr[idx == i, 0].mean() if np.sum(idx == i) >= 3 else np.nan for i in range(len(c))])
    return c, mu


def darton(h, u, umf=UMF_SIM, four_sqrtAD=0.03):
    return 0.54 * (u - umf) ** 0.4 * (h + four_sqrtAD) ** 0.8 * G ** -0.2


def fig8():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    edges = np.arange(0, 0.40, 0.03)
    for ax, (lab, u) in zip(axs, UGS):
        for ck, kd, name in KINDS:
            if load(u, kd) is None:
                continue
            sizes = bub(u, kd)[3]
            if len(sizes):
                c, mu = _binned(sizes, edges)
                ax.plot(c * 1e3, mu * 1e3, "o-", color=C[ck], ms=4, label=name)
        h = np.linspace(0.0, 0.35, 60)
        ax.plot(h * 1e3, darton(h, float(u)) * 1e3, color=C["ink2"], ls="--", label="Darton et al.")
        ax.set_xlabel("bed height (mm)")
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("bubble diameter (mm)")
    fig.suptitle("Fig. 8 — bubble size along the bed height (2-D equivalent diameter)", y=1.02)
    save(fig, "fig08_bubble_size.png")


def fig9():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    edges = np.arange(0, 0.40, 0.04)
    for ax, (lab, u) in zip(axs, UGS):
        for ck, kd, name in KINDS:
            if load(u, kd) is None:
                continue
            r = bub(u, kd)[4]
            if len(r):
                c, mu = _binned(r[:, [0, 1]], edges)
                ax.plot(c * 1e3, mu, "o-", color=C[ck], ms=4, label=name)
        h = np.linspace(0.02, 0.35, 60)
        D = darton(h, float(u))
        ax.plot(h * 1e3, 0.71 * np.sqrt(G * D), color=C["ink2"], ls="--", label="Davidson 0.71√(gD$_b$), D$_b$ from Darton")
        ax.plot(h * 1e3, 0.71 * np.sqrt(G * D) + (float(u) - UMF_SIM), color=C["ink2"], ls=":",
                label="Davidson + (u$_0$ − u$_{mf}$)")
        ax.set_xlabel("bed height (mm)")
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("bubble rise velocity (m/s)")
    fig.suptitle("Fig. 9 — bubble rise velocity (Hilligardt–Werther not drawn: its coefficients are not given in the paper)", y=1.02)
    save(fig, "fig09_bubble_rise.png")


# ------------------------------------------------------------------ Fig. 10–14
def _vec_panel(ax, d, name):
    xs, ys, mx, my, rx, ry = field_maps(d)
    xc, yc = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    e = d["eps"][win(d)].mean(0)
    ax.imshow(e.T, origin="lower", extent=[0, float(d["W"]), 0, float(d["H"])], cmap="Blues_r", vmin=0.4, vmax=1.0,
              aspect="equal", interpolation="bilinear", alpha=0.55)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    ax.quiver(X, Y, mx, my, color=C["ink"], angles="xy", width=0.006, pivot="mid")
    ax.axhline(YPROBE, color=C["twoway"], lw=0.8, ls=":")
    ax.set_ylim(0, 0.40); ax.set_title(name, fontsize=9); ax.grid(False); ax.tick_params(labelsize=7)


def fig_vel(num, u, lab):
    """Figs 10, 12, 13: time-averaged ε / particle-velocity vectors and mean, RMS velocity profiles at y = 0.0672 m."""
    if not any(load(u, kd) is not None for _, kd, _ in KINDS):
        return
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1], hspace=0.30, wspace=0.35)
    for i, (ck, kd, name) in enumerate(KINDS):
        d = load(u, kd)
        if d is None:
            continue
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        _vec_panel(ax, d, name)
    titles = ["(a) mean x-velocity", "(b) RMS of x-velocity", "(c) mean y-velocity", "(d) RMS of y-velocity"]
    keys = ["mean_x", "rms_x", "mean_y", "rms_y"]
    for j, (t, key) in enumerate(zip(titles, keys)):
        ax = fig.add_subplot(gs[1, j])
        for ck, kd, name in KINDS:
            d = load(u, kd)
            if d is None:
                continue
            p = profile(d)
            ax.plot(p["x"] * 100, p[key], "o-", color=C[ck], ms=4, label=name)
        ax.set_title(t, fontsize=9); ax.set_xlabel("x (cm)")
        ax.set_ylabel("m/s")
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"Fig. {num} — {lab}: time-averaged ε$_g$ (shade) and particle velocity (arrows); profiles at y = {YPROBE} m "
                 f"(dotted line), sampling cell 4.57 cm × 4.57 cm; experiment not included", y=0.97, fontsize=10)
    save(fig, f"fig{num:02d}_particle_velocity_{u}.png")


def fig11():
    u = "3.28"
    if not any(load(u, kd) is not None for _, kd, _ in KINDS):
        return
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1], hspace=0.30, wspace=0.35)
    for i, (ck, kd, name) in enumerate(KINDS):
        d = load(u, kd)
        if d is None:
            continue
        xs, ys, mx, my, rx, ry = field_maps(d)
        for j, (arr, nm) in enumerate(((rx, "RMS x-velocity"), (ry, "RMS y-velocity"))):
            ax = fig.add_subplot(gs[0, 2 * i + j])
            im = ax.pcolormesh(xs, ys, arr.T, cmap="Blues", vmin=0, shading="flat")
            ax.axhline(YPROBE, color=C["twoway"], lw=0.8, ls=":")
            ax.set_aspect("equal"); ax.set_ylim(0, 0.40); ax.grid(False); ax.tick_params(labelsize=7)
            ax.set_title(f"{name}\n{nm}", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    for j, (t, key) in enumerate((("(a) RMS of x-velocity", "rms_x"), ("(b) RMS of y-velocity", "rms_y"))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        for ck, kd, name in KINDS:
            d = load(u, kd)
            if d is not None:
                p = profile(d)
                ax.plot(p["x"] * 100, p[key], "o-", color=C[ck], ms=4, label=name)
        ax.set_title(t, fontsize=9); ax.set_xlabel("x (cm)"); ax.set_ylabel("m/s"); ax.legend(fontsize=8)
    fig.suptitle(f"Fig. 11 — time-averaged RMS of particle velocity at 3 u$_{{mf}}$ and profile at y = {YPROBE} m "
                 "(experiment not included)", y=0.97)
    save(fig, "fig11_rms_velocity_3.28.png")


def fig14():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, (lab, u) in zip(axs, UGS):
        for ck, kd, name in KINDS:
            d = load(u, kd)
            if d is not None:
                p = profile(d)
                ax.plot(p["x"] * 100, p["theta"], "o-", color=C[ck], ms=4, label=name)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("x (cm)"); ax.legend(fontsize=8)
    axs[0].set_ylabel("granular temperature θ$_p$ (m$^2$/s$^2$), 2-D")
    fig.suptitle("Fig. 14 — particle granular temperature at y = 0.0672 m (paper simulation 0.01 → 0.05, experiment 0.007 → 0.025 m²/s²)",
                 y=1.02, fontsize=10)
    save(fig, "fig14_granular_temperature.png")


# ------------------------------------------------------------------ Appendix figures
def figA1(t=13.0):
    u = "3.28"
    for ck, kd, name in KINDS:
        d = load(u, kd)
        if d is None:
            continue
        k = _frame_at(d, t)
        fig, axs = plt.subplots(1, 4, figsize=(10, 5.2), gridspec_kw=dict(wspace=0.1))
        for ax, thr in zip(axs, (0.70, 0.75, 0.80, 0.90)):
            lab, Z, h = bubble_map(d["eps"][k], thr)
            img = np.zeros(Z.shape)
            img[Z >= thr] = 1                                   # connected to freeboard or not: shade separately
            img[lab > 0] = 2
            ax.imshow(img.T, origin="lower", extent=[0, float(d["W"]), 0, float(d["H"])], cmap=plt.matplotlib.colors.ListedColormap([C["oneway"], "#cde2fb", C["twoway"]]),
                      vmin=0, vmax=2, interpolation="nearest", aspect="equal")
            ax.set_ylim(0, 0.45); ax.set_title(f"threshold {thr:.2f}", fontsize=9); ax.grid(False); ax.tick_params(labelsize=7)
            if thr != 0.70:
                ax.set_yticks([])
        fig.suptitle(f"Fig. A1 — effect of threshold on bubble and emulsion phases, 3 u$_{{mf}}$, t = {t:g} s, {name}\n"
                     "dark blue: emulsion; pale: ε ≥ threshold connected to the freeboard; orange: bubbles inside the bed", y=1.0, fontsize=9)
        save(fig, f"figA1_threshold_images_{kd}.png")


def figA2():
    u = "3.28"
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    edges = np.arange(0, 0.40, 0.03)
    for a, (ck, kd, name) in zip(ax, KINDS):
        if load(u, kd) is None:
            continue
        for thr, ls in ((0.70, ":"), (0.75, "-"), (0.80, "--"), (0.90, "-.")):
            sizes = bub(u, kd, thr)[3]
            if len(sizes):
                c, mu = _binned(sizes, edges)
                a.plot(c * 1e3, mu * 1e3, color=C[ck], ls=ls, label=f"ε = {thr:.2f}")
        a.set_title(name, fontsize=10); a.set_xlabel("bed height (mm)"); a.legend(fontsize=8)
    ax[0].set_ylabel("bubble diameter (mm)")
    fig.suptitle("Fig. A2 — effect of the threshold on bubble size, 3 u$_{mf}$ (thresholds 0.70, 0.75, 0.80, 0.90; every 5th frame except 0.75)", y=1.02, fontsize=10)
    save(fig, "figA2_threshold_bubble_size.png")


def figA3():
    u = "3.28"
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.0))
    for ck, kd, name in KINDS:
        d = load(u, kd)
        if d is None:
            continue
        for cell, ls in ((0.0457, "-"), (0.023, "--"), (0.013, ":")):
            p = profile(d, cell=cell)
            lab = f"{name}, {cell*100:.2g} cm" if True else None
            axs[0].plot(p["x"] * 100, p["mean_y"], color=C[ck], ls=ls, label=lab)
            axs[1].plot(p["x"] * 100, p["rms_y"], color=C[ck], ls=ls)
            axs[2].plot(p["x"] * 100, p["theta"], color=C[ck], ls=ls)
    for ax, t in zip(axs, ("(a) mean y-velocity", "(b) RMS of y-velocity", "(c) granular temperature")):
        ax.set_title(t, fontsize=9); ax.set_xlabel("x (cm)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Fig. A3 — effect of the sampling-cell size (4.57, 2.3, 1.3 cm) at y = 0.0672 m, 3 u$_{mf}$", y=1.02)
    save(fig, "figA3_sampling_cell_size.png")


FIGS = {"3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8, "9": fig9,
        "10": lambda: fig_vel(10, "3.28", "3 u$_{mf}$"), "11": fig11,
        "12": lambda: fig_vel(12, "2.19", "2 u$_{mf}$"), "13": lambda: fig_vel(13, "4.38", "4 u$_{mf}$"),
        "14": fig14, "A1": figA1, "A2": figA2, "A3": figA3}
