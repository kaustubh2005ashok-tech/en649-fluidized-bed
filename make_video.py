"""
make_video.py — side-by-side video of the hold runs at 1, 2, 3 and 4 u_mf: our one-way and quasi-two-way DEM beds
(particles coloured by speed) next to the paper's snapshot (Liu & van Wachem 2019, Fig. 5, t = 14.5 s) and a running
ΔP / bed-weight trace.

    python make_video.py --paper ../poster/panels/paper_p4_0.jpeg            # -> video/results_1to4umf.mp4
    python make_video.py --paper ... --preview                               # one PNG per segment only

Needs out/hold_full_U{1.10,2.19,3.28,4.38}_{1way,2way}.npz (run_hold.py) and imageio-ffmpeg.
"""
import argparse
import os
import subprocess
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import imageio_ffmpeg

C = dict(oneway="#2a78d6", twoway="#eb6834", ink="#0b0b0b", ink2="#52514e", grid="#e4e3de", surf="#fcfcfb",
         good="#1a7f37", warn="#b26b00", bad="#c62828", head="#0f2a4a")
plt.rcParams.update({"figure.facecolor": C["surf"], "axes.facecolor": C["surf"], "text.color": C["ink"],
                     "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "font.size": 13})
FPS = 25
W_PX, H_PX, DPI = 1920, 1080, 100
PAPER_W = 1075.0                                         # paper's theoretical ΔP (Pa) = its bed weight / area

# per case: u_g, file tag, paper ΔP mean (Pa, None = not reported), crop of paper Fig. 5 (row fractions), |v| range,
# expected behaviour, our verdict (filled for 1 u_mf from the data)
CASES = [
    dict(n=1, ug=1.10, tag="U1.10", paper_dP=None, crop=None, vmax=0.5,
         expect=["Incipient fluidization: drag just balances the bed weight, so ΔP ≈ bed weight / area.",
                 "The bed expands slightly; no bubbles; particles stay almost still.",
                 "The paper has no snapshot at 1 u_mf (u_mf = 1.00 m/s sim, 1.05 m/s expt)."]),
    dict(n=2, ug=2.19, tag="U2.19", paper_dP=1013, crop=(0.0, 0.325), vmax=1.5,
         expect=["Bubbling: discrete bubbles form at the distributor, grow, rise and erupt at the surface.",
                 "Bed surface ≈ 0.3 m; mean ΔP 1013 Pa (0.94 × weight), swinging ±500 Pa at 2–5 Hz."],
         verdict=("partial", "Mean ΔP matches (0.99 × weight) and the bed fluidizes, but it forms one large "
                  "central void with solids rising at the walls, and ΔP fluctuations are ~10× too small.")),
    dict(n=3, ug=3.28, tag="U3.28", paper_dP=996, crop=(0.325, 0.665), vmax=2.2,
         expect=["Vigorous bubbling: larger bubbles and strong mixing; surface ≈ 0.4 m.",
                 "Mean ΔP 996 Pa (0.93 × weight): still the bed weight, because gas bypasses through bubbles."],
         verdict=("bad", "Fails: the whole bed lifts as a plug and is pressed against the lid; ΔP ≈ 6 × weight. "
                  "The 1-D gas model cannot let gas bypass the bed as bubbles.")),
    dict(n=4, ug=4.38, tag="U4.38", paper_dP=914, crop=(0.665, 1.0), vmax=2.6,
         expect=["Near-slugging: very large bubbles, particles thrown up to ≈ 0.45 m.",
                 "Mean ΔP 914 Pa (0.85 × weight)."],
         verdict=("bad", "Fails: plug lift against the lid; ΔP ≈ 12 × weight (same cause as 3 u_mf).")),
]


def load(tag, kind):
    f = f"out/hold_full_{tag}_{kind}.npz"
    return np.load(f) if os.path.exists(f) else None


def verdict_1umf(r1, r2):
    rat = [float(np.mean(r["dP"][r["t"] > 2.5]) / r["dP_theory"]) for r in (r1, r2)]
    h = [float(np.mean(np.max(r["y"][r["t"] > 2.5], 1))) for r in (r1, r2)]
    ok = all(0.85 < x < 1.1 for x in rat)
    return ("good" if ok else "partial",
            f"ΔP = {rat[0]:.2f} × weight (one-way), {rat[1]:.2f} × weight (quasi-two-way); top particle at "
            f"{h[0]*1e3:.0f} / {h[1]*1e3:.0f} mm. " + ("Matches the expected incipient state." if ok else ""))


def bed_axes(fig, rect, r, title, color, vmax):
    ax = fig.add_axes(rect)
    W, H, dp = float(r["W"]), float(r["H"]), float(r["dp"])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_yticks(np.arange(0, H + 1e-9, 0.1))
    ax.tick_params(labelsize=10)
    for s in ax.spines.values():
        s.set_color(C["ink2"])
    ax.set_title(title, color=color, fontsize=15, fontweight="bold")
    n = r["x"].shape[1]
    d = 1.2 * dp                                         # drawn 20% oversize: ~4 px circles otherwise look sparse
    ec = EllipseCollection(np.full(n, d), np.full(n, d), np.zeros(n), units="xy", offsets=np.zeros((n, 2)),
                           offset_transform=ax.transData, cmap="turbo", linewidths=0)
    ec.set_clim(0, vmax)
    ax.add_collection(ec)
    return ax, ec


def wrap(lines, width):
    return "\n".join("• " + textwrap.fill(s, width, subsequent_indent="   ") for s in lines)


def segment_fig(case, r1, r2, paper):
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    fig.patches.append(plt.Rectangle((0, 0.925), 1, 0.075, transform=fig.transFigure, color=C["head"], zorder=-1))
    fig.text(0.015, 0.962, f"{case['n']} u$_{{mf}}$   (u$_g$ = {case['ug']:.2f} m/s)", color="white", fontsize=26,
             fontweight="bold", va="center")
    ttxt = fig.text(0.985, 0.962, "", color="white", fontsize=20, ha="right", va="center", family="monospace")
    fig.text(0.40, 0.962, "2-D DEM gas-fluidized bed  ·  reproduction of Liu & van Wachem (2019)", color="#c9d6e6",
             fontsize=14, va="center")

    ax1, ec1 = bed_axes(fig, [0.02, 0.30, 0.17, 0.58], r1, "Ours: one-way", C["oneway"], case["vmax"])
    ax2, ec2 = bed_axes(fig, [0.185, 0.30, 0.17, 0.58], r2, "Ours: quasi-two-way", C["twoway"], case["vmax"])
    ax2.set_yticklabels([])
    cax = fig.add_axes([0.36, 0.36, 0.008, 0.46])
    cb = fig.colorbar(ec1, cax=cax)
    cb.set_label("particle speed |v| (m/s)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    fig.text(0.19, 0.255, "x (m)  ·  bed width 0.23 m, height 0.55 m (lid)", ha="center", fontsize=11,
             color=C["ink2"])

    axp = fig.add_axes([0.405, 0.30, 0.355, 0.58])
    axp.set_axis_off()
    if case["crop"] is not None:
        h = paper.shape[0]
        a, b = case["crop"]
        axp.imshow(paper[int(a * h):int(b * h)])
        axp.set_title("Paper (should look like this): Fig. 5, t = 14.5 s  [static]", fontsize=15,
                      fontweight="bold", color=C["ink"])
        fig.text(0.583, 0.275, "left: particles by speed |v|  ·  right: gas volume fraction (red = dense)",
                 ha="center", fontsize=11, color=C["ink2"])
    else:
        axp.set_title("Paper: no snapshot at 1 u$_{mf}$", fontsize=15, fontweight="bold")
        axp.text(0.5, 0.5, "Expected: a packed bed that\njust becomes fluid —\nno bubbles, almost no motion.",
                 ha="center", va="center", fontsize=20, color=C["ink2"], transform=axp.transAxes)

    fig.text(0.775, 0.875, "What should happen", fontsize=17, fontweight="bold", va="top")
    fig.text(0.775, 0.835, wrap(case["expect"], 38), fontsize=13, va="top", linespacing=1.35)
    kind, msg = case["verdict"]
    col = dict(good=C["good"], partial=C["warn"], bad=C["bad"])[kind]
    lab = dict(good="✔ What we get", partial="◐ What we get", bad="✘ What we get")[kind]
    fig.text(0.775, 0.55, lab, fontsize=17, fontweight="bold", va="top", color=col)
    fig.text(0.775, 0.51, textwrap.fill(msg, 40), fontsize=13, va="top", linespacing=1.35)

    axd = fig.add_axes([0.06, 0.06, 0.92, 0.17])
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)
    axd.grid(True, color=C["grid"], lw=0.6)
    tmax = float(r1["t"][-1])
    allr = np.concatenate([r1["dP"] / r1["dP_theory"], r2["dP"] / r2["dP_theory"]])
    ytop = max(1.6, 1.1 * float(np.percentile(allr, 99.5)))
    axd.set_xlim(0, tmax)
    axd.set_ylim(0, ytop)
    axd.axhline(1.0, color=C["ink2"], lw=1, ls=":")
    axd.text(tmax * 0.005, 1.0, " bed weight", va="bottom", fontsize=10, color=C["ink2"])
    if case["paper_dP"]:
        axd.axhline(case["paper_dP"] / PAPER_W, color=C["good"], lw=1.6, ls="--",
                    label=f"paper mean ({case['paper_dP']} Pa)")
    l1, = axd.plot([], [], color=C["oneway"], lw=1.4, label="one-way")
    l2, = axd.plot([], [], color=C["twoway"], lw=1.4, label="quasi-two-way")
    cur = axd.axvline(0, color=C["ink"], lw=0.8)
    axd.set_xlabel("time (s)", fontsize=12)
    axd.set_ylabel("ΔP / bed weight", fontsize=12)
    axd.legend(loc="upper left", fontsize=11, frameon=False, ncol=3)
    axd.tick_params(labelsize=10)

    def update(k):
        for r, ec, ln in ((r1, ec1, l1), (r2, ec2, l2)):
            ec.set_offsets(np.column_stack([r["x"][k], r["y"][k]]))
            ec.set_array(np.hypot(r["vx"][k], r["vy"][k]))
            ln.set_data(r["t"][:k + 1], r["dP"][:k + 1] / r["dP_theory"])
        cur.set_xdata([r1["t"][k]])
        ttxt.set_text(f"t = {r1['t'][k]:5.2f} s")
    return fig, update


def card(lines, sub=None):
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=C["head"])
    y = 0.62
    for i, s in enumerate(lines):
        fig.text(0.5, y, s, ha="center", va="center", color="white", fontsize=40 if i == 0 else 22,
                 fontweight="bold" if i == 0 else "normal")
        y -= 0.1 if i == 0 else 0.06
    if sub:
        fig.text(0.5, 0.12, sub, ha="center", color="#c9d6e6", fontsize=16)
    return fig


def grab(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, help="page-4 image of the paper (Fig. 5)")
    ap.add_argument("--out", default="video/results_1to4umf.mp4")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    paper = plt.imread(a.paper)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    ff = None
    if not a.preview:
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-s", f"{W_PX}x{H_PX}", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", a.out], stdin=subprocess.PIPE)

    def hold(fig, sec):
        b = grab(fig)
        for _ in range(int(sec * FPS)):
            ff.stdin.write(b)
        plt.close(fig)

    if ff:
        hold(card(["Gas-fluidized bed: our DEM vs the reference paper",
                   "1, 2, 3 and 4 u$_{mf}$  ·  left: our one-way and quasi-two-way runs  ·  right: what the paper shows",
                   "Kaustubh Ashok (24B1522)  ·  EN 649  ·  Instructor: Prof. Manaswita Bose  ·  IIT Bombay"],
                  "Reference: Liu & van Wachem (2019), Powder Technol. 343, 145–158  ·  played at real time"), 4)

    for case in CASES:
        r1, r2 = load(case["tag"], "1way"), load(case["tag"], "2way")
        if r1 is None or r2 is None:
            print("missing data for", case["tag"], "- skipped")
            continue
        if case["n"] == 1:
            case["verdict"] = verdict_1umf(r1, r2)
        fig, update = segment_fig(case, r1, r2, paper)
        nk = min(len(r1["t"]), len(r2["t"]))
        step = max(1, int(round(float(r1["sample_hz"]) / FPS)))
        if a.preview:
            update(nk - 1)
            fig.savefig(a.out.replace(".mp4", f"_preview_{case['n']}umf.png"), dpi=DPI)
            plt.close(fig)
            continue
        for k in range(0, nk, step):
            update(k)
            ff.stdin.write(grab(fig))
        update(nk - 1)
        hold(fig, 2)
        print("rendered", case["tag"], flush=True)

    if ff:
        hold(card(["Summary",
                   "1 u$_{mf}$ and 2 u$_{mf}$: mean ΔP equals the bed weight, as in the paper",
                   "2 u$_{mf}$: one large void instead of discrete bubbles; ΔP fluctuations too small",
                   "3 and 4 u$_{mf}$: the bed lifts as a plug (ΔP 6–12 × weight) — the 1-D gas model has no bubble bypass",
                   "Next step: a 2-D gas pressure (Darcy–Ergun) solver so gas can flow around the bubbles"]), 6)
        ff.stdin.close()
        ff.wait()
        print("wrote", a.out)


if __name__ == "__main__":
    main()
