"""
animate.py — three-panel GIF: particle field (colour = speed), voidage field, live ΔP(U_g).

    python animate.py --tag demo                     # ramp 0 → 2.0 m/s over 1.2 s then hold
    python animate.py --tag demo --Ug_max 1.6 --nramp 100 --nhold 60

Illustration only — the ramp here is far faster than the paper's 0.2 m/s², so the
curve is not the quantitative result; use run_ramp.py + analysis.py for that.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from params import Params
from bed import load_state
from sim import step
from drag import voidage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--Ug_max", type=float, default=2.0)
    ap.add_argument("--nramp", type=int, default=115)
    ap.add_argument("--nhold", type=int, default=70)
    ap.add_argument("--sub", type=int, default=3000, help="DEM steps per frame")
    a = ap.parse_args()

    state, meta = load_state(f"out/state_{a.tag}.npz")
    p = Params(W=float(meta["W"]), H=float(meta["H"]), nrow=int(meta["nrow"]),
               en=float(meta["en"]), mu_f=float(meta["mu_f"]), seed=int(meta["seed"]))
    x, y, vx, vy = state
    N = x.size
    args = p.kernel_args(N)
    theo = p.dP_theory(N)
    ymax = 1000 * min(p.H, 2.2 * y.max())
    NF = a.nramp + a.nhold
    frame_dt = a.sub * p.dt

    os.makedirs("out/frames", exist_ok=True)
    Ugs, dPs, files = [], [], []
    for f in range(NF):
        Ug = a.Ug_max * min(f, a.nramp - 1) / (a.nramp - 1)
        F = step(x, y, vx, vy, N, Ug, a.sub, 1, *args)
        if f < a.nramp:
            Ugs.append(Ug); dPs.append(F / (p.W * p.dp))
        sp = np.sqrt(vx ** 2 + vy ** 2)
        eps = voidage(x, y, N, p.NCX, p.NCY, p.dx, p.dy, p.Vp, p.dp, p.eps_min)

        fig, ax = plt.subplots(1, 3, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [1.05, 1.05, 1.5]})
        ax[0].scatter(x * 1e3, y * 1e3, s=17.0, c=sp, cmap="viridis", vmin=0, vmax=2.0, edgecolors="none")
        ax[0].set_xlim(0, p.W * 1e3); ax[0].set_ylim(0, ymax); ax[0].set_aspect("equal")
        ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("y (mm)"); ax[0].set_title("particles (colour = speed)", fontsize=9)
        ax[1].imshow(eps.T, origin="lower", cmap="RdYlBu", vmin=0.4, vmax=1.0,
                     extent=[0, p.W * 1e3, 0, p.H * 1e3], aspect="equal", interpolation="bilinear")
        ax[1].set_ylim(0, ymax); ax[1].set_xlabel("x (mm)"); ax[1].set_yticklabels([])
        ax[1].set_title(r"gas volume fraction $\varepsilon_g$", fontsize=9)
        ax[2].plot(Ugs, dPs, "-", color="#1f6f8b", lw=1.6)
        ax[2].plot(Ugs[-1], dPs[-1], "o", color="#c1440e", ms=6)
        ax[2].axhline(theo, ls="--", color="k", lw=1)
        ax[2].text(0.06, theo * 1.03, "bed weight / area", fontsize=8)
        ax[2].set_xlim(0, a.Ug_max); ax[2].set_ylim(0, theo * 1.35)
        ax[2].set_xlabel(r"superficial velocity $U_g$ (m/s)"); ax[2].set_ylabel(r"$\Delta P_{bed}$ (Pa)")
        ax[2].set_title("fluidization curve", fontsize=9); ax[2].grid(alpha=0.3)
        fig.suptitle(f"2-D DEM fluidized bed  ·  N = {N},  $d_p$ = {p.dp*1e3:.3f} mm,  Gidaspow drag   |   "
                     f"t = {f*frame_dt:.2f} s,  $U_g$ = {Ug:.2f} m/s" + ("" if f < a.nramp else "  (holding)"), fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fn = f"out/frames/f{f:04d}.png"
        fig.savefig(fn, dpi=95); plt.close(fig); files.append(fn)
        if f % 25 == 0:
            print(f"frame {f:3d}  Ug={Ug:.2f}  dP={dPs[-1]:.0f} Pa  h={y.max()*1e3:.0f} mm")

    out = f"out/fluidized_bed_{a.tag}.gif"
    imageio.mimsave(out, [imageio.imread(fn) for fn in files], duration=0.07, loop=0)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
