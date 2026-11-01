"""
run_hold.py — constant-U_g run for the paper's flow-statistics figures (Liu & van Wachem 2019, Figs 3–15).

U_g is switched on at t = 0 and held for t_hold (paper: 15 s; statistics from 5–15 s). At sample_hz (50 Hz)
it stores ΔP, every particle's (x, y, v_x, v_y), the voidage field and the row-averaged solids flux q_s
(needed for the two-way gas velocity u_g = (U_g − q_s)/ε_g). Writes out/hold_<out_tag>.npz.

    python run_hold.py --Ug 3.28 --nfluid 5                    # one-way, 3 u_mf
    python run_hold.py --Ug 3.28 --nfluid 5 --two_way          # quasi-two-way (1-D volume flux)
    python run_hold.py --Ug 3.28 --t_hold 1.0 --out_tag test   # quick check
"""
import argparse
import time
import numpy as np
from params import Params
from bed import load_state
from sim import step
from drag import voidage, solids_flux_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full", help="settled bed out/state_<tag>.npz")
    ap.add_argument("--Ug", type=float, required=True, help="fluidization velocity, m/s (paper: 2.19, 3.28, 4.38)")
    ap.add_argument("--t_hold", type=float, default=15.0)
    ap.add_argument("--ramp_t", type=float, default=2.0,
                    help="s over which U_g is ramped linearly from 0 (an instant step throws the whole bed "
                         "against the ceiling in this 1-D gas model); statistics use t > 5 s")
    ap.add_argument("--nfluid", type=int, default=5)
    ap.add_argument("--two_way", action="store_true")
    ap.add_argument("--out_tag", default=None)
    a = ap.parse_args()
    out_tag = a.out_tag or f"{a.tag}_U{a.Ug:.2f}_{'2way' if a.two_way else '1way'}"

    state, meta = load_state(f"out/state_{a.tag}.npz")
    p = Params(W=float(meta["W"]), H=float(meta["H"]), nrow=int(meta["nrow"]), en=float(meta["en"]),
               mu_f=float(meta["mu_f"]), kn=float(meta["kn"]), seed=int(meta["seed"]),
               nfluid=a.nfluid, two_way=int(a.two_way))
    x, y, vx, vy = state
    N = x.size
    args = p.kernel_args(N)
    print(p.summary())
    print(f"hold U_g={a.Ug} m/s (ramped over {a.ramp_t} s) for {a.t_hold} s,{'two-way' if p.two_way else 'one-way'}, nfluid={p.nfluid}, "
          f"N={N}, output hold_{out_tag}.npz", flush=True)

    nsub = max(1, int(round(1.0 / (p.sample_hz * p.dt))))
    n = int(round(a.t_hold * p.sample_hz))
    t = np.arange(1, n + 1) / p.sample_hz
    dP = np.empty(n)
    X = np.empty((n, N), np.float32)
    Y = np.empty((n, N), np.float32)
    VX = np.empty((n, N), np.float32)
    VY = np.empty((n, N), np.float32)
    EPS = np.empty((n, p.NCX, p.NCY), np.float32)
    QS = np.empty((n, p.NCY), np.float32)

    t0 = time.time()
    last = 0
    for k in range(n):
        Ug_k = a.Ug * min(1.0, k / (a.ramp_t * p.sample_hz)) if a.ramp_t > 0 else a.Ug
        F = step(x, y, vx, vy, N, Ug_k, nsub, 1, *args)
        dP[k] = F / (p.W * p.dp)
        X[k], Y[k], VX[k], VY[k] = x, y, vx, vy
        EPS[k] = voidage(x, y, N, p.NCX, p.NCY, p.dx, p.dy, p.Vp, p.dp, p.eps_min)
        QS[k] = solids_flux_rows(y, vy, N, p.NCY, p.dy, p.W, p.Vp, p.dp)
        pct = int(100 * (k + 1) / n)
        if pct > last and pct % 2 == 0:
            last = pct
            el = time.time() - t0
            print(f"  [{pct:3d}%] t={t[k]:6.2f}/{a.t_hold:.1f} s  dP={dP[k]:7.1f} Pa  h={y.max()*1e3:6.1f} mm  "
                  f"elapsed {el/60:6.1f} min  ETA {el*(100-pct)/pct/60:6.1f} min", flush=True)

    out = f"out/hold_{out_tag}.npz"
    np.savez_compressed(out, t=t, dP=dP, x=X, y=Y, vx=VX, vy=VY, eps=EPS, qs=QS, Ug=a.Ug, N=N, W=p.W, H=p.H,
                        dp=p.dp, m=p.m, g=p.g, dx=p.dx, dy=p.dy, NCX=p.NCX, NCY=p.NCY, Vp=p.Vp,
                        dP_theory=p.dP_theory(N), nfluid=p.nfluid, two_way=p.two_way, seed=p.seed,
                        eps_mf=float(meta["eps_mf"]), h0=float(meta["h0"]), sample_hz=p.sample_hz)
    print(f"done in {(time.time()-t0)/60:.1f} min, saved {out}")


if __name__ == "__main__":
    main()
