"""
run_ramp.py — paper protocol (Liu & van Wachem §4.2): U_g ramped 0 → Ug_max at 0.2 m/s²
then back to 0, ΔP sampled at 50 Hz. Writes out/ramp_<tag>.npz.

    python run_ramp.py                       # uses out/state_full.npz
    python run_ramp.py --tag demo            # uses out/state_demo.npz
    python run_ramp.py --tag demo --Ug_max 2.0 --rate 0.5   # faster ramp for development

Wall-clock: the full bed at the paper's 0.2 m/s² to 2.2 m/s is 22 s of physical time
≈ 6.6 M DEM steps ≈ 2–3 h on a laptop. Use --rate 1.0 while developing.
"""
import argparse
import time
import numpy as np
from params import Params
from bed import load_state
from sim import ramp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--Ug_max", type=float, default=None)
    ap.add_argument("--rate", type=float, default=None, help="ramp rate m/s^2 (paper: 0.2)")
    a = ap.parse_args()

    state, meta = load_state(f"out/state_{a.tag}.npz")
    kw = dict(W=float(meta["W"]), H=float(meta["H"]), nrow=int(meta["nrow"]),
              en=float(meta["en"]), mu_f=float(meta["mu_f"]), kn=float(meta["kn"]),
              seed=int(meta["seed"]))
    if a.Ug_max is not None:
        kw["Ug_max"] = a.Ug_max
    if a.rate is not None:
        kw["ramp_rate"] = a.rate
    p = Params(**kw)
    N = state[0].size
    args = p.kernel_args(N)

    print(p.summary())
    print(f"N={N}  eps_mf={float(meta['eps_mf']):.4f}  ramp to {p.Ug_max} m/s at {p.ramp_rate} m/s^2  "
          f"({2*p.Ug_max/p.ramp_rate:.1f} s physical, {int(2*p.Ug_max/p.ramp_rate/p.dt)/1e6:.1f} M steps)")

    t0 = time.time()
    r = ramp(state, p, args)
    print(f"done in {(time.time()-t0)/60:.1f} min")

    out = f"out/ramp_{a.tag}.npz"
    np.savez_compressed(out, **r, N=N, W=p.W, dp=p.dp, m=p.m, g=p.g,
                        eps_mf=float(meta["eps_mf"]), h0=float(meta["h0"]),
                        dP_theory=p.dP_theory(N), en=p.en, mu_f=p.mu_f, seed=p.seed,
                        ramp_rate=p.ramp_rate, Ug_max=p.Ug_max)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
