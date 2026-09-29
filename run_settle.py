"""
run_settle.py — build the lattice, settle under gravity, save out/state_<tag>.npz.

    python run_settle.py                 # full bed  (0.23 m, ~3,380 particles)
    python run_settle.py --demo          # small bed (0.09 m, 624 particles)
    python run_settle.py --tag myrun --en 0.78 --mu 0.45
"""
import argparse
import time
import numpy as np
from params import Params
from bed import build_bed, bed_voidage, save_state
from sim import settle
from drag import ergun_umf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="small 0.09 m bed")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--en", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--t_settle", type=float, default=0.5)
    a = ap.parse_args()

    kw = {}
    if a.demo:
        kw.update(W=0.09, H=0.34, nrow=26)
    if a.en is not None:
        kw["en"] = a.en
    if a.mu is not None:
        kw["mu_f"] = a.mu
    if a.seed is not None:
        kw["seed"] = a.seed
    p = Params(**kw)
    tag = a.tag or ("demo" if a.demo else "full")

    print(p.summary())
    state = build_bed(p)
    N = state[0].size
    args = p.kernel_args(N)
    print(f"N = {N} particles ({p.nrow} rows x {p.ncol} cols)")

    t0 = time.time()
    settle(state, p, args, t_settle=a.t_settle)
    eps, h = bed_voidage(state, p)
    umf_e = ergun_umf(eps, p.dp, p.rho_p, p.rho_g, p.mu_g, p.g)
    print(f"\nsettled in {time.time()-t0:.0f} s:  h = {h:.4f} m   eps_mf = {eps:.4f}   "
          f"dP_theory = {p.dP_theory(N):.1f} Pa   Ergun u_mf(eps_mf) = {umf_e:.3f} m/s")

    out = f"out/state_{tag}.npz"
    save_state(out, state, p, eps_mf=eps, h0=h)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
