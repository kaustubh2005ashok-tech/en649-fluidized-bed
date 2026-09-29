"""
run_sweep.py — sensitivity of u_mf to e_n and μ (and seed), parallelised with multiprocessing.

    python run_sweep.py                       # 3×3 grid on the demo bed, 6 workers
    python run_sweep.py --full --workers 4    # full bed (long: ~3 h per case / worker)
    python run_sweep.py --seeds 3             # scatter study at the default (e_n, μ)

Each case: settle → ramp → extract u_mf. Results appended to out/sweep.csv and
plotted to out/sweep.png. Set NUMBA_NUM_THREADS=1 per worker (done below) so the
workers do not fight over cores.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import argparse
import itertools
import time
import numpy as np
from multiprocessing import Pool


def one_case(case):
    en, mu, seed, demo, Ug_max, rate = case
    from params import Params
    from bed import build_bed, bed_voidage
    from sim import settle, ramp
    from analysis import extract_umf
    from drag import ergun_umf

    kw = dict(en=en, mu_f=mu, seed=seed, Ug_max=Ug_max, ramp_rate=rate)
    if demo:
        kw.update(W=0.09, H=0.34, nrow=26)
    p = Params(**kw)
    state = build_bed(p)
    N = state[0].size
    args = p.kernel_args(N)
    t0 = time.time()
    settle(state, p, args, t_settle=0.4, verbose=False)
    eps, h0 = bed_voidage(state, p)
    r = ramp(state, p, args, verbose=False)
    dn = r["branch"] == -1
    umf, _ = extract_umf(r["Ug"][dn], r["dP"][dn], p.dP_theory(N))
    return dict(en=en, mu=mu, seed=seed, N=N, eps_mf=eps, umf=umf,
                umf_ergun=ergun_umf(eps, p.dp, p.rho_p, p.rho_g, p.mu_g, p.g),
                wall=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=1, help="if >1, vary seed at default (e_n, mu) only")
    ap.add_argument("--Ug_max", type=float, default=2.0)
    ap.add_argument("--rate", type=float, default=1.0, help="ramp rate; use 0.2 for paper-spec")
    a = ap.parse_args()

    if a.seeds > 1:
        cases = [(0.84, 0.35, s, not a.full, a.Ug_max, a.rate) for s in range(1, a.seeds + 1)]
    else:
        ens = (0.78, 0.84, 0.90)
        mus = (0.25, 0.35, 0.45)
        cases = [(en, mu, 7, not a.full, a.Ug_max, a.rate) for en, mu in itertools.product(ens, mus)]

    print(f"{len(cases)} cases on {a.workers} workers ({'full' if a.full else 'demo'} bed)")
    with Pool(a.workers) as pool:
        res = pool.map(one_case, cases)

    hdr = "en,mu,seed,N,eps_mf,umf,umf_ergun,wall_s\n"
    new = not os.path.exists("out/sweep.csv")
    with open("out/sweep.csv", "a") as f:
        if new:
            f.write(hdr)
        for r in res:
            f.write(f"{r['en']},{r['mu']},{r['seed']},{r['N']},{r['eps_mf']:.4f},"
                    f"{r['umf']:.4f},{r['umf_ergun']:.4f},{r['wall']:.0f}\n")
    for r in res:
        print(f"  en={r['en']:.2f} mu={r['mu']:.2f} seed={r['seed']}  eps={r['eps_mf']:.3f}  "
              f"umf={r['umf']:.3f}  Ergun={r['umf_ergun']:.3f}  ({r['wall']/60:.1f} min)")

    if a.seeds == 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for mu in mus:
            pts = [(r["en"], r["umf"]) for r in res if r["mu"] == mu]
            pts.sort()
            ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", label=rf"$\mu$ = {mu}")
        ax.set_xlabel(r"$e_n$"); ax.set_ylabel(r"$u_{mf}$ (m/s)")
        ax.grid(alpha=0.3); ax.legend()
        ax.set_title("sensitivity of $u_{mf}$ to contact parameters", fontsize=10)
        fig.tight_layout(); fig.savefig("out/sweep.png", dpi=150)
        print("saved out/sweep.png")
    print("saved out/sweep.csv")


if __name__ == "__main__":
    main()
