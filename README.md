# EN 649 — 2-D DEM Simulation of a Gas-Fluidized Bed

**Kaustubh Ashok · 24B1522 · IIT Bombay, Autumn 2026**

Reproduces the minimum fluidization velocity reported by Liu & van Wachem (2019),
*Powder Technology* 343, 145–158, with a soft-sphere DEM solver written from scratch
in Python + Numba, coupled to unresolved Gidaspow gas drag on a 2-D monolayer
reduction of the NETL SSCP bed.

---

## Layout

```
en649_fluidized_bed/
├── params.py        all constants; Params() dataclass, derived quantities, kernel_args()
├── contact.py       spring-dashpot-slider kernels (particle–particle, particle–wall)
├── drag.py          voidage field, Gidaspow β, per-particle drag, Ergun/Wen–Yu references
├── sim.py           cell-list integrator step(); settle(), ramp(), hold() drivers
├── bed.py           lattice initialisation, cell-aligned test lattice, state I/O
├── verify.py        Checkpoints 1a / 1b / 2  — run every session
├── run_settle.py    build + settle bed  → out/state_<tag>.npz
├── run_ramp.py      paper-protocol velocity ramp → out/ramp_<tag>.npz
├── analysis.py      u_mf extraction, Checkpoint 3, figures
├── run_sweep.py     (e_n, μ, seed) sensitivity sweep, multiprocessing
├── animate.py       three-panel GIF (illustration only)
├── requirements.txt
└── out/             all outputs (tracked, so the repo mirrors the working folder)
```

Every physical number lives in `params.py`. Nothing else hard-codes a value.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python verify.py                     # must print ALL PASSED before anything else
```

First run of any script JIT-compiles the kernels (~20 s). `cache=True` persists
the compiled code in `__pycache__/` so later sessions are fast.

---

## Workflow

```bash
# 1. verification (30 s) — repeat after ANY change to contact.py / drag.py / sim.py
python verify.py

# 2. small bed for development (624 particles, 0.09 m wide)
python run_settle.py --demo
python run_ramp.py   --tag demo --Ug_max 2.0 --rate 1.0     # ~3 min
python analysis.py   --tag demo

# 3. full bed, paper protocol (≈3,380 particles, 0.23 m, 0.2 m/s² ramp)  — run overnight
python run_settle.py
python run_ramp.py                                          # ~3.5 h, prints % complete + ETA
python analysis.py

# 3b. same bed with the paper's 5:1 fluid:DEM update ratio (voidage refreshed every 5 DEM steps)
python run_ramp.py --nfluid 5 --out_tag full_5to1           # ~3.3 h, saves out/ramp_full_5to1.npz
python analysis.py --tag full_5to1

# 4. sensitivity (9 cases, 6 workers)
python run_sweep.py                       # demo bed, fast ramp
python run_sweep.py --seeds 3             # run-to-run scatter
python run_sweep.py --full --rate 0.2     # paper-spec, long

# 5. animation
python animate.py --tag demo
```

---

## Verification checkpoints

| # | Test | Pass criterion | Catches |
|---|------|----------------|---------|
| 1a | binary head-on collision | measured e_n within 2 % of input | damping derivation, timestep |
| 1b | gravity settling, no gas | KE decays monotonically to < 1e-5 J | **sign errors** in any force term |
| 2a | β algebra | β(ε, U/ε)·(U/ε) ≡ Ergun ΔP/L(U, ε) | superficial/interstitial mix-ups |
| 2b | cell-aligned frozen lattice | Σ drag / (W d_p L) = Ergun ΔP/L | voidage bookkeeping, force assembly |
| 3 | fluidized plateau (analysis.py) | mean ΔP within 10 % of N m g / (W d_p) | drag summation, mass accounting |

1b caught three separate sign errors during development; 2a would have caught the
missing 1/ε_g factor that made the first u_mf 60 % too high. Do not skip them.

---

## Model summary

**Contact** — linear spring-dashpot-slider. η_n from e_n via
ζ = −ln e_n / √(ln² e_n + π²), η_n = 2ζ√(k_n m_eff). Coulomb-capped tangential
dashpot. Walls use the same law with infinite mass. **Rotation is not integrated**
(see Limitations).

**Timestep** — Δt = t_c / 50 with t_c = π / (ω₀√(1−ζ²)) → 3.3 µs. The paper uses
2×10⁻⁵ s ≈ t_c/8 with a Verlet integrator; symplectic Euler here needs the tighter step.

**Voidage** — monolayer slab convention: ε_g = 1 − ΣV_p / (Δx Δy d_p). Fluid cells 3 d_p.
Clamped to [0.36, 1].

**Drag** — Gidaspow: Ergun below ε_g = 0.8, Wen–Yu above. Written for interstitial slip
so that ΔP/L = β(u_g − u_s); the Ergun branch therefore carries ε² and ε¹ in the
denominators. Particle-phase term: F_d = β(u_g − v_p)V_p/ε_s (paper Eq. 3).

**Coupling scheme** — one-way, unresolved. U_g is prescribed at the distributor and
u_g = U_g/ε_g per cell. The voidage field (the "fluid update") is refreshed every
`nfluid` DEM steps (`Params.nfluid`, `--nfluid` on `run_ramp.py`); the drag on each
particle is applied every DEM step from the latest field and the particle's current
velocity. `nfluid = 1` (default) is the original **1 : 1** scheme; `nfluid = 5` mimics
the paper's **5 : 1** ratio. The gas momentum equation and its reaction term S_p
(paper Eq. 2) are **not** solved; the paper's 5 : 1 ratio comes from their coupled
Navier–Stokes solve, so here it only staggers the voidage update.

**u_mf extraction** — descending branch only (the ascending branch carries the
overshoot spike). Fixed-bed region fitted to a U + b U² (Ergun form — these particles
are at Re_p ≈ 200, inertial regime), plateau to a constant; u_mf is the intersection.

**Reference values** — `drag.ergun_umf(eps)` solves the full Ergun balance numerically.
Always compare against Ergun **at the measured settled voidage**, not a textbook 0.42:
u_mf ∝ ε^{3/2} in this regime, so a 2-D bed at ε ≈ 0.46–0.48 sits 10–20 % above a
3-D bed at ε ≈ 0.42 for the same particles.

---

## Demo-bed result (fast ramp, for pipeline validation only)

`run_ramp.py --tag demo --Ug_max 2.0 --rate 1.0`:

| | value |
|---|---|
| N, ε_mf | 624, 0.481 |
| plateau ΔP | 96.8 % of N m g/(W d_p) |
| u_mf (descending fit) | 1.07 m/s |
| Ergun at ε_mf | 1.28 m/s |
| paper simulation / experiment | 1.00 / 1.05 m/s |

The 1 m/s² ramp is 5× faster than the paper's and the descending branch is noisy; the
fit lands close to experiment partly by luck. The quantitative result is the full bed
at 0.2 m/s², and the honest comparison is against Ergun at the measured ε_mf.

---

## Full-bed results (paper protocol: N = 3,380, W = 0.23 m, 0.2 m/s² ramp to 2.2 m/s)

Same settled bed (ε_mf = 0.441) for both cases. Files: `out/*_full.*` and `out/*_full_5to1.*`.

| | 1 : 1 (`nfluid=1`) | 5 : 1 (`nfluid=5`) |
|---|---|---|
| plateau ΔP vs N m g/(W d_p) | 99.9 % | 97.5 % |
| u_mf (descending fit) | **0.989 m/s** | **1.005 m/s** |
| vs Ergun at ε_mf (1.104 m/s) | −10.5 % | −9.0 % |
| vs paper simulation (1.00 m/s) | −1.1 % | +0.5 % |
| vs NETL experiment (1.05 m/s) | −5.9 % | −4.3 % |
| wall-clock | 206 min | 200 min |

Each is a single run, so the 1.6 % difference between the two cases is not shown to be
significant; run `run_sweep.py --seeds 3` for run-to-run scatter.

Animations (`out/fluidized_bed_demo.gif`, `out/fluidized_bed_full.gif`, from
`animate.py`) are illustrations at a much faster ramp than the paper's.

---

## Known limitations (state these in the report)

1. **2-D monolayer** — no front/back walls, looser packing than 3-D; shifts u_mf up
   via ε^{3/2}.
2. **One-way coupling** — gas cannot redistribute around dense regions; bubbles are
   weaker and smaller than in resolved CFD-DEM.
3. **No rotation / no tangential history spring** — rotational ODE is stiff at this k_n
   (I ≈ 2×10⁻¹¹ kg m²) and diverges under a dashpot-only tangential law. Adding a
   history spring is the next physics upgrade.
4. **Hard ceiling** at y = H instead of a pressure outlet; keep U_g below elutriation.
5. **Gidaspow discontinuity** at ε_g = 0.8 (paper notes the same).

---

## Reference

D. Liu, B. van Wachem, "Comprehensive assessment of the accuracy of CFD-DEM
simulations of bubbling fluidized beds", Powder Technology 343 (2019) 145–158.
https://doi.org/10.1016/j.powtec.2018.11.025
