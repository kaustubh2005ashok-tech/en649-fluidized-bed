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

Three coupling variants, **5 runs each** (seeds 7, 11, 12, 13, 14). For each seed one settled
bed (ε_mf = 0.432–0.445) is shared by all three variants, so the variants are compared on
identical beds.

- **1 : 1** — voidage and drag refreshed every DEM step (`nfluid=1`), nearest-cell binning.
- **5 : 1** — voidage refreshed every 5 DEM steps (`nfluid=5`), nearest-cell binning.
- **5 : 1 + interpolation** — as 5 : 1, with cloud-in-cell voidage and bilinear voidage at
  each particle. **An extension beyond the proposal; its code is not part of this repository**
  (only its result files are), so these runs cannot be reproduced from here.

| | 1 : 1 | 5 : 1 | 5 : 1 + interpolation |
|---|---|---|---|
| u_mf, mean ± SD (descending fit) | **0.995 ± 0.012 m/s** | **0.992 ± 0.011 m/s** | **1.037 ± 0.006 m/s** |
| standard error of the mean | 0.005 | 0.005 | 0.003 |
| vs Ergun at each bed's own ε_mf | −9.5 % | −9.7 % | −5.6 % |
| vs paper simulation (1.00 m/s) | −0.5 % | −0.8 % | +3.7 % |
| vs NETL experiment (1.05 m/s) | −5.2 % | −5.5 % | −1.2 % |
| plateau ΔP vs N m g/(W d_p), mean | 99.1 % | 98.0 % | 98.1 % |

u_mf per seed (m/s):

| seed | ε_mf | Ergun u_mf | 1 : 1 | 5 : 1 | 5 : 1 + interp. |
|---|---|---|---|---|---|
| 7 | 0.441 | 1.104 | 0.9885 | 1.0051 | 1.0445 |
| 11 | 0.445 | 1.121 | 1.0155 | 0.9921 | 1.0292 |
| 12 | 0.432 | 1.066 | 0.9847 | 0.9764 | 1.0378 |
| 13 | 0.443 | 1.112 | 0.9927 | 0.9988 | 1.0399 |
| 14 | 0.438 | 1.092 | 0.9933 | 0.9865 | 1.0356 |

Paired comparisons (same bed per seed, paired t-test, n = 5):

| comparison | mean difference | p |
|---|---|---|
| 5 : 1 − 1 : 1 | −0.003 m/s (−0.3 %) | 0.67 |
| interpolation − 5 : 1 | +0.046 m/s (+4.6 %) | 0.001 |
| interpolation − 1 : 1 | +0.043 m/s (+4.3 %) | 0.005 |

- The 1 : 1 and 5 : 1 results are **not distinguishable**: the difference changes sign from
  seed to seed and is well inside the ≈ 0.012 m/s run-to-run scatter.
- Interpolation raises u_mf by about 4–5 %, in all five seeds. That is larger than the
  scatter, so it is a real effect of the voidage discretisation. Its closer agreement with
  the experiment should not be read as the interpolated scheme being more correct: the 2-D
  monolayer and one-way coupling also shift u_mf.
- Files: seed 7 is `out/*_full.*`, `out/*_full_5to1.*`, `out/*_full_5to1_interp.*`; other
  seeds add `_s<seed>` (for example `out/results_full_5to1_interp_s11.txt`).
- Wall-clock: a single full ramp took 3.3–3.5 h; with three ramps running in parallel on 3
  cores each took about 3.8–4.6 h.

Animations (`out/fluidized_bed_demo.gif`, `out/fluidized_bed_full.gif`, from
`animate.py`) are illustrations at a much faster ramp than the paper's.

---

## Deviations from the project proposal

The revised proposal (EN649_Proposal_Revised.pdf) specifies the method below. Differences
in the code, and why:

| Item | Proposal | This code | Reason |
|---|---|---|---|
| DEM timestep | 2×10⁻⁵ s (≈ t_c/8, as in the paper) | 3.3 µs (t_c/50) | symplectic Euler needs the tighter step |
| Domain height | 0.40 m | 0.55 m | `params.py` default `H`; the ceiling is a hard wall, not a pressure outlet |
| Particle count | ≈ 3,700 | 3,380 (52 rows × 65 cols) | `params.py` default `nrow = 52` |
| Static bed height | ≈ 0.167 m | 0.146 m | settled 2-D monolayer packing, ε_mf = 0.441 |
| Ergun reference | 1.02 m/s at ε_mf = 0.42 | 1.104 m/s at the measured ε_mf = 0.441 | u_mf ∝ ε^{3/2}; the 2-D bed packs looser |
| Fluid time step | 10⁻⁴ s in the paper (Appendix A.2) | 1 : 1 with the DEM step, or 5 : 1 via `--nfluid 5` (16.6 µs) | no gas momentum solve here; 5 : 1 only staggers the voidage update |
| Gas phase | one-dimensional, unresolved, prescribed U_g | as proposed | — |
| Cross-check | LIGGGHTS "if time permits" | not done | — |

Not in the proposal but implemented: an `--nfluid` option, a live percentage/ETA display
in `run_ramp.py`, and a demo-bed pipeline for development.

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
