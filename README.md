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

# 3c. same, with 1-D volume-flux (two-way) coupling; output defaults to <tag>_2way
python run_ramp.py --nfluid 5 --two_way --out_tag full_5to1_2way
python analysis.py --tag full_5to1_2way

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

**Two-way option (1-D volume-flux coupling)** — `--two_way` on `run_ramp.py`
(`Params.two_way`, default 0 = the one-way model above; all results above the two-way
section use the one-way model). The gas phase satisfies 1-D incompressible volume
continuity, ε_g u_g + ε_s v_s = U_g, with the solids flux averaged over each row of fluid
cells and refreshed together with the voidage field. The gas momentum equation is not
solved: the particle reaction S_p acts on the pressure profile, which is recovered as
ΔP = ΣF_d/(W d_p). Lateral gas redistribution around dense regions is not modelled. In code,
for each row of fluid cells, q_s = Σ V_p v_y / (W Δy d_p) and u_g = (U_g − q_s)/ε_g
replaces U_g/ε_g (`drag.solids_flux_rows`, `drag.apply_drag_2way`); u_g is allowed to go
negative locally and is not clamped, and no separate −V_p ∇p force is added (the Ergun
branch of β is already the total-force form). With all particles at rest the model reduces
exactly to the one-way one (Checkpoint 4a).

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

Six coupling variants, **10 runs each** (seeds 7, 11, 12, 13, 14, 15, 16, 17, 18, 19). For each seed one
settled bed (ε_mf = 0.432–0.452) is shared by all six variants, so the variants are compared on
identical beds.

- **1 : 1 / 5 : 1 / 10 : 1** — voidage refreshed every 1 / 5 / 10 DEM steps (`--nfluid 1|5|10`),
  nearest-cell binning. 5 : 1 mimics the paper's fluid : DEM ratio; 10 : 1 is a further test.
- **+ interp.** — as above, with cloud-in-cell voidage and bilinear voidage at each particle.
  **An extension beyond the proposal; its code is not part of this repository** (only its result
  files are), so these three variants cannot be reproduced from here.

| | 1 : 1 | 5 : 1 | 10 : 1 | 1 : 1 + interp. | 5 : 1 + interp. | 10 : 1 + interp. |
|---|---|---|---|---|---|---|
| u_mf, mean ± SD (descending fit) | **0.993 ± 0.011 m/s** | **0.991 ± 0.008 m/s** | **0.997 ± 0.013 m/s** | **1.037 ± 0.016 m/s** | **1.040 ± 0.006 m/s** | **1.039 ± 0.009 m/s** |
| standard error of the mean | 0.003 | 0.003 | 0.004 | 0.005 | 0.002 | 0.003 |
| 95 % CI of the mean (m/s) | 0.985–1.001 | 0.985–0.996 | 0.988–1.006 | 1.025–1.048 | 1.036–1.045 | 1.033–1.046 |
| vs Ergun at each bed's own ε_mf | -10.7 % | -10.9 % | -10.3 % | -6.8 % | -6.5 % | -6.5 % |
| vs paper simulation (1.00 m/s) | -0.7 % | -0.9 % | -0.3 % | +3.7 % | +4.0 % | +3.9 % |
| vs NETL experiment (1.05 m/s) | -5.4 % | -5.7 % | -5.0 % | -1.3 % | -0.9 % | -1.0 % |
| plateau ΔP vs N m g/(W d_p), mean | 98.7 % | 98.0 % | 99.0 % | 98.4 % | 98.5 % | 98.4 % |

u_mf per seed (m/s):

| seed | ε_mf | Ergun u_mf | 1 : 1 | 5 : 1 | 10 : 1 | 1 : 1 + interp. | 5 : 1 + interp. | 10 : 1 + interp. |
|---|---|---|---|---|---|---|---|---|
| 7 | 0.441 | 1.104 | 0.9885 | 1.0051 | 0.9895 | 1.0386 | 1.0445 | 1.0358 |
| 11 | 0.445 | 1.121 | 1.0155 | 0.9921 | 1.0150 | 1.0087 | 1.0292 | 1.0265 |
| 12 | 0.432 | 1.066 | 0.9847 | 0.9764 | 1.0044 | 1.0364 | 1.0378 | 1.0514 |
| 13 | 0.443 | 1.111 | 0.9927 | 0.9988 | 0.9987 | 1.0489 | 1.0399 | 1.0474 |
| 14 | 0.438 | 1.091 | 0.9933 | 0.9865 | 0.9864 | 1.0472 | 1.0356 | 1.0372 |
| 15 | 0.449 | 1.138 | 0.9922 | 0.9934 | 1.0206 | 1.0346 | 1.0431 | 1.0297 |
| 16 | 0.440 | 1.099 | 1.0009 | 0.9864 | 0.9833 | 1.0556 | 1.0382 | 1.0318 |
| 17 | 0.447 | 1.131 | 0.9965 | 0.9811 | 0.9979 | 1.0541 | 1.0479 | 1.0415 |
| 18 | 0.442 | 1.109 | 0.9727 | 0.9906 | 0.9849 | 1.0187 | 1.0378 | 1.0378 |
| 19 | 0.452 | 1.153 | 0.9924 | 0.9947 | 0.9917 | 1.0224 | 1.0487 | 1.0553 |

Paired comparisons (same bed per seed, paired t-test, n = 10; difference as a percentage of the
second case, with its 95 % confidence interval):

| comparison | mean difference | 95 % CI | p |
|---|---|---|---|
| 5 : 1 − 1 : 1 | -0.002 m/s (-0.2 %) | -1.2 % to +0.7 % | 0.59 |
| 10 : 1 − 1 : 1 | +0.004 m/s (+0.4 %) | -0.5 % to +1.4 % | 0.33 |
| 10 : 1 − 5 : 1 | +0.007 m/s (+0.7 %) | -0.4 % to +1.8 % | 0.20 |
| 5 : 1 − 1 : 1 (interp.) | +0.004 m/s (+0.4 %) | -0.7 % to +1.4 % | 0.45 |
| 10 : 1 − 1 : 1 (interp.) | +0.003 m/s (+0.3 %) | -0.9 % to +1.5 % | 0.61 |
| 10 : 1 − 5 : 1 (interp.) | -0.001 m/s (-0.1 %) | -0.7 % to +0.5 % | 0.76 |
| interp. − none, at 1 : 1 | +0.044 m/s (+4.4 %) | +3.0 % to +5.8 % | < 0.001 |
| interp. − none, at 5 : 1 | +0.050 m/s (+5.0 %) | +4.3 % to +5.7 % | < 0.001 |
| interp. − none, at 10 : 1 | +0.042 m/s (+4.2 %) | +3.0 % to +5.5 % | < 0.001 |

- **The fluid : DEM update ratio has no detectable effect** on u_mf over 1 : 1 – 10 : 1, with or
  without interpolation (every p ≥ 0.2). Any effect is bounded by about ±1.5 %; a repeated-measures
  test across the three ratios gives p = 0.91 without and 0.79 with interpolation. The 95 %
  intervals cannot exclude a difference of about 1 %, so this is "no effect larger than ≈ 1.5 %",
  not proof of zero effect.
- **Interpolation raises u_mf by about 4–5 %** at every ratio (p < 0.001, 95 % interval about
  +3 % to +6 %). That is far larger than the run-to-run scatter, so it is a real effect of the
  voidage discretisation. Its closer agreement with the experiment should not be read as the
  interpolated scheme being more correct: the 2-D monolayer and one-way coupling also shift u_mf.
- Run-to-run (bed-to-bed) SD of u_mf is 0.006–0.016 m/s (0.6–1.6 %).
- Files: seed 7 is `out/*_full.*`, `out/*_full_5to1.*`, `out/*_full_10to1.*` and the `_interp`
  variants (including `_1to1_interp`); other seeds add `_s<seed>`, e.g.
  `out/results_full_10to1_interp_s15.txt`.
- Wall-clock: a single full ramp took 3.3–3.5 h; with five or six ramps running in parallel on 3
  cores each took about 4.5–6.3 h.

### Two-way (1-D volume-flux) coupling vs one-way, 5 : 1, nearest-cell, same beds

Branch `feature/two-way-volume-flux`. Same 15 settled beds (seeds 7, 11–24) and protocol as above; the only
difference is `--two_way` (u_g = (U_g − q_s)/ε_g instead of U_g/ε_g). Files: `out/*_full_5to1_2way*`.

| seed | ε_mf | Ergun u_mf | one-way 5 : 1 | two-way 5 : 1 | difference | one-way plateau | two-way plateau |
|---|---|---|---|---|---|---|---|
| 7 | 0.441 | 1.104 | 1.0051 | 1.0021 | -0.0030 | 97.5 % | 98.3 % |
| 11 | 0.445 | 1.121 | 0.9921 | 0.9944 | +0.0023 | 98.5 % | 97.9 % |
| 12 | 0.432 | 1.066 | 0.9764 | 0.9886 | +0.0122 | 97.1 % | 97.3 % |
| 13 | 0.443 | 1.111 | 0.9988 | 0.9919 | -0.0069 | 99.0 % | 97.9 % |
| 14 | 0.438 | 1.091 | 0.9865 | 0.9765 | -0.0100 | 98.0 % | 97.6 % |
| 15 | 0.449 | 1.138 | 0.9934 | 0.9888 | -0.0046 | 98.0 % | 99.2 % |
| 16 | 0.440 | 1.099 | 0.9864 | 0.9803 | -0.0061 | 97.5 % | 98.9 % |
| 17 | 0.447 | 1.131 | 0.9811 | 1.0061 | +0.0250 | 97.9 % | 98.5 % |
| 18 | 0.442 | 1.109 | 0.9906 | 0.9859 | -0.0047 | 98.0 % | 98.3 % |
| 19 | 0.452 | 1.153 | 0.9947 | 0.9829 | -0.0118 | 98.7 % | 97.5 % |
| 20 | 0.445 | 1.121 | 0.9896 | 0.9792 | -0.0104 | 98.4 % | 96.9 % |
| 21 | 0.445 | 1.122 | 1.0058 | 0.9838 | -0.0220 | 98.7 % | 98.3 % |
| 22 | 0.442 | 1.109 | 0.9963 | 1.0039 | +0.0076 | 99.0 % | 99.3 % |
| 23 | 0.435 | 1.078 | 1.1039 ⚠ | 1.0128 | -0.0911 | 127.6 % ⚠ | 98.9 % |
| 24 | 0.445 | 1.119 | 0.9837 | 0.9891 | +0.0054 | 97.2 % | 97.1 % |

⚠ Seed 23, one-way 5 : 1: plateau ΔP is 1,155 Pa = **127.6 % of the bed weight** (peak 1,647 Pa), which **fails
Checkpoint 3** (plateau within 10 %). It is the only failure among all 105 full-bed runs, and the six other
variants on the same bed behave normally (plateaus 878–901 Pa). The cause was not diagnosed (a bed pinned
against the hard ceiling, limitation 4, is one candidate). Its u_mf (1.104 m/s) is therefore not trusted, and
results are given both with and without it.

| | one-way, all 15 | two-way, all 15 | one-way, 14 (no seed 23) | two-way, 14 (no seed 23) |
|---|---|---|---|---|
| u_mf, mean ± SD | **0.999 ± 0.030 m/s** | **0.991 ± 0.011 m/s** | **0.991 ± 0.008 m/s** | **0.990 ± 0.009 m/s** |
| vs Ergun at each bed's own ε_mf | -10.1 % | -10.8 % | -11.0 % | -11.1 % |
| vs paper simulation (1.00 m/s) | -0.1 % | -0.9 % | -0.9 % | -1.0 % |
| vs NETL experiment (1.05 m/s) | -4.9 % | -5.6 % | -5.6 % | -5.8 % |
| plateau ΔP vs N m g/(W d_p), mean | 100.1 % | 98.1 % | 98.1 % | 98.1 % |

Paired difference (two-way − one-way):

| | n | mean difference | 95 % CI | paired t, p | Wilcoxon, p | up / down |
|---|---|---|---|---|---|---|
| all 15 beds | 15 | -0.0079 m/s (-0.79 %) | -2.2 % to +0.6 % | 0.25 | 0.28 | 5 / 10 |
| excluding seed 23 | 14 | -0.0019 m/s (-0.19 %) | -0.9 % to +0.5 % | 0.55 | 0.46 | 5 / 9 |

- **No detectable effect of the 1-D volume-flux coupling on u_mf.** The mean shift is −0.2 % (14 valid pairs) or
  −0.8 % (all 15), both inside the 95 % interval around zero; the paired scatter is about 0.012 m/s (1.2 %), and
  the sign is mixed. Any effect is bounded by about ±1 % (excluding seed 23).
- Plateau ΔP is unchanged (98.1 % of bed weight in both, excluding seed 23).
- This is consistent with the expectation that in a batch bed upward and downward solids motion largely cancel
  in the row-averaged flux. The fixed-bed region is unchanged by construction (particles at rest, Checkpoint 4a).
- The flux calculation adds no measurable run time (−0.4 % demo bed, −0.7 % full bed; within noise).
- Scope: nearest-cell voidage only; the interpolated variants above were not run in two-way mode.


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
2. **One-way coupling (default) / 1-D volume-flux coupling (`--two_way`)** — gas cannot
   redistribute laterally around dense regions; bubbles are weaker and smaller than in
   resolved CFD-DEM. Context, for the emulsion at u_mf (ε_g = 0.441): particle response time
   τ_p = ρ_p ε_s/β ≈ 0.28 s; Stokes number St = τ_p/τ_f ≈ 1.9 at the bed-transit scale
   (H/U_g ≈ 0.15 s) and 0.6–1.4 at the bubble scale (2–5 Hz); solids volume fraction
   φ_v ≈ 0.56; mass loading ≈ 1,200; |S_p| ≈ 5,100 Pa/m against ≈ 19 Pa/m of gas convective
   inertia. The Stokes number does **not** justify one-way coupling (the bed is in the
   four-way-coupling regime of Elghobashi's map); the justification for the 1-D treatment
   rests on gas incompressibility (plateau ΔP ≈ 905 Pa is ≈ 0.9 % of atmospheric) and on
   the aim being the quasi-static ΔP–U_g curve and u_mf.
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
