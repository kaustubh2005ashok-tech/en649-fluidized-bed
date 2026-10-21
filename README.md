# EN 649 — 2-D DEM Simulation of a Gas-Fluidized Bed

**Kaustubh Ashok · 24B1522 · IIT Bombay, Autumn 2026**

Reproduces the minimum fluidization velocity reported by Liu & van Wachem (2019),
*Powder Technology* 343, 145–158, with a soft-sphere DEM solver written from scratch
in Python + Numba, coupled to Gidaspow gas drag on a 2-D monolayer reduction of the NETL SSCP bed.

**The main model is the 1-D volume-flux (two-way) coupling** (`--two_way`): the gas velocity responds to
the solids flux through volume continuity, u_g = (U_g − q_s)/ε_g, with the gas momentum equation not
solved. The original one-way model (u_g = U_g/ε_g) is kept as the baseline and as the code's default
(`two_way = 0`), so every earlier result stays reproducible. Headline: u_mf = 0.990 ± 0.009 m/s over 14
beds, against 1.05 m/s in the experiment and 1.00 m/s in the paper's simulation (see Full-bed results).

---

## Layout

```
en649_fluidized_bed/
├── params.py        all constants; Params() dataclass, derived quantities, kernel_args()
├── contact.py       spring-dashpot-slider kernels (particle–particle, particle–wall)
├── drag.py          voidage field, Gidaspow β, per-particle drag (one-way and two-way), Ergun/Wen–Yu references
├── sim.py           cell-list integrator step(); settle(), ramp(), hold() drivers
├── bed.py           lattice initialisation, cell-aligned test lattice, state I/O
├── verify.py        Checkpoints 1a / 1b / 2 / 4  — run every session
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

# 3. full bed, paper protocol (≈3,380 particles, 0.23 m, 0.2 m/s² ramp) — MAIN CASE: two-way, 5:1
python run_settle.py
python run_ramp.py --nfluid 5 --two_way --out_tag full_5to1_2way   # ~3.5–5 h, prints % complete + ETA
python analysis.py --tag full_5to1_2way

# 3b. one-way baselines on the same settled bed
python run_ramp.py --nfluid 5 --out_tag full_5to1           # one-way, paper's 5:1 update ratio
python analysis.py --tag full_5to1
python run_ramp.py                                          # one-way, 1:1 (tag full)
python analysis.py

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
| 4 | 1-D volume-flux coupling (4a–4d) | rest ≡ one-way to 1e-12; flux bookkeeping and conservation to 1e-12; force with moving solids matches the hand value to 1e-9, including negative slip | wrong q_s accumulation, sign of u_rel under two-way |

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

**Coupling scheme — main model: 1-D volume-flux (two-way)** (`--two_way`, `Params.two_way = 1`).
The gas phase satisfies 1-D incompressible volume continuity, ε_g u_g + ε_s v_s = U_g, with the solids
flux averaged over each row of fluid cells and refreshed together with the voidage field. The gas
momentum equation is not solved: the particle reaction S_p acts on the pressure profile, which is
recovered as ΔP = ΣF_d/(W d_p). Lateral gas redistribution around dense regions is not modelled. In
code, for each row of fluid cells, q_s = Σ V_p v_y / (W Δy d_p) and u_g = (U_g − q_s)/ε_g replaces
U_g/ε_g (`drag.solids_flux_rows`, `drag.apply_drag_2way`); u_g is allowed to go negative locally and is
not clamped, and no separate −V_p ∇p force is added (the Ergun branch of β is already the total-force
form). With all particles at rest the model reduces exactly to the one-way one (Checkpoint 4a).

**Baseline: one-way** (the code's default, `two_way = 0`) — U_g is prescribed at the distributor and
u_g = U_g/ε_g per cell. In both models the voidage field (the "fluid update", and q_s when two-way) is
refreshed every `nfluid` DEM steps (`Params.nfluid`, `--nfluid` on `run_ramp.py`); the drag on each
particle is applied every DEM step from the latest field and the particle's current velocity.
`nfluid = 1` (default) is the original **1 : 1** scheme; `nfluid = 5` mimics the paper's **5 : 1** ratio.
The gas momentum equation and its reaction term S_p (paper Eq. 2) are **not** solved; the paper's 5 : 1
ratio comes from their coupled Navier–Stokes solve, so here it only staggers the voidage update.

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

**Main result — 1-D volume-flux (two-way) coupling at 5 : 1, 14 beds:** u_mf = **0.990 ± 0.009 m/s** (mean ± SD; 95 % CI of the mean 0.984–0.995), −5.8 % vs the NETL experiment (1.05 m/s), −1.0 % vs the paper's simulation (1.00 m/s), and −11.1 % vs Ergun at each bed's own ε_mf. Plateau ΔP is 98.1 % of the bed weight.

Seven coupling variants on **14 beds** (seeds 7, 11–22, 24; **bed 23 excluded — see below**). For each seed one
settled bed (ε_mf = 0.432–0.452) is shared by all seven variants, so every comparison is paired on identical beds.

- **two-way 5 : 1** — the main model: gas velocity responds to the solids flux, u_g = (U_g − q_s)/ε_g
  (`--nfluid 5 --two_way`), voidage and q_s refreshed every 5 DEM steps, nearest-cell binning.
- **1 : 1 / 5 : 1 / 10 : 1** — the one-way baseline, u_g = U_g/ε_g, voidage refreshed every 1 / 5 / 10 DEM
  steps (`--nfluid 1|5|10`), nearest-cell binning. 5 : 1 is the like-for-like baseline for the two-way run.
- **+ interp.** — one-way as above, with cloud-in-cell voidage and bilinear voidage at each particle.
  **An extension beyond the proposal; its code is not part of this repository** (only its result files are),
  so these three variants cannot be reproduced from here. They were not run in two-way mode.

| | **two-way 5 : 1** | 1 : 1 | 5 : 1 | 10 : 1 | 1 : 1 + interp. | 5 : 1 + interp. | 10 : 1 + interp. |
|---|---|---|---|---|---|---|---|
| u_mf, mean ± SD (descending fit) | **0.990 ± 0.009 m/s** | **0.994 ± 0.011 m/s** | **0.991 ± 0.008 m/s** | **0.996 ± 0.012 m/s** | **1.036 ± 0.014 m/s** | **1.037 ± 0.008 m/s** | **1.041 ± 0.011 m/s** |
| standard error of the mean | 0.002 | 0.003 | 0.002 | 0.003 | 0.004 | 0.002 | 0.003 |
| 95 % CI of the mean (m/s) | 0.984–0.995 | 0.988–1.000 | 0.987–0.996 | 0.989–1.003 | 1.028–1.045 | 1.032–1.042 | 1.034–1.047 |
| vs Ergun at each bed's own ε_mf | −11.1 % | −10.7 % | −11.0 % | −10.6 % | −6.9 % | −6.9 % | −6.5 % |
| vs paper simulation (1.00 m/s) | −1.0 % | −0.6 % | −0.9 % | −0.4 % | +3.6 % | +3.7 % | +4.1 % |
| vs NETL experiment (1.05 m/s) | −5.8 % | −5.3 % | −5.6 % | −5.1 % | −1.3 % | −1.2 % | −0.9 % |
| plateau ΔP vs N m g/(W d_p), mean | 98.1 % | 98.5 % | 98.1 % | 98.7 % | 98.4 % | 98.4 % | 98.5 % |

u_mf per bed (m/s):

| seed | ε_mf | Ergun u_mf | **two-way 5 : 1** | 1 : 1 | 5 : 1 | 10 : 1 | 1 : 1 + interp. | 5 : 1 + interp. | 10 : 1 + interp. |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 0.441 | 1.104 | 1.0021 | 0.9885 | 1.0051 | 0.9895 | 1.0386 | 1.0445 | 1.0358 |
| 11 | 0.445 | 1.121 | 0.9944 | 1.0155 | 0.9921 | 1.0150 | 1.0087 | 1.0292 | 1.0265 |
| 12 | 0.432 | 1.066 | 0.9886 | 0.9847 | 0.9764 | 1.0044 | 1.0364 | 1.0378 | 1.0514 |
| 13 | 0.443 | 1.111 | 0.9919 | 0.9927 | 0.9988 | 0.9987 | 1.0489 | 1.0399 | 1.0474 |
| 14 | 0.438 | 1.091 | 0.9765 | 0.9933 | 0.9865 | 0.9864 | 1.0472 | 1.0356 | 1.0372 |
| 15 | 0.449 | 1.138 | 0.9888 | 0.9922 | 0.9934 | 1.0206 | 1.0346 | 1.0431 | 1.0297 |
| 16 | 0.440 | 1.099 | 0.9803 | 1.0009 | 0.9864 | 0.9833 | 1.0556 | 1.0382 | 1.0318 |
| 17 | 0.447 | 1.131 | 1.0061 | 0.9965 | 0.9811 | 0.9979 | 1.0541 | 1.0479 | 1.0415 |
| 18 | 0.442 | 1.109 | 0.9859 | 0.9727 | 0.9906 | 0.9849 | 1.0187 | 1.0378 | 1.0378 |
| 19 | 0.452 | 1.153 | 0.9829 | 0.9924 | 0.9947 | 0.9917 | 1.0224 | 1.0487 | 1.0553 |
| 20 | 0.445 | 1.121 | 0.9792 | 0.9962 | 0.9896 | 0.9835 | 1.0310 | 1.0178 | 1.0644 |
| 21 | 0.445 | 1.122 | 0.9838 | 1.0067 | 1.0058 | 0.9969 | 1.0205 | 1.0274 | 1.0290 |
| 22 | 0.442 | 1.109 | 1.0039 | 0.9828 | 0.9963 | 0.9851 | 1.0441 | 1.0414 | 1.0462 |
| 24 | 0.445 | 1.119 | 0.9891 | 1.0015 | 0.9837 | 1.0060 | 1.0473 | 1.0323 | 1.0371 |

Paired comparisons (same bed per seed, paired t-test, n = 14; difference as a percentage of the second case,
with its 95 % confidence interval):

| comparison | mean difference | 95 % CI | p |
|---|---|---|---|
| **two-way 5 : 1 − one-way 5 : 1** (like for like) | −0.002 m/s (−0.2 %) | −0.9 % to +0.5 % | 0.55 |
| two-way 5 : 1 − one-way 1 : 1 | −0.005 m/s (−0.5 %) | −1.3 % to +0.4 % | 0.28 |
| two-way 5 : 1 − one-way 10 : 1 | −0.006 m/s (−0.6 %) | −1.4 % to +0.1 % | 0.10 |
| one-way 5 : 1 − 1 : 1 | −0.003 m/s (−0.3 %) | −1.0 % to +0.5 % | 0.47 |
| one-way 10 : 1 − 1 : 1 | +0.002 m/s (+0.2 %) | −0.5 % to +0.9 % | 0.56 |
| one-way 10 : 1 − 5 : 1 | +0.005 m/s (+0.5 %) | −0.4 % to +1.4 % | 0.29 |
| 5 : 1 − 1 : 1 (interp.) | +0.001 m/s (+0.1 %) | −0.7 % to +0.9 % | 0.80 |
| 10 : 1 − 1 : 1 (interp.) | +0.005 m/s (+0.4 %) | −0.5 % to +1.4 % | 0.35 |
| 10 : 1 − 5 : 1 (interp.) | +0.004 m/s (+0.3 %) | −0.5 % to +1.1 % | 0.37 |
| interp. − none, at 1 : 1 | +0.042 m/s (+4.3 %) | +3.1 % to +5.4 % | < 0.001 |
| interp. − none, at 5 : 1 | +0.046 m/s (+4.6 %) | +3.9 % to +5.3 % | < 0.001 |
| interp. − none, at 10 : 1 | +0.045 m/s (+4.5 %) | +3.4 % to +5.6 % | < 0.001 |

- **The 1-D volume-flux coupling has no detectable effect on u_mf.** Against the like-for-like one-way 5 : 1 run the
  mean shift is −0.19 % (95 % CI −0.9 % to +0.5 %, p = 0.55). The two-way mean is
  slightly lower than all three one-way ratios (by 0.2–0.65 %), but no difference is significant (smallest p =
  0.098, against one-way 10 : 1; the comparisons are not corrected for multiplicity) and the scatter between
  beds (≈ 0.012 m/s) exceeds the shift. Any effect is bounded by about ±1.4 %. This is the expected behaviour
  for a batch bed, where upward and downward solids motion largely cancel in the row-averaged flux; the
  fixed-bed region is identical by construction (particles at rest, Checkpoint 4a).
- **The fluid : DEM update ratio has no detectable effect** over 1 : 1 – 10 : 1 in the one-way model, with or
  without interpolation (all p ≥ 0.29; repeated-measures test p = 0.93 without, 0.85 with interpolation).
- **Interpolation raises u_mf by about 4–5 %** at every ratio (p < 0.001, 95 % interval about +3 % to +6 %),
  far above the bed-to-bed scatter, so it is a real effect of the voidage discretisation. The closer agreement
  with the experiment should not be read as the interpolated scheme being more correct: the 2-D monolayer and
  the 1-D gas treatment also shift u_mf.
- The two-way and one-way (non-interpolated) variants sit about 5–6 % below the experiment (two-way: −5.8 %) and about 11 % below Ergun at the measured ε_mf; the interpolated
  variants sit about 1 % below the experiment and 7 % below Ergun. (A 2-D bed packs looser than 3-D and
  u_mf ∝ ε^{3/2}; see Known limitations.)
- Bed-to-bed SD of u_mf is 0.008–0.014 m/s (0.8–1.4 %). Wall-clock: a full ramp took 3.3–3.5 h alone and about
  4.5–6.3 h each with five or six running in parallel on 3 cores (the 15 two-way runs took 14.3 h in total).
- Files: seed 7 is `out/*_full_5to1_2way.*` (two-way) and `out/*_full.*`, `out/*_full_5to1.*`, `out/*_full_10to1.*` and the
  `_interp` variants (one-way); other seeds add `_s<seed>`, e.g. `out/results_full_5to1_2way_s11.txt`.

### Excluded bed: seed 23

Bed 23 is excluded from **every** result above, for all variants, because the **one-way 5 : 1 run on it fails
Checkpoint 3**: its fluidized plateau is 1,155 Pa = **127.6 % of the bed weight** (peak ΔP 1,647 Pa) against a ±10 %
criterion, which produces a spurious u_mf of 1.104 m/s (about 10 % above the other beds, 0.976–1.006 m/s). It is the
only failure among all 105 full-bed runs (15 beds × 7 variants); all 98 runs on the other 14 beds pass.

- **Why the whole bed, not just that run:** so that every comparison stays paired on the same beds. The other six
  variants on bed 23 pass Checkpoint 3 (plateaus 878–901 Pa, two-way 895 Pa), so the bed itself is *not shown to be
  defective*; the exclusion is driven by the failing run and by keeping the design paired.
- **Cause not diagnosed.** A bed pinned against the hard ceiling (limitation 4) is one candidate; it was not checked.
- **Nothing was deleted:** all bed-23 result files remain in `out/` (`*_s23*`) and are in the repository history.
- **Sensitivity:** including bed 23 (n = 15) changes two-way − one-way 5 : 1 from −0.19 % to −0.79 % (95 % CI −2.2 % to +0.6 %, p = 0.25), still no detectable difference; the one-way 5 : 1 mean would be 0.999 ± 0.030 m/s and the two-way 0.991 ± 0.011 m/s.
- Earlier versions of this README reported 10 beds (seeds 7, 11–19); the table above uses 14 beds.


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
| Gas phase | one-dimensional, unresolved, prescribed U_g | main model adds 1-D volume-flux response, u_g = (U_g − q_s)/ε_g; the proposed one-way U_g/ε_g is kept as the baseline | gas velocity responds to the solids flux; still no gas momentum solve |
| Cross-check | LIGGGHTS "if time permits" | not done | — |

Not in the proposal but implemented: the `--two_way` volume-flux option, an `--nfluid` option, a live percentage/ETA display
in `run_ramp.py`, and a demo-bed pipeline for development.

---

## Known limitations (state these in the report)

1. **2-D monolayer** — no front/back walls, looser packing than 3-D; shifts u_mf up
   via ε^{3/2}.
2. **1-D gas treatment (two-way volume flux, main; one-way baseline)** — gas cannot
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
