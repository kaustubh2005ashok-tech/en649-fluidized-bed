# Presentation material — reproduction of Liu & van Wachem (2019)

2-D DEM fluidized bed (N = 3380, Numba), one-way vs quasi-two-way (1-D volume-flux) coupling.
Constant-U_g holds of 15 s (ramped over 2 s), statistics from t > 5 s.

## Contents
- `figures/` — Figs 1–14 and A1–A3 of the paper, regenerated (`fig05`, `fig06`, `figA1` have one file per coupling).
- `data/summary.csv`, `data/dP_*.csv` — mean/SD of bed pressure drop and the full ΔP time series of the six runs.
- `scripts/` — `run_hold.py` (hold run), `paper_figs.py`, `paper_flow.py` (figures).
- `Lethe_problems_and_comparison.pdf` — Lethe CFD-DEM problems and comparison with this code.

## Results (ΔP, t > 5 s; bed weight / area = 905 Pa; paper: 1013 / 996 / 914 Pa)
| u_g (m/s) | one-way (Pa) | quasi-two-way (Pa) |
|---|---|---|
| 2.19 (2 u_mf) | 899 ± 39 | 899 ± 28 |
| 3.28 (3 u_mf) | 5570 ± 62 | 5241 ± 84 |
| 4.38 (4 u_mf) | 11042 ± 94 | 11132 ± 155 |

## Limitations (read before using the figures)
- **Only 2 u_mf is a valid fluidized state.** At 3 and 4 u_mf the 1-D gas model has no bypass, so the whole bed lifts as a
  plug and is pressed against the ceiling (ΔP 6–12x the bed weight). The bubble fraction of ~0.81 in Fig. 7 is the gap under
  that plug, not bubbles; figures built on bubbles/velocities at 3 and 4 u_mf inherit this.
- No NETL experimental data: Figs 10–15 show simulation only; Fig. 15 (error vs experiment) is not made.
- The Gidaspow in `drag.py` differs from Table A1 as printed in its voidage convention (see Fig. 1, dashed curve).
- Single bed (seed 7) per condition; no error bars on the flow statistics.
- Known cosmetic issues: Fig. 5 panel overlap, label overlap in Fig. 3.
- Raw trajectories (`out/hold_full_*.npz`, ~35 MB each) are not uploaded; regenerate with `run_hold.py` (~3 h each).
