"""
bed.py — initial configurations and state save/load.
"""
import numpy as np
from params import Params


def build_bed(p: Params):
    """
    Regular lattice, nrow × ncol, pitch lattice_gap·d_p, small random jitter.
    Returns state = [x, y, vx, vy] (float64 arrays, modified in place by the kernels).
    """
    rng = np.random.default_rng(p.seed)
    gap = p.lattice_gap * p.dp
    ncol = p.ncol
    x0 = (p.W - (ncol - 1) * gap) / 2
    jit = p.jitter * p.dp
    xs = np.empty(p.nrow * ncol)
    ys = np.empty(p.nrow * ncol)
    k = 0
    for r in range(p.nrow):
        for c in range(ncol):
            xs[k] = x0 + c * gap + rng.uniform(-jit, jit)
            ys[k] = p.R + 0.002 + r * gap + rng.uniform(-jit, jit)
            k += 1
    return [xs, ys, np.zeros(k), np.zeros(k)]


def lattice_layer(p: Params, npc: int, ncells_y: int = 4):
    """
    Frozen lattice aligned to the fluid grid, npc particles per cell side, so every
    occupied cell holds exactly npc² particles and the voidage is uniform:
        eps = 1 − npc² V_p / (Δx Δy d_p).
    Occupies ncells_y full rows of cells starting at a cell boundary (row 1).
    Returns (state, eps_exact, layer_height).
    """
    sx = p.dx / npc
    sy = p.dy / npc
    assert sx >= p.dp and sy >= p.dp, "npc too large: particles would overlap"
    xs, ys = [], []
    for jc in range(1, 1 + ncells_y):
        for ic in range(p.NCX):
            for a in range(npc):
                for b in range(npc):
                    xs.append((ic + (a + 0.5) / npc) * p.dx)
                    ys.append((jc + (b + 0.5) / npc) * p.dy)
    eps = 1.0 - npc * npc * p.Vp / (p.dx * p.dy * p.dp)
    return [np.array(xs), np.array(ys), np.zeros(len(xs)), np.zeros(len(xs))], eps, ncells_y * p.dy


def bed_voidage(state, p: Params):
    """Mean monolayer voidage of the settled bed, 1 − N V_p / (W h d_p)."""
    x, y, _, _ = state
    h = y.max()
    return 1.0 - x.size * p.Vp / (p.W * h * p.dp), h


def save_state(path, state, p: Params, **extra):
    x, y, vx, vy = state
    np.savez_compressed(path, x=x, y=y, vx=vx, vy=vy,
                        W=p.W, H=p.H, nrow=p.nrow, en=p.en, mu_f=p.mu_f, kn=p.kn, seed=p.seed,
                        **extra)


def load_state(path):
    d = np.load(path)
    return [d["x"].copy(), d["y"].copy(), d["vx"].copy(), d["vy"].copy()], d
