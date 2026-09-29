"""
params.py — single source of truth for every physical and numerical parameter.

All values default to Liu & van Wachem (2019), Powder Technology 343, 145–158,
Tables 1–2 (NETL Small Scale Challenge Problem, Geldart D particles).

Usage
-----
    from params import Params
    p = Params()                       # defaults (full 0.23 m bed)
    p = Params(W=0.09, nrow=26)        # small demo bed
    p = Params(en=0.78, mu_f=0.45)     # sensitivity case
    args = p.kernel_args(N)            # tuple handed to the njit kernels

Everything derived (eta_n, t_c, dt, m, V_p, grid spacing ...) is recomputed
in __post_init__, so changing one input keeps the whole set consistent.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Params:
    # ---------------- material (Table 1) ----------------
    dp: float = 3.256e-3       # particle diameter, m
    rho_p: float = 1131.0      # particle density, kg/m^3
    rho_g: float = 1.21        # gas density, kg/m^3
    mu_g: float = 1.8e-5       # gas viscosity, Pa.s
    g: float = 9.81            # gravity, m/s^2

    # ---------------- contact (Table 1) ----------------
    kn: float = 3654.0         # normal stiffness, N/m
    kt_ratio: float = 2.0 / 7.0  # k_t / k_n  (Silbert; paper gives 1044/3654 = 2/7 exactly)
    en: float = 0.84           # normal restitution coefficient
    mu_f: float = 0.35         # Coulomb friction coefficient
    dt_factor: float = 50.0    # dt = t_c / dt_factor   (paper uses ~t_c/8; 50 is safer for symplectic Euler)

    # ---------------- domain ----------------
    W: float = 0.23            # bed width, m   (SSCP width)
    H: float = 0.55            # domain height, m
    cell_dp: float = 3.0       # fluid cell size in particle diameters (paper: 3 d_p)

    # ---------------- initial packing ----------------
    nrow: int = 52             # rows in the initial lattice (52 rows ≈ 3,380 particles at W = 0.23)
    lattice_gap: float = 1.06  # lattice pitch / d_p
    jitter: float = 0.01       # random jitter / d_p
    seed: int = 7

    # ---------------- drag ----------------
    eps_switch: float = 0.8    # Gidaspow branch switch
    eps_min: float = 0.36      # voidage clamp (random close packing)

    # ---------------- ramp protocol (paper §4.2) ----------------
    ramp_rate: float = 0.2     # m/s^2
    Ug_max: float = 2.2        # m/s, top of ramp (≈ 2 u_mf for the 2-D bed)
    sample_hz: float = 50.0    # ΔP sampling frequency (paper: 50 Hz)

    # ---------------- derived (filled in __post_init__) ----------------
    R: float = field(init=False)
    m: float = field(init=False)
    meff: float = field(init=False)
    Iner: float = field(init=False)
    Vp: float = field(init=False)
    ap: float = field(init=False)
    kt: float = field(init=False)
    zeta: float = field(init=False)
    etan: float = field(init=False)
    etat: float = field(init=False)
    omega0: float = field(init=False)
    tc: float = field(init=False)
    dt: float = field(init=False)
    NCX: int = field(init=False)
    NCY: int = field(init=False)
    dx: float = field(init=False)
    dy: float = field(init=False)

    def __post_init__(self):
        self.R = self.dp / 2
        self.Vp = np.pi / 6 * self.dp ** 3          # sphere volume (mass/voidage bookkeeping)
        self.ap = np.pi / 4 * self.dp ** 2          # disc area (2-D geometry)
        self.m = self.rho_p * self.Vp
        self.meff = self.m / 2                      # equal spheres
        self.Iner = 0.4 * self.m * self.R ** 2
        self.kt = self.kt_ratio * self.kn

        # damping from restitution: zeta = -ln e / sqrt(ln^2 e + pi^2)
        L = np.log(self.en)
        self.zeta = -L / np.sqrt(L * L + np.pi ** 2)
        self.etan = 2 * self.zeta * np.sqrt(self.kn * self.meff)
        self.etat = self.etan

        # contact time and timestep
        self.omega0 = np.sqrt(self.kn / self.meff)
        self.tc = np.pi / (self.omega0 * np.sqrt(1 - self.zeta ** 2))
        self.dt = self.tc / self.dt_factor

        # fluid grid (cells of ~cell_dp particle diameters)
        self.NCX = max(3, int(round(self.W / (self.cell_dp * self.dp))))
        self.NCY = max(3, int(round(self.H / (self.cell_dp * self.dp))))
        self.dx = self.W / self.NCX
        self.dy = self.H / self.NCY

    # ---------------- helpers ----------------
    @property
    def ncol(self) -> int:
        return int((self.W - 0.004) / (self.lattice_gap * self.dp))

    def kernel_args(self, N: int):
        """Tuple of scalars + workspace arrays handed to the njit kernels."""
        ncell = self.NCX * self.NCY
        head = np.full(ncell, -1, np.int64)
        nxt = np.full(N, -1, np.int64)
        return (self.dt, self.R, self.m, self.Iner, self.kn, self.kt,
                self.etan, self.etat, self.mu_f, self.g,
                self.W, self.H, self.NCX, self.NCY, self.dx, self.dy,
                self.dp, self.rho_g, self.mu_g, self.Vp, self.ap,
                self.eps_switch, self.eps_min,
                ncell, head, nxt, self.dx, self.dy)

    def dP_theory(self, N: int) -> float:
        """Bed weight per unit (monolayer) cross-section: N m g / (W d_p)."""
        return N * self.m * self.g / (self.W * self.dp)

    def summary(self) -> str:
        return (f"d_p={self.dp*1e3:.3f} mm  rho_p={self.rho_p:.0f}  kn={self.kn:.0f}  kt={self.kt:.0f}  "
                f"en={self.en}  mu={self.mu_f}\n"
                f"eta_n={self.etan:.4e} N.s/m  zeta={self.zeta:.4f}  t_c={self.tc:.3e} s  "
                f"dt={self.dt:.3e} s  (t_c/dt={self.tc/self.dt:.0f})\n"
                f"W={self.W} m  H={self.H} m  grid {self.NCX}x{self.NCY}  cell={self.dx*1e3:.1f} mm")


if __name__ == "__main__":
    print(Params().summary())
    print()
    print(Params(W=0.09, H=0.34, nrow=26).summary())
