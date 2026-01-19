"""
spectrum_monitor.py

Lightweight Spectrum Monitoring Module (snapshot-first)
-------------------------------------------------------

Goal
- Offline "training": learn a reusable KL basis Φ on a fixed frequency grid and store a ModelPack.
- Online inference: given IQ snapshots, compute Welch PSD on that grid, infer a smoothed PSD approximation,
  and extract emitters (connected occupied components) with bandwidth + peak potency.

Key contracts
- WelchConfig stores fs_hz explicitly (NOT dt). Snapshot length may vary, fs must NOT.
- ModelPack defines the frequency grid f_hz and Welch parameters used to build it.
- Online inference does not fit kernel params; it uses the stored Φ and kernel params.

Uncertainty
- SnapshotResult can optionally include μξ and Σξ (R-space posterior). This allows cheap Monte Carlo
  to produce CIs for emitter features later.

Thresholding
- Uses empirical quantile threshold over "quiet" PSD bins to improve robustness vs real Welch statistics.

Dependencies: numpy, scipy
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.linalg as npl
from scipy.optimize import minimize
from scipy.signal import get_window, welch
from scipy.stats import norm

Array = np.ndarray


# -------------------------
# Helpers
# -------------------------

def symmetrize(M: Array) -> Array:
    return 0.5 * (M + M.T)


def mad(x: Array) -> float:
    x = np.asarray(x).reshape(-1)
    m = np.median(x)
    return float(np.median(np.abs(x - m)))


def Qinv(p: float) -> float:
    """Inverse Q-function: Q^{-1}(p) = isf(p) = Φ^{-1}(1-p)."""
    return float(norm.isf(p))


def matern32_from_r(r: Array, ell: float) -> Array:
    a = np.sqrt(3.0) * r / max(float(ell), 1e-30)
    return (1.0 + a) * np.exp(-a)


def composite_matern32_kernel_1d(x: Array, ell_short: float, ell_long: float, a1: float, a2: float) -> Array:
    """
    Composite kernel:
      k = a1^2 * Mat32(ell_short) + a2^2 * Mat32(ell_long)
    """
    x = np.asarray(x).reshape(-1)
    r = np.abs(x[:, None] - x[None, :])
    return (a1**2) * matern32_from_r(r, ell_short) + (a2**2) * matern32_from_r(r, ell_long)


def build_phi_kl_from_kernel(K: Array, w: Array, R: int) -> Tuple[Array, Array]:
    """
    Weighted KL recipe:
      Kw_ij = K_ij * sqrt(w_i w_j)
      eig(Kw) -> (lam, V), descending
      phi = V / sqrt(w)
      Phi_KL = phi * sqrt(lam)
    Returns:
      Phi_KL (M,R), lam (R,)
    """
    w = np.asarray(w).reshape(-1)
    sqrtw = np.sqrt(np.maximum(w, 1e-30))
    Kw = symmetrize(K * (sqrtw[:, None] * sqrtw[None, :]))

    evals, evecs = npl.eigh(Kw)  # ascending
    idx = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx], 0.0)
    evecs = evecs[:, idx]

    R = int(max(1, min(R, len(w))))
    lam = evals[:R]
    V = evecs[:, :R]

    phi = V / sqrtw[:, None]
    Phi_KL = phi * np.sqrt(lam)[None, :]
    return Phi_KL, lam


def log_marginal_likelihood_y(y: Array, A: Array, sigma_eps: float) -> float:
    """
    y = A ξ + ε, ξ~N(0,I), ε~N(0, σ^2 I)
    Woodbury/det lemma with B = I + A^T A / σ^2.
    """
    y = np.asarray(y).reshape(-1)
    N, R = A.shape
    sigma2 = float(sigma_eps**2)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return -np.inf

    B = symmetrize(np.eye(R) + (A.T @ A) / sigma2)
    try:
        L = npl.cholesky(B)
    except npl.LinAlgError:
        return -np.inf

    Aty = A.T @ y
    v = npl.solve(L, Aty)
    BinvAty = npl.solve(L.T, v)

    yTy = float(y @ y)
    corr = float(Aty @ BinvAty)
    quad = (1.0 / sigma2) * yTy - (1.0 / (sigma2**2)) * corr

    logdetB = 2.0 * float(np.sum(np.log(np.diag(L) + 1e-30)))
    logdetC = N * np.log(sigma2) + logdetB

    return float(-0.5 * (quad + logdetC + N * np.log(2.0 * np.pi)))


def posterior_xi_cholesky(y: Array, A: Array, sigma_eps: float) -> Tuple[Array, Array]:
    """
    Posterior over ξ using Cholesky (no explicit inverse):
      Σ = (I + A^T A / σ^2)^-1
      μ = Σ (A^T y / σ^2)
    """
    y = np.asarray(y).reshape(-1)
    R = A.shape[1]
    sigma2 = float(sigma_eps**2)
    if sigma2 <= 0:
        raise ValueError("sigma_eps must be > 0")

    S = symmetrize(np.eye(R) + (A.T @ A) / sigma2)
    L = npl.cholesky(S)  # S = L L^T

    b = (A.T @ y) / sigma2
    v = npl.solve(L, b)
    mu = npl.solve(L.T, v)

    # Sigma = S^{-1}
    I = np.eye(R)
    V = npl.solve(L, I)
    Sigma = V.T @ V
    return mu, Sigma


def diag_Phi_Sigma_PhiT(Phi: Array, Sigma: Array) -> Array:
    PS = Phi @ Sigma
    return np.sum(PS * Phi, axis=1)


def connected_components_1d(mask: Array) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    comps: List[Tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        comps.append((i, j))
        i = j + 1
    return comps


def postprocess_mask_1d(mask: Array, *, max_gap_bins: int = 2, min_width_bins: int = 3) -> Array:
    """
    Stabilize occupancy mask:
      1) Fill gaps of length <= max_gap_bins inside occupied regions
      2) Remove components with width < min_width_bins
    """
    m = np.asarray(mask, dtype=bool).copy().reshape(-1)
    n = len(m)
    if n == 0:
        return m

    # 1) fill small gaps
    if max_gap_bins > 0:
        i = 0
        while i < n:
            if m[i]:
                i += 1
                continue
            j = i
            while j < n and (not m[j]):
                j += 1
            gap_len = j - i
            left_on = (i - 1 >= 0 and m[i - 1])
            right_on = (j < n and m[j])
            if left_on and right_on and gap_len <= max_gap_bins:
                m[i:j] = True
            i = j

    # 2) remove tiny components
    comps = connected_components_1d(m)
    for a, b in comps:
        if (b - a + 1) < int(min_width_bins):
            m[a:b + 1] = False

    return m


# -------------------------
# Data containers
# -------------------------

@dataclass
class WelchConfig:
    fc_hz: float
    fs_hz: float
    nperseg: int
    noverlap: int
    window: str
    detrend: bool
    average: str


@dataclass
class KernelParamsComposite:
    ell_short_hz: float
    ell_long_hz: float
    a1: float
    a2: float
    sigma_eps: float  # likelihood noise scale used for inference


@dataclass
class ModelPack:
    """
    What you deliver to your partner.
    This fully defines the "format contract" + inference basis.
    """
    version: str
    welch: WelchConfig
    R: int
    f_hz: Array          # (M,)
    w: Array             # (M,)
    Phi_KL: Array        # (M,R)
    kernel_params: KernelParamsComposite

    # Online detection defaults
    p_fa_bin: float = 1e-3
    pi0: float = 0.95

    # Mask post-processing defaults (in bins)
    max_gap_bins: int = 2
    min_width_bins: int = 3

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            version=self.version,
            welch=np.array(list(asdict(self.welch).items()), dtype=object),
            R=np.int64(self.R),
            f_hz=self.f_hz,
            w=self.w,
            Phi_KL=self.Phi_KL,
            kernel_params=np.array(list(asdict(self.kernel_params).items()), dtype=object),
            p_fa_bin=np.float64(self.p_fa_bin),
            pi0=np.float64(self.pi0),
            max_gap_bins=np.int64(self.max_gap_bins),
            min_width_bins=np.int64(self.min_width_bins),
        )

    @staticmethod
    def load(path: Union[str, Path]) -> "ModelPack":
        path = Path(path)
        z = np.load(path, allow_pickle=True)

        welch_items = dict(z["welch"].tolist())
        kp_items = dict(z["kernel_params"].tolist())

        welch = WelchConfig(
            fc_hz=float(welch_items["fc_hz"]),
            fs_hz=float(welch_items["fs_hz"]),
            nperseg=int(welch_items["nperseg"]),
            noverlap=int(welch_items["noverlap"]),
            window=str(welch_items["window"]),
            detrend=bool(welch_items["detrend"]),
            average=str(welch_items["average"]),
        )
        kp = KernelParamsComposite(
            ell_short_hz=float(kp_items["ell_short_hz"]),
            ell_long_hz=float(kp_items["ell_long_hz"]),
            a1=float(kp_items["a1"]),
            a2=float(kp_items["a2"]),
            sigma_eps=float(kp_items["sigma_eps"]),
        )
        return ModelPack(
            version=str(z["version"]),
            welch=welch,
            R=int(z["R"]),
            f_hz=np.asarray(z["f_hz"]),
            w=np.asarray(z["w"]),
            Phi_KL=np.asarray(z["Phi_KL"]),
            kernel_params=kp,
            p_fa_bin=float(z["p_fa_bin"]),
            pi0=float(z["pi0"]),
            max_gap_bins=int(z.get("max_gap_bins", 2)),
            min_width_bins=int(z.get("min_width_bins", 3)),
        )
    def with_truncated_R(self, R_new: int) -> "ModelPack":
        R_new = int(R_new)
        if R_new < 1 or R_new > int(self.Phi_KL.shape[1]):
            raise ValueError(f"R_new must be in [1, {self.Phi_KL.shape[1]}].")
        return replace(self, R=R_new, Phi_KL=self.Phi_KL[:, :R_new].copy())

@dataclass
class SnapshotResult:
    f_hz: Array
    psd_welch: Array          # raw Welch PSD (linear)
    mu_psd: Array             # posterior mean PSD (linear)
    sigma_psd_diag: Array     # posterior marginal var diag
    snf_used: float
    tau_bin: float
    pi_bin: Array
    occ_mask: Array
    emitters: List[Dict[str, float]]

    # Optional: expose latent posterior so you can MC sample for emitter CIs
    mu_xi: Optional[Array] = None
    Sigma_xi: Optional[Array] = None


# -------------------------
# Offline training
# -------------------------

class SpectrumModelTrainer:
    """
    Offline training: fit kernel hyperparams + build KL basis Φ on a fixed grid.
    """

    def __init__(self, random_seed: int = 0):
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def welch_psd(x: Array, welch_cfg: WelchConfig) -> Tuple[Array, Array, float]:
        x = np.asarray(x).reshape(-1)
        if not np.iscomplexobj(x):
            raise ValueError("IQ must be complex.")
        if len(x) < int(welch_cfg.nperseg):
            raise ValueError("IQ snapshot shorter than nperseg.")
        x = x - np.mean(x)

        fs_hz = float(welch_cfg.fs_hz)
        win = get_window(welch_cfg.window, welch_cfg.nperseg, fftbins=True)

        f_bb, Pxx = welch(
            x,
            fs=fs_hz,
            window=win,
            nperseg=welch_cfg.nperseg,
            noverlap=welch_cfg.noverlap,
            nfft=welch_cfg.nperseg,
            detrend=welch_cfg.detrend,
            return_onesided=False,
            scaling="density",
            average=welch_cfg.average,
        )
        f_bb = np.fft.fftshift(f_bb)
        Pxx = np.fft.fftshift(Pxx)
        f_rf = float(welch_cfg.fc_hz) + f_bb
        return f_rf, Pxx, fs_hz

    @staticmethod
    def weights_uniform(f_hz: Array) -> Array:
        f_hz = np.asarray(f_hz).reshape(-1)
        df = float(np.mean(np.diff(f_hz)))
        return np.full_like(f_hz, df, dtype=float)

    @staticmethod
    def estimate_noise_floor(psd_mat: Array, quiet_frac: float = 0.25) -> Tuple[Array, Array]:
        """
        Returns per-snapshot:
          noise_floor[t] = median PSD over quiet bins
          sigma_eps_hint[t] = 1.4826 * MAD over quiet bins
        """
        Y = np.asarray(psd_mat)
        if Y.ndim == 1:
            Y = Y[None, :]
        T, _ = Y.shape

        med_over_time = np.median(Y, axis=0)
        thr = np.quantile(med_over_time, quiet_frac)
        quiet_mask = med_over_time <= thr

        noise_floor = np.zeros(T)
        sigma_eps_hint = np.zeros(T)
        for t in range(T):
            yq = Y[t, quiet_mask]
            noise_floor[t] = float(np.median(yq))
            sigma_eps_hint[t] = float(1.4826 * mad(yq))
        return noise_floor, sigma_eps_hint

    def fit_composite_kernel(
        self,
        f_hz: Array,
        w: Array,
        Y: Array,
        sigma_eps_hint: Array,
        R: int,
        fit_indices: Sequence[int] = (0, 25, 50, 75),
        maxiter: int = 60,
        n_restarts: int = 3,
        bounds_scale: Optional[Dict[str, float]] = None,
    ) -> KernelParamsComposite:
        f_hz = np.asarray(f_hz).reshape(-1)
        w = np.asarray(w).reshape(-1)
        B_total = float(np.sum(w))

        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y[None, :]
        T, _ = Y.shape

        fit_idx = np.array([i for i in fit_indices if 0 <= i < T], dtype=int)
        if fit_idx.size == 0:
            fit_idx = np.array([0], dtype=int)

        sig0 = float(np.median(np.asarray(sigma_eps_hint).reshape(-1)))
        if not np.isfinite(sig0) or sig0 <= 0:
            sig0 = float(np.std(Y))

        cfg = dict(
            ell_short_min=1e-4,  # fractions of B_total
            ell_short_max=0.2,
            delta_min=1e-4,
            delta_max=2.0,
            a_min=1e-3,
            a_max=1e3,
            sigma_min=0.1,       # multipliers of sig0
            sigma_max=10.0,
        )
        if bounds_scale:
            cfg.update(bounds_scale)

        ell_short_min = cfg["ell_short_min"] * B_total
        ell_short_max = cfg["ell_short_max"] * B_total
        delta_min = cfg["delta_min"] * B_total
        delta_max = cfg["delta_max"] * B_total
        a_min, a_max = float(cfg["a_min"]), float(cfg["a_max"])
        sigma_min, sigma_max = float(cfg["sigma_min"] * sig0), float(cfg["sigma_max"] * sig0)

        bounds = [
            (np.log(ell_short_min), np.log(ell_short_max)),
            (np.log(delta_min), np.log(delta_max)),
            (np.log(a_min), np.log(a_max)),
            (np.log(a_min), np.log(a_max)),
            (np.log(sigma_min), np.log(sigma_max)),
        ]

        def unpack(u: Array) -> Tuple[float, float, float, float, float]:
            ell_s = float(np.exp(u[0]))
            ell_l = float(ell_s + np.exp(u[1]))
            a1 = float(np.exp(u[2]))
            a2 = float(np.exp(u[3]))
            sig = float(np.exp(u[4]))
            return ell_s, ell_l, a1, a2, sig

        def neg_sum_logml(u: Array) -> float:
            ell_s, ell_l, a1, a2, sig = unpack(u)
            K = composite_matern32_kernel_1d(f_hz, ell_s, ell_l, a1, a2)
            Phi, _ = build_phi_kl_from_kernel(K, w, R)
            ll_sum = 0.0
            for t in fit_idx:
                ll = log_marginal_likelihood_y(Y[t], Phi, sig)
                if not np.isfinite(ll):
                    return 1e30
                ll_sum += ll
            return -float(ll_sum)

        u_base = np.array([
            np.log(0.02 * B_total),
            np.log(0.15 * B_total),
            np.log(1.0),
            np.log(1.0),
            np.log(np.clip(sig0, sigma_min, sigma_max)),
        ], dtype=float)

        best = None
        for r in range(max(1, int(n_restarts))):
            if r == 0:
                u0 = u_base.copy()
            else:
                u0 = np.array([self.rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)
            res = minimize(
                neg_sum_logml,
                x0=u0,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=int(maxiter)),
            )
            if best is None or res.fun < best.fun:
                best = res

        assert best is not None
        ell_s, ell_l, a1, a2, sig = unpack(best.x)
        return KernelParamsComposite(ell_short_hz=ell_s, ell_long_hz=ell_l, a1=a1, a2=a2, sigma_eps=sig)

    def train_from_iq(
        self,
        xs: Sequence[Array],
        welch_cfg: WelchConfig,
        R: int = 20,
        quiet_frac: float = 0.25,
        fit_kernel: bool = True,
        fit_indices: Sequence[int] = (0, 25, 50, 75),
        n_restarts: int = 3,
        version: str = "modelpack_v2",
    ) -> ModelPack:
        f0 = None
        Y = []
        for x in xs:
            f_rf, Pxx, _ = self.welch_psd(x, welch_cfg)
            if f0 is None:
                f0 = np.asarray(f_rf)
            else:
                if not np.allclose(f0, f_rf, atol=0.0, rtol=1e-12):
                    raise ValueError("Frequency grid mismatch across training snapshots.")
            Y.append(Pxx)

        psd_mat = np.vstack(Y) if len(Y) > 1 else np.asarray(Y[0])[None, :]
        f = np.asarray(f0)
        w = self.weights_uniform(f)

        _, sigma_eps_hint = self.estimate_noise_floor(psd_mat, quiet_frac=quiet_frac)

        if fit_kernel:
            kp = self.fit_composite_kernel(
                f_hz=f, w=w, Y=psd_mat, sigma_eps_hint=sigma_eps_hint, R=R,
                fit_indices=fit_indices, n_restarts=n_restarts
            )
        else:
            B_total = float(np.sum(w))
            kp = KernelParamsComposite(
                ell_short_hz=0.02 * B_total,
                ell_long_hz=0.2 * B_total,
                a1=1.0,
                a2=1.0,
                sigma_eps=float(np.median(sigma_eps_hint)),
            )

        K = composite_matern32_kernel_1d(f, kp.ell_short_hz, kp.ell_long_hz, kp.a1, kp.a2)
        Phi_KL, _ = build_phi_kl_from_kernel(K, w, R)


        return ModelPack(
            version=version,
            welch=welch_cfg,
            R=R,
            f_hz=f,
            w=w,
            Phi_KL=Phi_KL,
            kernel_params=kp,
        )
    def build_modelpack_from_kernel(
        self,
        welch_cfg: WelchConfig,
        kernel_params: KernelParamsComposite,
        R: int,
    ):
        # 1. Build grid by running Welch once on dummy noise
        M = welch_cfg.nperseg
        df = welch_cfg.fs_hz / welch_cfg.nperseg
        f_bb = np.fft.fftshift(np.fft.fftfreq(M, d=1/welch_cfg.fs_hz))
        f_hz = welch_cfg.fc_hz + f_bb

        # 2. Quadrature weights
        w = np.full_like(f_hz, df)

        # 3. Kernel matrix
        K = composite_matern32_kernel_1d(
            f_hz,
            kernel_params.ell_short_hz,
            kernel_params.ell_long_hz,
            kernel_params.a1,
            kernel_params.a2,
        )

        # 4. KL basis
        Phi_KL, _ = build_phi_kl_from_kernel(K, w, R)

        return ModelPack(
            version="rebuild_from_tuning",
            welch=welch_cfg,
            R=R,
            f_hz=f_hz,
            w=w,
            Phi_KL=Phi_KL,
            kernel_params=kernel_params,
        )


# -------------------------
# Online inference (snapshot)
# -------------------------

class SpectrumMonitor:
    """
    Online monitor using a saved ModelPack (snapshot mode).
    """

    def __init__(self, model: ModelPack):
        self.model = model

    def _assert_grid_match(self, f_hz: Array) -> None:
        f_hz = np.asarray(f_hz).reshape(-1)
        f0 = self.model.f_hz.reshape(-1)
        if f_hz.shape != f0.shape or not np.allclose(f_hz, f0, atol=0.0, rtol=1e-12):
            raise ValueError("Frequency grid mismatch: IQ/Welch config does not match the trained ModelPack.")

    def welch_psd(self, x: Array) -> Tuple[Array, Array, float]:
        cfg = self.model.welch
        x = np.asarray(x).reshape(-1)
        if not np.iscomplexobj(x):
            raise ValueError("IQ must be complex.")
        if len(x) < int(cfg.nperseg):
            raise ValueError("IQ snapshot shorter than nperseg.")
        x = x - np.mean(x)

        fs_hz = float(cfg.fs_hz)
        win = get_window(cfg.window, cfg.nperseg, fftbins=True)

        f_bb, Pxx = welch(
            x,
            fs=fs_hz,
            window=win,
            nperseg=cfg.nperseg,
            noverlap=cfg.noverlap,
            nfft=cfg.nperseg,
            detrend=cfg.detrend,
            return_onesided=False,
            scaling="density",
            average=cfg.average,
        )
        f_bb = np.fft.fftshift(f_bb)
        Pxx = np.fft.fftshift(Pxx)
        f_rf = float(cfg.fc_hz) + f_bb
        return f_rf, Pxx, fs_hz

    @staticmethod
    def estimate_snf_quiet(psd: Array, quiet_frac: float = 0.25) -> float:
        psd = np.asarray(psd).reshape(-1)
        thr = np.quantile(psd, float(quiet_frac))
        quiet = psd <= thr
        if np.sum(quiet) == 0:
            return float(np.median(psd))
        return float(np.median(psd[quiet]))

    @staticmethod
    def tau_from_quiet_quantile(psd: Array, quiet_mask: Array, p_fa_bin: float) -> float:
        psd = np.asarray(psd).reshape(-1)
        qm = np.asarray(quiet_mask, dtype=bool).reshape(-1)
        if np.sum(qm) == 0:
            return float(np.quantile(psd, 1.0 - float(p_fa_bin)))
        return float(np.quantile(psd[qm], 1.0 - float(p_fa_bin)))

    @staticmethod
    def occupancy_prob(mu: Array, var_diag: Array, tau: float) -> Array:
        mu = np.asarray(mu).reshape(-1)
        sig = np.sqrt(np.maximum(np.asarray(var_diag).reshape(-1), 1e-30))
        z = (float(tau) - mu) / sig
        return 1.0 - norm.cdf(z)

    @staticmethod
    def occ_mask_from_pi(pi: Array, pi0: float) -> Array:
        return np.asarray(pi).reshape(-1) >= float(pi0)

    @staticmethod
    def emitters_from_mask(f_hz: Array, w: Array, mu_psd: Array, occ_mask: Array) -> List[Dict[str, float]]:
        f = np.asarray(f_hz).reshape(-1)
        w = np.asarray(w).reshape(-1)
        mu = np.asarray(mu_psd).reshape(-1)
        occ = np.asarray(occ_mask, dtype=bool).reshape(-1)

        comps = connected_components_1d(occ)
        out: List[Dict[str, float]] = []
        for i0, i1 in comps:
            idx = np.arange(i0, i1 + 1)
            fmin = float(np.min(f[idx]))
            fmax = float(np.max(f[idx]))
            bw = float(fmax - fmin)

            i_peak = int(idx[np.argmax(mu[idx])])
            f_peak = float(f[i_peak])
            peak_psd = float(mu[i_peak])

            weights = w[idx] * np.maximum(mu[idx], 0.0)
            denom = float(np.sum(weights)) if float(np.sum(weights)) > 0 else float(np.sum(w[idx]))
            f_centroid = float(np.sum(f[idx] * weights) / denom)

            out.append(dict(
                f_start_hz=fmin,
                f_stop_hz=fmax,
                bw_hz=bw,
                f_peak_hz=f_peak,
                peak_psd=peak_psd,
                f_centroid_hz=f_centroid,
                i_start=int(i0),
                i_stop=int(i1),
            ))
        return out

    def _static_posterior(self, y: Array, sigma_eps: Optional[float] = None) -> Tuple[Array, Array, Array, Array]:
        y = np.asarray(y).reshape(-1)
        Phi = self.model.Phi_KL
        sig = float(sigma_eps if sigma_eps is not None else self.model.kernel_params.sigma_eps)
        mu_xi, Sigma_xi = posterior_xi_cholesky(y, Phi, sig)
        mu_psd = Phi @ mu_xi
        var_psd_diag = diag_Phi_Sigma_PhiT(Phi, Sigma_xi)
        return mu_xi, Sigma_xi, mu_psd, var_psd_diag

    def infer_snapshot(
        self,
        x: Array,
        *,
        snf: Optional[float] = None,
        quiet_frac: float = 0.25,
        p_fa_bin: Optional[float] = None,
        pi0: Optional[float] = None,
        sigma_eps: Optional[float] = None,
        return_latent: bool = True,
    ) -> SnapshotResult:
        f_rf, psd, _ = self.welch_psd(x)
        self._assert_grid_match(f_rf)

        y = np.asarray(psd).reshape(-1)

        # Quiet mask and noise floor estimate
        thr = np.quantile(y, float(quiet_frac))
        quiet_mask = y <= thr
        snf_used = float(snf) if snf is not None else float(np.median(y[quiet_mask]) if np.any(quiet_mask) else np.median(y))

        # Threshold from empirical quiet-bin quantile
        pfa = float(p_fa_bin) if p_fa_bin is not None else float(self.model.p_fa_bin)
        tau = self.tau_from_quiet_quantile(y, quiet_mask, pfa)

        # Inference
        sig_eps = float(sigma_eps) if sigma_eps is not None else float(self.model.kernel_params.sigma_eps)
        mu_xi, Sigma_xi, mu_psd, var_diag = self._static_posterior(y, sigma_eps=sig_eps)

        # Occupancy + postprocess
        pi0v = float(pi0) if pi0 is not None else float(self.model.pi0)
        pi_bin = self.occupancy_prob(mu_psd, var_diag, tau)
        occ0 = self.occ_mask_from_pi(pi_bin, pi0v)

        occ_mask = postprocess_mask_1d(
            occ0,
            max_gap_bins=int(self.model.max_gap_bins),
            min_width_bins=int(self.model.min_width_bins),
        )

        emitters = self.emitters_from_mask(self.model.f_hz, self.model.w, mu_psd, occ_mask)

        return SnapshotResult(
            f_hz=self.model.f_hz.copy(),
            psd_welch=y,
            mu_psd=mu_psd,
            sigma_psd_diag=var_diag,
            snf_used=snf_used,
            tau_bin=float(tau),
            pi_bin=pi_bin,
            occ_mask=occ_mask,
            emitters=emitters,
            mu_xi=(mu_xi if return_latent else None),
            Sigma_xi=(Sigma_xi if return_latent else None),
        )
