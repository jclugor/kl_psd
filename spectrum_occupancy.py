"""
Spectrum Occupancy + Per-Emitter Bandwidth

What this module does
- Input: one IQ signal or a time-indexed list of IQ snapshots (e.g., 100 snapshots).
- Converts IQ -> Welch PSD on a consistent frequency grid (fc ± fs/2).
- Builds a KL basis Φ_KL from a composite two-lengthscale Matérn-3/2 kernel on frequency.
- Learns kernel hyperparameters by marginal likelihood (Woodbury), optionally with restarts.
- Computes PSD posterior: s|y ~ N(μ_s|y, Σ_s|y) via ξ posterior and Σ_s|y = Φ Σ_ξ Φ^T.
- Computes occupancy probabilities:
    per-bin:   π_i = P(s_i > τ_i | y)
    per-band:  π_B = P(p_B > τ_B | y)  where p_B = c_B^T s (average PSD or total power)
- Calibrates thresholds τ_i / τ_B using a noise-only noise-floor PSD level s_nf (not Σ_ε):
    τ_B ≈ s_nf * ( 1 + sqrt(2/|B|) * Q^{-1}(P_fa) )
    τ_i is |B|=1 special case.
- Extracts emitters and bandwidths:
    - Builds occupied set O_b (by bin probabilities or plug-in rule).
    - Finds connected components of O_b to obtain multiple disjoint emitters.
    - Per emitter: f_min, f_max, B_occ, peak frequency, centroid frequency.
- SNR estimation from reconstructed PSD using unoccupied bins:
    - Choose π_N cutoff, Nb = {i: π_i ≤ π_N}
    - s_bn = weighted mean of μ_s|y over Nb (or use provided s_nf)
    - P_sig, P_noise, SNR (linear and dB) computed over occupied support.
- Optional credible intervals via posterior sampling:
    - sample ξ ~ N(μ_ξ|y, Σ_ξ|y), build s^(m) = Φ ξ^(m)
    - compute features per sample, return quantiles

Dependencies: numpy, scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    """Inverse Q-function: Q^{-1}(p) = Φ^{-1}(1-p) = isf(p)."""
    return float(norm.isf(p))


def matern32_from_r(r: Array, ell: float) -> Array:
    a = np.sqrt(3.0) * r / ell
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
    Document KL recipe:
      Kw_ij = K_ij * sqrt(w_i w_j)
      eig(Kw) -> (lam, V), descending
      phi = V / sqrt(w)
      Phi_KL = phi * sqrt(lam)
    Returns: Phi_KL (M,R), lam (R,)
    """
    w = np.asarray(w).reshape(-1)
    sqrtw = np.sqrt(w)
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
    Static log marginal likelihood:
      y = A ξ + ε, ξ~N(0,I), ε~N(0, σ^2 I)
    Uses Woodbury/det lemma with B = I + A^T A / σ^2.
    """
    y = np.asarray(y).reshape(-1)
    N, R = A.shape
    sigma2 = float(sigma_eps**2)
    if sigma2 <= 0:
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


def posterior_xi(y: Array, A: Array, sigma_eps: float) -> Tuple[Array, Array]:
    """
    Posterior over ξ:
      Σ_ξ|y = (I + A^T A / σ^2)^-1
      μ_ξ|y = Σ_ξ|y (A^T y / σ^2)
    """
    y = np.asarray(y).reshape(-1)
    R = A.shape[1]
    sigma2 = float(sigma_eps**2)
    S = np.eye(R) + (A.T @ A) / sigma2
    Sigma = npl.inv(S)
    mu = Sigma @ ((A.T @ y) / sigma2)
    return mu, Sigma


def diag_Phi_Sigma_PhiT(Phi: Array, Sigma: Array) -> Array:
    """Efficient diagonal of Phi Sigma Phi^T."""
    PS = Phi @ Sigma
    return np.sum(PS * Phi, axis=1)


def band_weight_vector(
    w: Array,
    band_idx: Array,
    mode: str = "avg_psd",
) -> Array:
    """
    Build c_B for band statistic p_B = c_B^T s.
    mode:
      - "avg_psd": c_i = w_i / sum_{j in B} w_j    (units PSD)
      - "total_power": c_i = w_i                  (units power)
    """
    w = np.asarray(w).reshape(-1)
    c = np.zeros_like(w, dtype=float)
    if band_idx.size == 0:
        return c
    if mode == "avg_psd":
        c[band_idx] = w[band_idx] / np.sum(w[band_idx])
    elif mode == "total_power":
        c[band_idx] = w[band_idx]
    else:
        raise ValueError("mode must be 'avg_psd' or 'total_power'")
    return c


def connected_components_1d(mask: Array) -> List[Tuple[int, int]]:
    """
    Return contiguous runs of True in a 1D boolean mask as (start_idx, end_idx) inclusive.
    """
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


# -------------------------
# Data containers
# -------------------------

@dataclass
class PSDData:
    f_hz: Array          # (M,)
    psd: Array           # (T,M) or (M,)
    fc_hz: float
    fs_hz: float
    dt_acq_s: float
    snapshot_dt_s: Optional[float] = None


@dataclass
class NoiseModel:
    quiet_mask: Array    # (M,) bool
    quiet_frac: float
    noise_floor: Array   # (T,) linear PSD units
    sigma_eps: Array     # (T,) linear PSD units (measurement noise scale for PSD estimate)


@dataclass
class KernelParamsComposite:
    ell_short_hz: float
    ell_long_hz: float
    a1: float
    a2: float
    sigma_eps: float     # scalar σ_ε used in MLL fit


@dataclass
class KLModel:
    R: int
    Phi_KL: Array        # (M,R)
    w: Array             # (M,)
    f_hz: Array          # (M,)
    kernel_params: KernelParamsComposite


@dataclass
class PosteriorPSD:
    mu_s: Array          # (M,)
    Sigma_xi: Array      # (R,R)
    mu_xi: Array         # (R,)
    sigma_s_diag: Array  # (M,) diag(Σ_s) = diag(Φ Σ_ξ Φ^T)


# -------------------------
# Main estimator
# -------------------------

class SpectrumOccupancyEstimator:
    """
    Updated estimator matching WBPSDEst.pdf “occupancy + bandwidth” feature set.
    """

    def __init__(
        self,
        fc_hz: float = 98e6,
        dt_acq_s: float = 0.01,
        snapshot_dt_s: Optional[float] = 1.0,
        nperseg: int = 8192,
        noverlap: Optional[int] = None,
        window: str = "hann",
        detrend: bool = False,
        average: str = "mean",
        clamp_psd_excess_nonneg: bool = True,
        random_seed: int = 0,
    ):
        self.fc_hz = float(fc_hz)
        self.dt_acq_s = float(dt_acq_s)
        self.snapshot_dt_s = snapshot_dt_s if snapshot_dt_s is None else float(snapshot_dt_s)
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap) if noverlap is not None else int(nperseg // 2)
        self.window = window
        self.detrend = detrend
        self.average = average
        self.clamp_psd_excess_nonneg = bool(clamp_psd_excess_nonneg)
        self.rng = np.random.default_rng(random_seed)

    # -------- IQ loading / PSD --------

    def load_iq_files(self, folder: Union[str, Path], indices: Sequence[int]) -> List[Array]:
        folder = Path(folder)
        xs: List[Array] = []
        for k in indices:
            x = np.load(folder / f"Samples 88 and 108MHz,time to read 0.01s, sample #{k}.npy")
            x = np.asarray(x).reshape(-1)
            if not np.iscomplexobj(x):
                raise ValueError(f"sample #{k}.npy is not complex IQ.")
            xs.append(x)
        return xs

    def welch_psd(self, x: Array) -> Tuple[Array, Array, float]:
        x = np.asarray(x).reshape(-1)
        x = x - np.mean(x)  # DC removal

        fs_hz = float(len(x) / self.dt_acq_s)
        win = get_window(self.window, self.nperseg, fftbins=True)

        f_bb, Pxx = welch(
            x,
            fs=fs_hz,
            window=win,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nperseg,
            detrend=self.detrend,
            return_onesided=False,
            scaling="density",
            average=self.average,
        )
        f_bb = np.fft.fftshift(f_bb)
        Pxx = np.fft.fftshift(Pxx)
        f_rf = self.fc_hz + f_bb
        return f_rf, Pxx, fs_hz

    def build_psd_matrix(self, xs: Sequence[Array]) -> PSDData:
        Y: List[Array] = []
        f0: Optional[Array] = None
        fs0: Optional[float] = None

        for t, x in enumerate(xs):
            f_rf, Pxx, fs_hz = self.welch_psd(x)
            if f0 is None:
                f0 = f_rf
                fs0 = fs_hz
            else:
                if not np.allclose(f0, f_rf):
                    raise ValueError("Frequency grids differ across snapshots (unexpected).")
                if abs(fs_hz - float(fs0)) > 1e-6 * float(fs0):
                    raise ValueError("Sample rate differs across snapshots (unexpected).")
            Y.append(Pxx)

        psd = np.vstack(Y) if len(Y) > 1 else np.asarray(Y[0])
        return PSDData(
            f_hz=np.asarray(f0),
            psd=psd,
            fc_hz=self.fc_hz,
            fs_hz=float(fs0),
            dt_acq_s=self.dt_acq_s,
            snapshot_dt_s=self.snapshot_dt_s,
        )

    # -------- weights / noise --------

    def weights_uniform(self, f_hz: Array) -> Array:
        f_hz = np.asarray(f_hz).reshape(-1)
        df = float(np.mean(np.diff(f_hz)))
        return np.full_like(f_hz, df, dtype=float)

    def estimate_noise_model(self, psd_data: PSDData, quiet_frac: float = 0.25) -> NoiseModel:
        Y = np.asarray(psd_data.psd)
        if Y.ndim == 1:
            Y = Y[None, :]
        T, M = Y.shape

        med_over_time = np.median(Y, axis=0)
        thr = np.quantile(med_over_time, quiet_frac)
        quiet_mask = med_over_time <= thr

        noise_floor = np.zeros(T)
        sigma_eps = np.zeros(T)

        for t in range(T):
            yq = Y[t, quiet_mask]
            noise_floor[t] = float(np.median(yq))
            sigma_eps[t] = float(1.4826 * mad(yq))

        return NoiseModel(
            quiet_mask=quiet_mask,
            quiet_frac=float(quiet_frac),
            noise_floor=noise_floor,
            sigma_eps=sigma_eps,
        )

    def psd_excess(self, psd_data: PSDData, noise: NoiseModel) -> Array:
        Y = np.asarray(psd_data.psd)
        if Y.ndim == 1:
            floor = float(np.asarray(noise.noise_floor).reshape(-1)[0])
            Yex = Y - floor
            return np.maximum(Yex, 0.0) if self.clamp_psd_excess_nonneg else Yex

        floor = np.asarray(noise.noise_floor).reshape(-1, 1)
        Yex = Y - floor
        return np.maximum(Yex, 0.0) if self.clamp_psd_excess_nonneg else Yex

    # -------- composite kernel fitting / KL --------

    def fit_composite_kernel(
        self,
        f_hz: Array,
        w: Array,
        Y_excess: Array,
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

        Y = np.asarray(Y_excess)
        if Y.ndim == 1:
            Y = Y[None, :]
        T, M = Y.shape

        fit_idx = np.array([i for i in fit_indices if 0 <= i < T], dtype=int)
        if fit_idx.size == 0:
            fit_idx = np.array([0], dtype=int)

        sig0 = float(np.median(np.asarray(sigma_eps_hint).reshape(-1)))
        if sig0 <= 0:
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

        # u = [log_ell_short, log_delta, log_a1, log_a2, log_sigma]
        bounds = [
            (np.log(ell_short_min), np.log(ell_short_max)),
            (np.log(delta_min), np.log(delta_max)),
            (np.log(a_min), np.log(a_max)),
            (np.log(a_min), np.log(a_max)),
            (np.log(sigma_min), np.log(sigma_max)),
        ]

        def unpack(u: Array) -> Tuple[float, float, float, float, float]:
            ell_s = float(np.exp(u[0]))
            ell_l = float(ell_s + np.exp(u[1]))  # enforce ell_long > ell_short
            a1 = float(np.exp(u[2]))
            a2 = float(np.exp(u[3]))
            sig = float(np.exp(u[4]))
            return ell_s, ell_l, a1, a2, sig

        def neg_sum_logml(u: Array) -> float:
            ell_s, ell_l, a1, a2, sig = unpack(u)
            K = composite_matern32_kernel_1d(f_hz, ell_s, ell_l, a1, a2)
            Phi_KL, _ = build_phi_kl_from_kernel(K, w, R)
            A = Phi_KL  # Theta = I in PSD domain
            ll_sum = 0.0
            for t in fit_idx:
                ll = log_marginal_likelihood_y(Y[t], A, sig)
                if not np.isfinite(ll):
                    return 1e30
                ll_sum += ll
            return -float(ll_sum)

        # baseline init
        u_base = np.array([
            np.log(0.02 * B_total),                    # ell_short
            np.log(0.15 * B_total),                    # delta -> ell_long
            np.log(1.0),                               # a1
            np.log(1.0),                               # a2
            np.log(np.clip(sig0, sigma_min, sigma_max))# sigma
        ], dtype=float)

        best = None
        for r in range(max(1, n_restarts)):
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
        return KernelParamsComposite(
            ell_short_hz=ell_s,
            ell_long_hz=ell_l,
            a1=a1,
            a2=a2,
            sigma_eps=sig,
        )

    def build_kl_model(
        self,
        psd_data: PSDData,
        noise: NoiseModel,
        R: int = 20,
        fit_kernel: bool = True,
        fit_indices: Sequence[int] = (0, 25, 50, 75),
        kernel_params: Optional[KernelParamsComposite] = None,
        n_restarts: int = 3,
    ) -> KLModel:
        f = psd_data.f_hz
        w = self.weights_uniform(f)
        Yex = self.psd_excess(psd_data, noise)

        if kernel_params is None:
            if fit_kernel:
                kernel_params = self.fit_composite_kernel(
                    f_hz=f,
                    w=w,
                    Y_excess=Yex,
                    sigma_eps_hint=noise.sigma_eps,
                    R=R,
                    fit_indices=fit_indices,
                    n_restarts=n_restarts,
                )
            else:
                B_total = float(np.sum(w))
                kernel_params = KernelParamsComposite(
                    ell_short_hz=0.02 * B_total,
                    ell_long_hz=0.2 * B_total,
                    a1=1.0,
                    a2=1.0,
                    sigma_eps=float(np.median(noise.sigma_eps)),
                )

        K = composite_matern32_kernel_1d(f, kernel_params.ell_short_hz, kernel_params.ell_long_hz, kernel_params.a1, kernel_params.a2)
        Phi_KL, _ = build_phi_kl_from_kernel(K, w, R)
        return KLModel(R=R, Phi_KL=Phi_KL, w=w, f_hz=f, kernel_params=kernel_params)

    # -------- posterior PSD (single snapshot) --------

    def posterior_psd(
        self,
        y: Array,
        kl: KLModel,
        sigma_eps: Optional[float] = None,
    ) -> PosteriorPSD:
        """
        Compute posterior s|y in the static model with Theta=I, s≈Φ ξ, ξ~N(0,I), ε~N(0,σ^2 I).
        Returns μ_s, diag(Σ_s), plus ξ posterior (μ_ξ, Σ_ξ).
        """
        y = np.asarray(y).reshape(-1)
        Phi = kl.Phi_KL
        sig = float(sigma_eps if sigma_eps is not None else kl.kernel_params.sigma_eps)

        mu_xi, Sigma_xi = posterior_xi(y, Phi, sig)
        mu_s = Phi @ mu_xi
        sigma_s_diag = diag_Phi_Sigma_PhiT(Phi, Sigma_xi)
        return PosteriorPSD(mu_s=mu_s, Sigma_xi=Sigma_xi, mu_xi=mu_xi, sigma_s_diag=sigma_s_diag)

    # -------- thresholds + occupancy --------

    def thresholds_from_snf(
        self,
        snf: float,
        p_fa: float,
        band_size_bins: int,
    ) -> float:
        """
        Threshold for average-PSD statistic (units: PSD):
          τ_B ≈ s_nf * ( 1 + sqrt(2/|B|) * Q^{-1}(P_fa) )
        """
        B = max(1, int(band_size_bins))
        return float(snf * (1.0 + np.sqrt(2.0 / B) * Qinv(p_fa)))

    def bin_thresholds_from_snf(
        self,
        snf: float,
        p_fa_bin: float,
        M: int,
    ) -> float:
        """
        Per-bin τ_i is the |B|=1 special case.
        """
        return float(snf * (1.0 + np.sqrt(2.0) * Qinv(p_fa_bin)))

    def occupancy_prob_bin(
        self,
        mu_s: Array,
        sigma_s_diag: Array,
        tau_i: Union[float, Array],
    ) -> Array:
        """
        π_i = P(s_i > τ_i | y) = 1 - Φ((τ_i - μ_i)/σ_i)
        """
        mu_s = np.asarray(mu_s).reshape(-1)
        sig = np.sqrt(np.maximum(np.asarray(sigma_s_diag).reshape(-1), 1e-30))
        tau = np.asarray(tau_i) if np.ndim(tau_i) else float(tau_i)
        z = (tau - mu_s) / sig
        return 1.0 - norm.cdf(z)

    def band_posterior(
        self,
        post: PosteriorPSD,
        kl: KLModel,
        bands_hz: Sequence[Tuple[float, float]],
        statistic: str = "avg_psd",
    ) -> Dict[str, Array]:
        """
        Band statistic posterior:
          p_B|y ~ N( μ_B, σ_B^2 ), μ_B = c^T μ_s,  σ_B^2 = c^T Σ_s c
        We compute σ_B^2 without forming Σ_s:
          Σ_s = Φ Σ_ξ Φ^T, so σ_B^2 = (Φ^T c)^T Σ_ξ (Φ^T c)
        """
        f = kl.f_hz
        w = kl.w
        Phi = kl.Phi_KL
        mu_s = post.mu_s
        Sigma_xi = post.Sigma_xi

        centers = []
        muB = []
        stdB = []
        nbins = []

        for (flo, fhi) in bands_hz:
            idx = np.where((f >= flo) & (f < fhi))[0]
            if idx.size == 0:
                continue
            c = band_weight_vector(w, idx, mode=statistic)
            m = float(c @ mu_s)
            v = float((Phi.T @ c) @ (Sigma_xi @ (Phi.T @ c)))
            centers.append(0.5 * (flo + fhi))
            muB.append(m)
            stdB.append(float(np.sqrt(max(v, 0.0))))
            nbins.append(int(idx.size))

        return dict(
            centers_hz=np.array(centers),
            muB=np.array(muB),
            stdB=np.array(stdB),
            nbins=np.array(nbins, dtype=int),
        )

    def occupancy_prob_band(
        self,
        muB: Array,
        stdB: Array,
        tauB: Union[float, Array],
    ) -> Array:
        muB = np.asarray(muB).reshape(-1)
        stdB = np.asarray(stdB).reshape(-1)
        tau = np.asarray(tauB) if np.ndim(tauB) else float(tauB)
        z = (tau - muB) / np.maximum(stdB, 1e-30)
        return 1.0 - norm.cdf(z)

    # -------- emitter features (bandwidth, peaks, centroid, SNR) --------

    def estimate_snf_from_unoccupied_bins(
        self,
        mu_s: Array,
        w: Array,
        pi_i: Array,
        pi_N: float = 0.1,
        snf_fallback: Optional[float] = None,
    ) -> float:
        """
        s_bn := weighted mean of μ_s over Nb = {i : π_i <= π_N}.
        If Nb is empty, fall back to snf_fallback or global weighted mean of μ_s.
        """
        mu_s = np.asarray(mu_s).reshape(-1)
        w = np.asarray(w).reshape(-1)
        pi_i = np.asarray(pi_i).reshape(-1)

        Nb = np.where(pi_i <= float(pi_N))[0]
        if Nb.size == 0:
            if snf_fallback is not None:
                return float(snf_fallback)
            return float(np.sum(w * mu_s) / np.sum(w))
        return float(np.sum(w[Nb] * mu_s[Nb]) / np.sum(w[Nb]))

    def occupied_set_from_pi(
        self,
        pi_i: Array,
        pi0: float = 0.95,
    ) -> Array:
        """Occupied bins O_b as boolean mask: π_i >= π0."""
        return np.asarray(pi_i).reshape(-1) >= float(pi0)

    def occupied_set_plugin(
        self,
        mu_s: Array,
        tau_i: Union[float, Array],
    ) -> Array:
        """Plug-in occupied bins: μ_i > τ_i."""
        mu_s = np.asarray(mu_s).reshape(-1)
        tau = np.asarray(tau_i) if np.ndim(tau_i) else float(tau_i)
        return mu_s > tau

    def per_emitter_features_from_mask(
        self,
        f_hz: Array,
        w: Array,
        mu_s: Array,
        occ_mask: Array,
    ) -> List[Dict[str, float]]:
        """
        Connected components of occ_mask => emitters.
        For each component:
          f_min, f_max, B_occ
          peak frequency (argmax μ in component)
          centroid frequency (power-weighted)
        """
        f_hz = np.asarray(f_hz).reshape(-1)
        w = np.asarray(w).reshape(-1)
        mu_s = np.asarray(mu_s).reshape(-1)
        occ_mask = np.asarray(occ_mask, dtype=bool).reshape(-1)

        comps = connected_components_1d(occ_mask)
        out: List[Dict[str, float]] = []

        for (i0, i1) in comps:
            idx = np.arange(i0, i1 + 1)
            fmin = float(np.min(f_hz[idx]))
            fmax = float(np.max(f_hz[idx]))
            Bocc = float(fmax - fmin)

            # peak in component
            i_peak = int(idx[np.argmax(mu_s[idx])])
            f_peak = float(f_hz[i_peak])

            # centroid in component (using μ_s as weights)
            weights = w[idx] * np.maximum(mu_s[idx], 0.0)
            denom = float(np.sum(weights)) if float(np.sum(weights)) > 0 else float(np.sum(w[idx]))
            f_centroid = float(np.sum(f_hz[idx] * weights) / denom)

            out.append(dict(
                f_start_hz=fmin,
                f_stop_hz=fmax,
                bw_hz=Bocc,
                f_peak_hz=f_peak,
                f_centroid_hz=f_centroid,
                i_start=int(i0),
                i_stop=int(i1),
            ))
        return out

    def expected_occupied_bandwidth(
        self,
        pi_i: Array,
        w: Array,
    ) -> float:
        """E[B_occ|y] ≈ sum_i π_i w_i (works for uniform or non-uniform grids)."""
        pi_i = np.asarray(pi_i).reshape(-1)
        w = np.asarray(w).reshape(-1)
        return float(np.sum(pi_i * w))

    def snr_over_occupied_support(
        self,
        mu_s: Array,
        w: Array,
        occ_mask: Array,
        snf: float,
    ) -> Tuple[float, float, float, float]:
        """
        Over occupied bins O_b:
          P_sig = sum_{i in Ob} w_i * (μ_i - s_nf)_+
          P_noise = s_nf * sum_{i in Ob} w_i
          SNR = P_sig / P_noise
        Returns: (P_sig, P_noise, SNR_lin, SNR_dB)
        """
        mu_s = np.asarray(mu_s).reshape(-1)
        w = np.asarray(w).reshape(-1)
        occ = np.asarray(occ_mask, dtype=bool).reshape(-1)

        idx = np.where(occ)[0]
        if idx.size == 0:
            return 0.0, 0.0, 0.0, -np.inf

        P_sig = float(np.sum(w[idx] * np.maximum(mu_s[idx] - float(snf), 0.0)))
        P_noise = float(float(snf) * np.sum(w[idx]))
        if P_noise <= 0:
            return P_sig, P_noise, np.inf, np.inf
        SNR = P_sig / P_noise
        return P_sig, P_noise, float(SNR), float(10.0 * np.log10(SNR + 1e-30))

    # -------- credible intervals via posterior sampling --------

    def sample_features(
        self,
        post: PosteriorPSD,
        kl: KLModel,
        occ_mask: Array,
        snf: float,
        n_samples: int = 200,
        q_lo: float = 0.025,
        q_hi: float = 0.975,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Sample ξ^(m) ~ N(μ_ξ, Σ_ξ), compute s^(m)=Φ ξ^(m), then compute:
          - expected occupied bandwidth via hard mask: B_occ = fmax - fmin (if mask nonempty)
          - peak frequency, centroid frequency within mask
          - SNR over occupied support
        Returns credible intervals (q_lo,q_hi) for these scalar features.
        """
        Phi = kl.Phi_KL
        f = kl.f_hz
        w = kl.w
        occ_mask = np.asarray(occ_mask, dtype=bool).reshape(-1)

        if np.sum(occ_mask) == 0:
            return {}

        # Sample ξ
        L = npl.cholesky(symmetrize(post.Sigma_xi) + 1e-12 * np.eye(kl.R))
        Z = self.rng.standard_normal(size=(kl.R, int(n_samples)))
        Xi_samp = post.mu_xi[:, None] + L @ Z

        # Compute features per sample
        idx = np.where(occ_mask)[0]
        fmin = float(np.min(f[idx]))
        fmax = float(np.max(f[idx]))

        Bocc = []
        fpeak = []
        fcent = []
        snr_db = []

        for m in range(Xi_samp.shape[1]):
            s = Phi @ Xi_samp[:, m]
            # ensure nonneg for weights where needed
            s_pos = np.maximum(s, 0.0)

            # peak within occupied bins
            i_pk = int(idx[np.argmax(s_pos[idx])])
            fpeak.append(float(f[i_pk]))

            # centroid within occupied bins
            weights = w[idx] * s_pos[idx]
            denom = float(np.sum(weights)) if float(np.sum(weights)) > 0 else float(np.sum(w[idx]))
            fcent.append(float(np.sum(f[idx] * weights) / denom))

            # bandwidth based on support endpoints (mask fixed)
            Bocc.append(float(fmax - fmin))

            # SNR over occupied support (mask fixed)
            _, _, _, snr_dB = self.snr_over_occupied_support(s_pos, w, occ_mask, snf)
            snr_db.append(float(snr_dB))

        def ci(arr):
            arr = np.asarray(arr, dtype=float)
            return (float(np.quantile(arr, q_lo)), float(np.quantile(arr, q_hi)))

        return dict(
            bw_hz=ci(Bocc),
            f_peak_hz=ci(fpeak),
            f_centroid_hz=ci(fcent),
            snr_db=ci(snr_db),
        )

    # -------- high-level run --------

    def run_from_folder(
        self,
        folder: Union[str, Path],
        indices: Sequence[int],
        R: int = 20,
        quiet_frac: float = 0.25,
        fit_kernel: bool = True,
        fit_indices: Sequence[int] = (0, 25, 50, 75),
        n_restarts: int = 3,
        # occupancy choices
        occ_method: str = "per_bin_prob",     # "per_bin_prob" or "plugin"
        pi0: float = 0.95,                    # probability threshold for occupied bins
        # threshold calibration
        p_fa_bin: float = 1e-3,               # per-bin false alarm
        p_fa_band: float = 1e-3,              # per-band false alarm
        # noise floor s_nf (optional)
        snf_global: Optional[float] = None,   # if provided, used directly as noise-floor PSD level
        snf_mode: str = "quiet_floor",        # "quiet_floor" or "posterior_unoccupied"
        pi_N: float = 0.1,                    # cutoff to define noise bins Nb when estimating s_nf from posterior
        # band outputs (optional)
        band_bw_hz: float = 200e3,
        f_min_hz: float = 88e6,
        f_max_hz: float = 108e6,
        band_statistic: str = "avg_psd",      # "avg_psd" or "total_power"
        # credible intervals
        n_ci_samples: int = 0,                # 0 disables sampling CIs
    ) -> Dict[str, object]:
        """
        Full pipeline:
          IQ -> PSD -> noise model -> KL model -> per-snapshot posterior -> occupancy -> emitters + bandwidth

        Returns dict with keys:
          psd_data, noise, kl, Y_excess,
          per_snapshot: list of per-snapshot results:
            mu_s, pi_i, occ_mask, snf_used, expected_Bocc,
            emitters: list of per-emitter dicts with bandwidth, peak, centroid, SNR
            (optional) emitter_ci: credible intervals per emitter (if n_ci_samples>0)
            (optional) band-level occupancy arrays (centers, pi_B, etc.)
        """
        xs = self.load_iq_files(folder, indices)
        psd_data = self.build_psd_matrix(xs)
        noise = self.estimate_noise_model(psd_data, quiet_frac=quiet_frac)
        Yex = self.psd_excess(psd_data, noise)

        kl = self.build_kl_model(
            psd_data=psd_data,
            noise=noise,
            R=R,
            fit_kernel=fit_kernel,
            fit_indices=fit_indices,
            n_restarts=n_restarts,
        )

        f = kl.f_hz
        w = kl.w

        # build uniform bands (for band outputs)
        bw = float(band_bw_hz)
        edges = np.arange(float(f_min_hz), float(f_max_hz) + bw, bw)
        bands = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]

        # iterate snapshots
        Y = np.asarray(Yex)
        if Y.ndim == 1:
            Y = Y[None, :]
        T = Y.shape[0]

        per_snapshot: List[Dict[str, object]] = []

        for t in range(T):
            y = Y[t].reshape(-1)

            # posterior over PSD excess on grid
            post = self.posterior_psd(y, kl, sigma_eps=kl.kernel_params.sigma_eps)

            # determine s_nf (noise-floor PSD level) used for threshold calibration and SNR
            if snf_global is not None:
                snf_used = float(snf_global)
            else:
                if snf_mode == "quiet_floor":
                    # use quiet-bin floor estimate from raw PSD (works well as initial snf)
                    snf_used = float(noise.noise_floor[t])
                elif snf_mode == "posterior_unoccupied":
                    # iterative-ish: use an initial snf from quiet floor to compute pi_i, then re-estimate from Nb
                    snf_init = float(noise.noise_floor[t])
                    tau_i_init = self.bin_thresholds_from_snf(snf_init, p_fa_bin, M=len(f))
                    pi_i_init = self.occupancy_prob_bin(post.mu_s, post.sigma_s_diag, tau_i_init)
                    snf_used = self.estimate_snf_from_unoccupied_bins(
                        mu_s=post.mu_s, w=w, pi_i=pi_i_init, pi_N=pi_N, snf_fallback=snf_init
                    )
                else:
                    raise ValueError("snf_mode must be 'quiet_floor' or 'posterior_unoccupied'")

            # per-bin thresholds and occupancy probabilities
            tau_i = self.bin_thresholds_from_snf(snf_used, p_fa_bin, M=len(f))
            pi_i = self.occupancy_prob_bin(post.mu_s, post.sigma_s_diag, tau_i)

            # occupied set
            if occ_method == "per_bin_prob":
                occ_mask = self.occupied_set_from_pi(pi_i, pi0=pi0)
            elif occ_method == "plugin":
                occ_mask = self.occupied_set_plugin(post.mu_s, tau_i=tau_i)
            else:
                raise ValueError("occ_method must be 'per_bin_prob' or 'plugin'")

            # expected occupied bandwidth (soft)
            expected_Bocc = self.expected_occupied_bandwidth(pi_i, w)

            # per-emitter features (hard support via connected components)
            emitters = self.per_emitter_features_from_mask(f, w, post.mu_s, occ_mask)

            # add SNR per emitter (computed over that emitter's support)
            for em in emitters:
                em_mask = np.zeros_like(occ_mask, dtype=bool)
                em_mask[int(em["i_start"]): int(em["i_stop"]) + 1] = True
                P_sig, P_noise, SNR_lin, SNR_dB = self.snr_over_occupied_support(post.mu_s, w, em_mask, snf_used)
                em.update(dict(
                    P_sig=P_sig,
                    P_noise=P_noise,
                    snr_lin=SNR_lin,
                    snr_db=SNR_dB,
                ))

            # optional credible intervals per emitter (sampling)
            emitter_ci: List[Dict[str, object]] = []
            if int(n_ci_samples) > 0 and len(emitters) > 0:
                for em in emitters:
                    em_mask = np.zeros_like(occ_mask, dtype=bool)
                    em_mask[int(em["i_start"]): int(em["i_stop"]) + 1] = True
                    ci = self.sample_features(
                        post=post,
                        kl=kl,
                        occ_mask=em_mask,
                        snf=snf_used,
                        n_samples=int(n_ci_samples),
                    )
                    emitter_ci.append(ci)

            # band-level occupancy probability (optional output; useful for reporting per 200 kHz channel)
            band_post = self.band_posterior(post, kl, bands_hz=bands, statistic=band_statistic)

            # band thresholds τ_B depend on |B| (bins per band). We can approximate |B| by nbins in each band.
            tauB = np.array([
                self.thresholds_from_snf(snf_used, p_fa_band, int(nB))
                for nB in band_post["nbins"]
            ])
            pi_B = self.occupancy_prob_band(band_post["muB"], band_post["stdB"], tauB)

            per_snapshot.append(dict(
                t=t,
                snf_used=snf_used,
                tau_i=tau_i,
                pi_i=pi_i,
                occ_mask=occ_mask,
                expected_Bocc_hz=expected_Bocc,
                mu_s=post.mu_s,
                sigma_s_diag=post.sigma_s_diag,
                emitters=emitters,
                emitter_ci=emitter_ci if int(n_ci_samples) > 0 else None,
                band=dict(
                    centers_hz=band_post["centers_hz"],
                    muB=band_post["muB"],
                    stdB=band_post["stdB"],
                    tauB=tauB,
                    pi_B=pi_B,
                    band_bw_hz=bw,
                    statistic=band_statistic,
                )
            ))

        return dict(
            psd_data=psd_data,
            noise=noise,
            Y_excess=Yex,
            kl=kl,
            per_snapshot=per_snapshot,
        )
