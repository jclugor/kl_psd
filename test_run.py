import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from spectrum_monitor import ModelPack, SpectrumMonitor

MODEL_PATH = Path("monitor_pack_v1_fc98_dt0p01_n8192.npz")
DATA_DIR = Path("data")
PATTERN = "Samples 88 and 108MHz,time to read 0.01s, sample #{k}.npy"

def to_db(x, floor=1e-30):
    return 10.0 * np.log10(np.maximum(x, floor))

def plot_snapshot(res, title="Snapshot", db=True):
    f_mhz = res.f_hz / 1e6
    y_welch = res.psd_welch
    y_mu = res.mu_psd
    y_std = np.sqrt(np.maximum(res.sigma_psd_diag, 0.0))

    if db:
        yw = to_db(y_welch)
        ym = to_db(y_mu)
        yhi = to_db(y_mu + 2*y_std)
        ylo = to_db(np.maximum(y_mu - 2*y_std, 1e-30))
        ytau = to_db(np.full_like(y_mu, res.tau_bin))
        ylab = "PSD (dB)"
    else:
        yw, ym = y_welch, y_mu
        yhi, ylo = y_mu + 2*y_std, np.maximum(y_mu - 2*y_std, 0.0)
        ytau = np.full_like(y_mu, res.tau_bin)
        ylab = "PSD (linear)"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(f_mhz, yw, label="Welch PSD")
    ax.plot(f_mhz, ym, label="Posterior mean PSD")
    ax.fill_between(f_mhz, ylo, yhi, alpha=0.2, label="±2σ (marginal)")
    ax.plot(f_mhz, ytau, "--", label="τ (bin threshold)")

    ymax = np.max(ym)
    for k, em in enumerate(res.emitters):
        f0 = em["f_start_hz"] / 1e6
        f1 = em["f_stop_hz"] / 1e6
        ax.axvspan(f0, f1, alpha=0.15)

        fp = em["f_peak_hz"] / 1e6
        idx = int(np.argmin(np.abs(f_mhz - fp)))
        ax.plot([fp], [ym[idx]], marker="o")

        bw_khz = em["bw_hz"] / 1e3
        peak_psd = em["peak_psd"]
        peak_txt = f"{to_db(np.array([peak_psd]))[0]:.1f} dB" if db else f"{peak_psd:.3g}"
        ax.text(0.5*(f0+f1), ymax, f"Em{k+1}: {bw_khz:.1f} kHz, peak {peak_txt}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.show()

def main():
    model = ModelPack.load(MODEL_PATH)
    mon = SpectrumMonitor(model, enable_kf=False)

    # choose a few files to test
    for k in [0, 1, 2]:
        fp = DATA_DIR / PATTERN.format(k=k)
        x = np.load(fp)
        res = mon.infer_snapshot(x, snf=None, quiet_frac=0.25)
        print(f"\n--- Snapshot #{k} ---")
        print("Emitters:", len(res.emitters))
        for em in res.emitters:
            print(em)
        plot_snapshot(res, title=f"Snapshot #{k}")

if __name__ == "__main__":
    main()
