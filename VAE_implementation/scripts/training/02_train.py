#!/usr/bin/env python3
"""
02_train.py ??? Train VAE for PSD compression (1024 bins) with robust local saving/resume.

Features:
- Loads preprocessed dataset (.npz) + splits from data/processed/...
- Builds encoder/decoder (mirror architecture, latent_dim=32)
- Objective: recon (MSE) + beta * KL
- Training recon uses mu (deterministic), consistent with edge inference z???mu
- Beta warm-up schedule
- Optional "peaky 50/50" oversampling (data-level emphasis, no loss change)
- Robust saving/resume: saves encoder/decoder weights separately (Keras 3 safe)
- End-of-run promotion: compares run BEST vs GLOBAL_BEST, replaces only if better

Usage:
  python VAE_implementation/scripts/training/02_train.py --config VAE_implementation/configs/vae_default.yaml
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

import tensorflow as tf  # type: ignore[import-untyped]
from tensorflow import keras  # type: ignore[import-untyped]
from tensorflow.keras import layers  # type: ignore[import-untyped]


# =============================
# Repo helpers
# =============================
def repo_root() -> Path:
    """Return the project root for repository-relative config paths."""

    return Path(__file__).resolve().parents[3]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_text_int(path: Path, default: int = 0) -> int:
    try:
        return int(path.read_text().strip())
    except Exception:
        return default


def safe_read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def safe_write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


# =============================
# Architecture (must match eval)
# =============================
def _has_conv1d_transpose() -> bool:
    return hasattr(layers, "Conv1DTranspose")


def build_encoder(
    input_bins: int = 1024, latent_dim: int = 32, include_dense128: bool = True
) -> keras.Model:
    """
    Encoder:
      Input: (1024,1)
      Conv1D(16,k=5,s=2) + LeakyReLU -> (512,16)
      Conv1D(32,k=3,s=2) + LeakyReLU -> (256,32)
      Flatten
      (optional) Dense(128)+LeakyReLU
      Dense(latent_dim) -> mu
      Dense(latent_dim) -> logvar
    """
    x_in = keras.Input(shape=(input_bins, 1), name="x_in")

    x = layers.Conv1D(16, kernel_size=5, strides=2, padding="same", name="enc_conv1")(
        x_in
    )
    x = layers.LeakyReLU(alpha=0.2, name="enc_lrelu1")(x)

    x = layers.Conv1D(32, kernel_size=3, strides=2, padding="same", name="enc_conv2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="enc_lrelu2")(x)

    x = layers.Flatten(name="enc_flatten")(x)

    if include_dense128:
        x = layers.Dense(128, name="enc_dense")(x)
        x = layers.LeakyReLU(alpha=0.2, name="enc_lrelu_dense")(x)

    mu = layers.Dense(latent_dim, name="z_mu")(x)
    logvar = layers.Dense(latent_dim, name="z_logvar")(x)

    return keras.Model(x_in, [mu, logvar], name="encoder")


def build_decoder(input_bins: int = 1024, latent_dim: int = 32) -> keras.Model:
    """
    Decoder (mirror capacity):
      z (latent_dim)
      Dense -> (256,32)
      Conv1DTranspose(32,k=3,s=2) -> (512,32)
      Conv1DTranspose(16,k=5,s=2) -> (1024,16)
      Conv1D(1,k=1,sigmoid) -> (1024,1)
    """
    z_in = keras.Input(shape=(latent_dim,), name="z_in")

    x = layers.Dense(256 * 32, name="dec_dense")(z_in)  # 8192
    x = layers.Reshape((256, 32), name="dec_reshape")(x)  # (256,32)

    if _has_conv1d_transpose():
        Conv1DTranspose = layers.Conv1DTranspose
        x = Conv1DTranspose(
            32, kernel_size=3, strides=2, padding="same", name="dec_deconv1"
        )(x)
        x = layers.LeakyReLU(alpha=0.2, name="dec_lrelu1")(x)

        x = Conv1DTranspose(
            16, kernel_size=5, strides=2, padding="same", name="dec_deconv2"
        )(x)
        x = layers.LeakyReLU(alpha=0.2, name="dec_lrelu2")(x)
    else:
        x = layers.UpSampling1D(size=2, name="dec_ups1")(x)
        x = layers.Conv1D(32, kernel_size=3, padding="same", name="dec_conv1")(x)
        x = layers.LeakyReLU(alpha=0.2, name="dec_lrelu1")(x)

        x = layers.UpSampling1D(size=2, name="dec_ups2")(x)
        x = layers.Conv1D(16, kernel_size=5, padding="same", name="dec_conv2")(x)
        x = layers.LeakyReLU(alpha=0.2, name="dec_lrelu2")(x)

    x_hat = layers.Conv1D(1, kernel_size=1, activation="sigmoid", name="x_hat")(x)
    return keras.Model(z_in, x_hat, name="decoder")


# =============================
# VAE model (subclassed)
# =============================
class Sampling(keras.layers.Layer):
    """Reparameterization trick: z = mu + sigma * eps"""

    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        sigma = tf.exp(0.5 * logvar)
        return mu + sigma * eps


class VAE(keras.Model):
    """
    Objective:
      recon = mean_batch(sum_bins((x-x_hat)^2))
      kl = mean_batch(sum_lat(-0.5*(1+logvar-mu^2-exp(logvar))))
      loss = recon + beta*kl

    Peak-friendly: reconstruction uses mu directly during training (deterministic).
    """

    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        sampler: keras.layers.Layer,
        beta_init: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.beta = tf.Variable(
            beta_init, dtype=tf.float32, trainable=False, name="beta"
        )

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, x, training=False):
        mu, logvar = self.encoder(x, training=training)
        z = self.sampler([mu, logvar]) if training else mu
        return self.decoder(z, training=training)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            mu, logvar = self.encoder(x, training=True)

            # IMPORTANT: use mu for reconstruction (better peaks, less blur)
            x_hat = self.decoder(mu, training=True)

            recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=[1, 2]))
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            )
            loss = recon + self.beta * kl

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(loss)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "beta": self.beta,
        }

    def test_step(self, x):
        mu, logvar = self.encoder(x, training=False)
        x_hat = self.decoder(mu, training=False)

        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=[1, 2]))
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        )
        loss = recon + self.beta * kl

        self.total_loss_tracker.update_state(loss)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "beta": self.beta,
        }


# =============================
# Callbacks
# =============================
class BetaWarmUp(keras.callbacks.Callback):
    """Linear warm-up beta from 0 to beta_final over warmup_epochs."""

    def __init__(self, beta_final: float, warmup_epochs: int):
        super().__init__()
        self.beta_final = float(beta_final)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs <= 0:
            beta = self.beta_final
        else:
            t = min(1.0, epoch / float(self.warmup_epochs))
            beta = t * self.beta_final
        self.model.beta.assign(beta)


class SaveEncDec(keras.callbacks.Callback):
    """
    Saves encoder/decoder weights each epoch (latest), and saves run-best by val_loss.
    Also updates last_epoch.txt for resume.
    """

    def __init__(
        self,
        enc_latest: Path,
        dec_latest: Path,
        enc_best: Path,
        dec_best: Path,
        epoch_file: Path,
        best_file: Path,
    ):
        super().__init__()
        self.enc_latest = Path(enc_latest)
        self.dec_latest = Path(dec_latest)
        self.enc_best = Path(enc_best)
        self.dec_best = Path(dec_best)
        self.epoch_file = Path(epoch_file)
        self.best_file = Path(best_file)

        self.best_val = float("inf")
        if self.best_file.exists():
            try:
                self.best_val = float(
                    json.loads(self.best_file.read_text())["best_val_loss"]
                )
            except Exception:
                self.best_val = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss", None)

        # Always save latest
        self.model.encoder.save_weights(self.enc_latest)
        self.model.decoder.save_weights(self.dec_latest)
        self.epoch_file.write_text(str(epoch + 1))

        # Save best (run-level)
        if val_loss is not None and float(val_loss) < self.best_val:
            self.best_val = float(val_loss)
            safe_write_json(self.best_file, {"best_val_loss": self.best_val})
            self.model.encoder.save_weights(self.enc_best)
            self.model.decoder.save_weights(self.dec_best)
            print(
                f"\n[RUN BEST] val_loss={self.best_val:.6f} -> saved enc_best/dec_best\n"
            )


def promote_to_global_best(
    run_best_val: float, run_dir: Path, enc_best: Path, dec_best: Path, global_dir: Path
) -> None:
    """
    Compare run_best_val vs global_best_val stored in GLOBAL_BEST/best_info.json.
    Replace GLOBAL_BEST weights only if run_best_val is better (lower).
    """
    ensure_dir(global_dir)
    g_enc = global_dir / "enc_best.weights.h5"
    g_dec = global_dir / "dec_best.weights.h5"
    g_info = global_dir / "best_info.json"

    global_best_val = float("inf")
    if g_info.exists():
        try:
            global_best_val = float(json.loads(g_info.read_text())["best_val_loss"])
        except Exception:
            global_best_val = float("inf")

    print(
        f"\n[COMPARE] run_best={run_best_val:.6f} vs global_best={global_best_val:.6f}"
    )

    if run_best_val < global_best_val and enc_best.exists() and dec_best.exists():
        shutil.copy2(enc_best, g_enc)
        shutil.copy2(dec_best, g_dec)
        info = {
            "best_val_loss": float(run_best_val),
            "source_run_dir": str(run_dir),
        }
        safe_write_json(g_info, info)
        print("[GLOBAL UPDATE] Promoted this run to GLOBAL_BEST ???")
    else:
        print("[GLOBAL UPDATE] Kept previous GLOBAL_BEST (this run not better)")


# =============================
# Data loading
# =============================
@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    ds_train: tf.data.Dataset
    ds_val: tf.data.Dataset
    ds_test: tf.data.Dataset
    steps_per_epoch: int
    val_steps: int


def load_dataset(processed_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    npz_path = processed_dir / "dataset_psd_1024_norm.npz"
    if not npz_path.exists():
        candidates = sorted(
            processed_dir.glob("*.npz"), key=lambda p: p.stat().st_size, reverse=True
        )
        if not candidates:
            raise FileNotFoundError(f"No .npz found in {processed_dir}")
        npz_path = candidates[0]

    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data:
        raise KeyError(f"NPZ keys: {list(data.keys())} ??? expected 'X'.")
    X = data["X"].astype(np.float32)
    freqs = data["freqs_hz"] if "freqs_hz" in data else None
    print("[DATA] Loaded:", npz_path)
    print("[DATA] X shape:", X.shape, "dtype:", X.dtype)
    return X, freqs


def load_splits(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits_dir = processed_dir / "splits"
    tr = np.load(splits_dir / "train_idx.npy")
    va = np.load(splits_dir / "val_idx.npy")
    te = np.load(splits_dir / "test_idx.npy")
    return tr, va, te


def make_datasets(
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    steps_multiplier: int,
    use_peaky_5050: bool,
    peaky_quantile: float,
    seed: int,
) -> DatasetBundle:
    X_train = X[train_idx][..., None]
    X_val = X[val_idx][..., None]
    X_test = X[test_idx][..., None]

    autotune = tf.data.AUTOTUNE

    base_steps = int(math.ceil(len(X_train) / batch_size))
    steps_per_epoch = max(1, base_steps * max(1, int(steps_multiplier)))
    val_steps = int(math.ceil(len(X_val) / batch_size))

    if not use_peaky_5050:
        ds_train = (
            tf.data.Dataset.from_tensor_slices(X_train)
            .shuffle(min(len(X_train), 20000), seed=seed, reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(autotune)
        )
    else:
        mx = X_train.max(axis=1).squeeze(-1)
        thr = np.quantile(mx, peaky_quantile)
        idx_peaky = np.where(mx >= thr)[0]
        idx_all = np.arange(len(X_train))
        print(
            f"[DATA] Peaky quantile={peaky_quantile} -> peaky frames {len(idx_peaky)}/{len(idx_all)}"
        )

        ds_peaky = (
            tf.data.Dataset.from_tensor_slices(X_train[idx_peaky])
            .shuffle(
                min(len(idx_peaky), 20000), seed=seed, reshuffle_each_iteration=True
            )
            .repeat()
            .batch(batch_size)
            .prefetch(autotune)
        )

        ds_all = (
            tf.data.Dataset.from_tensor_slices(X_train[idx_all])
            .shuffle(min(len(idx_all), 20000), seed=seed, reshuffle_each_iteration=True)
            .repeat()
            .batch(batch_size)
            .prefetch(autotune)
        )

        ds_train = tf.data.Dataset.sample_from_datasets(
            [ds_peaky, ds_all], weights=[0.5, 0.5]
        ).prefetch(autotune)

    ds_val = (
        tf.data.Dataset.from_tensor_slices(X_val).batch(batch_size).prefetch(autotune)
    )
    ds_test = (
        tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size).prefetch(autotune)
    )

    return DatasetBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
    )


# =============================
# Main
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", required=True, help="YAML config path (repo-relative or absolute)."
    )
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    cfg = yaml.safe_load(cfg_path.read_text())

    processed_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]
    ensure_dir(models_dir)

    input_bins = int(cfg.get("preprocess", {}).get("target_bins", 1024))

    tcfg = cfg["train"]
    batch_size = int(tcfg.get("batch_size", 256))
    epochs = int(tcfg.get("epochs", 800))
    steps_multiplier = int(tcfg.get("steps_multiplier", 1))
    lr = float(tcfg.get("lr", 3e-4))
    clipnorm = float(tcfg.get("clipnorm", 1.0))
    beta_final = float(tcfg.get("beta_final", 1.0))
    beta_warmup_epochs = int(tcfg.get("beta_warmup_epochs", 80))
    resume = bool(tcfg.get("resume", True))
    run_name = str(tcfg.get("run_name", "run_current"))

    use_peaky_5050 = bool(tcfg.get("use_peaky_5050", True))
    peaky_quantile = float(tcfg.get("peaky_quantile", 0.80))
    seed = int(tcfg.get("seed", 2026))

    include_dense128 = bool(tcfg.get("include_dense128", True))

    global_best_dir = models_dir / "GLOBAL_BEST"
    runs_dir = models_dir / "runs"
    run_dir = runs_dir / run_name
    ensure_dir(runs_dir)
    ensure_dir(run_dir)

    enc_latest = run_dir / "enc_latest.weights.h5"
    dec_latest = run_dir / "dec_latest.weights.h5"
    enc_best = run_dir / "enc_best.weights.h5"
    dec_best = run_dir / "dec_best.weights.h5"
    epoch_file = run_dir / "last_epoch.txt"
    best_file = run_dir / "best_val_loss.json"
    logs_dir = run_dir / "logs"
    ensure_dir(logs_dir)

    # Load data
    X, _ = load_dataset(processed_dir)
    train_idx, val_idx, test_idx = load_splits(processed_dir)

    bundle = make_datasets(
        X=X,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=batch_size,
        steps_multiplier=steps_multiplier,
        use_peaky_5050=use_peaky_5050,
        peaky_quantile=peaky_quantile,
        seed=seed,
    )
    print(
        "[TRAIN] steps_per_epoch:",
        bundle.steps_per_epoch,
        "| val_steps:",
        bundle.val_steps,
    )

    # Build models
    latent_dim = 32
    encoder = build_encoder(
        input_bins=input_bins, latent_dim=latent_dim, include_dense128=include_dense128
    )
    decoder = build_decoder(input_bins=input_bins, latent_dim=latent_dim)
    sampler = Sampling()

    vae = VAE(encoder, decoder, sampler, beta_init=0.0, name="vae_psd_1024")
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
    vae.compile(optimizer=optimizer)

    # Build once
    _ = vae(tf.zeros((1, input_bins, 1), dtype=tf.float32), training=False)
    print("[MODEL] vae.built =", vae.built)

    # Resume or fresh start
    if not resume:
        epoch_file.write_text("0")
        if best_file.exists():
            best_file.unlink(missing_ok=True)
        initial_epoch = 0
        print("[START] resume=False -> training from epoch 0 with fresh weights.")
    else:
        initial_epoch = read_text_int(epoch_file, default=0)
        if enc_latest.exists() and dec_latest.exists():
            print("[RESUME] Loading enc/dec latest weights...")
            encoder.load_weights(enc_latest)
            decoder.load_weights(dec_latest)
        else:
            print(
                "[RESUME] No latest weights found -> training from epoch 0 with fresh weights."
            )
            initial_epoch = 0

    print("[TRAIN] initial_epoch =", initial_epoch, "| epochs =", epochs)

    callbacks = [
        BetaWarmUp(beta_final=beta_final, warmup_epochs=beta_warmup_epochs),
        SaveEncDec(
            enc_latest=enc_latest,
            dec_latest=dec_latest,
            enc_best=enc_best,
            dec_best=dec_best,
            epoch_file=epoch_file,
            best_file=best_file,
        ),
        keras.callbacks.TensorBoard(log_dir=str(logs_dir)),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=True),
        keras.callbacks.TerminateOnNaN(),
        # FIX: add mode="min" so Keras knows how to compare
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_recon_loss",
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    fit_kwargs = dict(
        x=bundle.ds_train,
        validation_data=bundle.ds_val,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    if use_peaky_5050:
        fit_kwargs["steps_per_epoch"] = bundle.steps_per_epoch
        fit_kwargs["validation_steps"] = bundle.val_steps

    _ = vae.fit(**fit_kwargs)

    run_best_val = float("inf")
    if best_file.exists():
        try:
            run_best_val = float(json.loads(best_file.read_text())["best_val_loss"])
        except Exception:
            run_best_val = float("inf")

    promote_to_global_best(
        run_best_val=run_best_val,
        run_dir=run_dir,
        enc_best=enc_best,
        dec_best=dec_best,
        global_dir=global_best_dir,
    )

    print("\n[DONE]")
    print("Run dir:", run_dir)
    print("Latest:", enc_latest, dec_latest)
    print("Best:", enc_best, dec_best)
    print("Global best dir:", global_best_dir)


if __name__ == "__main__":
    main()
