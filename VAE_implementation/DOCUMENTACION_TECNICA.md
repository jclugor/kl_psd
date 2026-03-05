# VAE_implementation

Pipeline completo para compresion de espectro PSD (1024 bins) usando VAE, exportacion INT8 para edge (Raspberry) y transmision UDP al servidor.

## 1. Objetivo del proyecto

Este proyecto comprime senales PSD en dos pasos:

1. `Encoder VAE` transforma cada PSD de 1024 bins en un vector latente `mu` de 32 dimensiones.
2. Ese latente se cuantiza a `int8`, se empaqueta y se envia por red (UDP) con un formato binario ligero.

En servidor:

1. Se recibe y desempaqueta `mu_int8`.
2. Se de-cuantiza a `float`.
3. El `Decoder` reconstruye la PSD aproximada de 1024 bins.

Resultado: mucho menos trafico por frame, manteniendo forma espectral util.

## 2. Contexto teorico

## 2.1 VAE (Variational Autoencoder)

El encoder produce dos vectores por frame de entrada `x`:

- `mu(x)`
- `logvar(x)`

Distribucion latente aproximada:

`q(z|x) = N(mu, diag(exp(logvar)))`

Perdida usada en entrenamiento (`02_train.py`):

- Reconstruccion (MSE suma por bin):
  - `recon = mean_batch(sum_bins((x - x_hat)^2))`
- Regularizacion KL:
  - `kl = mean_batch(sum_lat(-0.5*(1 + logvar - mu^2 - exp(logvar))))`
- Total:
  - `loss = recon + beta * kl`

## 2.2 Inferencia deterministica

Aunque un VAE clasico samplea `z`, aqui para inferencia se usa `z ~= mu` (deterministico). Esto reduce blur y estabiliza picos espectrales, alineado con edge deployment.

## 2.3 Cuantizacion INT8

`04_export_tflite.py` exporta `encoder_mu_only` a TFLite INT8 (PTQ). El sistema usa `scale` y `zero_point` del tensor de salida para recuperar `mu_float` en servidor:

`mu_float = (mu_int8 - zero_point) * scale`

## 2.4 Codificacion temporal

`07_pack_unpack.py` codifica por bloque:

- `mu0` absoluto
- deltas `mu[t]-mu[t-1]` (int8 con clip)

Esto explota correlacion temporal entre frames para mejorar compresion.

## 3. Estructura del proyecto

```text
VAE_implementation/
  configs/
    vae_default.yaml
  scripts/
    00_get_data.ps1
    01_preprocess.py
    02_train.py
    03_eval.py
    04_export_tflite.py
    05_entropy_stats.py
    06_codec_benchmark.py
    07_pack_unpack.py
    08_udp_sender.py
    08_udp_receiver.py
    09_udp_sender_indexed.py
    09_udp_receiver_plot_compare.py
    10_udp_sender_prod.py
    10_udp_receiver_prod.py
  models/
    GLOBAL_BEST/
    runs/run_current/
  VAE_test.ipynb
```

## 4. Requisitos de entorno

Desde la raiz `kl_psd`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install tensorflow pandas pyyaml
```

Nota: `requirements.txt` actual no incluye explicitamente `tensorflow`, `pandas`, `pyyaml`, pero los scripts los requieren.

## 5. Configuracion principal

Archivo: `VAE_implementation/configs/vae_default.yaml`

Campos clave:

- `paths.raw_dataset_dir`: dataset crudo CSV
- `paths.processed_dir`: salida preprocesada `.npz` y `splits`
- `paths.models_dir`: pesos, reportes y exportaciones
- `preprocess.target_bins`: dimension final (1024)
- `preprocess.normalize_mode`: `global_minmax | per_frame_minmax | none`
- `preprocess.split`: particion train/val/test
- `train.batch_size`, `train.epochs`, `train.lr`, `train.beta_final`, `train.beta_warmup_epochs`, `train.resume`, `train.run_name`

## 6. Pipeline recomendado (paso a paso)

Ejecutar desde la raiz del repo (`kl_psd`).

## 6.1 Descargar/actualizar dataset

```powershell
.\VAE_implementation\scripts\00_get_data.ps1
```

## 6.2 Preprocesar dataset

```powershell
python VAE_implementation/scripts/01_preprocess.py --config VAE_implementation/configs/vae_default.yaml
```

Salidas:

- `data/processed/psd_1024/dataset_psd_1024_norm.npz`
- `data/processed/psd_1024/metadata.csv`
- `data/processed/psd_1024/splits/*.npy`

## 6.3 Entrenar VAE

```powershell
python VAE_implementation/scripts/02_train.py --config VAE_implementation/configs/vae_default.yaml
```

Que hace:

- Entrena encoder/decoder con warmup de beta.
- Guarda `latest` y `best` por run.
- Compara mejor run contra `GLOBAL_BEST` y promueve si mejora.

Artefactos de run:

- `models/runs/<run_name>/enc_latest.weights.h5`
- `models/runs/<run_name>/dec_latest.weights.h5`
- `models/runs/<run_name>/enc_best.weights.h5`
- `models/runs/<run_name>/dec_best.weights.h5`
- `models/runs/<run_name>/history.csv`

## 6.4 Evaluar reconstruccion

```powershell
python VAE_implementation/scripts/03_eval.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

Salidas en `eval/`:

- `metrics.json`
- `recon_random.png`
- `recon_peaky.png`
- `hist_values.png`
- `waterfall_orig.png`
- `waterfall_recon.png`

## 6.5 Exportar encoder INT8 (TFLite)

```powershell
python VAE_implementation/scripts/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

Salidas:

- `encoder_mu_int8.tflite`
- `encoder_mu.keras` (si el entorno lo permite)
- `export_report.json`

## 6.6 Analisis de entropia latente

```powershell
python VAE_implementation/scripts/05_entropy_stats.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --split train --keyframe_every 30
```

Mide:

- `H(mu)` y `H(delta)` en bits/simbolo
- cota teorica de bytes/frame

Salida: `entropy_report.json`.

## 6.7 Benchmark de codec end-to-end

```powershell
python VAE_implementation/scripts/06_codec_benchmark.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --split train --keyframe_every 30
```

Compara `none`, `zlib`, `lzma`, `bz2` en:

- bytes/frame
- tiempos de compresion/descompresion
- unpack
- tiempo end-to-end
- metrica de reconstruccion

Salida: `codec_report.json`.

## 7. Explicacion por script

## 7.1 `00_get_data.ps1`

- Clona o actualiza el repo de dataset CSV en `data/raw/...`.
- Especifico para PowerShell/Windows.

## 7.2 `01_preprocess.py`

- Lee CSV y parsea columna `pxx`.
- Reescala a 1024 bins (promedio o interpolacion).
- Normaliza segun config.
- Genera `.npz`, `metadata.csv` y splits.

## 7.3 `02_train.py`

- Define arquitectura encoder/decoder 1D Conv.
- Entrena VAE con `recon + beta*KL`.
- Warmup de beta.
- Resume robusto (`last_epoch`, `latest weights`).
- Promocion automatica a `GLOBAL_BEST`.

## 7.4 `03_eval.py`

- Carga pesos (`run best`, `run latest` o `GLOBAL_BEST`).
- Evalua en test split.
- Reporta perdida total y metricas de picos (`topk_peak_mse`, `peak_bias`).
- Exporta plots de overlays, histograma y waterfall.

## 7.5 `04_export_tflite.py`

- Toma encoder entrenado.
- Crea modelo `encoder_mu_only`.
- PTQ INT8 con `representative_dataset`.
- Valida error `mu_float` vs `mu_tflite`.

## 7.6 `05_entropy_stats.py`

- Corre encoder INT8 sobre un split.
- Calcula histogramas de simbolos para `mu` y `delta(mu)`.
- Entrega limites teoricos de tasa (bytes/frame).

## 7.7 `06_codec_benchmark.py`

- Benchmark completo por etapas:
  - encode INT8
  - packetize delta
  - compress/decompress
  - unpack
  - recon decoder
- Es la referencia principal de tradeoff tasa-tiempo.

## 7.8 `07_pack_unpack.py`

- Define formato binario de paquete UDP (`KLP1`).
- Header fijo de 24 bytes (big-endian).
- Funciones:
  - `encode_mu_block` / `decode_mu_block`
  - `pack_packet` / `unpack_packet`
  - `decode_packet_to_mu`
- Incluye `--self_test`.

## 7.9 `08_udp_sender.py`

- Simulador edge basico.
- Toma PSD del split, infiere `mu_int8`, empaqueta y envia UDP.

## 7.10 `08_udp_receiver.py`

- Receptor basico.
- Decodifica paquetes y opcionalmente reconstruye PSD.
- Usa de-cuantizacion placeholder (`/128`) en modo reconstruct (mejorado en etapa 9).

## 7.11 `09_udp_sender_indexed.py`

- Sender mejorado para evaluacion.
- Incluye indices globales de frames en payload (`meta v1`) para comparacion exacta contra originales en servidor.

## 7.12 `09_udp_receiver_plot_compare.py`

- Receiver de evaluacion completo.
- De-cuantiza con `scale/zp` reales del TFLite.
- Reconstruye y compara contra PSD original por indice.
- Genera overlays, waterfalls y `udp_metrics.json`.

## 7.13 `10_udp_sender_prod.py`

- Sender de produccion (edge/Raspberry).
- Envia solo header + bloque `mu` comprimido (sin indices/debug).
- Soporta fuente `dataset` o `npy_dir`.

## 7.14 `10_udp_receiver_prod.py`

- Receptor de produccion (server side) para paquetes sin indices.
- Decodifica `mu_int8` con `decode_packet_to_mu`.
- De-cuantiza usando `scale/zp` reales del `encoder_mu_int8.tflite`.
- Reconstruye PSD con `dec_best.weights.h5`.
- Opcional: invierte normalizacion a escala original (`--invert_norm_to_original`) usando `gmin/gmax`.
- Guarda en `<model_dir>/udp_prod/`:
  - `waterfall_recon.png`
  - `psd_last.png`
  - `recon_last_psd.npy`
  - `udp_prod_metrics.json`

## 8. Pruebas UDP recomendadas

## 8.1 Prueba basica (etapa 8)

Terminal A:

```powershell
python VAE_implementation/scripts/08_udp_receiver.py --config VAE_implementation/configs/vae_default.yaml --bind_ip 127.0.0.1 --port 5005 --use_global_best --reconstruct
```

Terminal B:

```powershell
python VAE_implementation/scripts/08_udp_sender.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --split test --block_len 30 --n_blocks 50 --zlib_level 1
```

## 8.2 Prueba con comparacion/plots (etapa 9)

Terminal A:

```powershell
python VAE_implementation/scripts/09_udp_receiver_plot_compare.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --idle_stop_s 5
```

Terminal B:

```powershell
python VAE_implementation/scripts/09_udp_sender_indexed.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

## 8.3 Prueba de produccion (etapa 10)

Terminal A (receiver servidor):

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 0.0.0.0 --port 5005 --idle_stop_s 10 --save_every_packets 10
```

Terminal B (sender edge/simulador):

```powershell
python VAE_implementation/scripts/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

## 9. Estado actual (artefactos encontrados)

En `models/GLOBAL_BEST` ya existen:

- `enc_best.weights.h5`, `dec_best.weights.h5`
- `encoder_mu_int8.tflite`
- `export_report.json`
- `entropy_report.json`
- `codec_report.json`
- `udp_eval/udp_metrics.json` y plots

Valores reportados (referencia rapida):

- `best_val_loss`: `0.1272793114`
- PTQ `mu_mae`: `0.0014709432`
- codec `zlib level 1`: ~`28.13 bytes/frame`
- UDP eval (3 paquetes x 30 frames): ~`34.91 bytes/frame`

## 10. Troubleshooting rapido

- Error `No .npz found`: ejecutar `01_preprocess.py`.
- Error `TFLite model not found`: ejecutar `04_export_tflite.py`.
- Error `Decoder weights not found`: entrenar (`02_train.py`) o usar `--use_global_best` con pesos disponibles.
- Sin paquetes UDP: validar `bind_ip`, `dest_ip`, puerto y firewall.
- Shape distinta de 1024 en `npy_dir` para `10_udp_sender_prod.py`: cada archivo debe ser `(1024,)`.

## 11. Notas de evolucion

- El pipeline de produccion (sender + receiver etapa 10) ya esta cerrado.
- Recomendado: mover dependencias reales a un `requirements` especifico del modulo VAE (incluyendo TensorFlow, pandas y PyYAML).
- Recomendado: documentar `VAE_test.ipynb` cuando se congele su flujo.
