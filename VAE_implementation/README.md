# VAE_implementation

Guia rapida para ejecutar el pipeline de compresion PSD con VAE (version estable sin Kalman).

Documentacion completa:
- `VAE_implementation/DOCUMENTACION_TECNICA.md`
- `VAE_implementation/IMPLEMENTACION_REAL.md`

## Flujo rapido

Desde la raiz del repo (`kl_psd`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para edge con fuente SDR por ZMQ:

```powershell
pip install pyzmq
```

1. Descargar dataset:

```powershell
.\VAE_implementation\scripts\00_get_data.ps1
```

2. Preprocesar:

```powershell
python VAE_implementation/scripts/01_preprocess.py --config VAE_implementation/configs/vae_default.yaml
```

3. Entrenar:

```powershell
python VAE_implementation/scripts/02_train.py --config VAE_implementation/configs/vae_default.yaml
```

4. Evaluar:

```powershell
python VAE_implementation/scripts/03_eval.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

5. Exportar encoder INT8 (TFLite):

```powershell
python VAE_implementation/scripts/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

## Produccion (local test)

Receiver:

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original
```

Sender (simulado con dataset):

```powershell
python VAE_implementation/scripts/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

Comparacion (guardar original + reconstruccion):

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
```

## Produccion (SDR real)

Servidor (receiver):

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 0.0.0.0 --port 5005 --save_every_packets 10 --idle_stop_s 10 --invert_norm_to_original
```

Edge (bridge ZMQ PSD -> UDP):

```bash
python VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

Si la PSD viene en otra llave JSON:

```bash
--psd_key <nombre_llave>
```

## Si ya tienes modulo PSD, que falta

- Validar que el emisor PSD conecte a `ipc:///tmp/ane_psd.ipc` (PAIR).
- Confirmar escala de entrada (original dB/dBm o `--already_normalized`).
- Verificar red UDP y firewall.
- Definir ejecucion continua (servicio + logs).

## Notas

- Ruta principal actual: `10_udp_sender_prod.py` + `10_udp_receiver_prod.py`.
- Integracion SDR en vivo: `11_edge_hackrf_psd_zmq_to_udp.py`.
- Etapas `08` y `09` se mantienen para pruebas/diagnostico.
- Config principal: `VAE_implementation/configs/vae_default.yaml`.

