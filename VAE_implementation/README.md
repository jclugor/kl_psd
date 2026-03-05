# VAE_implementation

Guia rapida para ejecutar el pipeline de compresion PSD con VAE.

Documentacion completa (teoria, arquitectura, detalle script por script):
- `VAE_implementation/DOCUMENTACION_TECNICA.md`

## Flujo rapido

Desde la raiz del repo (`kl_psd`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Raspberry/edge puedes usar runtime liviano opcional:

```powershell
pip install tflite-runtime
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

6. Entropia y benchmark:

```powershell
python VAE_implementation/scripts/05_entropy_stats.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --split train --keyframe_every 30
python VAE_implementation/scripts/06_codec_benchmark.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --split train --keyframe_every 30
```

7. Probar formato de paquete (wire format, opcional):

```powershell
python VAE_implementation/scripts/07_pack_unpack.py --self_test --n_frames 120 --block_len 30 --zlib_level 1
```

8. UDP basico (etapa 8, opcional):

Receiver (Terminal A):

```powershell
python VAE_implementation/scripts/08_udp_receiver.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --reconstruct
```

Sender (Terminal B):

```powershell
python VAE_implementation/scripts/08_udp_sender.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

## UDP (test local)

Receiver (Terminal A):

```powershell
python VAE_implementation/scripts/09_udp_receiver_plot_compare.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --idle_stop_s 5
```

Sender (Terminal B):

```powershell
python VAE_implementation/scripts/09_udp_sender_indexed.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

## UDP (produccion, etapa 10)

Receiver prod (servidor, Terminal A):

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 0.0.0.0 --port 5005 --idle_stop_s 10 --save_every_packets 10
```

Sender prod (edge/simulador, Terminal B):

```powershell
python VAE_implementation/scripts/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

## Notas

- `10_udp_sender_prod.py` esta listo para envio productivo.
- `10_udp_receiver_prod.py` ya esta implementado y guarda salidas en `udp_prod/`.
- Config principal: `VAE_implementation/configs/vae_default.yaml`.
