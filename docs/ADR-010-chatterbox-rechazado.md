# ADR-010 — Chatterbox Multilingual rechazado como TTS de producción

- **Estado:** Rechazado (2026-09-07)
- **Contexto:** Fase 2 necesita TTS offline con clonación de voz que conviva con Whisper int8 (~1 GB) en una GTX 1650 Ti de 4 GB, dentro del presupuesto ADR-003.
- **Candidato:** `chatterbox-tts==0.1.7` (MIT, Resemble AI). XTTS-v2 descartado antes: pide 4-6 GB él solo + licencia no-comercial (CPML).

## Entorno medido

- Laptop: Ryzen 5 4600H / GTX 1650 Ti 4 GB (driver 610.74) / Windows 11 + Python 3.11
- venv principal + `pip install --no-deps` del set TTS (protege torch/numpy base):
  `chatterbox-tts==0.1.7`, `transformers==5.2.0`, `diffusers==0.29.0`, `librosa==0.11.0`,
  `safetensors==0.5.3`, `conformer==0.3.2`, `s3tokenizer==0.3.0`, `resemble-perth==1.0.1`,
  `omegaconf==2.3.1`, `einops==0.8.2`, `pykakasi==2.3.0`, `pyloudnorm==0.2.0`,
  `spacy-pkuseg==1.0.1`, `onnx==1.22.0`, `numba==0.67.0`, `llvmlite==0.49.0`,
  `audioread==3.1.0`, `scikit-learn==1.9.0`, `pooch==1.9.0`, `soxr==1.1.0`,
  `lazy-loader==0.5`, `msgpack==1.2.2`, `antlr4-python3-runtime==4.9.3`,
  `jaconv==0.5.0`, `ml-dtypes==0.5.4`, `narwhals==2.25.0`, `threadpoolctl==3.6.0`,
  `decorator==5.3.1`, `cloudpickle==3.1.2`, `deprecated==1.3.1`, `typer-slim==0.24.0`,
  `importlib-metadata==9.0.1`, `zipp==4.1.0`, salvo `tokenizers==0.22.2`
  (transformers 5.2.0 lo exige; el resto corre sobre torch 2.13+cu132 y numpy 2.4.4,
  más laxos que los pins de upstream).
- Commit medido: `bf785c3` (PR #10). Pesos desde HuggingFace cache local.

## Comandos reproducibles

```powershell
$env:PYTHONPATH = "src"
venv\Scripts\python.exe -m pip check        # limpio salvo gradio (solo GUI) + 3 overrides probados
venv\Scripts\python.exe -c "import chatterbox.mtl_tts; print('import ok')"
# Smoke (ref: seno 220 Hz 10 s, 24 kHz, mono — sintético, sin voz personal):
echo '{"texto": "hola mundo", "ref_audio": "ref.wav"}' | python scripts/tts_worker.py --out-dir out_smoke
```

## Medición (GPU, `time.perf_counter`, `torch.cuda.memory_allocated/max_memory_allocated`)

| Qué | Resultado |
|---|---|
| Carga modelo (cache HF) | 22 579 ms |
| VRAM tras carga | 3 075 MB |
| Frase 1 (cold) | 26 119 ms |
| Frase 2 (warm) | 11 613 ms |
| VRAM pico | 3 712 MB |
| WAV salida | mono, 24 kHz, 81 600 frames (~3.4 s), válido |

## Decisión

**Rechazado.** Warm 11 613 ms = 38.7× el presupuesto TTS (300 ms) y 5.8× el techo total (2 s);
3 712 MB de 4 096 MB no dejan espacio para Whisper. Alternativas honestas: turnos
(descargar un modelo para cargar el otro), GPU mayor, o un candidato que quepa con
Whisper (<1 GB VRAM residente y <300 ms/frase).

## Trazabilidad

- PR #10 (historial: adapter + worker + tests del prototipo, eliminados de `src/` al rechazar)
- `requirements-tts.txt` eliminado con el prototipo; receta archivada en este ADR
- `pip check` limpio salvo avisos documentados arriba
