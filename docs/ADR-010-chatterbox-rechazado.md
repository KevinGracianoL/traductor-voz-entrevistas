# ADR-010 — Chatterbox Multilingual rechazado como TTS de producción

- **Estado:** Rechazado (2026-09-07)
- **Contexto:** Fase 2 necesita TTS offline con clonación de voz que conviva con Whisper int8 (~1 GB) en una GTX 1650 Ti de 4 GB, dentro del presupuesto ADR-003.
- **Candidato:** `chatterbox-tts==0.1.7` (MIT, Resemble AI). XTTS-v2 descartado antes: pide 4-6 GB él solo + licencia no-comercial (CPML).

## Procedencia exacta de la medición

- **Base:** `967e9d9` — su `requirements.txt` no contiene ningún pin TTS
  (verificado: 0 coincidencias), así que `pip install -r requirements.txt` resuelve limpio.
- **Set experimental:** los pins TTS (luego fijados en `a8ef6aa`) instalados con
  `--no-deps` para proteger torch/numpy base. Entre `967e9d9` y `a8ef6aa` no hay
  ningún cambio en `src/` (solo config), así que el código medido es el mismo.
- **Máquina:** Ryzen 5 4600H / GTX 1650 Ti 4 GB (driver 610.74) / Windows 11 + Python 3.11.

## Procedimiento autocontenido (reproduce la medición)

```powershell
git fetch origin
git checkout 967e9d9          # base limpia: requirements.txt SIN pins TTS
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
# Set TTS sin resolver dependencias (protege torch 2.13+cu132 / numpy 2.4.4):
pip install --no-deps chatterbox-tts==0.1.7 transformers==5.2.0 diffusers==0.29.0 `
  librosa==0.11.0 safetensors==0.5.3 conformer==0.3.2 s3tokenizer==0.3.0 `
  resemble-perth==1.0.1 omegaconf==2.3.1 einops==0.8.2 pykakasi==2.3.0 `
  pyloudnorm==0.2.0 spacy-pkuseg==1.0.1 onnx==1.22.0 numba==0.67.0 `
  llvmlite==0.49.0 audioread==3.1.0 scikit-learn==1.9.0 pooch==1.9.0 `
  soxr==1.1.0 lazy-loader==0.5 msgpack==1.2.2 antlr4-python3-runtime==4.9.3 `
  jaconv==0.5.0 ml-dtypes==0.5.4 narwhals==2.25.0 threadpoolctl==3.6.0 `
  decorator==5.3.1 cloudpickle==3.1.2
pip install tokenizers==0.22.2   # transformers 5.2.0 lo exige (repo trae 0.23.1)
python -m pip check              # ver salida real abajo: entorno experimental
python -c "import chatterbox.mtl_tts; print('import ok')"
```

`pip check` real durante la medición (entorno experimental fuera de metadata, no "limpio"):

```
chatterbox-tts 0.1.7 requires gradio, which is not installed.
diffusers 0.29.0 requires importlib-metadata, which is not installed.
pykakasi 2.3.0 requires deprecated, which is not installed.
transformers 5.2.0 requires typer-slim, which is not installed.
chatterbox-tts 0.1.7 has requirement numpy<2.0.0,>=1.24.0; python_version < "3.13", but you have numpy 2.4.4.
chatterbox-tts 0.1.7 has requirement torch==2.6.0; python_version < "3.14", but you have torch 2.13.0+cu132.
chatterbox-tts 0.1.7 has requirement torchaudio==2.6.0; python_version < "3.14", but you have torchaudio 2.11.0.
```

(Gradio es sólo GUI y no se usa; los 4 paquetes menores se instalaron después sin cambiar el resultado. La inferencia corre pese a los 3 overrides de versión — probado, no supuesto.)

## Entrada determinista: `ref.wav` sintético (sin voz personal)

```python
import math, struct, wave

sr, dur = 24000, 10.0
frames = bytearray()
for i in range(int(sr * dur)):
    t = i / sr
    v = 0.5 * math.sin(2 * math.pi * 220 * t) + 0.25 * math.sin(2 * math.pi * 440 * t)
    frames += struct.pack("<h", max(-32768, min(32767, int(v * 32767))))
with wave.open("ref.wav", "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    f.writeframes(bytes(frames))
```

## Smoke + validación + medición

```powershell
$env:PYTHONPATH = "src"
# En a8ef6aa existen src/traductor/tts/ y scripts/tts_worker.py (se archivaron después).
venv\Scripts\python.exe -c "
import sys, time; sys.path.insert(0, 'src')
import torch
from traductor.tts.chatterbox import cargar_modelo, sintetizar
t0=time.perf_counter(); m=cargar_modelo(device='cuda'); t1=time.perf_counter()
print('carga ms:', round((t1-t0)*1000), '| VRAM MB:', round(torch.cuda.memory_allocated()/1024**2))
def intervalo(t0):
    torch.cuda.synchronize()  # drena trabajo previo y pendiente: se mide cómputo, no lanzamiento
    return (time.perf_counter()-t0)*1000.0

torch.cuda.synchronize()  # también ANTES de arrancar el cronómetro

for i,txt in enumerate(['hola mundo, esto es una prueba','cuéntame sobre un bug difícil']):
    a=time.perf_counter(); wav,sr=sintetizar(txt,'ref.wav',m)
    print(f'frase{i+1} ms:', round(intervalo(a)), 'sr:', sr)
print('VRAM pico MB:', round(torch.cuda.max_memory_allocated()/1024**2))
from traductor.tts.worker import guardar_wav
from pathlib import Path
guardar_wav(Path('out_smoke.wav'), wav, sr)
"
# Validar WAV con stdlib (el worker del commit medido lo guarda igual):
venv\Scripts\python.exe -c "
import wave
f = wave.open('out_smoke.wav','rb')
print(f.getnchannels(), f.getframerate(), f.getnframes())
"
```

Método: latencia con `time.perf_counter()` alrededor de cada fase; VRAM con
`torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()` (pico del proceso).

## Resultados (salida real)

| Qué | Resultado (con `synchronize`, worktree `967e9d9`) |
|---|---|
| Carga modelo (cache HF) | 23 442 ms |
| VRAM tras carga | 3 075 MB |
| Frase 1 (cold) | 53 227 ms |
| Frase 2 (warm) | 17 397 ms |
| VRAM pico | 3 712 MB |
| WAV salida | mono, 24 kHz, 83 520 frames (~3.5 s), válido |

Nota: una primera medición sin `synchronize()` dio 26/11 s (subestima: mide el
lanzamiento de kernels, no su término). Los valores de arriba, sincronizados,
son los válidos.

## Decisión

**Rechazado.** Warm 17 397 ms = 58× el presupuesto TTS (300 ms) y 8.7× el techo total (2 s);
3 712 MB de 4 096 MB no dejan espacio para Whisper. Alternativas honestas: turnos
(descargar un modelo para cargar el otro), GPU mayor, o un candidato que quepa con
Whisper (<1 GB VRAM residente y <300 ms/frase).

## Trazabilidad

- PR #10 (historial: adapter + worker + tests del prototipo, eliminados de `src/` al rechazar)
- Prototipo recuperable: `git checkout 967e9d9 -- src/traductor/tts scripts/tts_worker.py tests/test_tts_*`
- Pesos del modelo: cache de HuggingFace (`~/.cache/huggingface`), no versionados
