<div align="center">

# 🎙️ Traductor de Voz en Tiempo Real
### ES ↔ EN para entrevistas de trabajo — local, privado, verificable

> **Pipeline:** `audio → VAD → ASR → traducción → TTS → audio` — con texto siempre visible para detectar errores antes de responder.

[![CI](https://github.com/KevinGracianoL/traductor-voz-entrevistas/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinGracianoL/traductor-voz-entrevistas/actions)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%2013.2-EE4C2C?style=flat-square&logo=pytorch)
![Ruff](https://img.shields.io/badge/Ruff-checked-000000?style=flat-square)
![mypy strict](https://img.shields.io/badge/mypy-strict-2A6DB5?style=flat-square)
![coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square)
![mutants 139/139](https://img.shields.io/badge/mutants-139%2F139-brightgreen?style=flat-square)
![License MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)

**Portafolio → [KevinGracianoL](https://github.com/KevinGracianoL) · Proyecto guía [entrenamiento-dev](https://github.com/KevinGracianoL/entrenamiento-dev) · Hecho para entrevistas reales, no demos**

[Instalación](#-instalación) · [Uso](#️-uso) · [Arquitectura](#️-arquitectura) · [Calidad](#-calidad-5-gates-1-contrato) · [ADRs](#-decisiones-adr)

</div>

---

<div align="center">

### ✨ Demo

*Teleprompter ES+EN en vivo durante Zoom/Meet — [https://traductor-demo.kevingraciano.dev](https://traductor-demo.kevingraciano.dev) (feed demo) + `http://localhost:8000` (pipeline real).*

> **Privacidad:** el pipeline real corre en tu laptop. El audio nunca sale de la máquina. La demo en VM es solo feed estático.

</div>

---

## 🎯 Por qué este proyecto

En una entrevista en inglés, un error de traducción no es un bug — es la respuesta equivocada. Este traductor prioriza **texto verificable** sobre voz sintética indistinguible, y **latencia medida en tu hardware** sobre benchmarks de RTX 4090.

**Para reclutadores:** cada línea es testeable, cada decisión tiene evidencia (ADRs), y `mutmut` demuestra que los tests no solo ejecutan líneas — las verifican.

---

## ✅ Estado actual

| Paso | Estado | Qué entrega |
|---|---|---|
| **1 — Hardware** | ✅ | `torch.cuda.is_available()`, VRAM libre/total, `RealtimeSTT` `tiny` `int8` (TU117) |
| **2 — Traducción** | ✅ | `argos-translate` `EN↔ES` offline en CPU, `ARGOS_COMPUTE_TYPE=default` |
| **3 — Medición** | ✅ | `src/traductor/latencia/` (reloj inyectable, `p50` mediana, `p95=None` si `n<20`) |
| **4 — Audio virtual** | ✅ | `src/traductor/audio/virtual.py` (ruta por nombre, VB-CABLE, 139/139) |
| **5 — Teleprompter** | ✅ | `src/traductor/ui/` en `localhost:8000` + Caddy demo en `fuerzafiel` |

---

## 🏗️ Arquitectura

```mermaid
flowchart LR
    M1[🎤 Micrófono] --> VAD
    M2[🔊 Audio virtual] --> VAD
    VAD --> ASR[Whisper int8<br/>400-600 ms]
    ASR --> TR[argos CPU<br/>~150 ms]
    TR --> TP[📺 Teleprompter<br/>ES+EN]
    TR --> TTS[TTS 300 ms]
    TTS --> SPK[🔊 Altavoz virtual]
```

**Presupuesto ADR-003:** techo **1.5–2 s** total. Si no cabe, se recorta calidad, nunca latencia.

---

## 🛠️ Stack

| Capa | Tech | Nota |
|---|---|---|
| **ASR** | `RealtimeSTT` + `faster-whisper` `int8` | TU117 sin Tensor Cores → FP16 emulado, INT8 en cores enteros |
| **Traducción** | `argos-translate` + `ctranslate2` | Offline, CPU, gratis |
| **Medición** | `time.perf_counter` inyectable | Testeable sin hardware |
| **Calidad** | `ruff` `mypy --strict` `pytest` `mutmut` | 100% cov, 139/139 mutantes |

---

## 💻 Requisitos

- **Hardware referencia:** Ryzen 5 4600H / GTX 1650 Ti 4 GB (TU117) / 24 GB RAM — 0 ms de red
- Python 3.11+, CUDA 13.2, Windows 10/11, micrófono
- Descartados: `fuerzafiel` VM (947 MB RAM), server Medellín i5-3230M (sin AVX2), GPU remota MX (peaje por chunk)

---

## 🚀 Instalación

```powershell
# 1. Entorno
python -m venv venv; .\venv\Scripts\Activate.ps1

# 2. Deps (PyTorch con CUDA)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
python setup_dlls.py

# 3. Verificar que local == CI
ruff check .; ruff format --check .; mypy .; pytest
```

---

## ▶️ Uso

```powershell
python scripts/verificar_hardware.py  # → CUDA: True | GTX 1650 Ti | VRAM 3.2/4 GB | mic → texto
python scripts/demo_traduccion.py     # → "Tell me about a hard bug..." ↔ "Háblame de un bug..."
```

```python
from traductor.latencia.presupuesto import cabe_en_presupuesto
from traductor.latencia.medidor import medir_tiempo
from traductor.traduccion.argos import traducir
import time

# ¿Cabe en 1 s?
cabe_en_presupuesto({"asr": 500, "traduccion": 150, "tts": 300}, 1000)  # True

# Medir (tests usan reloj falso, prod usa perf_counter)
texto, ms = medir_tiempo(lambda: traducir("hello", "en", "es"), clock=time.perf_counter)
```

---

## 📁 Estructura

```
├── src/traductor/
│   ├── hardware/cuda.py        # verifica GPU/VRAM
│   ├── audio/captura.py        # mic → texto (RealtimeSTT)
│   ├── audio/virtual.py        # ruta determinista por nombre
│   ├── traduccion/argos.py     # EN↔ES offline
│   └── latencia/
│       ├── presupuesto.py      # ¿cabe? ¿quién es más lento?
│       └── medidor.py          # reloj inyectable, p50/p95 honesto
├── scripts/
│   ├── verificar_hardware.py   # wrapper fino hardware
│   └── demo_traduccion.py      # wrapper fino traducción
├── tests/                      # 39 tests, 100% cov, 139/139 mutantes
├── .github/workflows/ci.yml
├── pyproject.toml              # ruff + mypy strict + pytest --cov-fail-under=90
└── requirements.txt
```

---

## ✅ Calidad — 5 gates, 1 contrato

| Pregunta | Herramienta | Config |
|---|---|---|
| ¿Legible/sin bug? | **ruff** | `select = ["E","F","B","SIM","UP","I","S"]` |
| ¿Tipos encajan? | **mypy --strict** | `ignore_missing_imports` para RealtimeSTT |
| ¿Hace lo que digo? | **pytest** | `--cov-fail-under=90` en `pyproject.toml` |
| ¿Qué no probé? | **coverage** | `100%` |
| ¿Detectaría un bug? | **mutmut** | `139/139` con `pytest_add_cli_args = ["--no-cov"]` |

> `mutmut` necesita `fork` → WSL. En CI (Ubuntu) el gate falla si `survived > 0`. Verificado rompiendo `<=`→`<` y `*1000`→`/1000` a mano.

---

## 📋 ADRs — decisiones con evidencia

| # | Decisión | Por qué |
|---|---|---|
| 001 | Cascada, no end-to-end | Texto verificable > latencia mínima |
| 002 | Audio virtual a nivel SO | Funciona con cualquier Meet/Zoom sin API |
| 003 | Techo 1.5–2 s | Recorta calidad, nunca latencia |
| 004 | INT8, no FP16 | TU117 sin Tensor Cores, FP16 emulado |
| 005 | Local, no remoto | AVX2+CUDA+0 ms gana a geografía |
| 006 | Sobre RealtimeSTT | VAD/ASR commodity, nosotros orquestamos |
| 007 | Teleprompter primero | Semanas vs meses, honestidad en entrevista |
| 008 | Fallback automático | Una entrevista no es un log |
| 009 | Dirección por fuente | Determinista, 0 ms, sin detector que falle en code-switching |

---

## 🗺️ Roadmap

- [x] **Paso 1 — Hardware** — CUDA + VRAM + mic → texto (`src/traductor/hardware/`, `audio/captura.py`)
- [x] **Paso 2 — Traducción** — `argos` offline EN↔ES (`src/traductor/traduccion/`)
- [x] **Paso 3 — Medición** — presupuesto + medidor honesto (`p95`, `exc.elapsed_ms`, `139/139` mutantes)
- [x] **Paso 4 — Audio virtual** — ruta por nombre, VB-CABLE (`src/traductor/audio/virtual.py`)
- [ ] **Paso 5 — Teleprompter** — UI en vivo + deploy micro VM (nginx + TLS) — *siguiente*
- [ ] Fase 2 — TTS opcional
- [ ] Fase 3 — Conversión de voz (timbre de Kevin)

---

<div align="center">

**Hecho por [Kevin Graciano](https://github.com/KevinGracianoL) — aprendiendo en público, midiendo en mi hardware.**

*Cero credenciales y cero audios de entrevistas reales en el historial.*

</div>
