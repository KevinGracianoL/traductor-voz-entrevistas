# Traductor de Voz en Tiempo Real — ES ↔ EN para Entrevistas

> Pipeline local en tu laptop: **audio → VAD → ASR → traducción → TTS → audio**. Texto siempre visible para verificar errores antes de responder.

[![CI](https://github.com/KevinGracianoL/traductor-voz-entrevistas/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinGracianoL/traductor-voz-entrevistas/actions)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Ruff](https://img.shields.io/badge/ruff-checked-green)
![mypy strict](https://img.shields.io/badge/mypy-strict-blue)
![coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)
![mutants 84/84](https://img.shields.io/badge/mutants-84%2F84-brightgreen)

Proyecto real de [entrenamiento-dev](https://github.com/KevinGracianoL/entrenamiento-dev) — cada decisión con evidencia y cada línea testeable.

---

## ✨ Qué hace hoy

- **Paso 1 — Verifica hardware:** `torch.cuda.is_available()`, VRAM libre vs total, micrófono → texto con `RealtimeSTT` (`tiny`, `int8` en GTX 1650 Ti TU117).
- **Paso 2 — Traduce offline:** `argos-translate` bidireccional `EN↔ES` en CPU, sin red. `ARGOS_COMPUTE_TYPE=default` (con `auto` generaba basura en `es→en`).
- **Paso 3 — Mide y decide:** `latencia/presupuesto.py` (¿cabe en el techo?) + `latencia/medidor.py` (¿cuánto tarda de verdad? con reloj inyectable, `p50` mediana, `p95=None` si `n<20`).

Próximo: audio virtual en Windows → teleprompter en vivo.

---

## 🏗️ Arquitectura

```
[ Micrófono ] ─┐
               ├─► VAD ─► ASR (Whisper int8) ─► Traducción (argos CPU) ─► [ Teleprompter ES+EN ]
[ Audio virtual]┘                                          └─► TTS (opcional) ─► [ Altavoz virtual ]
```

**Presupuesto ADR-003:** techo 1.5–2 s total. ASR 400–600 ms, traducción ~150 ms, TTS ~300 ms. Si no cabe, se recorta calidad, nunca latencia.

---

## 💻 Requisitos

- **Laptop Kevin:** Ryzen 5 4600H / GTX 1650 Ti 4 GB (TU117, sin Tensor Cores) / 24 GB RAM — cómputo local, 0 ms de red.
- Python 3.11+, CUDA 13.2, Windows 10/11
- Micrófono funcional

> Descartados como cómputo: micro VM `fuerzafiel` (947 MB RAM, sin GPU), server Medellín i5-3230M (sin AVX2), GPU remota México (peaje de red por chunk).

---

## 🚀 Instalación

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
python setup_dlls.py

# Verificar gates (local = CI)
ruff check .; ruff format --check .; mypy .; pytest
```

---

## ▶️ Uso

```powershell
python paso1_verificar.py   # CUDA + VRAM + "the blue dog eats screws" → texto
python paso2_traducir.py    # "Tell me about a hard bug..." ↔ "Háblame de un bug..."
```

```python
from latencia.presupuesto import cabe_en_presupuesto, degradar_configuracion
from latencia.medidor import medir_tiempo, resumen_estadisticas

# ¿Cabe?
cabe_en_presupuesto({"asr": 500, "traduccion": 150, "tts": 300}, 1000)  # True

# Medir con reloj falso (tests) o real (producción)
res, ms = medir_tiempo(lambda: traducir("hello", "en", "es"), clock=time.perf_counter)
```

---

## 📁 Estructura

```
.
├── paso1_verificar.py      # Paso 1: CUDA + RealtimeSTT
├── paso2_traducir.py       # Paso 2: argos ES↔EN
├── latencia/
│   ├── presupuesto.py      # Decisión: ¿cabe? ¿quién es más lento? (sin torch)
│   └── medidor.py          # Medición: reloj inyectable, p50/p95 honesto
├── tests/
│   ├── test_presupuesto.py # 13 tests, frontera inclusiva, converge
│   └── test_medidor.py     # 11 tests, excepción con elapsed_ms, p95 None
├── .github/workflows/ci.yml
├── pyproject.toml          # ruff + mypy strict + pytest --cov-fail-under=90 + mutmut --no-cov
└── requirements.txt
```

---

## ✅ Calidad — 5 gates, 1 contrato

| Pregunta | Herramienta | Gate |
|---|---|---|
| ¿Legible/sin bug obvio? | `ruff` | `select = ["E","F","B","SIM","UP","I","S"]` |
| ¿Tipos encajan? | `mypy --strict` | `ignore_missing_imports` para RealtimeSTT |
| ¿Hace lo que digo? | `pytest` | `--cov-fail-under=90` en `pyproject.toml` |
| ¿Qué no probé? | `coverage` | `100%` |
| ¿Tests detectarían un bug? | `mutmut` | `84/84` con `pytest_add_cli_args = ["--no-cov"]` |

> `mutmut` no corre en Windows nativo (necesita `fork` → WSL). En CI corre en Ubuntu y el gate falla si `survived > 0`.

---

## 📋 ADRs

| # | Decisión | Por qué |
|---|---|---|
| 001 | Pipeline en cascada, no end-to-end | Texto verificable antes de responder |
| 002 | Audio virtual a nivel SO | Funciona con Zoom/Teams/Meet sin API |
| 003 | Techo 1.5–2 s | Si no cabe, recorta calidad |
| 004 | INT8, no FP16 | TU117 sin Tensor Cores, FP16 emulado y más lento |
| 005 | Local, no remoto | AVX2+CUDA+0 ms gana a geografía |
| 006 | Sobre RealtimeSTT | VAD+ASR resueltos, nosotros orquestamos |
| 007 | Teleprompter primero, TTS opcional | Entregable en semanas, honestidad en entrevista |
| 008 | Fallback automático | Una entrevista no es un log |
| 009 | Dirección por fuente de audio | Determinista, 0 ms extra, sin detector que titubee en code-switching |

---

## 📄 Licencia

MIT — ver `LICENSE` (si aplica).

> Repo de portafolio. Cero credenciales y cero audios de entrevistas reales en el historial.
