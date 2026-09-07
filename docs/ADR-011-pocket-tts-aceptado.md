# ADR-011 — Pocket TTS aceptado como TTS (CPU, proceso persistente)

- **Estado:** Aceptado (2026-09-07). Revierte el bloqueo de Fase 2 tras ADR-010.
- **Contexto:** Chatterbox rechazado (11.6-17.4 s/frase, 3.6 GB VRAM). Se necesita TTS offline con clonación que deje la VRAM libre para Whisper.
- **Candidato:** `pocket-tts==3.0.2` (Kyutai), modelos 6 capas `english`/`spanish` en CPU, voces precalculadas (`alba`/`lola`).
- **Licencia:** código MIT; pesos con atribución CC-BY-4.0 + condiciones de Hugging Face (aceptar al descargar).

## Por qué no los otros

- **Piper** (fallback si Pocket fallaba): innecesario, Pocket pasa los umbrales.
- **Kokoro:** no clona voz + fallo abierto de voces españolas silenciosas en Windows 11.
- **Supertonic 3:** autores anunciaron archivo y fin de soporte.

## Entorno medido

- Ryzen 5 4600H / GTX 1650 Ti 4 GB (driver 610.74) / Windows 11 + Python 3.11.
- `pip install pocket-tts==3.0.2` resuelve sobre torch 2.13+cu132 y numpy 2.4.4; `pip check` limpio; `import` ok.

## Medición (CPU, `time.perf_counter`, 10 corridas warm por idioma)

| Qué | EN (`alba`) | ES (`lola`) |
|---|---|---|
| TTFA p50 | 152.1 ms | 169.9 ms |
| TTFA p95 | 155.4 ms | 177.7 ms |
| RTF p50 | 0.461 | 0.458 |
| Audio medio | 1.96 s | 2.64 s |

RAM pico del proceso: 1 899 MB (CPU, VRAM intacta para Whisper).

## Concurrente real (Whisper tiny int8 GPU + Argos CPU + Pocket CPU, 5 frases de entrevista)

| Etapa | p50 |
|---|---|
| ASR | ~390 ms |
| Traducción | ~65 ms (warm) |
| TTS TTFA | ~156 ms |
| **Pipeline p95** | **~1 209 ms ≤ 2 000 ms** |
| VRAM pico (nvidia-smi) | 620 MB (incluye escritorio) |

(Primera frase fría ~3.8 s por warm-up; steady-state arriba. WAV EN/ES validados: mono 24 kHz PCM16.)

## Decisión

**Aceptado.** TTFA p95 ≤ 400 ms y RTF ≤ 1 se cumplen con margen; pipeline p95 cabe en el techo ADR-003. Arquitectura: proceso persistente en CPU, modelos EN+ES cargados, voz precalculada, streaming desde el primer chunk (`src/traductor/tts/pocket.py`, sin reutilizar el worker de Chatterbox).

## Trazabilidad

- PR #11 (adapter + tests + este ADR). PR #10: rechazo de Chatterbox.
- Escucha pendiente por Kevin de las frases reales generadas (smoke) antes de dar Fase 2 por cerrada.
