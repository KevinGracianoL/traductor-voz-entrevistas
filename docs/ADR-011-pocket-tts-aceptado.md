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

## Medición (CPU, `time.perf_counter`, 20 corridas warm por idioma + warm-up previo separado)

Muestras crudas versionadas en `docs/medicion-pocket-tts.json` (comando/datos/salida).

| Qué | EN (`alba`) | ES (`lola`) |
|---|---|---|
| TTFA p50 (n=20) | 149.6 ms | 160.5 ms |
| TTFA p95 (n=20) | 159.7 ms | 175.7 ms |
| RTF p50 | 0.461 | 0.458 |

RAM pico del proceso: 1 899 MB (CPU, VRAM intacta para Whisper).

## Concurrente real (Whisper tiny int8 GPU + Argos CPU + Pocket CPU, 20 frases de entrevista)

| Etapa | warm p50 |
|---|---|
| ASR (con `synchronize`) | 370.4 ms |
| Traducción | 55.7 ms |
| TTS total | 820.2 ms |
| **Pipeline cold[0]** | **3 771.8 ms (warm-up, separado)** |
| **Pipeline warm p95 (n=19)** | **1 581.1 ms ≤ 2 000 ms** |
| VRAM pico (nvidia-smi) | 620 MB (incluye escritorio) |

WAV EN/ES validados: mono 24 kHz PCM16. Escucha por Kevin (2026-09-07): "está bien" en ambos idiomas.

## Decisión

**Aceptado.** TTFA p95 ≤ 400 ms y RTF ≤ 1 se cumplen con margen; pipeline p95 cabe en el techo ADR-003. Arquitectura: proceso persistente en CPU, modelos EN+ES cargados, voz precalculada, streaming desde el primer chunk (`src/traductor/tts/pocket.py`, sin reutilizar el worker de Chatterbox).

## Trazabilidad

- PR #11 (adapter + tests + este ADR). PR #10: rechazo de Chatterbox.
- Escucha pendiente por Kevin de las frases reales generadas (smoke) antes de dar Fase 2 por cerrada.
