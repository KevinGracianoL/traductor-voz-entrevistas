# ADR-012 - Benchmark ASR bidireccional (Propuesto)

- **Estado:** Propuesto (2026-09-08) — metodología y harness listos (PR #13); **medición pendiente en la máquina objetivo**.
- **Contexto:** el pipeline necesita transcripción en las dos direcciones: la que se escucha (EN→ES, subtítulos) y la que se habla (ES→EN, para traducir). Hoy el tubo usa RealtimeSTT (faster-whisper tiny int8). Antes de fijar el motor hay que medir en la GPU real (GTX 1650 Ti 4 GB, presupuesto ADR-003), no en benchmarks de otra gente: mismo criterio que ADR-010.
- **Candidatos a medir:** `faster-whisper` (tiny, int8/FP16, ya en el venv) vs `moonshine` (tiny/base, ONNX, útil para on-device). RealtimeSTT **no se borra** hasta que un ADR fije el ganador.
- **Decisión propuesta:** decidir el motor con un benchmark reproducible por idioma (es/en) y motor, con dos métricas:

  - **WER** (`traductor.asr.wer`): precisión de la transcripción contra una referencia (ground truth), normalizada por palabras.
  - **Latencia p50/p95** (reuso de `latencia.medidor`): `p95=None` con `n<20` — con pocas muestras es igual a `max` y engaña (ADR-003).

  Procedimiento: `scripts/benchmark_asr.py` + manifest de audios de referencia en `scripts/audio/manifesto.json`, `n≥20` por celda. La salida (tabla por `engine|idioma`) se pega aquí como evidencia.
- **Criterio (por idioma):** gana el motor con menor WER que quepa en el presupuesto de ASR (p50 ≤ ~600 ms en esta GPU). Empate en WER → gana latencia. Si ningún motor cabe, se degrada tamaño de modelo, nunca se sacrifica latencia (ADR-003).
- **Consecuencias:**
  - El benchmark queda como gate de regresión: si un futuro cambio de motor empeora WER o latencia en la máquina objetivo, se detecta re-corriendo `scripts/benchmark_asr.py`.
  - Si gana moonshine, se propone un ADR para reemplazar RealtimeSTT (con su propio PR); si gana faster-whisper, se mantiene.
  - El harness y las métricas son testeables en CI (WER, agregación, tabla); la medición en sí es de hardware y no corre en CI.