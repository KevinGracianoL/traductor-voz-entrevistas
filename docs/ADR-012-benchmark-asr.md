# ADR-012 - Benchmark ASR bidireccional (Propuesto)

- **Estado:** Propuesto (2026-09-08) — metodología y harness listos (PR #13); **medición pendiente en la máquina objetivo**.
- **Contexto:** el pipeline necesita transcripción en las dos direcciones: la que se escucha (EN→ES, subtítulos) y la que se habla (ES→EN, para traducir). Hoy el tubo usa RealtimeSTT (faster-whisper tiny int8). Antes de fijar el motor hay que medir en la GPU real (GTX 1650 Ti 4 GB, presupuesto ADR-003), no en benchmarks de otra gente: mismo criterio que ADR-010.
- **Candidatos a medir (decisión de arquitectura, PR #16):** **Moonshine Small ES/EN INT8 en CPU** para ambas direcciones (la que se habla y la que se escucha) + **faster-whisper Small multilingüe en GPU** solo como comparación del benchmark (si Small no es estable, faster-whisper Base). RealtimeSTT **no se borra** hasta que un ADR fije el ganador. **No usar WhisperX en la ruta en vivo.**
- **Corpus de aceptación (por separado, no un solo promedio):** español colombiano espontáneo; inglés de entrevista; términos técnicos; audio con ruido; cold start y warm; p50/p95; **CPU, RAM y VRAM**; omisiones, sustituciones y **alucinaciones** (WER cubre las dos primeras como ediciones; las alucinaciones se miden aparte).
- **Decisión propuesta:** decidir el motor con un benchmark reproducible por idioma (es/en) y motor, con dos métricas:

  - **WER** (`traductor.asr.wer`): precisión de la transcripción contra una referencia (ground truth), normalizada por palabras.
  - **Latencia p50/p95** (reuso de `latencia.medidor`): `p95=None` con `n<20` — con pocas muestras es igual a `max` y engaña (ADR-003).

  Procedimiento: `scripts/benchmark_asr.py` + manifest de audios de referencia en `scripts/audio/manifesto.json`, `n≥20` y `wer_n≥20` por celda. La salida (tabla por `engine|idioma`) se pega aquí como evidencia.
- **Normalización del WER (regla escrita, no implícita):** ambos lados (referencia y transcripción) pasan por `normalizar_texto` antes de comparar: **minúsculas, sin puntuación, espacios colapsados**. Esto evita que un motor que puntúa/mayusculiza (faster-whisper) puntúe peor que uno que no, aunque transcriba igual (r1 del PR #13). Decisiones deliberadas:
  - **Números siguen contando error:** "tres" ≠ "3".
  - **Tildes siguen contando error:** "café" ≠ "cafe". Consecuencia práctica: las **referencias del manifest deben llevar las tildes reales** del audio; si no, una muestra bien transcrita puntúa WER por un acento y el benchmark favorece al motor peor.
  - Cualquier cambio a esta regla es un cambio de métrica y exige su propio ADR.
- **Criterio (por idioma):** gana el motor con menor WER que quepa en el presupuesto de ASR (p50 ≤ ~600 ms en esta GPU). Empate en WER → gana latencia. Si ningún motor cabe, se degrada tamaño de modelo, nunca se sacrifica latencia (ADR-003).
- **Consecuencias:**
  - El benchmark queda como gate de regresión: si un futuro cambio de motor empeora WER o latencia en la máquina objetivo, se detecta re-corriendo `scripts/benchmark_asr.py`.
  - Si gana moonshine, se propone un ADR para reemplazar RealtimeSTT (con su propio PR); si gana faster-whisper, se mantiene.
  - El harness y las métricas son testeables en CI (WER, agregación, tabla); la medición en sí es de hardware y no corre en CI.
  - Tamaño del conjunto: con `n≥20`/`wer_n≥20` y 4 celdas (2 idiomas × 2 motores) hacen falta **≥80 muestras con referencia**. Mirar `wer_n` antes que el WER al leer la tabla.