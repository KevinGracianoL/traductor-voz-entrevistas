# ADR-014 - Gates de aceptación del motor TTS (Propuesto)

- **Estado:** Propuesto (2026-09-08) — criterios completos y harness listos (PRs #15/#16); **medición pendiente** en la máquina objetivo con XTTS-v2 (candidato primario, ADR-011).
- **Contexto:** para que ADR-011 pase de "Propuesto" a un motor concreto, hace falta un **criterio de aceptación objetivo**. Sin él, la elección del motor sería una opinión más. Las mediciones se hacen en la GPU real (GTX 1650 Ti 4 GB, presupuesto ADR-003), con el ASR co-residente — mismo criterio que ADR-010 y ADR-012.
- **Cadena de decisión:** se prueba primero **XTTS-v2** (fork `coqui-tts`, pesos CPML — aceptable por uso personal no comercial, declarado en ADR-011). **Orden práctico: correr SOLO el gate de VRAM primero** (smoke de ~5 min, con el ASR co-residente) antes de montar el venv completo — es el gate que más probablemente tumbe al candidato en esta GPU de 4 GB (la objeción de VRAM del descarte anterior, convertida en gate, no borrada). **Si falla cualquiera de los bloqueantes, se rechaza** (sin cuantización agresiva que empeore la voz) y se prueba **Supertonic 3 CPU + OpenVoice V2** (candidato B). El que XTTS se instale y se mida no lo declara aceptado: la aceptación es el resultado de los gates.
- **Criterios de aceptación (todos deben pasar; `None` = sin medir → FALLA):**

  - **TTFA caliente p95 < 400 ms.** TTFA = time to first audio. Con el contrato actual (no streaming) TTFA ≈ latencia de la primera síntesis; XTTS soporta streaming y el primer audio **no puede bloquearse esperando toda la frase**. `p95` con `n≥20` (misma honestidad que ADR-003/012).
  - **VRAM total (ASR + motor) < 3.2 GB** (3276.8 MiB). El "~1 GB" del ASR es un **supuesto a medir en la primera corrida**, no una cifra verificada.
  - **RAM total < 18 GB.**
  - **Pipeline warm p95 < 2 s** (cierre del turno: ASR ≤300 ms + traducción ≤200 ms + primer fragmento ≤350 ms + ruteo ≤100 ms → ~1.3 s, dentro del objetivo de 1.5–2 s del ADR-003).
  - **Sin OOM.**
  - **Sin crecimiento sostenido de memoria** durante la sesión.
  - **Sin palabras añadidas, omitidas, repetidas ni sonidos metálicos** en el corpus de aceptación.
  - **La voz del usuario es reconocible** en una prueba A/B.
  - **Endurance: completa 90 minutos continuos** (ADR-019) sin cuelgues ni respuestas atrasadas.

- **Cómo se mide:** `scripts/medir_gates_tts.py` — carga el motor candidato y **faster-whisper co-residente**, hace `warm-up`, `n≥20` síntesis calientes con reloj inyectable y evalúa `evaluar_gates`. La VRAM se mide con **`vram_ocupada_mib()`** (`mem_get_info` a nivel driver, una sola llamada sobre el dispositivo actual): incluye lo que CTranslate2 reserva fuera del allocator de torch. **Whisper se calienta con su PRIMERA `transcribe()`** (CT2 reserva workspace/KV/beam ahí) y el generador se drena con `list()`. **Auto-verificación del instrumento:** la foto de Whisper se toma justo después de su warm-up (delta aislado, impreso siempre, `raise` si < 50 MiB); la foto de co-residencia después de las síntesis TTFA; `--warmup-audio` obligatorio y validado en argparse (falla en el primer segundo); 0 segmentos = `raise`. Los gates de sesión (OOM, memoria, artefactos, A/B, endurance) los mide la corrida larga del ADR-019 y se alimentan como valores al módulo. Prueba de cierre: VRAM vs `nvidia-smi` ±100 MiB. Un gate sin medir (`None`) **falla**: un criterio que no se puede evaluar no pasa en silencio.
- **Consecuencias:**
  - Un candidato que pase todos los gates se propone como decisión de ADR-011 (motor concreto) con su propio PR.
  - Un candidato que no pase se descarta con la evidencia pegada aquí (mismo patrón que ADR-010) y se prueba el siguiente de la cadena (B: Supertonic + OpenVoice V2).
  - El módulo `gates.py` queda como gate de regresión: si un futuro cambio empeora alguna métrica, se detecta re-corriendo el harness.