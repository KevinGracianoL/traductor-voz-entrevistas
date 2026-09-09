# ADR-019 - Gates de sesión y endurance del flujo (Propuesto)

- **Estado:** Propuesto (2026-09-09) — criterios definidos; **la corrida larga NO se ha ejecutado**: la aprobación FINAL del motor (ADR-011) y del flujo (ADR-015) queda condicionada a estos gates. Pendiente no es aprobado.
- **Contexto:** los gates del ADR-014 (TTFA, VRAM, RAM, pipeline end-to-end) miden corridas de segundos. Las fugas de memoria, las respuestas atrasadas y los fallos de disponibilidad solo aparecen en sesiones largas: la evidencia del ADR-014 (tasa de fallo 5 % en la etapa de traducción; artefacto del decodificador en ventanas específicas) demuestra que lo que no se ve en segundos sí aparece en minutos.
- **Criterios de aceptación de la corrida larga (todos deben pasar; `None` = sin medir = FALLA, misma regla del ADR-014):**

  - **Endurance: completa 90 minutos continuos** de flujo real (mic → ASR → Argos → TTS → CABLE → Meet/Zoom según el nivel de la escalera del ADR-015) **sin cuelgues, sin respuestas atrasadas y sin pérdida de dispositivos**.
  - **Sin OOM** durante la sesión.
  - **Sin crecimiento sostenido de memoria:** la RAM (y la VRAM del worker TTS) no crecen de forma monótona; se mide con `psutil`/`vram_ocupada_mib` en intervalos regulares y se compara el inicio con el final (con margen de ruido anotado).
  - **Sin artefactos de palabras:** validación ASR-de-retorno sobre el audio sintetizado (ADR-015): palabras añadidas, omitidas o repetidas, clipping, silencios anómalos y duración absurda se detectan y rechazan antes del micrófono virtual. Los candidatos del A/B del ADR-014 (`in→and`, `start→stop`) se siguen aquí.
  - **Voz reconocible en A/B:** la firma el usuario ESCUCHANDO (ningún script la firma). Criterio de escucha: timbre, RITMO (que no suene apurada), pausas naturales y palabras técnicas claras.
  - **Degradación automática sin reiniciar la llamada:** la escalera del ADR-015 baja de nivel ante fallos (voz clonada → voz genérica → subtítulos) sin perder la llamada.
  - **Recuperación automática:** los workers (TTS, traducción) con watchdog se reinician y re-enrolan sin intervención.

- **Cómo se mide:** corrida real de 90 minutos con el flujo `outgoing_es_to_en` (ADR-015) + navegador/cámara/Meet abiertos (el contexto de RAM se anota, regla del ADR-014); timestamps por etapa (patrón `RegistroEtapas`); log estructurado de fallos y degradaciones; `nvidia-smi` y `psutil` periódicos. Los valores alimentan `MedicionTts` (campos de sesión) y el harness emite el veredicto final.
- **Prerequisito de arranque (bloqueante, ADR-014):** la etapa de traducción valida su modelo spacy `mwt` precargado y funciona sin red ANTES de aceptar la llamada.
- **Consecuencias:**
  - El motor (ADR-011) pasa de "aceptado por los gates medidos" a "aprobado" solo con esta corrida.
  - Los artefactos y las fugas que aparezcan se convierten en tests de regresión (patrón del proyecto: cada bug real deja su test).
  - La evidencia de esta corrida se pega en este ADR con el mismo patrón de registro que el ADR-014.