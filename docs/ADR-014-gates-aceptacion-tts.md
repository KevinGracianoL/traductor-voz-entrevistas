# ADR-014 - Gates de aceptación del motor TTS

- **Estado:** gates con el sub-gate de TTFA **corregido** (presupuesto DERIVADO de etapas medidas, 2026-09-09 — evidencia abajo). XTTS-v2 y candidato B **medidos y RECHAZADOS** por el gate mal especificado / por TTFA arquitectural; el rechazo de XTTS quedó **invalidado** y su aprobación pendiente de pipeline end-to-end + gates de sesión. El módulo `gates.py` y el harness quedan como gate de regresión.
- **Contexto:** para que ADR-011 pase de "Propuesto" a un motor concreto, hace falta un **criterio de aceptación objetivo**. Sin él, la elección del motor sería una opinión más. Las mediciones se hacen en la GPU real (GTX 1650 Ti 4 GB, presupuesto ADR-003), con el ASR co-residente — mismo criterio que ADR-010 y ADR-012.
- **Cadena de decisión:** se prueba primero **XTTS-v2** (fork `coqui-tts`, pesos CPML — aceptable por uso personal no comercial, declarado en ADR-011). **Orden práctico: correr SOLO el gate de VRAM primero** (smoke de ~5 min, con el ASR co-residente) antes de montar el venv completo — es el gate que más probablemente tumbe al candidato en esta GPU de 4 GB (la objeción de VRAM del descarte anterior, convertida en gate, no borrada). **Si falla cualquiera de los bloqueantes, se rechaza** (sin cuantización agresiva que empeore la voz) y se prueba **Supertonic 3 CPU + OpenVoice V2** (candidato B). El que XTTS se instale y se mida no lo declara aceptado: la aceptación es el resultado de los gates.
- **Criterios de aceptación (todos deben pasar; `None` = sin medir → FALLA):**

  - **TTFA (time to first audio) = PRIMER CHUNK reproducible, caliente p95 < presupuesto DERIVADO:** `2000 ms − (ASR p95 + traducción p95 + ruteo p95)`, cada etapa MEDIDA en el hardware con n≥20 (corrección 2026-09-09: el "< 400 ms" literal era un sub-presupuesto inventado). Etapa sin medir → presupuesto no derivable → FALLA.
  - **VRAM total (ASR + motor) < 3.2 GB** (3276.8 MiB). El "~1 GB" del ASR es un **supuesto a medir en la primera corrida**, no una cifra verificada.
  - **RAM total < 18 GB.** Definición: `psutil.virtual_memory().used` de TODA la máquina — el veredicto depende de qué más esté abierto; anotar el contexto al correr (como con `nvidia-smi`).
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

## Evidencia — XTTS-v2 rechazado por TTFA (2026-09-08)

**Entorno:** Ryzen 5 4600H / GTX 1650 Ti 4 GB / Windows. Motor: XTTS-v2 vía `coqui-tts` 0.27.5 (fork idiap), torch 2.14.0+cu132, en venv propio (venv-tts). ASR co-residente: faster-whisper tiny int8. Referencia de voz: `scripts/audio/voz_kevin.wav` (13 s, 16 kHz mono). Harness con `--warmup-audio`; `warm-up: 3 segmentos`, `delta VRAM (Whisper) = 110 MiB`.

| Gate | Medido | Límite | Estado |
|---|---|---|---|
| TTFA síntesis completa (contrato no-streaming, p95 n=20) | **3574.6 ms** | < 400 ms | FALLA |
| TTFA primer chunk (streaming real `inference_stream`, n=5) | rango **655–889 ms** (p95 no válido con n<20) | < 400 ms | FALLA |
| VRAM co-residente (XTTS + Whisper) | 3010.9 MiB | < 3276.8 MiB | PASA |
| RAM total (máquina) | 11804.9 MiB | < 18432 MiB | PASA |

**Lectura honesta del TTFA:** el gate define TTFA = time to first audio; con el contrato no-streaming se mide la síntesis completa (3574 ms). XTTS soporta streaming (`inference_stream`, 6 chunks por frase) y el **primer chunk más rápido observado (655 ms) ya supera el límite por 60 %** — el rechazo no depende del `n` (no se reporta p95 con n=5; la regla del ADR exige n≥20). Las latents de condicionamiento (756 ms) se calculan una vez y no entran al presupuesto por turno (ADR-011).

**Contexto de RAM (nota):** el valor 11804.9 MiB es el uso de TODA la máquina (`psutil.virtual_memory().used`) y **no se anotó qué había abierto** durante la corrida. Hoy sobra holgura (~6.6 GB), pero el candidato B puede quedar al filo: **las corridas futuras deben anotar el contexto** (navegador, Meet, etc.) como se hace con `nvidia-smi`.

**Veredicto:** un bloqueante basta → **XTTS-v2 RECHAZADO** (sin cuantización agresiva). Sigue **Supertonic 3 CPU + OpenVoice V2 (candidato B)** — este ADR queda como gate de regresión y registro del go/no-go.

## Evidencia — candidato B (Supertonic 3 CPU + OpenVoice V2) rechazado por TTFA (2026-09-09)

**Entorno:** Ryzen 5 4600H / GTX 1650 Ti 4 GB / Windows. Motor: Supertonic 3 (SDK `supertonic` 1.3.1, ONNX CPU, voz M1) + OpenVoice V2 (converter, CPU, watermark off) vía `BackendB` (`src/traductor/tts/backend_b.py`), torch 2.14.0+cu132 en venv-tts. ASR co-residente: faster-whisper tiny int8 (CUDA). Referencia: `scripts/audio/voz_kevin.wav` (13 s). Harness: `--motor b --warmup-audio voz_kevin.wav --referencia voz_kevin.wav`; texto: "Thank you for the question. My experience with distributed systems is one of my strongest points." (fragmento real del flujo ES→EN). `warm-up: 3 segmentos`, `delta VRAM (Whisper) = 110 MiB`. El `target_se` (timbre de Kevin) se extrae una vez por perfil (~2 s con VAD silero, en el warm-up, no entra al turno); el `src_se` (timbre de la voz base M1) también se cachea.

| Gate | Medido | Límite | Estado |
|---|---|---|---|
| TTFA caliente p95 (n=20, pipeline completo por turno) | **7885.8 ms** | < 400 ms | FALLA |
| VRAM co-residente (Whisper sola: el motor es CPU) | 910.9 MiB | < 3276.8 MiB | PASA |
| RAM total (máquina) | 11531.8 MiB | < 18432 MiB | PASA |

**Contexto de RAM (anotado, regla del ADR):** durante la corrida estaban abiertos opencode (~1 GB), VSCode (×2), WhatsApp, Edge webview, Windows Defender y el explorador — el 11531.8 MiB incluye esa máquina normal de trabajo; el harness en sí añade ~2 GB sobre la línea base (~9.5 GB sin él).

**Lectura honesta del TTFA:** el pipeline B es **no-streaming por construcción**: Supertonic sintetiza el fragmento completo (~1.75 s caliente para 6.4 s de audio) y OpenVoice V2 convierte el timbre del fragmento completo (~5-6 s). No existe "primer chunk" en ninguna etapa — el TTFA es la síntesis entera. **7885.8 ms es 20× el límite** y el número es estable (n=20, p95). La latencia no depende del streaming: es el costo del convertidor sobre el audio completo.

**Veredicto:** un bloqueante basta → **candidato B RECHAZADO**. Con XTTS y B descartados, la cadena del ADR-011 está agotada (Pocket ya estaba descartado). La escalera del ADR-015 apunta al nivel 3 (voz inglesa **genérica** Supertonic/Piper + subtítulos) — sin clonación de timbre, la etapa de conversión desaparece y queda solo el synth (~1.75 s para 6.4 s de audio, aún por encima del TTFA de 400 ms para fragmentos completos; la decisión de flujo decide cómo partir el audio y qué gate aplica). La clonación de voz (Fase 3) queda condicionada a un motor que cumpla los gates: la evidencia de este ADR es el criterio, no la opinión.

## Corrección — el sub-gate de TTFA era inválido y se reemplaza por uno DERIVADO (2026-09-09)

**Qué se corrige:** el gate "TTFA < 400 ms" de la evidencia de 2026-09-08 era un **sub-presupuesto inventado** (ASR 300 + traducción 200 + primer fragmento 350 + ruteo 100 = 950 ms de etapas "presupuestadas"), nunca validado contra el total. Con el propio número medido de XTTS streaming (primer chunk 889 ms peor caso, n=5), la suma contra el objetivo end-to-end de ADR-003 cabe: **300 + 200 + 889 + 100 = 1489 ms < 2000 ms**. La evidencia anterior NO se borra (registro del error); este apartado corrige el criterio y pega la medición nueva. Regla del usuario que dirige el proyecto: **nunca fijar un número a ojo** — el presupuesto se deriva de etapas medidas.

**Criterio nuevo (implementado en `gates.py` como cálculo, no literal):**

```
presupuesto_ttfa = 2000 ms − (ASR p95 + traducción p95 + ruteo p95)
```

Cada etapa debe estar MEDIDA en el hardware (n≥20, p95, reloj inyectable del harness); una etapa sin medir hace el presupuesto no derivable y el gate FALLA. TTFA se redefine como **primer chunk de audio reproducible** (streaming caliente para XTTS).

**Etapas medidas (2026-09-09, máquina objetivo, venv-tts, faster-whisper co-residente, `warm-up: 3 segmentos`):**

| Etapa | Supuesto anterior | Medido ahora (p95, n≥20) | Veredicto |
|---|---|---|---|
| ASR (faster-whisper tiny int8, es) | 300 ms | **768.5 ms** | el supuesto subestimaba 2.5× |
| Traducción (Argos ES→EN, 20 frases distintas) | 200 ms | **176.6 ms** | supuesto razonable (argos 1.11 cachea el mismo texto: se miden frases distintas como los turnos reales) |
| Ruteo a VB-CABLE | 100 ms | **sin medir — VB-CABLE NO instalado** (instrumento listo: drenado real hasta que el dispositivo consume el chunk + auto-verificación del piso físico `frames/sample_rate` con `raise` si el p95 sale por debajo — patrón delta VRAM; tests en `test_harness_gates.py`) | gate FALLA por regla (sin medir no pasa) |
| TTFA 1er chunk (XTTS `inference_stream`, n≥20, caliente) | 655–889 ms (n=5) | **691.1 ms** | el n=5 de la corrida anterior no cumplía la regla n≥20; el p95 n≥20 confirma el orden de magnitud |
| Pipeline end-to-end (audio → 1er audio en el micrófono virtual) | inferido sumando supuestos (nunca medido) | **sin medir — VB-CABLE ausente** (instrumento listo: UNA corrida encadenada con cortes DISTINTOS de audio por iteración — los turnos reales difieren y argos cachea texto idéntico —, primer chunk + ruteo con drenado) | gate FALLA por regla; la inferencia es exactamente lo que produjo el rechazo equivocado |
| VRAM co-residente | — | **2906.9 MiB** | PASA (< 3276.8) |
| RAM total (máquina, contexto anotado) | — | **13198.6 MiB** | PASA (< 18432) |

**Presupuesto derivado con lo medido:** 2000 − 768.5 − 176.6 − ruteo → **1054.9 ms con ruteo = 0** (o 954.9 ms con el supuesto de 100 ms). El TTFA medido (691.1 ms) cabe en cualquiera de los dos — pero el gate FALLA hoy porque **el ruteo no se puede medir** (VB-CABLE no está instalado; instrumento listo y verificado: drenado real + auto-verificación del piso físico con `raise`, pipeline end-to-end por cortes distintos) y el pipeline end-to-end sigue sin medir. XTTS **NO queda aprobado**: la corrección invalida el rechazo anterior, no fabrica un pase — la aprobación requiere instalar VB-CABLE y cerrar estas dos mediciones + los gates de sesión (OOM, memoria, artefactos, A/B — lo firma el usuario escuchando —, endurance 90 min). Próximo paso: instalar VB-CABLE y re-correr el harness (`--motor xtts --warmup-audio voz_kevin.wav --referencia voz_kevin.wav`); el instrumento cierra las dos mediciones pendientes con su auto-verificación.

**Hallazgos de la medición:**
- `ARGOS_COMPUTE_TYPE` sin "default" hace que argos es→en produzca basura ("mainstream" en bucle); el wrapper `traductor.traduccion.argos` ya lo fija — el harness exige salida correcta (sanity) antes de medir la etapa.
- Argos cachea el mismo texto (0.0 ms): medir con frases distintas, como los turnos reales.
- torchaudio/torchcodec necesita las DLL compartidas de FFmpeg en `PATH` (build BtbN, mismo requisito que la evidencia XTTS).
