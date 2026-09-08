# ADR-013 - Worker TTS aislado y enrolamiento (Propuesto)

- **Estado:** Propuesto (2026-09-08) — protocolo de worker y enrolamiento listos (PR #14); **motor sin elegir** (ADR-011) y medición pendiente en la máquina objetivo.
- **Contexto:** el TTS de ES→EN (hablar con el timbre de Kevin) necesita dos cosas que este PR fija antes de elegir motor:
  1. **Enrolamiento:** convertir muestras grabadas del usuario en un `VoiceProfile` validado y persistido.
  2. **Aislamiento:** el motor TTS debe poder cargarse/descargarse en su **propio proceso**, sin comprometer la VRAM que ya ocupa Whisper (ADR-010 midió la co-residencia como problema real).
- **Decisión propuesta:**

  - **`TiendaPerfilesJson`** (`traductor.tts.tienda_json`): implementación real de `VoiceProfileStore` (contrato ADR-011) sobre un directorio, un `{id}.json` por perfil, UTF-8 explícito. Un archivo corrupto no se silencia: falla claro.
  - **`enrolar`** (`traductor.tts.enrolamiento`): `enrolar(id, nombre, muestras) -> VoiceProfile` validando que cada muestra exista y sea un archivo antes de aceptarla.
  - **Worker aislado** (`traductor.tts.worker`): un proceso que tiene un `TTSBackend` inyectado (el motor real aún no existe, ADR-011). Protocolo: jobs JSON-line por la entrada (`texto`, `perfil_id`, `salida`) y resultados JSON-line por la salida (`ResultadoOk`/`ResultadoError`, tipos explícitos). **Un worker no muere por un job malo**: los fallos esperados (perfil ausente, síntesis que falla) vuelven como `ResultadoError`, no como excepción.

- **Por qué `Resultado` (TypedDict) y no excepciones en el worker:** el worker es una frontera entre procesos; los fallos esperados se representan como valores (`ok: false, error`) para que el llamador decida, y los errores internos del motor se traducen a ese formato en la frontera. Coincide con el estilo de los contratos del ADR-011.
- **Consecuencias:**
  - El motor real (candidato apache-2.0, p. ej. CosyVoice 2) se inyecta como `TTSBackend` cuando un ADR lo elija; el protocolo del worker queda fijado y testeable sin GPU.
  - La medición de latencia del TTS (gates del PR #15) se alimenta del `elapsed_ms` que ya reporta `ResultadoOk`.
  - Enrolamiento y tienda son puro FS: testeables en CI con `tmp_path`; el worker con fakes del backend y la tienda.