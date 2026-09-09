# ADR-013 - Worker TTS aislado y enrolamiento (Propuesto)

- **Estado:** Propuesto (2026-09-08) — protocolo de worker y enrolamiento listos (PR #14); **motor sin elegir** (ADR-011, cadena agotada — ver ADR-014/README Fase 2g) y medición pendiente en la máquina objetivo.
- **Contexto:** el TTS de ES→EN (hablar con el timbre de Kevin) necesita dos cosas que este PR fija antes de elegir motor:
  1. **Enrolamiento:** convertir muestras grabadas del usuario en un `VoiceProfile` validado y persistido.
  2. **Aislamiento:** el motor TTS debe poder cargarse/descargarse en su **propio proceso**, sin comprometer la VRAM que ya ocupa Whisper (ADR-010 midió la co-residencia como problema real).
- **Decisión propuesta:**

  - **`TiendaPerfilesJson`** (`traductor.tts.tienda_json`): implementación real de `VoiceProfileStore` (contrato ADR-011) sobre un directorio, un `{id}.json` por perfil, UTF-8 explícito. Un archivo corrupto no se silencia: `listar` lo omite (para poder enumerar los sanos) pero `listar_errores` lo reporta con su ruta, y `obtener` falla claro indicando el archivo.
  - **`enrolar`** (`traductor.tts.enrolamiento`): `enrolar(id, nombre, muestras) -> VoiceProfile` validando que cada muestra exista y sea un archivo antes de aceptarla.
  - **Worker aislado** (`traductor.tts.worker`): un proceso que tiene un `TTSBackend` inyectado (el motor real aún no existe, ADR-011). Protocolo: jobs JSON-line por la entrada (`texto`, `perfil_id`, `salida` relativa) y resultados JSON-line por la salida (`ResultadoOk`/`ResultadoError`, tipos explícitos).

- **Frontera entre procesos (reglas de entrada, no implícitas):** el worker y la tienda no confían en quien les manda datos (r1 del PR #14, mismo principio que ADR-012):
  - **`perfil_id` con lista blanca** `^[A-Za-z0-9_-]+$` en `TiendaPerfilesJson._ruta`: default-deny sobre conjunto cerrado. Un id como `../secreto` falla con `ValueError` antes de tocar el FS — nunca lee, escribe ni borra fuera del directorio. (Blacklist de `..` no: siempre se escapa.)
  - **`job.salida` confinado** al `directorio_salida` del worker: `resolve()` + `is_relative_to`. Rutas absolutas o con `..` fuera se rechazan con `ResultadoError`.
  - **El worker no muere por un job malo:** los fallos esperados —parseo, perfil ausente o inválido, tienda que falla, síntesis que falla, escritura que falla (permisos, disco lleno), salida fuera del directorio— vuelven como `ResultadoError`. La escritura (mkdir/write_bytes) está dentro del manejo de error, no fuera.

- **Por qué `Resultado` (TypedDict) y no excepciones en el worker:** el worker es una frontera entre procesos; los fallos esperados se representan como valores (`ok: false, error`) para que el llamador decida, y los errores internos del motor se traducen a ese formato en la frontera. Coincide con el estilo de los contratos del ADR-011.
- **Consecuencias:**
  - El motor real (candidato apache-2.0, p. ej. CosyVoice 2) se inyecta como `TTSBackend` cuando un ADR lo elija; el protocolo del worker queda fijado y testeable sin GPU.
  - La medición de latencia del TTS (gates del PR #15) se alimenta del `elapsed_ms` que ya reporta `ResultadoOk`.
  - Enrolamiento y tienda son puro FS: testeables en CI con `tmp_path`; el worker con fakes del backend y la tienda.

- **Nota de diseño (2026-09-09, observación del review del PR #18):** el cache de timbre de `BackendB` se indexa por `(id, rutas)` — cubre re-enrolar con grabaciones NUEVAS, pero no SOBREESCRIBIR una ruta existente: la ruta no cambia y se seguiría sirviendo el timbre viejo. Hoy no importa (ningún motor está en el flujo, ADR-014), pero cuando un backend con cache entre al worker de vida larga, decidir explícitamente entre: (a) `st_mtime_ns` + tamaño de cada muestra como parte de la clave, (b) hash del contenido, o (c) invalidación explícita del cache desde la tienda al re-enrolar. Relacionado: validar los pesos en `__init__` deja dos canales de fallo (paquetes → `Salud`, pesos → excepción) — para un worker que debe REPORTAR salud en vez de morir al arrancar, la presencia de pesos debería ser otra rama de `verificar_salud`.
