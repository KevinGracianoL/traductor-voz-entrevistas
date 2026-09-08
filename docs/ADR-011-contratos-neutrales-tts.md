# ADR-011 - Contratos neutrales de TTS (Propuesto)

- **Estado:** Propuesto (2026-09-08) — se aprueba cuando un backend real (worker TTS) lo implemente sin forzar cambios de contrato.
- **Contexto:** Fase 2 necesita TTS offline con clonación de voz (timbre de Kevin, ES→EN) que conviva con Whisper int8 (~1 GB) en una GTX 1650 Ti de 4 GB, dentro del presupuesto ADR-003. Chatterbox fue rechazado con mediciones (ADR-010) y XTTS-v2 quedó descartado antes: pide 4–6 GB él solo en una GPU que además carga Whisper, y sus pesos son CPML (no comercial). El pipeline (traducción → TTS → audio virtual) no debe depender del proveedor final mientras se evalúa el candidato.
- **Dirección de backend (criterio, no elección):** la licencia que importa es la de los **pesos**, no la del repo — la trampa de F5-TTS (repo MIT, pesos cc-by-nc-4.0) y de XTTS-v2 (CPML). Los candidatos conocidos con pesos permisivos (CosyVoice 2, apache-2.0) y el rechazo medido de Chatterbox (MIT, ADR-010) se evaluarán contra estos contratos, no antes de ellos. Por eso este PR habla de un **worker TTS genérico**, no de XTTS.
- **Decisión propuesta:** definir contratos neutrales en `src/traductor/tts/` y escribir los PRs siguientes (benchmark ASR bidireccional, worker TTS, gates de latencia) contra ellos:

  - **`VoiceProfile`** (frozen dataclass): identidad del usuario + rutas de muestras de referencia para el enrolamiento. Valida `id`, `nombre` y al menos una muestra.
  - **`VoiceProfileStore`** (Protocol): ciclo de vida de perfiles — `listar`, `obtener`, `guardar`, `eliminar`. Separa enrolamiento del backend.
  - **`TTSBackend`** (Protocol): `sintetizar(texto, perfil) -> AudioResult`, `verificar_salud() -> Salud`, `cerrar()`. Ciclo de vida explícito; el healthcheck es la puerta para los gates de latencia (PR #15).
  - **`AudioResult` / `Salud`** (frozen dataclasses): audio (bytes + formato + duración opcional) y estado del backend (disponible + detalle). `Salud(disponible=False)` exige un `detalle`: un backend caído sin explicación no sirve de puerta de gates.

- **Por qué `sintetizar(texto, perfil)` y no un backend atado a un perfil:** el perfil viaja explícito en cada llamada; el caller decide qué voz usar sin que el contrato esconda estado global. El costo (re-resolución del perfil por llamada) lo paga la implementación, no el contrato.
- **Consecuencias:**
  - Este PR **no instancia ningún backend** ni añade dependencias pesadas. Los fakes de los tests son la prueba de que el contrato es satisfacible.
  - Si el worker TTS obliga a cambiar una firma, se cambia aquí con su propio ADR, antes de que haya integraciones que migrar.
  - La dirección EN→ES (escuchar) no usa TTS: va por subtítulos (ADR-001). Estos contratos cubren ES→EN (hablar).