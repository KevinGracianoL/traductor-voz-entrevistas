"""Backend XTTS-v2 sobre el fork mantenido `coqui-tts` (idiap) — candidato primario.

Implementa `TTSBackend` (ADR-011): `sintetizar(texto, perfil)`, `verificar_salud`,
`cerrar`. El import del motor es lazy: en CI `coqui_tts` no está instalado y
`verificar_salud()` lo reporta (disponible=False con detalle) — el estado sin
cargar es el que se prueba en CI; la síntesis real corre en la máquina objetivo
(harness del ADR-014) y las líneas que tocan el modelo van `# pragma: no cover`.

El perfil llega pre-enrolado (ADR-011): `perfil.muestras` son las referencias
de voz; XTTS las usa para clonar el timbre en el idioma de salida.
"""

from __future__ import annotations

from typing import Any

from traductor.tts.modelos import AudioResult, Salud, VoiceProfile

MODELO_XTTS = "tts_models/multilingual/multi-dataset/xtts_v2"


class BackendXtts:
    """TTSBackend sobre XTTS-v2. `idioma_salida`: "en" para ES→EN (ADR-015)."""

    def __init__(self, idioma_salida: str = "en") -> None:
        self._idioma_salida = idioma_salida
        self._tts: Any | None = None

    def _cargar(self) -> Any:  # pragma: no cover - requiere coqui_tts + GPU
        """Carga el modelo una sola vez (lazy). Raises: RuntimeError."""
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise RuntimeError(
                    "coqui-tts no instalado: instala el fork idiap/coqui-ai-TTS "
                    "(venv propio del TTS, ver ADR-011/014)"
                ) from exc
            try:
                self._tts = TTS(MODELO_XTTS, device="cuda")
            except Exception as exc:
                raise RuntimeError(f"XTTS-v2 no cargó: {exc}") from exc
        return self._tts

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:  # pragma: no cover
        """Sintetiza `texto` con el timbre de `perfil.muestras`.

        Raises:
            RuntimeError: si el motor no está disponible (ver `verificar_salud`).
        """
        tts = self._cargar()
        import numpy as np

        wav = tts.tts(
            texto,
            speaker_wav=list(perfil.muestras),
            language=self._idioma_salida,
            split_sentences=True,
        )
        muestras = np.asarray(wav, dtype=np.float32)
        return AudioResult(
            datos=muestras.tobytes(),
            formato="pcm_f32le",
            duracion_s=float(len(muestras)) / 24000.0,
        )

    def verificar_salud(self) -> Salud:
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                return Salud(disponible=False, detalle=f"coqui-tts no instalado: {exc}")
            try:  # pragma: no cover - solo con coqui-tts instalado
                self._tts = TTS(MODELO_XTTS, device="cuda")
            except Exception as exc:  # pragma: no cover - GPU/modelo ausente
                return Salud(disponible=False, detalle=f"XTTS-v2 no cargó: {exc}")
        return Salud(disponible=True, detalle=f"XTTS-v2 listo ({MODELO_XTTS})")  # pragma: no cover

    def cerrar(self) -> None:
        """Libera el modelo. Idempotente: llamar dos veces no falla."""
        self._tts = None
