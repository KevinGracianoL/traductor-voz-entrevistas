"""TTS con Pocket TTS (Kyutai) en CPU — adapter standalone (ADR-011).

No reutiliza el worker de Chatterbox: la frontera aquí es distinta (CPU,
proceso persistente, voz precalculada, streaming desde el primer chunk).
El caller posee la instancia (cargada una vez); sintetizar nunca recarga.
"""

from __future__ import annotations

import struct
import time
from collections.abc import Callable, Iterable
from typing import Any


class PocketTTS:
    """Modelo cargado + voz precalculada. Una instancia por idioma."""

    def __init__(self, model: Any, voice_state: dict[str, Any], sr: int) -> None:
        self._model = model
        self._voice_state = voice_state
        self._sr = sr

    @classmethod
    def cargar(cls, language: str = "english", voice: str | None = None) -> PocketTTS:
        """Carga modelo 6 capas en CPU + precalcula voz (una vez por proceso)."""
        from pocket_tts import TTSModel
        from pocket_tts.default_parameters import get_default_voice_for_language

        model = TTSModel.load_model(language=language)
        nombre_voz = voice or get_default_voice_for_language(language)
        state = model._cached_get_state_for_audio_prompt(nombre_voz)
        return cls(model, state, model.sample_rate)

    def sintetizar(
        self,
        texto: str,
        clock: Callable[[], float] = time.perf_counter,
    ) -> tuple[bytes, int, float]:
        """Texto -> (pcm16 bytes, sr, ttfa_ms). Mide hasta el primer chunk."""
        if not texto or not texto.strip():
            raise ValueError("texto vacío")
        t0 = clock()
        chunks: list[Any] = []
        t_first: float | None = None
        for ch in self._model.generate_audio_stream(self._voice_state, texto):
            if t_first is None:
                t_first = clock()
            chunks.append(ch)
        if t_first is None:
            raise ValueError("modelo no generó ningún chunk")
        ttfa_ms = (t_first - t0) * 1000.0
        return _a_pcm16(chunks), self._sr, ttfa_ms


def _a_pcm16(chunks: Iterable[Any]) -> bytes:
    """Chunks (tensores con .tolist o listas) -> PCM16 mono. Solo stdlib."""
    plano: list[float] = []
    for ch in chunks:
        datos = ch.tolist() if hasattr(ch, "tolist") else list(ch)
        if isinstance(datos, (list, tuple)) and datos and isinstance(datos[0], (list, tuple)):
            for fila in datos:
                plano.extend(float(x) for x in fila)
        elif isinstance(datos, (list, tuple)):
            plano.extend(float(x) for x in datos)
        else:
            plano.append(float(datos))
    return struct.pack(
        f"<{len(plano)}h",
        *(max(-32768, min(32767, int(round(x * 32767)))) for x in plano),
    )
