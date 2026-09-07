"""Servidor TTS persistente: posee modelos EN+ES, atiende N solicitudes con 1 carga.

Frontera entre el adapter (`pocket.py`, puro) y el proceso de producción:
cargar una vez al arrancar, sintetizar muchas veces sin recargar.
"""

from __future__ import annotations

from traductor.tts.pocket import PocketTTS


class ServidorTTS:
    """Dos instancias PocketTTS (una por idioma), cargadas una sola vez."""

    def __init__(self, en: PocketTTS, es: PocketTTS) -> None:
        self._en = en
        self._es = es

    @classmethod
    def cargar(cls) -> ServidorTTS:
        """Carga EN+ES una vez (lento, ~1 min en CPU). Llamar al arrancar."""
        return cls(PocketTTS.cargar("english"), PocketTTS.cargar("spanish"))

    def atender(self, texto: str, idioma: str = "es") -> tuple[bytes, int]:
        """Sintetiza con el modelo ya cargado. Devuelve (pcm16, sr)."""
        if idioma not in ("es", "en"):
            raise ValueError(f"idioma debe ser 'es' o 'en': {idioma!r}")
        tts = self._es if idioma == "es" else self._en
        pcm, sr, _ = tts.sintetizar(texto)
        return pcm, sr
