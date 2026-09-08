"""Contrato neutral de un backend TTS.

Un `TTSBackend` sintetiza texto a audio para un `VoiceProfile`. El pipeline
depende de ESTE protocolo, no de un proveedor concreto: el worker XTTS y los
gates de latencia (PRs siguientes) se escriben contra él.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from traductor.tts.modelos import AudioResult, Salud, VoiceProfile


@runtime_checkable
class TTSBackend(Protocol):
    """Backend de síntesis de voz. Ciclo de vida explícito: `cerrar()`."""

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:
        """Sintetiza `texto` con el timbre de `perfil`.

        Raises:
            RuntimeError: si el backend no está disponible (ver `verificar_salud`).
        """
        ...

    def verificar_salud(self) -> Salud:
        """True si el backend está listo para sintetizar (modelo cargado, GPU ok)."""
        ...

    def cerrar(self) -> None:
        """Libera recursos (modelo, VRAM). Idempotente: llamar dos veces no falla."""
        ...
