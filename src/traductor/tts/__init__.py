"""Contratos neutrales de TTS — modelos, backend y tienda de perfiles.

El pipeline depende de estos contratos, no de un proveedor concreto (ADR-011,
estado Propuesto). No se instancia ningún backend en este paquete.
"""

from traductor.tts.backend import TTSBackend
from traductor.tts.modelos import AudioResult, Salud, VoiceProfile
from traductor.tts.tienda import VoiceProfileStore

__all__ = ["AudioResult", "Salud", "TTSBackend", "VoiceProfile", "VoiceProfileStore"]
