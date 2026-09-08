"""TTS — contratos neutrales (ADR-011), enrolamiento y worker aislado (ADR-013).

El pipeline depende de estos contratos, no de un proveedor concreto. El motor
real se inyecta (aún sin elegir, ADR-011); el worker aislado (ADR-013) corre
cualquier `TTSBackend` en su propio proceso.
"""

from traductor.tts.backend import TTSBackend
from traductor.tts.enrolamiento import enrolar
from traductor.tts.modelos import AudioResult, Salud, VoiceProfile
from traductor.tts.tienda import VoiceProfileStore
from traductor.tts.tienda_json import TiendaPerfilesJson
from traductor.tts.worker import Job, procesar_job
from traductor.tts.worker import main as main_worker

__all__ = [
    "AudioResult",
    "Job",
    "Salud",
    "TTSBackend",
    "TiendaPerfilesJson",
    "VoiceProfile",
    "VoiceProfileStore",
    "enrolar",
    "main_worker",
    "procesar_job",
]
