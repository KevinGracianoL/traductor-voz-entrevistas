"""TTS — contratos neutrales (ADR-011), enrolamiento y worker aislado (ADR-013).

El pipeline depende de estos contratos, no de un proveedor concreto. El motor
real se inyecta (aún sin elegir, ADR-011); el worker aislado (ADR-013) corre
cualquier `TTSBackend` en su propio proceso.
"""

from traductor.tts.backend import TTSBackend
from traductor.tts.backend_xtts import MODELO_XTTS, BackendXtts
from traductor.tts.enrolamiento import enrolar
from traductor.tts.gates import (
    GateResultado,
    MedicionTts,
    cabe_en_gates,
    evaluar_gates,
    resumen_gates,
)
from traductor.tts.modelos import AudioResult, Salud, VoiceProfile
from traductor.tts.tienda import VoiceProfileStore
from traductor.tts.tienda_json import TiendaPerfilesJson
from traductor.tts.worker import Job, procesar_job
from traductor.tts.worker import main as main_worker

__all__ = [
    "AudioResult",
    "BackendXtts",
    "GateResultado",
    "Job",
    "MODELO_XTTS",
    "MedicionTts",
    "Salud",
    "TTSBackend",
    "TiendaPerfilesJson",
    "VoiceProfile",
    "VoiceProfileStore",
    "cabe_en_gates",
    "enrolar",
    "evaluar_gates",
    "main_worker",
    "procesar_job",
    "resumen_gates",
]
