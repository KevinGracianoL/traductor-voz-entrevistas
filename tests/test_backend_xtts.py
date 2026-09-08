"""Tests del BackendXtts — estado sin cargar (el que corre en CI).

El motor real (coqui-tts + modelo XTTS-v2) no está en CI: lo que se prueba es
que el wrapper cumple el contrato en estado no cargado (healthcheck honesto,
cierre idempotente, error claro al sintetizar sin motor). La síntesis real va
en la máquina objetivo (harness del ADR-014).
"""

import pytest

from traductor.tts.backend import TTSBackend
from traductor.tts.backend_xtts import MODELO_XTTS, BackendXtts
from traductor.tts.modelos import VoiceProfile

PERFIL = VoiceProfile(id="kevin-es", nombre="Kevin", muestras=("ref1.wav", "ref2.wav"))


def test_backend_satisface_el_contrato() -> None:
    assert isinstance(BackendXtts(), TTSBackend)


def test_idioma_salida_por_defecto_es_en() -> None:
    assert BackendXtts()._idioma_salida == "en"
    assert BackendXtts(idioma_salida="es")._idioma_salida == "es"


def test_verificar_salud_sin_motor_reporta_indisponible() -> None:
    """En CI (sin coqui-tts) el healthcheck es honesto: False con detalle."""
    salud = BackendXtts().verificar_salud()
    assert salud.disponible is False
    assert "coqui-tts no instalado" in salud.detalle


def test_sintetizar_sin_motor_error_claro() -> None:
    backend = BackendXtts()
    with pytest.raises(
        RuntimeError,
        match=(
            "^coqui-tts no instalado: instala el fork idiap/coqui-ai-TTS "
            r"\(venv propio del TTS, ver ADR-011/014\)$"
        ),
    ):
        backend.sintetizar("hello", PERFIL)


def test_cerrar_idempotente() -> None:
    backend = BackendXtts()
    backend.cerrar()
    backend.cerrar()
    assert backend._tts is None


def test_modelo_es_xtts_v2() -> None:
    assert MODELO_XTTS == "tts_models/multilingual/multi-dataset/xtts_v2"
