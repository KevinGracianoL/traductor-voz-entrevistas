"""Tests del BackendXtts — estado sin cargar (el que corre en CI).

El motor real (coqui-tts + modelo XTTS-v2) no está en CI: se prueba el
contrato en estado no cargado (healthcheck sin efectos secundarios, cierre
idempotente, error claro al sintetizar sin motor) y la conversión PCM pura
(unidad de duración fijada: len/sr, no len/sr/1000).
"""

import pytest

from traductor.tts.backend import TTSBackend
from traductor.tts.backend_xtts import MODELO_XTTS, BackendXtts, pcm_a_audio_result
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
    assert salud.detalle.startswith("coqui-tts no instalado: ")


def test_verificar_salud_no_carga_el_modelo() -> None:
    """verificar_salud no tiene efectos secundarios: no carga (r1 PR #16)."""
    backend = BackendXtts()
    backend.verificar_salud()
    assert backend._tts is None


def test_verificar_salud_con_modelo_cargado() -> None:
    """Simulando el modelo cargado, el healthcheck dice disponible (r2 PR #16)."""
    backend = BackendXtts()
    backend._tts = object()  # sin el motor real: solo la rama cargada
    salud = backend.verificar_salud()
    assert salud.disponible is True
    assert MODELO_XTTS in salud.detalle


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


def test_pcm_a_audio_result_bytes_y_duracion() -> None:
    """Unidad de duración: len/sr (si fuera len/sr/1000, este test lo caza)."""
    muestras = [0.0, 0.5, -0.5, 1.0]
    audio = pcm_a_audio_result(muestras)
    assert audio.formato == "pcm_f32le"
    assert len(audio.datos) == 4 * 4  # 4 muestras float32
    assert audio.duracion_s == pytest.approx(4 / 24000)


def test_pcm_a_audio_result_sr_personalizado() -> None:
    audio = pcm_a_audio_result([0.0, 0.25], sr=16000)
    assert audio.duracion_s == pytest.approx(2 / 16000)
