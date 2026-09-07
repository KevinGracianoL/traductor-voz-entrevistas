"""Tests del adapter Pocket TTS — sin modelo ni CPU real."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from traductor.tts.pocket import PocketTTS, _a_pcm16


def _inject_pocket() -> MagicMock:
    """Inyecta pocket_tts falso (CI no instala la dep pesada)."""
    pkg = types.ModuleType("pocket_tts")
    mock_cls = MagicMock()
    mock_model = MagicMock()
    mock_model.sample_rate = 24000
    mock_cls.load_model.return_value = mock_model
    params = types.ModuleType("pocket_tts.default_parameters")
    params.get_default_voice_for_language = MagicMock(  # type: ignore[attr-defined]
        side_effect=lambda lang: {"english": "alba", "spanish": "lola"}[lang]
    )
    pkg.TTSModel = mock_cls  # type: ignore[attr-defined]
    sys.modules["pocket_tts"] = pkg
    sys.modules["pocket_tts.default_parameters"] = params
    return mock_cls


class FakeChunk:
    """Tensor mínimo: solo .tolist()."""

    def __init__(self, valores: object) -> None:
        self._v = valores

    def tolist(self) -> object:
        return self._v


def _tts() -> PocketTTS:
    model = MagicMock()
    model.generate_audio_stream.side_effect = lambda *a, **k: iter([FakeChunk([0.0, 0.5])])
    return PocketTTS(model, {"voz": 1}, 24000)


def test_sintetizar_devuelve_pcm_sr_ttfa() -> None:
    tts = _tts()
    pcm, sr, ttfa = sintetizar_con_reloj(tts, "hola")
    assert sr == 24000
    assert ttfa == pytest.approx(150.0)
    import struct

    assert struct.unpack("<2h", pcm) == (0, 16384)


def sintetizar_con_reloj(tts: PocketTTS, texto: str) -> tuple[bytes, int, float]:
    """Reloj falso: t0=1.0, primer chunk a 1.15."""
    tiempos = iter([1.0, 1.15, 1.2])
    return tts.sintetizar(texto, clock=lambda: next(tiempos))


def test_sintetizar_vacio_lanza() -> None:
    with pytest.raises(ValueError) as excinfo:
        _tts().sintetizar("")
    assert str(excinfo.value) == "texto vacío"


def test_sintetizar_sin_chunks_lanza() -> None:
    model = MagicMock()
    model.generate_audio_stream.return_value = iter([])
    with pytest.raises(ValueError) as excinfo:
        PocketTTS(model, {}, 24000).sintetizar("hola")
    assert str(excinfo.value) == "modelo no generó ningún chunk"


def test_sintetizar_reutiliza_modelo() -> None:
    tts = _tts()
    tts.sintetizar("uno")
    tts.sintetizar("dos")
    assert tts._model.generate_audio_stream.call_count == 2


def test_cargar_defaults_english_alba() -> None:
    mock_cls = _inject_pocket()
    try:
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_model._cached_get_state_for_audio_prompt.return_value = {"v": 1}
        mock_cls.load_model.return_value = mock_model
        tts = PocketTTS.cargar()
        mock_cls.load_model.assert_called_once_with(language="english")
        mock_model._cached_get_state_for_audio_prompt.assert_called_once_with("alba")
        assert tts._sr == 24000
    finally:
        sys.modules.pop("pocket_tts", None)
        sys.modules.pop("pocket_tts.default_parameters", None)


def test_cargar_spanish_lola() -> None:
    mock_cls = _inject_pocket()
    try:
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_model._cached_get_state_for_audio_prompt.return_value = {"v": 2}
        mock_cls.load_model.return_value = mock_model
        PocketTTS.cargar(language="spanish")
        mock_cls.load_model.assert_called_once_with(language="spanish")
        mock_model._cached_get_state_for_audio_prompt.assert_called_once_with("lola")
    finally:
        sys.modules.pop("pocket_tts", None)
        sys.modules.pop("pocket_tts.default_parameters", None)


def test_a_pcm16_recorta_y_aplana() -> None:
    import struct

    pcm = _a_pcm16([FakeChunk([[2.0, -2.0]]), FakeChunk(0.5)])
    assert struct.unpack("<3h", pcm) == (32767, -32768, 16384)
