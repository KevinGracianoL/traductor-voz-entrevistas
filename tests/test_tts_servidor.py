"""Tests del servidor persistente: 1 carga + N solicitudes."""

from unittest.mock import MagicMock, patch

import pytest

from traductor.tts.servidor import ServidorTTS


def _tts_mock() -> MagicMock:
    m = MagicMock()
    m.sintetizar.return_value = (b"\x00\x00", 24000, 150.0)
    return m


def test_cargar_una_vez_atender_varias() -> None:
    """Una carga (2 modelos) + N atenciones sin recargar."""
    with patch("traductor.tts.servidor.PocketTTS") as mock_cls:
        en, es = MagicMock(), MagicMock()
        en.sintetizar.return_value = (b"\x00\x00", 24000, 100.0)
        es.sintetizar.return_value = (b"\x00\x00", 24000, 100.0)
        mock_cls.cargar.side_effect = [en, es]
        srv = ServidorTTS.cargar()
        assert mock_cls.cargar.call_count == 2
        mock_cls.cargar.assert_any_call("english")
        mock_cls.cargar.assert_any_call("spanish")
        srv.atender("hola")
        srv.atender("hello", "en")
        srv.atender("adios")
        assert mock_cls.cargar.call_count == 2  # sin recargas
        assert en.sintetizar.call_count == 1
        assert es.sintetizar.call_count == 2


def test_atender_enrute_por_idioma() -> None:
    en, es = _tts_mock(), _tts_mock()
    srv = ServidorTTS(en, es)
    srv.atender("hello", "en")
    srv.atender("hola", "es")
    assert en.sintetizar.call_count == 1
    assert es.sintetizar.call_count == 1


def test_atender_idioma_invalido_lanza() -> None:
    srv = ServidorTTS(_tts_mock(), _tts_mock())
    with pytest.raises(ValueError) as excinfo:
        srv.atender("hola", "fr")
    assert str(excinfo.value) == "idioma debe ser 'es' o 'en': 'fr'"


def test_atender_devuelve_pcm_y_sr() -> None:
    srv = ServidorTTS(_tts_mock(), _tts_mock())
    pcm, sr = srv.atender("hola")
    assert pcm == b"\x00\x00"
    assert sr == 24000
