"""Tests del adapter Pocket TTS — sin modelo ni CPU real."""

from __future__ import annotations

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
    # Args exactos: matar mutantes que quitan modelo/state/texto
    tts._model.generate_audio_stream.assert_called_once_with({"voz": 1}, "hola")


def test_sintetizar_stream_entrega_conforme_llega() -> None:
    """Regresión P0: el chunk 2 no se entrega hasta que el consumidor recibió el 1."""

    class StreamExigente:
        def __init__(self) -> None:
            self.recibido = False
            self.n = 0

        def __iter__(self) -> StreamExigente:
            return self

        def __next__(self) -> FakeChunk:
            self.n += 1
            if self.n == 1:
                return FakeChunk([0.0])
            if self.n == 2:
                assert self.recibido, "chunk1 no llegó al consumidor antes del chunk2"
                return FakeChunk([0.5])
            raise StopIteration

    model = MagicMock()
    model.generate_audio_stream.return_value = StreamExigente()
    tts = PocketTTS(model, {"voz": 1}, 24000)
    it = tts.sintetizar_stream("hola")
    c1 = next(it)
    stream = model.generate_audio_stream.return_value
    assert isinstance(stream, StreamExigente)
    stream.recibido = True
    c2 = next(it)
    import struct

    assert struct.unpack("<h", c1) == (0,)
    assert struct.unpack("<h", c2) == (16384,)
    with pytest.raises(StopIteration):
        next(it)


def sintetizar_con_reloj(tts: PocketTTS, texto: str) -> tuple[bytes, int, float]:
    """Reloj falso: t0=1.0, primer chunk a 1.15."""
    tiempos = iter([1.0, 1.15, 1.2])
    return tts.sintetizar(texto, clock=lambda: next(tiempos))


def test_sintetizar_stream_vacio_lanza() -> None:
    with pytest.raises(ValueError) as excinfo:
        list(_tts().sintetizar_stream("   "))
    assert str(excinfo.value) == "texto vacío"


def test_sintetizar_vacio_lanza() -> None:
    with pytest.raises(ValueError) as excinfo:
        _tts().sintetizar("")
    assert str(excinfo.value) == "texto vacío"
    with pytest.raises(ValueError) as excinfo2:
        _tts().sintetizar("   ")
    assert str(excinfo2.value) == "texto vacío"


def test_sintetizar_sin_chunks_lanza() -> None:
    model = MagicMock()
    model.generate_audio_stream.return_value = iter([])
    with pytest.raises(ValueError) as excinfo:
        PocketTTS(model, {}, 24000).sintetizar("hola")
    assert str(excinfo.value) == "modelo no generó ningún chunk"


def test_sintetizar_batch_concatena_chunks() -> None:
    """total += pcm (no =): con 2 chunks el batch trae ambos en orden."""
    import struct

    model = MagicMock()
    model.generate_audio_stream.return_value = iter([FakeChunk([0.0]), FakeChunk([0.5])])
    tts = PocketTTS(model, {"voz": 1}, 24000)
    pcm, _, _ = tts.sintetizar("hola")
    assert struct.unpack("<2h", pcm) == (0, 16384)


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
        assert tts._model is mock_model
        assert tts._voice_state == {"v": 1}
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


def test_a_pcm16_acepta_lista_sin_tolist() -> None:
    """Rama iterable sin .tolist(): listas crudas también valen."""
    import struct

    assert struct.unpack("<2h", _a_pcm16([[0.0, 0.5]])) == (0, 16384)


def test_a_pcm16_escala_exacta() -> None:
    """0.35*32767=11468 (con *32768 daría 11469)."""
    import struct

    assert struct.unpack("<h", _a_pcm16([FakeChunk([0.35])])) == (11468,)
