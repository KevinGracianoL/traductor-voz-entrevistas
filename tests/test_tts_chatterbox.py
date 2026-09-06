"""Tests para TTS Chatterbox — validación sin modelo ni GPU."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from traductor.tts.chatterbox import sintetizar


def _inject_chatterbox() -> MagicMock:
    """Inyecta chatterbox falso en sys.modules."""
    pkg = types.ModuleType("chatterbox")
    mtl = types.ModuleType("chatterbox.mtl_tts")
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()
    mock_cls = MagicMock()
    mock_cls.from_pretrained = MagicMock(return_value=mock_model)
    mtl.ChatterboxMultilingualTTS = mock_cls  # type: ignore[attr-defined]
    pkg.mtl_tts = mtl  # type: ignore[attr-defined]
    sys.modules["chatterbox"] = pkg
    sys.modules["chatterbox.mtl_tts"] = mtl
    return mock_cls


def test_sintetizar_vacio_lanza(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    with pytest.raises(ValueError, match="texto vacío"):
        sintetizar("", str(ref))
    with pytest.raises(ValueError, match="texto vacío"):
        sintetizar("   ", str(ref))


def test_sintetizar_ref_no_existe_lanza() -> None:
    with pytest.raises(FileNotFoundError, match="ref_audio no existe"):
        sintetizar("hello", "/no/existe.wav")


def test_sintetizar_exaggeration_fuera_rango_lanza(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    with pytest.raises(ValueError, match="exaggeration fuera"):
        sintetizar("hello", str(ref), exaggeration=2.0)
    with pytest.raises(ValueError, match="exaggeration fuera"):
        sintetizar("hello", str(ref), exaggeration=-0.1)


def test_sintetizar_ok_llama_generate(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    mock_cls = _inject_chatterbox()
    try:
        wav, sr = sintetizar("hello", str(ref), exaggeration=0.5, language_id="en")
        mock_cls.from_pretrained.assert_called_once_with(device="cuda")
        mock_model = mock_cls.from_pretrained.return_value
        mock_model.generate.assert_called_once()
        kwargs = mock_model.generate.call_args.kwargs
        assert kwargs["text"] == "hello"
        assert kwargs["audio_prompt_path"] == str(ref)
        assert kwargs["exaggeration"] == 0.5
        assert kwargs["language_id"] == "en"
        assert sr == 24000
        assert wav is mock_model.generate.return_value
    finally:
        sys.modules.pop("chatterbox", None)
        sys.modules.pop("chatterbox.mtl_tts", None)


def test_sintetizar_pasa_language_id(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    mock_cls = _inject_chatterbox()
    try:
        sintetizar("hola", str(ref), language_id="es")
        mock_model = mock_cls.from_pretrained.return_value
        mock_model.generate.assert_called_once()
        assert mock_model.generate.call_args.kwargs["language_id"] == "es"
    finally:
        sys.modules.pop("chatterbox", None)
        sys.modules.pop("chatterbox.mtl_tts", None)


def test_sintetizar_reutiliza_modelo(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Si se pasa modelo, no recarga (1 from_pretrained + 2 generate)."""
    fake_ref = tmp_path / "ref.wav"
    fake_ref.write_bytes(b"fake")
    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = MagicMock()

    wav1, _ = sintetizar("hola", fake_ref, modelo=mock_model)
    wav2, _ = sintetizar("hello", fake_ref, modelo=mock_model)
    assert mock_model.generate.call_count == 2
    assert wav1 is mock_model.generate.return_value
    assert wav2 is mock_model.generate.return_value
