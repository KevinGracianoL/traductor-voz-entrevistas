"""Tests para TTS Chatterbox — validación sin modelo ni GPU.

Chatterbox importa de forma perezosa; CI no instala la dep pesada.
Inyectamos módulos falsos en sys.modules.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from traductor.tts.chatterbox import cargar_modelo, sintetizar


def _inject_chatterbox() -> MagicMock:
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


def test_cargar_modelo_defaults() -> None:
    """cargar_modelo() sin args usa device='cuda' y multilingüe."""
    mock_cls = _inject_chatterbox()
    try:
        cargar_modelo()
        mock_cls.from_pretrained.assert_called_once_with(device="cuda")
    finally:
        sys.modules.pop("chatterbox", None)
        sys.modules.pop("chatterbox.mtl_tts", None)


def test_cargar_modelo_device_passthrough() -> None:
    mock_cls = _inject_chatterbox()
    try:
        cargar_modelo(device="cpu")
        mock_cls.from_pretrained.assert_called_once_with(device="cpu")
    finally:
        sys.modules.pop("chatterbox", None)
        sys.modules.pop("chatterbox.mtl_tts", None)


def test_sintetizar_vacio_lanza(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    mock_model = _mock_modelo()
    with pytest.raises(ValueError) as excinfo:
        sintetizar("", str(ref), mock_model)
    assert str(excinfo.value) == "texto vacío"
    with pytest.raises(ValueError) as excinfo2:
        sintetizar("   ", str(ref), mock_model)
    assert str(excinfo2.value) == "texto vacío"


def test_sintetizar_ref_no_existe_lanza() -> None:
    mock_model = _mock_modelo()
    with pytest.raises(FileNotFoundError) as excinfo:
        sintetizar("hello", "/no/existe.wav", mock_model)
    # Path normaliza distinto en Windows/Ubuntu: comparar contra el mismo Path
    assert str(excinfo.value) == f"ref_audio no existe: {Path('/no/existe.wav')}"


def test_sintetizar_ref_directorio_lanza(tmp_path: Path) -> None:
    """Un directorio no es un clip válido aunque exista."""
    with pytest.raises(FileNotFoundError) as excinfo:
        sintetizar("hello", tmp_path, _mock_modelo())
    assert str(excinfo.value) == f"ref_audio no existe: {tmp_path}"


def _mock_modelo() -> MagicMock:
    m = MagicMock()
    m.sr = 24000
    m.generate.return_value = MagicMock()
    return m


def test_sintetizar_exaggeration_frontera(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    # 0.0 y 1.0 son válidos (límites inclusivos)
    mock_model = _mock_modelo()
    sintetizar("hello", str(ref), mock_model, exaggeration=0.0)
    sintetizar("hello", str(ref), mock_model, exaggeration=1.0)
    assert mock_model.generate.call_count == 2

    with pytest.raises(ValueError) as excinfo:
        sintetizar("hello", str(ref), _mock_modelo(), exaggeration=-0.1)
    assert str(excinfo.value) == "exaggeration fuera de rango [0,1]: -0.1"
    with pytest.raises(ValueError) as excinfo:
        sintetizar("hello", str(ref), _mock_modelo(), exaggeration=1.1)
    assert str(excinfo.value) == "exaggeration fuera de rango [0,1]: 1.1"


def test_sintetizar_ok_llama_generate(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    mock_model = _mock_modelo()
    wav, sr = sintetizar("hello", str(ref), mock_model, exaggeration=0.5, language_id="en")
    mock_model.generate.assert_called_once()
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["text"] == "hello"
    assert kwargs["audio_prompt_path"] == str(ref)
    assert kwargs["exaggeration"] == 0.5
    assert kwargs["language_id"] == "en"
    assert sr == 24000
    assert wav is mock_model.generate.return_value


def test_sintetizar_defaults_language_en(tmp_path: Path) -> None:
    """Omitir language_id usa 'en' por defecto."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    mock_model = _mock_modelo()
    sintetizar("hello", str(ref), mock_model)
    kwargs = mock_model.generate.call_args.kwargs
    assert kwargs["language_id"] == "en"


def test_sintetizar_reutiliza_modelo(tmp_path: Path) -> None:
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
