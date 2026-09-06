"""Tests del contrato worker TTS — sin modelo ni GPU."""

import io
import json
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from traductor.tts.worker import guardar_wav, parse_request, run


def _modelo() -> MagicMock:
    m = MagicMock()
    m.sr = 24000
    m.generate.return_value = MagicMock()
    return m


def test_parse_request_ok() -> None:
    texto, ref, ex, lang = parse_request('{"texto": "hola", "ref_audio": "r.wav"}')
    assert (texto, ref, ex, lang) == ("hola", "r.wav", 0.5, "en")


def test_parse_request_completo() -> None:
    t, r, ex, lang = parse_request(
        '{"texto": "hi", "ref_audio": "r.wav", "exaggeration": 0.7, "language_id": "es"}'
    )
    assert (t, r, ex, lang) == ("hi", "r.wav", 0.7, "es")


def test_parse_request_invalido() -> None:
    with pytest.raises(ValueError):
        parse_request("no json {")
    with pytest.raises(ValueError):
        parse_request('{"texto": "", "ref_audio": "r.wav"}')
    with pytest.raises(ValueError):
        parse_request('{"texto": "hola"}')
    with pytest.raises(ValueError):
        parse_request('{"texto": "hola", "ref_audio": "r.wav", "exaggeration": "mucha"}')
    with pytest.raises(ValueError, match="objeto JSON"):
        parse_request("[1, 2]")
    with pytest.raises(ValueError, match="language_id"):
        parse_request('{"texto": "hola", "ref_audio": "r.wav", "language_id": ""}')


def test_guardar_wav_pcm16_valido(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    destino = tmp_path / "t.wav"
    guardar_wav(destino, np.zeros(240, dtype=np.float32), 24000)
    with wave.open(str(destino), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == 24000
        assert f.getnframes() == 240


def test_guardar_wav_acepta_tensor_torch(tmp_path: Path) -> None:
    """Rama hasattr(wav, 'cpu'): tensores torch exponen .cpu().numpy()."""
    np = pytest.importorskip("numpy")

    class FakeTensor:
        def cpu(self) -> object:
            return self

        def numpy(self) -> object:
            return np.ones(100, dtype=np.float32)

    destino = tmp_path / "t.wav"
    guardar_wav(destino, FakeTensor(), 16000)
    with wave.open(str(destino), "rb") as f:
        assert f.getframerate() == 16000
        assert f.getnframes() == 100


def test_main_carga_una_vez_y_atiende_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() es el smoke de venv-tts: una carga + loop hasta EOF."""
    import sys
    from unittest.mock import patch

    from traductor.tts.worker import main

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    out = tmp_path / "out"
    linea = io.StringIO(json.dumps({"texto": "hola", "ref_audio": str(ref)}))
    monkeypatch.setattr(sys, "stdin", linea)
    with patch("traductor.tts.worker.cargar_modelo", return_value=modelo) as mock_load:
        assert main(["--out-dir", str(out), "--device", "cpu"]) == 0
        mock_load.assert_called_once_with(device="cpu")
    assert modelo.generate.call_count == 1
    assert (out / "0000.wav").exists()


def test_run_procesa_y_sigue_con_errores(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    out = tmp_path / "out"
    entrada = io.StringIO(
        json.dumps({"texto": "hola", "ref_audio": str(ref)})
        + "\n\nlinea rota\n"
        + json.dumps({"texto": "hi", "ref_audio": str(ref), "language_id": "es"})
        + "\n"
    )
    salida = io.StringIO()
    n = run(entrada, salida, modelo, out)
    assert n == 3
    lineas = [json.loads(x) for x in salida.getvalue().strip().split("\n")]
    assert lineas[0]["ok"] is True
    assert lineas[0]["wav"].endswith("0000.wav")
    assert lineas[1]["ok"] is False
    assert lineas[2]["ok"] is True
    # El modelo se inyecta: run nunca recarga (generate x2, from_pretrained x0)
    assert modelo.generate.call_count == 2
    assert (out / "0000.wav").exists()
    assert (out / "0002.wav").exists()


def test_run_no_recarga_modelo(tmp_path: Path) -> None:
    """El worker recibe el modelo ya cargado: carga 0 veces por frase."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    out = tmp_path / "out"
    entrada = io.StringIO(json.dumps({"texto": "a", "ref_audio": str(ref)}) + "\n")
    run(entrada, io.StringIO(), modelo, out)
    modelo.generate.assert_called_once()
