"""Tests del contrato worker TTS — sin modelo ni GPU."""

import io
import json
import wave
from collections.abc import Iterator
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
    with pytest.raises(ValueError, match="no es JSON"):
        parse_request("no json {")
    with pytest.raises(ValueError) as e1:
        parse_request('{"texto": "", "ref_audio": "r.wav"}')
    assert str(e1.value) == "campo 'texto' requerido y no vacío"
    with pytest.raises(ValueError) as e2:
        parse_request('{"texto": "hola"}')
    assert str(e2.value) == "campo 'ref_audio' requerido"
    with pytest.raises(ValueError) as e2b:
        parse_request('{"texto": "hola", "ref_audio": ""}')
    assert str(e2b.value) == "campo 'ref_audio' requerido"
    with pytest.raises(ValueError) as e3:
        parse_request('{"texto": "hola", "ref_audio": "r.wav", "exaggeration": "mucha"}')
    assert str(e3.value) == "campo 'exaggeration' debe ser número"
    with pytest.raises(ValueError) as e5:
        parse_request("[1, 2]")
    assert str(e5.value) == "request debe ser un objeto JSON"
    with pytest.raises(ValueError) as e4:
        parse_request('{"texto": "hola", "ref_audio": "r.wav", "language_id": ""}')
    assert str(e4.value) == "campo 'language_id' debe ser string no vacío"


def test_guardar_wav_pcm16_valido(tmp_path: Path) -> None:
    destino = tmp_path / "t.wav"
    guardar_wav(destino, [0.0] * 240, 24000)
    with wave.open(str(destino), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == 24000
        assert f.getnframes() == 240


def test_guardar_wav_acepta_tensor_torch(tmp_path: Path) -> None:
    """Rama hasattr(wav, 'tolist'): tensores torch exponen .tolist()."""

    class FakeTensor:
        def tolist(self) -> list[list[float]]:
            return [[1.0] * 100]

    destino = tmp_path / "t.wav"
    guardar_wav(destino, FakeTensor(), 16000)
    with wave.open(str(destino), "rb") as f:
        assert f.getframerate() == 16000
        assert f.getnframes() == 100


def test_guardar_wav_recorta_fuera_de_rango(tmp_path: Path) -> None:
    """Clip a [-1, 1]: 2.0 -> 32767, -2.0 -> -32768."""
    import struct

    destino = tmp_path / "t.wav"
    guardar_wav(destino, [2.0, -2.0, 0.5, 0.35], 8000)
    with wave.open(str(destino), "rb") as f:
        # 0.35*32767=11468.45->11468 (con *32768 daría 11469)
        assert struct.unpack("<4h", f.readframes(4)) == (32767, -32768, 16384, 11468)


def test_run_flush_por_respuesta(tmp_path: Path) -> None:
    """Sin flush por respuesta, un cliente persistente se bloquea (deadlock)."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    out = tmp_path / "out"
    lineas = [
        json.dumps({"texto": "hola", "ref_audio": str(ref)}),
        "linea rota",
        json.dumps({"texto": "hi", "ref_audio": str(ref)}),
    ]
    flushes: list[int] = []

    class Salida(io.StringIO):
        def flush(self) -> None:
            flushes.append(1)
            super().flush()

    def gen() -> Iterator[str]:
        for i, linea in enumerate(lineas):
            if i > 0:
                assert len(flushes) == i, "respuesta anterior sin flush: deadlock"
            yield linea

    salida = Salida()
    assert run(gen(), salida, modelo, out) == 3
    assert len(flushes) == 3


def test_main_carga_una_vez_y_atiende_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    assert "frases: 1" in capsys.readouterr().err


def test_run_procesa_y_sigue_con_errores(tmp_path: Path) -> None:
    from unittest.mock import patch

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    out = tmp_path / "out"
    entrada = io.StringIO(
        json.dumps({"texto": "hola", "ref_audio": str(ref), "exaggeration": 0.7})
        + "\n\nlinea rota\n"
        + json.dumps({"texto": "hi", "ref_audio": str(ref), "language_id": "es"})
        + "\n"
    )
    salida = io.StringIO()
    with patch("time.perf_counter", side_effect=[1.0, 1.56789, 2.0, 2.25]):
        n = run(entrada, salida, modelo, out)
    assert n == 3
    lineas = [json.loads(x) for x in salida.getvalue().strip().split("\n")]
    assert lineas[0]["ok"] is True
    assert lineas[0]["wav"].endswith("0000.wav")
    assert lineas[0]["sr"] == 24000
    assert lineas[0]["ms"] == 567.9
    assert lineas[1]["ok"] is False
    assert "línea no es JSON" in lineas[1]["error"]
    assert lineas[2]["ok"] is True
    assert lineas[2]["wav"].endswith("0002.wav")
    assert lineas[2]["ms"] == 250.0
    # El modelo se inyecta: run nunca recarga (generate x2, from_pretrained x0)
    assert modelo.generate.call_count == 2
    llamadas = modelo.generate.call_args_list
    assert llamadas[0].kwargs["text"] == "hola"
    assert llamadas[0].kwargs["language_id"] == "en"
    assert llamadas[0].kwargs["audio_prompt_path"] == str(ref)
    assert llamadas[0].kwargs["exaggeration"] == 0.7
    assert llamadas[1].kwargs["text"] == "hi"
    assert llamadas[1].kwargs["language_id"] == "es"
    assert (out / "0000.wav").exists()
    assert (out / "0002.wav").exists()


def test_run_crea_dir_anidado(tmp_path: Path) -> None:
    """parents=True: crea intermedios que no existen."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    out = tmp_path / "a" / "b"
    entrada = io.StringIO(json.dumps({"texto": "hola", "ref_audio": str(ref)}) + "\n")
    assert run(entrada, io.StringIO(), modelo, out) == 1
    assert (out / "0000.wav").exists()


def test_run_dir_existente_no_falla(tmp_path: Path) -> None:
    """exist_ok=True: un out_dir pre-creado no lanza."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    out = tmp_path / "out"
    out.mkdir()
    entrada = io.StringIO(json.dumps({"texto": "hola", "ref_audio": str(ref)}) + "\n")
    assert run(entrada, io.StringIO(), modelo, out) == 1
    assert (out / "0000.wav").exists()


def test_main_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main([]) usa --out-dir out_tts y --device cuda por defecto."""
    import sys
    from unittest.mock import patch

    from traductor.tts.worker import main

    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    linea = io.StringIO(json.dumps({"texto": "hola", "ref_audio": str(ref)}))
    monkeypatch.setattr(sys, "stdin", linea)
    monkeypatch.chdir(tmp_path)
    with patch("traductor.tts.worker.cargar_modelo", return_value=modelo) as mock_load:
        assert main([]) == 0
        mock_load.assert_called_once_with(device="cuda")
    assert (tmp_path / "out_tts" / "0000.wav").exists()


def test_run_no_recarga_modelo(tmp_path: Path) -> None:
    """El worker recibe el modelo ya cargado: carga 0 veces por frase."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    modelo = _modelo()
    out = tmp_path / "out"
    entrada = io.StringIO(json.dumps({"texto": "a", "ref_audio": str(ref)}) + "\n")
    run(entrada, io.StringIO(), modelo, out)
    modelo.generate.assert_called_once()
