"""Tests de enrolamiento — muestras grabadas → VoiceProfile validado (PR #14)."""

import re
from pathlib import Path

import pytest

from traductor.tts.enrolamiento import enrolar
from traductor.tts.modelos import VoiceProfile


def test_enrolar_valido(tmp_path: Path) -> None:
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"RIFF")
    b.write_bytes(b"RIFF")
    perfil = enrolar("kevin-es", "Kevin", [a, b])
    assert perfil == VoiceProfile(id="kevin-es", nombre="Kevin", muestras=(str(a), str(b)))


def test_enrolar_acepta_strings(tmp_path: Path) -> None:
    a = tmp_path / "a.wav"
    a.write_bytes(b"RIFF")
    perfil = enrolar("kevin-es", "Kevin", [str(a)])
    assert perfil.muestras == (str(a),)


def test_enrolar_muestras_vacias_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^muestras vacías: se necesita al menos una muestra$"):
        enrolar("kevin-es", "Kevin", [])


def test_enrolar_muestra_faltante_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no encontradas"):
        enrolar("kevin-es", "Kevin", [tmp_path / "no-existe.wav"])


def test_enrolar_varias_faltantes_join_exacto(tmp_path: Path) -> None:
    esperado = f"{tmp_path / 'a-falta.wav'}, {tmp_path / 'b-falta.wav'}"
    with pytest.raises(ValueError, match=re.escape(esperado)):
        enrolar("kevin-es", "Kevin", [tmp_path / "a-falta.wav", tmp_path / "b-falta.wav"])


def test_enrolar_muestra_no_archivo_raise(tmp_path: Path) -> None:
    directorio = tmp_path / "carpeta"
    directorio.mkdir()
    with pytest.raises(ValueError, match="no son archivos"):
        enrolar("kevin-es", "Kevin", [directorio])


def test_enrolar_varios_no_archivo_join_exacto(tmp_path: Path) -> None:
    d1 = tmp_path / "c1"
    d2 = tmp_path / "c2"
    d1.mkdir()
    d2.mkdir()
    esperado = f"{d1}, {d2}"
    with pytest.raises(ValueError, match=re.escape(esperado)):
        enrolar("kevin-es", "Kevin", [d1, d2])


def test_enrolar_delega_validacion_del_perfil(tmp_path: Path) -> None:
    a = tmp_path / "a.wav"
    a.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="nombre"):
        enrolar("kevin-es", "   ", [a])
