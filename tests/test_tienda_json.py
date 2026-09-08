"""Tests de TiendaPerfilesJson — persistencia real de VoiceProfile (PR #14)."""

from pathlib import Path

import pytest

from traductor.tts.modelos import VoiceProfile
from traductor.tts.tienda import VoiceProfileStore
from traductor.tts.tienda_json import TiendaPerfilesJson


def test_tienda_implementa_el_contrato(tmp_path: Path) -> None:
    assert isinstance(TiendaPerfilesJson(tmp_path), VoiceProfileStore)


def test_tienda_guardar_y_obtener(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    perfil = VoiceProfile(id="kevin-es", nombre="Kevin", muestras=("a.wav", "b.wav"))
    tienda.guardar(perfil)
    assert tienda.obtener("kevin-es") == perfil


def test_tienda_obtener_ausente_none(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    assert tienda.obtener("no-existe") is None


def test_tienda_listar_vacio(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    assert tienda.listar() == []


def test_tienda_listar_sin_directorio(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path / "no-existe")
    assert tienda.listar() == []


def test_tienda_listar_varios(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    tienda.guardar(VoiceProfile(id="b", nombre="B", muestras=("1.wav",)))
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("2.wav",)))
    assert [p.id for p in tienda.listar()] == ["a", "b"]


def test_tienda_listar_ignora_no_json(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    (tmp_path / "nota.txt").write_text("hola", encoding="utf-8")
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("1.wav",)))
    assert [p.id for p in tienda.listar()] == ["a"]


def test_tienda_guardar_reemplaza(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    tienda.guardar(VoiceProfile(id="a", nombre="Antes", muestras=("1.wav",)))
    tienda.guardar(VoiceProfile(id="a", nombre="Después", muestras=("2.wav",)))
    perfil = tienda.obtener("a")
    assert perfil is not None
    assert perfil.nombre == "Después"


def test_tienda_eliminar(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("1.wav",)))
    assert tienda.eliminar("a") is True
    assert tienda.eliminar("a") is False
    assert tienda.obtener("a") is None


def test_tienda_archivo_utf8(tmp_path: Path) -> None:
    """Tildes sobreviven el round-trip: guardar → leer es idéntico."""
    tienda = TiendaPerfilesJson(tmp_path)
    perfil = VoiceProfile(id="a", nombre="Álvaro", muestras=("café.wav",))
    tienda.guardar(perfil)
    assert tienda.obtener("a") == perfil


def test_tienda_id_no_texto_raise(tmp_path: Path) -> None:
    """Un perfil con id no-texto (solo uno de los campos mal) falla claro."""
    tienda = TiendaPerfilesJson(tmp_path)
    (tmp_path / "a.json").write_bytes(b'{"id": 1, "nombre": "A", "muestras": ["1.wav"]}')
    with pytest.raises(ValueError, match="^perfil inválido: id y nombre deben ser texto$"):
        tienda.obtener("a")


def test_tienda_muestras_no_lista_raise(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path)
    (tmp_path / "a.json").write_bytes(b'{"id": "a", "nombre": "A", "muestras": "1.wav"}')
    with pytest.raises(ValueError, match="^perfil inválido: muestras debe ser una lista$"):
        tienda.obtener("a")


def test_tienda_archivo_corrupto_raise(tmp_path: Path) -> None:
    """Un perfil corrupto NO se silencia: listar/obtener fallan claro."""
    tienda = TiendaPerfilesJson(tmp_path)
    (tmp_path / "a.json").write_bytes(b"{no-json")
    with pytest.raises(ValueError):
        tienda.obtener("a")


def test_tienda_guardar_crea_directorio(tmp_path: Path) -> None:
    tienda = TiendaPerfilesJson(tmp_path / "sub" / "dir")
    tienda.guardar(VoiceProfile(id="a", nombre="A", muestras=("1.wav",)))
    assert (tmp_path / "sub" / "dir" / "a.json").exists()
