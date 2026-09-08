"""Tests de los contratos neutrales de TTS — dominio + protocolos + fakes.

El PR #12 NO implementa un backend TTS: define los contratos contra los que se
escribirán los PRs siguientes (worker TTS, gates de latencia). Estos tests
fijan la forma del contrato y prueban que un fake cualquiera lo satisface.

Nota sobre `isinstance` + `runtime_checkable`: solo comprueba que EXISTAN los
miembros, no sus firmas. La conformidad de firmas la exige `mypy .` en CI
(`mypy archivo.py` suelto no resuelve `traductor.*` y el gate se volvería hueco).
"""

import pytest

from traductor.tts.backend import TTSBackend
from traductor.tts.modelos import AudioResult, Salud, VoiceProfile
from traductor.tts.tienda import VoiceProfileStore


def test_perfil_valido() -> None:
    perfil = VoiceProfile(
        id="kevin-es",
        nombre="Kevin",
        muestras=("samples/a.wav", "samples/b.wav"),
    )
    assert perfil.id == "kevin-es"
    assert perfil.nombre == "Kevin"
    assert perfil.muestras == ("samples/a.wav", "samples/b.wav")


def test_perfil_quita_espacios() -> None:
    perfil = VoiceProfile(id="  kevin-es  ", nombre="  Kevin  ", muestras=("a.wav",))
    assert perfil.id == "kevin-es"
    assert perfil.nombre == "Kevin"


def test_perfil_sin_id_raise() -> None:
    with pytest.raises(ValueError, match="id"):
        VoiceProfile(id="", nombre="Kevin", muestras=("a.wav",))


def test_perfil_sin_nombre_raise() -> None:
    with pytest.raises(ValueError, match="nombre"):
        VoiceProfile(id="kevin-es", nombre="   ", muestras=("a.wav",))


def test_perfil_sin_muestras_raise() -> None:
    with pytest.raises(ValueError, match="muestras"):
        VoiceProfile(id="kevin-es", nombre="Kevin", muestras=())


def test_audio_result_valido() -> None:
    audio = AudioResult(datos=b"RIFF", formato="wav")
    assert audio.datos == b"RIFF"
    assert audio.formato == "wav"
    assert audio.duracion_s is None


def test_audio_result_con_duracion() -> None:
    audio = AudioResult(datos=b"RIFF", formato="wav", duracion_s=1.5)
    assert audio.duracion_s == 1.5


def test_audio_result_sin_formato_raise() -> None:
    with pytest.raises(ValueError, match="formato"):
        AudioResult(datos=b"RIFF", formato="  ")


def test_salud_disponible() -> None:
    salud = Salud(disponible=True)
    assert salud.disponible is True
    assert salud.detalle == ""


def test_salud_detalle() -> None:
    salud = Salud(disponible=False, detalle="GPU no encontrada")
    assert salud.disponible is False
    assert salud.detalle == "GPU no encontrada"


def test_salud_caida_sin_detalle_raise() -> None:
    with pytest.raises(ValueError, match="detalle"):
        Salud(disponible=False, detalle="")


class BackendFake:
    """Fake que cumple TTSBackend: prueba de que el contrato es satisfacible."""

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:
        return AudioResult(datos=f"audio-de-{texto}".encode(), formato="wav")

    def verificar_salud(self) -> Salud:
        return Salud(disponible=True, detalle="fake ok")

    def cerrar(self) -> None:
        return None


class SinCerrar:
    """Falta `cerrar`: isinstance detecta miembros ausentes, no firmas
    incompatibles: eso lo cubre mypy."""

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:
        return AudioResult(datos=b"x", formato="wav")

    def verificar_salud(self) -> Salud:
        return Salud(disponible=True)


def test_backend_fake_satisface_el_contrato() -> None:
    assert isinstance(BackendFake(), TTSBackend)


def test_objeto_sin_miembro_no_es_backend() -> None:
    """isinstance detecta el miembro ausente (no valida firmas: eso es mypy)."""
    assert not isinstance(SinCerrar(), TTSBackend)


def test_backend_fake_uso_real() -> None:
    perfil = VoiceProfile(id="kevin-es", nombre="Kevin", muestras=("a.wav",))
    backend = BackendFake()
    audio = backend.sintetizar("hola", perfil)
    assert audio.datos == b"audio-de-hola"
    assert backend.verificar_salud().disponible is True


class TiendaFake:
    """Fake que cumple VoiceProfileStore: memoria en proceso."""

    def __init__(self) -> None:
        self._perfiles: dict[str, VoiceProfile] = {}

    def listar(self) -> list[VoiceProfile]:
        return list(self._perfiles.values())

    def obtener(self, perfil_id: str) -> VoiceProfile | None:
        return self._perfiles.get(perfil_id)

    def guardar(self, perfil: VoiceProfile) -> None:
        self._perfiles[perfil.id] = perfil

    def eliminar(self, perfil_id: str) -> bool:
        return self._perfiles.pop(perfil_id, None) is not None


def test_tienda_fake_satisface_el_contrato() -> None:
    assert isinstance(TiendaFake(), VoiceProfileStore)


def test_tienda_fake_uso_real() -> None:
    perfil = VoiceProfile(id="kevin-es", nombre="Kevin", muestras=("a.wav",))
    tienda = TiendaFake()
    tienda.guardar(perfil)
    assert tienda.listar() == [perfil]
    assert tienda.obtener("kevin-es") == perfil
    assert tienda.obtener("otro") is None
    assert tienda.eliminar("kevin-es") is True
    assert tienda.eliminar("kevin-es") is False
    assert tienda.listar() == []
