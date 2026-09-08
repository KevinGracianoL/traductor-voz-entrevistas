"""Tests del worker TTS aislado — protocolo job/resultado con backend inyectado."""

import io
import json
from pathlib import Path

from traductor.tts.backend import TTSBackend
from traductor.tts.modelos import AudioResult, Salud, VoiceProfile
from traductor.tts.tienda import VoiceProfileStore
from traductor.tts.worker import Job, main, procesar_job


class BackendFake:
    """TTSBackend falso: devuelve audio fijo o lanza si se pide."""

    def __init__(self, fallar: bool = False) -> None:
        self._fallar = fallar
        self.sintesis: list[tuple[str, VoiceProfile]] = []

    def sintetizar(self, texto: str, perfil: VoiceProfile) -> AudioResult:
        if self._fallar:
            raise RuntimeError("GPU no disponible")
        self.sintesis.append((texto, perfil))
        return AudioResult(datos=f"audio-{texto}".encode(), formato="wav")

    def verificar_salud(self) -> Salud:
        return Salud(disponible=True)

    def cerrar(self) -> None:
        return None


class TiendaFake:
    """VoiceProfileStore en memoria."""

    def __init__(self, perfiles: dict[str, VoiceProfile] | None = None) -> None:
        self._perfiles = dict(perfiles or {})

    def listar(self) -> list[VoiceProfile]:
        return list(self._perfiles.values())

    def obtener(self, perfil_id: str) -> VoiceProfile | None:
        return self._perfiles.get(perfil_id)

    def guardar(self, perfil: VoiceProfile) -> None:
        self._perfiles[perfil.id] = perfil

    def eliminar(self, perfil_id: str) -> bool:
        return self._perfiles.pop(perfil_id, None) is not None


PERFIL = VoiceProfile(id="kevin-es", nombre="Kevin", muestras=("a.wav",))


def test_worker_implementa_el_contrato() -> None:
    assert isinstance(BackendFake(), TTSBackend)
    assert isinstance(TiendaFake(), VoiceProfileStore)


def test_procesar_job_ok(tmp_path: Path) -> None:
    backend = BackendFake()
    salida = tmp_path / "salida.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida=str(salida))
    resultado = procesar_job(job, backend, TiendaFake({"kevin-es": PERFIL}))
    assert resultado["ok"] is True
    assert resultado["salida"] == str(salida)
    assert salida.read_bytes() == b"audio-hola"
    assert resultado["elapsed_ms"] >= 0
    assert backend.sintesis == [("hola", PERFIL)]


def test_procesar_job_usa_reloj_inyectable(tmp_path: Path) -> None:
    salida = tmp_path / "salida.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida=str(salida))
    reloj = iter([0.0, 0.5])

    def clock() -> float:
        return next(reloj)

    resultado = procesar_job(
        job,
        BackendFake(),
        TiendaFake({"kevin-es": PERFIL}),
        clock=clock,
    )
    assert resultado["ok"] is True
    assert resultado["elapsed_ms"] == 500.0


def test_procesar_job_crea_directorio(tmp_path: Path) -> None:
    salida = tmp_path / "sub" / "dir" / "salida.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida=str(salida))
    resultado = procesar_job(job, BackendFake(), TiendaFake({"kevin-es": PERFIL}))
    assert resultado["ok"] is True
    assert salida.exists()


def test_procesar_job_perfil_ausente(tmp_path: Path) -> None:
    salida = tmp_path / "salida.wav"
    job = Job(texto="hola", perfil_id="otro", salida=str(salida))
    resultado = procesar_job(job, BackendFake(), TiendaFake())
    assert resultado["ok"] is False
    assert "perfil no encontrado" in resultado["error"]
    assert not salida.exists()


def test_procesar_job_backend_falla_no_mata(tmp_path: Path) -> None:
    salida = tmp_path / "salida.wav"
    job = Job(texto="hola", perfil_id="kevin-es", salida=str(salida))
    resultado = procesar_job(job, BackendFake(fallar=True), TiendaFake({"kevin-es": PERFIL}))
    assert resultado["ok"] is False
    assert "GPU no disponible" in resultado["error"]
    assert not salida.exists()


def test_main_procesa_jobs(tmp_path: Path) -> None:
    salida = tmp_path / "salida.wav"
    entrada = io.StringIO(
        json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": str(salida)}) + "\n"
    )
    salida_texto = io.StringIO()
    main(BackendFake(), TiendaFake({"kevin-es": PERFIL}), entrada=entrada, salida=salida_texto)
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert lineas[0]["ok"] is True
    assert salida.read_bytes() == b"audio-hola"


def test_main_job_invalido_continua(tmp_path: Path) -> None:
    salida = tmp_path / "salida.wav"
    entrada = io.StringIO(
        "{no-json}\n"
        + json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": str(salida)})
        + "\n"
    )
    salida_texto = io.StringIO()
    main(BackendFake(), TiendaFake({"kevin-es": PERFIL}), entrada=entrada, salida=salida_texto)
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert lineas[0]["ok"] is False
    assert "job inválido" in lineas[0]["error"]
    assert lineas[1]["ok"] is True


def test_main_continua_despues_de_linea_vacia(tmp_path: Path) -> None:
    salida = tmp_path / "salida.wav"
    entrada = io.StringIO(
        "\n" + json.dumps({"texto": "hola", "perfil_id": "kevin-es", "salida": str(salida)}) + "\n"
    )
    salida_texto = io.StringIO()
    main(BackendFake(), TiendaFake({"kevin-es": PERFIL}), entrada=entrada, salida=salida_texto)
    lineas = [json.loads(linea) for linea in salida_texto.getvalue().strip().splitlines()]
    assert lineas[0]["ok"] is True
    assert salida.exists()


def test_main_salta_lineas_vacias(tmp_path: Path) -> None:
    entrada = io.StringIO("\n\n")
    salida_texto = io.StringIO()
    main(BackendFake(), TiendaFake(), entrada=entrada, salida=salida_texto)
    assert salida_texto.getvalue() == ""
