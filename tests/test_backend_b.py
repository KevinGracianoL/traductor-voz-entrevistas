"""Tests del BackendB (Supertonic 3 + OpenVoice V2) — fakes stdlib-only.

En CI ni `supertonic` ni `openvoice` están instalados: se prueba el estado sin
cargar (healthcheck honesto sin efectos secundarios, cierre idempotente, error
claro al sintetizar) y la ORQUESTACIÓN del pipeline con fakes que reproducen
la API real (synthesize → buffer de muestras, save_audio/convert → WAV real
vía el módulo `wave` de la stdlib). Sin numpy/torch/soundfile: el venv de
mutmut (WSL) es mínimo y los fakes no necesitan más. El motor real corre en la
máquina objetivo (harness del ADR-014).

Los fakes son estrictos donde el motor real lo es: `extraer_se` falla si el
archivo no existe o si recibe otro conversor (como `se_extractor.get_se` real)
— así un mutante que escribe en la ruta equivocada se detecta.
"""

import os
import sys
import wave
from array import array
from pathlib import Path
from typing import Any

import pytest

from traductor.tts.backend import TTSBackend
from traductor.tts.backend_b import TEXTO_PROBE, VOZ_SUPERTONIC, BackendB
from traductor.tts.modelos import VoiceProfile

SR_SUPERTONIC = 44100


def _checkpoints(tmp_path: Path) -> str:
    """Dir de pesos mínimo para construir BackendB (validado en __init__)."""
    converter = tmp_path / "checkpoints" / "converter"
    converter.mkdir(parents=True)
    (converter / "config.json").write_text("{}", encoding="utf-8")
    (converter / "checkpoint.pth").write_bytes(b"pesos")
    return str(tmp_path / "checkpoints")


def _escribir_wav(path: str, sr: int, muestras: Any) -> Path:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(muestras.tobytes())
    return Path(path)


class _TtsFake:
    """Reproduce la API real de supertonic.TTS (v1.3.1): buffer + save_audio."""

    def __init__(self) -> None:
        self.estilo = object()
        self.textos: list[tuple[str, str]] = []
        self.guardados: list[str] = []

    def get_voice_style(self, voice_name: str) -> object:
        assert voice_name == VOZ_SUPERTONIC
        return self.estilo

    def synthesize(self, text: str, voice_style: object, lang: str) -> tuple[Any, object]:
        assert voice_style is not None  # el backend pasa el estilo cargado
        self.textos.append((text, lang))
        return array("h", [0] * SR_SUPERTONIC), [1.0]

    def save_audio(self, wav: Any, path: str) -> None:
        assert os.path.isabs(path)  # el backend siempre escribe en su tempdir
        self.guardados.append(path)
        _escribir_wav(path, SR_SUPERTONIC, wav)


class _ConversorFake:
    """Reproduce la API real de ToneColorConverter.convert (escribe WAV)."""

    def __init__(self) -> None:
        self.llamadas: list[tuple[str, object, object]] = []
        self.salidas: list[str] = []

    def convert(
        self,
        audio_src_path: str,
        src_se: object,
        tgt_se: object,
        output_path: str,
        tau: float = 0.3,
        message: str = "default",
    ) -> None:
        assert os.path.isabs(audio_src_path)  # como el motor real: rutas reales
        assert os.path.isabs(output_path)
        assert src_se is not None and tgt_se is not None  # el modelo real no acepta None
        self.llamadas.append((audio_src_path, src_se, tgt_se))
        self.salidas.append(output_path)
        with wave.open(audio_src_path, "rb") as w:
            frames = w.readframes(w.getnframes())
            sr = w.getframerate()
        _escribir_wav(output_path, sr, array("h", [0]) * (len(frames) // 2))


def _backend_con_fakes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[BackendB, _TtsFake, _ConversorFake, list[str], VoiceProfile, VoiceProfile]:
    backend = BackendB(dir_checkpoints=_checkpoints(tmp_path))
    tts = _TtsFake()
    conv = _ConversorFake()
    extracciones: list[str] = []

    ref_kevin = _escribir_wav(str(tmp_path / "ref1.wav"), SR_SUPERTONIC, array("h", [0] * 8000))
    ref_otro = _escribir_wav(str(tmp_path / "refA.wav"), SR_SUPERTONIC, array("h", [0] * 8000))
    perfil = VoiceProfile(id="kevin", nombre="Kevin", muestras=(str(ref_kevin),))
    perfil_2 = VoiceProfile(id="otro", nombre="Otro", muestras=(str(ref_otro),))

    def extraer(ruta: Path, _conv: object, _dir: str) -> object:
        assert _conv is conv  # el conversor real se usa siempre
        assert ruta.is_file()  # como get_se real: falla si el archivo no existe
        assert Path(_dir).is_dir()  # el dir de trabajo existe (get_se real)
        extracciones.append(ruta.name)
        return object()

    def cargar_supertonic() -> _TtsFake:
        backend._estilo = tts.estilo  # como el `_cargar_supertonic` real
        return tts

    monkeypatch.setattr(backend, "_cargar_supertonic", cargar_supertonic)
    monkeypatch.setattr(backend, "_cargar_conversor", lambda: conv)
    monkeypatch.setattr(backend, "_extraer_se", extraer)
    return backend, tts, conv, extracciones, perfil, perfil_2


def test_backend_satisface_el_contrato(tmp_path: Path) -> None:
    assert isinstance(BackendB(dir_checkpoints=_checkpoints(tmp_path)), TTSBackend)


def test_defaults_de_construccion(tmp_path: Path) -> None:
    dir_checkpoints = _checkpoints(tmp_path)
    backend = BackendB(dir_checkpoints=dir_checkpoints)
    assert backend._dir_checkpoints == Path(dir_checkpoints)
    assert backend._voz == "M1"
    assert backend._device == "cpu"
    assert backend._idioma_salida == "en"
    assert backend._tts is None
    assert backend._estilo is None
    assert backend._conversor is None
    assert backend._src_se_cache is None
    assert backend._se_por_perfil == {}


def test_construccion_valida_pesos_temprano(tmp_path: Path) -> None:
    """Sin los pesos de OpenVoice el error es claro en __init__, no dentro del
    converter (el DIR_CHECKPOINTS_B del script es un default de máquina)."""
    with pytest.raises(
        RuntimeError,
        match=r"^pesos de OpenVoice V2 incompletos en .*: faltan converter/",
    ):
        BackendB(dir_checkpoints=str(tmp_path / "sin-pesos"))
    solo_config = tmp_path / "solo-config"
    (solo_config / "converter").mkdir(parents=True)
    (solo_config / "converter" / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint.pth"):
        BackendB(dir_checkpoints=str(solo_config))


def test_verificar_salud_sin_motores_reporta_indisponible(tmp_path: Path) -> None:
    """En CI (sin supertonic/openvoice) el healthcheck es honesto: False+detalle."""
    salud = BackendB(dir_checkpoints=_checkpoints(tmp_path)).verificar_salud()
    assert salud.disponible is False
    assert salud.detalle.startswith("supertonic no instalado: ")


def test_verificar_salud_sin_openvoice_reporta_indisponible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """supertonic instalado pero openvoice no (estado actual de venv-tts)."""
    monkeypatch.setitem(sys.modules, "supertonic", object())
    salud = BackendB(dir_checkpoints=_checkpoints(tmp_path)).verificar_salud()
    assert salud.disponible is False
    assert salud.detalle.startswith("openvoice no instalado: ")


def test_verificar_salud_medio_cargado_reporta_indisponible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Solo UNO de los dos modelos cargado: sigue siendo no disponible (or, no and)."""
    backend = BackendB(dir_checkpoints=_checkpoints(tmp_path))
    backend._conversor = object()
    salud = backend.verificar_salud()
    assert salud.disponible is False
    assert "no instalado" in salud.detalle


def test_verificar_salud_no_carga_los_modelos(tmp_path: Path) -> None:
    backend = BackendB(dir_checkpoints=_checkpoints(tmp_path))
    backend.verificar_salud()
    assert backend._tts is None
    assert backend._conversor is None


def test_verificar_salud_con_modelos_cargados(tmp_path: Path) -> None:
    backend = BackendB(dir_checkpoints=_checkpoints(tmp_path))
    backend._tts = object()
    backend._estilo = object()
    backend._conversor = object()
    salud = backend.verificar_salud()
    assert salud.disponible is True
    assert VOZ_SUPERTONIC in salud.detalle


def test_sintetizar_sin_motores_error_claro(tmp_path: Path) -> None:
    backend = BackendB(dir_checkpoints=_checkpoints(tmp_path))
    perfil = VoiceProfile(id="x", nombre="X", muestras=("r.wav",))
    with pytest.raises(
        RuntimeError,
        match=(
            "^supertonic no instalado: pip install supertonic "
            r"\(venv propio del TTS, ver ADR-011/014\)$"
        ),
    ):
        backend.sintetizar("hello", perfil)


def test_cerrar_idempotente(tmp_path: Path) -> None:
    backend = BackendB(dir_checkpoints=_checkpoints(tmp_path))
    backend.cerrar()
    backend.cerrar()
    assert backend._tts is None
    assert backend._estilo is None
    assert backend._conversor is None


def test_constantes_del_candidato_b() -> None:
    assert VOZ_SUPERTONIC == "M1"
    assert TEXTO_PROBE


def test_sintetizar_pipeline_completo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend, tts, conv, extracciones, perfil, _ = _backend_con_fakes(monkeypatch, tmp_path)
    audio = backend.sintetizar("Thank you.", perfil)
    assert audio.formato == "wav"
    assert len(audio.datos) > 0
    assert audio.duracion_s == pytest.approx(1.0)
    assert tts.textos == [("Thank you.", "en"), (TEXTO_PROBE, "en")]
    assert len(conv.llamadas) == 1
    assert len(extracciones) == 2  # src_se + target_se


def test_rutas_de_archivos_temporales(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """El backend escribe en un dir temporal propio y el convertidor recibe el out.

    Mata los mutantes de strings de `mkdtemp`/`src.wav`/`out.wav`: el src va al
    dir temporal `backend-b-*` y el convertidor escribe `out.wav` en él.
    """
    backend, tts, conv, _extracciones, perfil, _ = _backend_con_fakes(monkeypatch, tmp_path)
    backend.sintetizar("One.", perfil)
    src, probe = tts.guardados
    assert Path(src).parent.name.startswith("backend-b-")  # tempdir del backend
    assert src.endswith("src.wav")
    assert probe.endswith("src_probe.wav")
    (salida,) = conv.salidas
    assert salida.endswith("out.wav")


def test_caches_src_se_y_target_se_por_perfil(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend, _tts, _conv, extracciones, perfil, perfil_2 = _backend_con_fakes(monkeypatch, tmp_path)
    backend.sintetizar("One.", perfil)
    backend.sintetizar("Two.", perfil)
    backend.sintetizar("Three.", perfil_2)
    # src_se: 1 extracción (probe) + target: 1 por perfil (kevin, otro)
    assert sorted(extracciones) == ["ref1.wav", "refA.wav", "src_probe.wav"]


def test_reenrolamiento_con_muestras_nuevas_recalcula(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mismo perfil.id con grabaciones nuevas (rutas distintas): recalcula.

    La clave del cache es (id, rutas): muestras nuevas → timbre nuevo. El caso
    de SOBREESCRIBIR la misma ruta NO está cubierto: devuelve el timbre viejo
    (limitación documentada en ADR-013).
    """
    backend, _tts, _conv, extracciones, perfil, _ = _backend_con_fakes(monkeypatch, tmp_path)
    backend.sintetizar("One.", perfil)
    nueva = _escribir_wav(str(tmp_path / "ref1-nueva.wav"), SR_SUPERTONIC, array("h", [0] * 8000))
    reenrolado = VoiceProfile(id=perfil.id, nombre=perfil.nombre, muestras=(str(nueva),))
    backend.sintetizar("Two.", reenrolado)
    assert sorted(extracciones) == ["ref1-nueva.wav", "ref1.wav", "src_probe.wav"]


def test_cerrar_limpia_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend, _tts, _conv, extracciones, perfil, _ = _backend_con_fakes(monkeypatch, tmp_path)
    backend.sintetizar("One.", perfil)
    backend.cerrar()
    assert backend._se_por_perfil == {}
    assert backend._src_se_cache is None
    assert backend._estilo is None
    backend.sintetizar("Two.", perfil)
    assert len(extracciones) == 4  # vuelve a extraer todo


def test_limpieza_ignora_errores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """El cleanup del tempdir es tolerante a fallos (locks de AV en Windows).

    `ignore_errors=True` es el contrato de resiliencia (lección de los locks de
    antivirus del setup_dlls): se espía `shutil.rmtree` y se afirma el kwargs —
    los mutantes que lo quitan o lo ponen a None/False son indistinguibles por
    comportamiento y solo este contrato los detecta.
    """
    import shutil

    backend, _tts, _conv, _extracciones, perfil, _ = _backend_con_fakes(monkeypatch, tmp_path)
    llamadas: list[dict[str, Any]] = []
    rmtree_real = shutil.rmtree

    def espia(ruta: str, **kwargs: Any) -> None:
        llamadas.append(kwargs)
        rmtree_real(ruta, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", espia)
    backend.sintetizar("One.", perfil)
    assert llamadas and all(kwargs == {"ignore_errors": True} for kwargs in llamadas)
