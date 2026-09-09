"""Tests de las partes puras del harness del ADR-014 (parser + composición).

El harness no corre en CI (requiere GPU); su parser y la composición de la
medición viven en `traductor.tts.harness` y se prueban aquí — incluido que
PUEDE emitir un "go" con los flags de sesión completos (r1 del PR #16,
bloqueante: solo sabía decir no-go).
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from traductor.tts.gates import cabe_en_gates
from traductor.tts.harness import (
    TEXTO_POR_DEFECTO,
    RegistroEtapas,
    _bytes_a_mib,
    componer_medicion,
    medir_ram_mib,
    parser_harness,
    perfil_benchmark,
    verificar_resolucion_audible,
    verificar_ruteo_primer_sample,
)


def _wav(tmp_path: Path, nombre: str = "voz.wav") -> Path:
    wav = tmp_path / nombre
    wav.write_bytes(b"RIFF")
    return wav


def test_parser_requiere_warmup_audio() -> None:
    with pytest.raises(SystemExit):
        parser_harness().parse_args([])


def test_parser_rechaza_wav_inexistente(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        parser_harness().parse_args(["--warmup-audio", str(tmp_path / "no.wav")])
    assert "el WAV de warm-up no existe" in capsys.readouterr().err


def test_parser_rechaza_referencia_inexistente(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        parser_harness().parse_args(
            ["--warmup-audio", str(_wav(tmp_path)), "--referencia", str(tmp_path / "no.wav")]
        )
    assert "el WAV de referencia no existe" in capsys.readouterr().err


def test_referencia_acepta_varias_muestras(tmp_path: Path) -> None:
    a = _wav(tmp_path, "a.wav")
    b = _wav(tmp_path, "b.wav")
    args = parser_harness().parse_args(
        ["--warmup-audio", str(_wav(tmp_path)), "--referencia", str(a), str(b)]
    )
    assert args.referencia == [a, b]


def test_motor_por_defecto_xtts_y_opcion_b(tmp_path: Path) -> None:
    args = parser_harness().parse_args(["--warmup-audio", str(_wav(tmp_path))])
    assert args.motor == "xtts"
    args = parser_harness().parse_args(["--warmup-audio", str(_wav(tmp_path)), "--motor", "b"])
    assert args.motor == "b"
    with pytest.raises(SystemExit):
        parser_harness().parse_args(["--warmup-audio", str(_wav(tmp_path)), "--motor", "nope"])


def test_motor_choices_y_nargs_de_referencia(tmp_path: Path) -> None:
    """Los valores de argparse exactos (mata los mutantes de choices/nargs)."""
    parser = parser_harness()
    acciones = {a.dest: a for a in parser._actions}
    assert acciones["motor"].choices == ("xtts", "b")
    assert acciones["referencia"].nargs == "+"
    for nombre in ("asr_p95", "traduccion_p95", "ruteo_p95"):
        assert acciones[nombre].type is float
    assert acciones["pipeline_p95"].type is float


def test_texto_por_defecto_y_personalizado(tmp_path: Path) -> None:
    args = parser_harness().parse_args(["--warmup-audio", str(_wav(tmp_path))])
    assert args.texto == TEXTO_POR_DEFECTO
    args = parser_harness().parse_args(
        ["--warmup-audio", str(_wav(tmp_path)), "--texto", "Thank you."]
    )
    assert args.texto == "Thank you."


def test_perfil_benchmark_usa_referencia_o_warmup(tmp_path: Path) -> None:
    a = _wav(tmp_path, "a.wav")
    b = _wav(tmp_path, "b.wav")
    args = parser_harness().parse_args(
        ["--warmup-audio", str(_wav(tmp_path)), "--referencia", str(a), str(b)]
    )
    perfil = perfil_benchmark(args)
    assert perfil.id == "benchmark"
    assert perfil.nombre == "Benchmark"
    assert perfil.muestras == (str(a), str(b))
    args = parser_harness().parse_args(["--warmup-audio", str(_wav(tmp_path))])
    assert perfil_benchmark(args).muestras == (str(args.warmup_audio),)


def test_parser_help_explica_flags() -> None:
    """Los helps de cada flag son los exactos (mata los mutantes de string)."""
    parser = parser_harness()
    helps = {a.dest: a.help for a in parser._actions}
    assert helps["warmup_audio"] == (
        "WAV de voz real para el warm-up de Whisper (obligatorio: sin el "
        "decoder ejercitado la VRAM subestima y el harness hace raise)"
    )
    assert helps["referencia"] == (
        "Muestras de referencia (1 o más) para el perfil TTS (default: --warmup-audio)"
    )
    assert helps["motor"] == (
        "Candidato a medir: xtts (primario ADR-011) o b (Supertonic 3 + OpenVoice V2)"
    )
    assert helps["texto"] == (
        "Texto sintetizado en cada repetición (la salida del flujo ES→EN es inglés)"
    )
    assert helps["asr_p95"] == (
        "ASR warm p95 en ms, MEDIDO en el hardware (presupuesto TTFA derivado)"
    )
    assert helps["traduccion_p95"] == (
        "Traducción Argos p95 en ms, MEDIDA (presupuesto TTFA derivado)"
    )
    assert helps["ruteo_p95"] == "Ruteo a VB-CABLE p95 en ms, MEDIDO (presupuesto TTFA derivado)"
    assert helps["pipeline_p95"] == "Pipeline warm p95 en ms, de la corrida de flujo (ADR-015)"
    assert helps["oom"] == "la corrida larga registró OOM (True = FALLA)"
    assert helps["memoria_estable"] == "la memoria no creció sostenidamente (True = PASA)"
    assert helps["artefactos"] == (
        "se detectaron palabras añadidas/omitidas/repetidas (True = FALLA)"
    )
    assert helps["voz_reconocible_ab"] == "la voz fue reconocible en A/B (True = PASA)"
    assert helps["endurance_90min"] == "completó 90 minutos continuos (True = PASA)"


def test_componer_medicion_sin_flags_de_sesion_queda_sin_medir(tmp_path: Path) -> None:
    args = parser_harness().parse_args(["--warmup-audio", str(_wav(tmp_path))])
    m = componer_medicion(300.0, 400.0, 150.0, 50.0, None, 2500.0, 12000.0, args)
    assert m.ttfa_caliente_p95_ms == 300.0
    assert m.asr_p95_ms == 400.0
    assert m.traduccion_p95_ms == 150.0
    assert m.ruteo_p95_ms == 50.0
    assert m.vram_mib == 2500.0
    assert m.ram_mib == 12000.0
    assert m.pipeline_p95_ms is None
    assert m.oom is None  # sin flag = sin medir = FALLA
    assert cabe_en_gates(m) is False  # sesión sin medir: FALLA


def test_componer_medicion_go_completo(tmp_path: Path) -> None:
    args = parser_harness().parse_args(
        [
            "--warmup-audio",
            str(_wav(tmp_path)),
            "--pipeline-p95",
            "1500",
            "--no-oom",
            "--memoria-estable",
            "--no-artefactos",
            "--voz-reconocible-ab",
            "--endurance-90min",
        ]
    )
    m = componer_medicion(300.0, 400.0, 150.0, 50.0, None, 2500.0, 12000.0, args)
    assert m.pipeline_p95_ms == 1500.0
    assert m.oom is False
    assert m.memoria_estable is True
    assert m.artefactos is False
    assert m.voz_reconocible_ab is True
    assert m.endurance_90min is True
    assert cabe_en_gates(m) is True  # el harness PUEDE emitir go


def test_componer_medicion_flags_negativos(tmp_path: Path) -> None:
    args = parser_harness().parse_args(
        [
            "--warmup-audio",
            str(_wav(tmp_path)),
            "--oom",
            "--no-memoria-estable",
            "--artefactos",
            "--no-voz-reconocible-ab",
            "--no-endurance-90min",
        ]
    )
    m = componer_medicion(300.0, 400.0, 150.0, 50.0, None, 2500.0, 12000.0, args)
    assert m.oom is True
    assert m.memoria_estable is False
    assert m.artefactos is True
    assert m.voz_reconocible_ab is False
    assert m.endurance_90min is False
    assert cabe_en_gates(m) is False


def test_componer_medicion_pipeline_medido_gana_al_flag(tmp_path: Path) -> None:
    """El pipeline MEDIDO (param) gana sobre --pipeline-p95 (flag de sesión)."""
    args = parser_harness().parse_args(
        ["--warmup-audio", str(_wav(tmp_path)), "--pipeline-p95", "1500"]
    )
    m = componer_medicion(300.0, 400.0, 150.0, 50.0, 1890.0, 2500.0, 12000.0, args)
    assert m.pipeline_p95_ms == 1890.0
    m = componer_medicion(300.0, 400.0, 150.0, 50.0, None, 2500.0, 12000.0, args)
    assert m.pipeline_p95_ms == 1500.0


def test_componer_medicion_ttfa_fuera_de_presupuesto_derivado_falla(tmp_path: Path) -> None:
    """Sin etapas medidas el presupuesto no se deriva: el TTFA FALLA aunque mida 1 ms."""
    args = parser_harness().parse_args(["--warmup-audio", str(_wav(tmp_path))])
    m = componer_medicion(1.0, None, None, None, None, 2500.0, 12000.0, args)
    assert cabe_en_gates(m) is False


def test_medir_ram_sin_psutil_devuelve_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil ausente (FORZADO, no depende de la máquina): None → FALLA (r3)."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert medir_ram_mib() is None


def test_bytes_a_mib_unidad() -> None:
    """1 GiB = 1024 MiB; un /1024**3 daría ~1 y el gate de RAM pasaría siempre."""
    assert _bytes_a_mib(1 * 1024**3) == 1024.0
    assert _bytes_a_mib(512 * 1024**2) == 512.0


def test_verificar_ruteo_acepta_primer_sample() -> None:
    """El ruteo es time-to-first-sample-audible: decenas de ms, nunca la duración."""
    verificar_ruteo_primer_sample(50.0, 1000.0)  # latencia de dispositivo
    verificar_ruteo_primer_sample(200.0, 1000.0)
    verificar_ruteo_primer_sample(999.0, 1000.0)  # justo debajo de la duración
    verificar_ruteo_primer_sample(0.5, 1000.0)  # > 0 pero mínimo: aún es medición


def test_verificar_ruteo_falla_si_mide_el_drenado() -> None:
    """GUARD OBLIGATORIO: medir el drenado completo (p95 >= duración del chunk)
    DEBE hacer fallar la verificación — es el error conocido que el guard
    existe para atrapar (medir hasta que el audio termina, no hasta que empieza)."""
    with pytest.raises(RuntimeError, match="drenado"):
        verificar_ruteo_primer_sample(1000.0, 1000.0)  # p95 == duración
    with pytest.raises(RuntimeError, match="drenado"):
        verificar_ruteo_primer_sample(1003.5, 1000.0)  # el número del error real
    with pytest.raises(RuntimeError, match="drenado"):
        verificar_ruteo_primer_sample(1500.0, 1000.0)


def test_verificar_ruteo_falla_si_no_mide_nada() -> None:
    """Piso: p95 indistinguible de cero = no se midió nada (instrumento roto)."""
    with pytest.raises(RuntimeError, match="cero"):
        verificar_ruteo_primer_sample(0.0, 1000.0)
    with pytest.raises(RuntimeError, match="cero"):
        verificar_ruteo_primer_sample(-5.0, 1000.0)
    with pytest.raises(RuntimeError, match="cero"):
        verificar_ruteo_primer_sample(None, 1000.0)


def test_verificar_resolucion_acepta_la_mitad() -> None:
    """>= resolución/2 es medible; la resolución entera también."""
    verificar_resolucion_audible(5.0, 10.0)  # == resolución / 2
    verificar_resolucion_audible(10.0, 10.0)
    verificar_resolucion_audible(19.4, 10.0)


def test_verificar_resolucion_falla_si_sub_resolucion() -> None:
    """0.1 ms con un detector de 10 ms es una medición que no ocurrió (PR #20)."""
    with pytest.raises(RuntimeError, match="sub-resolución"):
        verificar_resolucion_audible(0.1, 10.0)
    with pytest.raises(RuntimeError, match="sub-resolución"):
        verificar_resolucion_audible(4.9, 10.0)
    with pytest.raises(
        RuntimeError, match=r"^p95 de la frontera audible: sin medir \(instrumento roto\)$"
    ):
        verificar_resolucion_audible(None, 10.0)


def _reloj_falso(ticks: list[float]) -> Callable[[], float]:
    import itertools

    secuencia = itertools.chain(ticks, [ticks[-1]] * 1000)

    def reloj() -> float:
        return next(secuencia)

    return reloj


def test_registro_etapas_desglose_suma_al_total() -> None:
    """Las marcas consecutivas cierran el total EXACTO (reloj inyectable)."""
    reloj = _reloj_falso([0.0, 0.1, 0.3, 0.6])
    r = RegistroEtapas(clock=reloj)
    r.marcar("entrada")
    r.marcar("asr")
    r.marcar("traduccion")
    r.marcar("tts")
    desglose = r.desglose_ms()
    # cada delta pertenece a la etapa que TERMINÓ (la segunda marca del par)
    assert desglose["asr"] == pytest.approx(100.0)
    assert desglose["traduccion"] == pytest.approx(200.0)
    assert desglose["tts"] == pytest.approx(300.0)
    assert sum(desglose.values()) == pytest.approx(600.0)  # == total (0.6 - 0.0)
    assert r.total_ms() == pytest.approx(600.0)  # última marca - primera marca
    r.verificar_cierre(600.0)  # 100 % atribuido


def test_registro_etapas_cierre_incorrecto_raise() -> None:
    """Un total que no cierra con la suma de las etapas es un hueco sin atribuir."""
    reloj = _reloj_falso([0.0, 0.1])
    r = RegistroEtapas(clock=reloj)
    r.marcar("entrada")
    r.marcar("asr")
    with pytest.raises(
        RuntimeError,
        match=r"residuo sin atribuir: 400\.000 ms \(etapas suman 100\.000 ms, total 500\.000 ms\)",
    ):
        r.verificar_cierre(500.0)  # las etapas suman 100 ms, el total dice 500


def test_registro_etapas_cierre_desvio_intermedio_raise() -> None:
    """Un desvío entre el épsilon y 1 ms (0.5 ms) también es un hueco sin atribuir."""
    reloj = _reloj_falso([0.0, 0.0005])
    r = RegistroEtapas(clock=reloj)
    r.marcar("entrada")
    r.marcar("asr")
    with pytest.raises(RuntimeError, match="residuo sin atribuir: 0.500"):
        r.verificar_cierre(1.0)  # las etapas suman 0.5 ms, el total dice 1.0 ms


def test_registro_etapas_total_con_dos_marcas() -> None:
    """2 marcas = intervalo real (el mutante '<= 2' / '< 3' daría 0.0)."""
    reloj = _reloj_falso([2.0, 2.75])
    r = RegistroEtapas(clock=reloj)
    r.marcar("entrada")
    r.marcar("asr")
    assert r.total_ms() == pytest.approx(750.0)
    assert r.inicio_s() == 2.0


def test_registro_etapas_vacio() -> None:
    r = RegistroEtapas(clock=_reloj_falso([1.0]))
    assert r.desglose_ms() == {}
    assert r.total_ms() == 0.0
    assert r.inicio_s() == 1.0  # sin marcas: devuelve el reloj actual
    r.marcar("entrada")
    assert r.total_ms() == 0.0  # una sola marca: sin intervalo
    assert r.inicio_s() == 1.0  # la primera marca es la referencia


def test_registro_etapas_inicio_s() -> None:
    reloj = _reloj_falso([5.0, 5.2, 5.9])
    r = RegistroEtapas(clock=reloj)
    r.marcar("entrada")
    assert r.inicio_s() == 5.0
    r.marcar("asr")
    assert r.inicio_s() == 5.0  # sigue siendo la primera marca
