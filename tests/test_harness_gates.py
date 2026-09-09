"""Tests de las partes puras del harness del ADR-014 (parser + composición).

El harness no corre en CI (requiere GPU); su parser y la composición de la
medición viven en `traductor.tts.harness` y se prueban aquí — incluido que
PUEDE emitir un "go" con los flags de sesión completos (r1 del PR #16,
bloqueante: solo sabía decir no-go).
"""

import sys
from pathlib import Path

import pytest

from traductor.tts.gates import cabe_en_gates
from traductor.tts.harness import (
    TEXTO_POR_DEFECTO,
    _bytes_a_mib,
    componer_medicion,
    medir_ram_mib,
    parser_harness,
    perfil_benchmark,
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
    m = componer_medicion(300.0, 2500.0, 12000.0, args)
    assert m.ttfa_caliente_p95_ms == 300.0
    assert m.vram_mib == 2500.0
    assert m.ram_mib == 12000.0
    assert m.pipeline_p95_ms is None
    assert m.oom is None  # sin flag = sin medir = FALLA
    assert cabe_en_gates(m) is False


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
    m = componer_medicion(300.0, 2500.0, 12000.0, args)
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
    m = componer_medicion(300.0, 2500.0, 12000.0, args)
    assert m.oom is True
    assert m.memoria_estable is False
    assert m.artefactos is True
    assert m.voz_reconocible_ab is False
    assert m.endurance_90min is False
    assert cabe_en_gates(m) is False


def test_medir_ram_sin_psutil_devuelve_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil ausente (FORZADO, no depende de la máquina): None → FALLA (r3)."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert medir_ram_mib() is None


def test_bytes_a_mib_unidad() -> None:
    """1 GiB = 1024 MiB; un /1024**3 daría ~1 y el gate de RAM pasaría siempre."""
    assert _bytes_a_mib(1 * 1024**3) == 1024.0
    assert _bytes_a_mib(512 * 1024**2) == 512.0
