"""Tests de los gates de aceptación del motor TTS (ADR-014, Propuesto).

Los gates son el criterio escrito para aceptar un candidato (ADR-011: XTTS-v2
primario; Supertonic+OpenVoice V2 como B): TTFA < 400 ms, VRAM < 3.2 GB,
RAM < 18 GB, pipeline p95 < 2 s, sin OOM, memoria estable, sin artefactos,
voz reconocible A/B y endurance 90 min. None = sin medir = FALLA.
"""

from dataclasses import replace

import pytest

from traductor.tts.gates import (
    MedicionTts,
    cabe_en_gates,
    evaluar_gates,
    resumen_gates,
)

BUENA = MedicionTts(
    ttfa_caliente_p95_ms=300.0,
    vram_mib=2500.0,
    ram_mib=12000.0,
    pipeline_p95_ms=1500.0,
    oom=False,
    memoria_estable=True,
    artefactos=False,
    voz_reconocible_ab=True,
    endurance_90min=True,
)


def test_medicion_valida_negativos() -> None:
    with pytest.raises(ValueError, match="negativo"):
        MedicionTts(ttfa_caliente_p95_ms=-1.0)
    with pytest.raises(ValueError, match="negativo"):
        MedicionTts(vram_mib=-1.0)
    with pytest.raises(ValueError, match="negativo"):
        MedicionTts(ram_mib=-1.0)
    with pytest.raises(ValueError, match="negativo"):
        MedicionTts(pipeline_p95_ms=-1.0)


def test_medicion_valida_booleanos() -> None:
    # tipo inválido a propósito: prueba la validación de runtime
    with pytest.raises(ValueError, match="bool"):
        MedicionTts(oom="sí")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bool"):
        MedicionTts(memoria_estable=1)  # type: ignore[arg-type]


def test_medicion_todo_none_permitido() -> None:
    MedicionTts()


def test_gates_pasan_con_buena_medicion() -> None:
    assert all(g.pasa for g in evaluar_gates(BUENA))
    assert cabe_en_gates(BUENA) is True


def test_gate_ttfa_sobre_limite_falla() -> None:
    assert evaluar_gates(replace(BUENA, ttfa_caliente_p95_ms=450.0))[0].pasa is False


def test_gate_vram_sobre_limite_falla() -> None:
    assert evaluar_gates(replace(BUENA, vram_mib=3500.0))[1].pasa is False


def test_gate_ram_sobre_limite_falla() -> None:
    assert evaluar_gates(replace(BUENA, ram_mib=19000.0))[2].pasa is False


def test_gate_pipeline_sobre_limite_falla() -> None:
    assert evaluar_gates(replace(BUENA, pipeline_p95_ms=2500.0))[3].pasa is False


def test_gate_oom_hubo_falla() -> None:
    assert evaluar_gates(replace(BUENA, oom=True))[4].pasa is False


def test_gate_memoria_no_estable_falla() -> None:
    assert evaluar_gates(replace(BUENA, memoria_estable=False))[5].pasa is False


def test_gate_artefactos_hubo_falla() -> None:
    assert evaluar_gates(replace(BUENA, artefactos=True))[6].pasa is False


def test_gate_voz_no_reconocible_falla() -> None:
    assert evaluar_gates(replace(BUENA, voz_reconocible_ab=False))[7].pasa is False


def test_gate_endurance_no_completado_falla() -> None:
    assert evaluar_gates(replace(BUENA, endurance_90min=False))[8].pasa is False


def test_limites_exactos_fallan() -> None:
    """Límites estrictos: el valor exacto NO cabe."""
    assert evaluar_gates(replace(BUENA, ttfa_caliente_p95_ms=400.0))[0].pasa is False
    assert evaluar_gates(replace(BUENA, vram_mib=3.2 * 1024))[1].pasa is False
    assert evaluar_gates(replace(BUENA, ram_mib=18 * 1024))[2].pasa is False
    assert evaluar_gates(replace(BUENA, pipeline_p95_ms=2000.0))[3].pasa is False


def test_sin_medicion_todo_falla() -> None:
    medicion = MedicionTts()
    resultados = evaluar_gates(medicion)
    assert len(resultados) == 9
    assert all(g.pasa is False for g in resultados)
    assert cabe_en_gates(medicion) is False


def test_solo_ttfa_medido() -> None:
    medicion = MedicionTts(ttfa_caliente_p95_ms=300.0)
    resultados = evaluar_gates(medicion)
    assert resultados[0].pasa is True
    assert all(not g.pasa for g in resultados[1:])


def test_resumen_gates_output_exacto() -> None:
    tabla = resumen_gates(evaluar_gates(BUENA))
    assert tabla == (
        "gate                              medido                        límite  estado\n"
        "TTFA caliente p95                  300.0                      < 400 ms    PASA\n"
        "VRAM co-residente                 2500.0                      < 3.2 GB    PASA\n"
        "RAM total                        12000.0                       < 18 GB    PASA\n"
        "Pipeline warm p95                 1500.0                         < 2 s    PASA\n"
        "Sin OOM                               no                       sin OOM    PASA\n"
        "Memoria estable                       sí     sin crecimiento sostenido    PASA\n"
        "Sin artefactos de palabras            no     sin añadir/omitir/repetir    PASA\n"
        "Voz reconocible A/B                   sí                    prueba A/B    PASA\n"
        "Endurance 90 min                      sí              90 min continuos    PASA\n"
        "\n"
        "GATES: PASA"
    )


def test_resumen_gates_sin_medir_output_exacto() -> None:
    tabla = resumen_gates(evaluar_gates(MedicionTts()))
    assert tabla == (
        "gate                              medido                        límite  estado\n"
        "TTFA caliente p95              sin medir                      < 400 ms   FALLA\n"
        "VRAM co-residente              sin medir                      < 3.2 GB   FALLA\n"
        "RAM total                      sin medir                       < 18 GB   FALLA\n"
        "Pipeline warm p95              sin medir                         < 2 s   FALLA\n"
        "Sin OOM                        sin medir                       sin OOM   FALLA\n"
        "Memoria estable                sin medir     sin crecimiento sostenido   FALLA\n"
        "Sin artefactos de palabras     sin medir     sin añadir/omitir/repetir   FALLA\n"
        "Voz reconocible A/B            sin medir                    prueba A/B   FALLA\n"
        "Endurance 90 min               sin medir              90 min continuos   FALLA\n"
        "\n"
        "GATES: FALLA"
    )


def test_resumen_gates_vacio() -> None:
    tabla = resumen_gates([])
    assert tabla == (
        "gate                              medido                        límite  estado"
        "\n\n(sin gates)"
    )


def test_resumen_gates_con_fallo_dice_falla() -> None:
    tabla = resumen_gates(evaluar_gates(replace(BUENA, oom=True)))
    assert "GATES: FALLA" in tabla
