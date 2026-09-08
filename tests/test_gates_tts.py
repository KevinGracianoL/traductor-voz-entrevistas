"""Tests de los gates de aceptación del motor TTS (ADR-014, Propuesto).

Los gates son el criterio escrito para aceptar un candidato (ADR-011, sin
motor elegido): TTFA caliente p95 < 400 ms y VRAM co-residente < 3.2 GB.
Aquí se prueba la decisión pura; la medición en la GPU va en
`scripts/medir_gates_tts.py`.
"""

import pytest

from traductor.tts.gates import (
    MedicionTts,
    cabe_en_gates,
    evaluar_gates,
    resumen_gates,
)


def test_medicion_valida_negativos() -> None:
    with pytest.raises(ValueError, match="negativo"):
        MedicionTts(ttfa_caliente_p95_ms=-1.0)
    with pytest.raises(ValueError, match="negativo"):
        MedicionTts(vram_mib=-1.0)


def test_medicion_todo_none_permitido() -> None:
    MedicionTts()


def test_gates_pasan_bajo_limite() -> None:
    medicion = MedicionTts(ttfa_caliente_p95_ms=300.0, vram_mib=2500.0)
    resultados = evaluar_gates(medicion)
    assert all(g.pasa for g in resultados)
    assert cabe_en_gates(medicion) is True


def test_gate_ttfa_sobre_limite_falla() -> None:
    medicion = MedicionTts(ttfa_caliente_p95_ms=450.0, vram_mib=2500.0)
    resultados = evaluar_gates(medicion)
    assert resultados[0].pasa is False
    assert resultados[1].pasa is True
    assert cabe_en_gates(medicion) is False


def test_gate_vram_sobre_limite_falla() -> None:
    medicion = MedicionTts(ttfa_caliente_p95_ms=300.0, vram_mib=3500.0)
    resultados = evaluar_gates(medicion)
    assert resultados[0].pasa is True
    assert resultados[1].pasa is False


def test_gate_ttfa_en_limite_exacto_falla() -> None:
    """Límite estricto: 400.0 ms exactos NO cabe."""
    medicion = MedicionTts(ttfa_caliente_p95_ms=400.0, vram_mib=2500.0)
    assert evaluar_gates(medicion)[0].pasa is False


def test_gate_vram_en_limite_exacto_falla() -> None:
    """3.2 GB = 3276.8 MiB exactos NO caben (estricto)."""
    medicion = MedicionTts(ttfa_caliente_p95_ms=300.0, vram_mib=3.2 * 1024)
    assert evaluar_gates(medicion)[1].pasa is False


def test_gate_sin_medicion_falla_no_en_silencio() -> None:
    """None = no se pudo medir: el gate FALLA con motivo claro (n≥20)."""
    medicion = MedicionTts()
    resultados = evaluar_gates(medicion)
    assert all(not g.pasa for g in resultados)
    assert cabe_en_gates(medicion) is False


def test_gate_solo_ttfa() -> None:
    medicion = MedicionTts(ttfa_caliente_p95_ms=300.0)
    resultados = evaluar_gates(medicion)
    assert resultados[0].pasa is True
    assert resultados[1].pasa is False  # VRAM sin medir


def test_gate_solo_vram() -> None:
    medicion = MedicionTts(vram_mib=2500.0)
    resultados = evaluar_gates(medicion)
    assert resultados[0].pasa is False  # TTFA sin medir
    assert resultados[1].pasa is True


def test_resumen_gates_pasa() -> None:
    medicion = MedicionTts(ttfa_caliente_p95_ms=300.0, vram_mib=2500.0)
    tabla = resumen_gates(evaluar_gates(medicion))
    assert "PASA" in tabla
    assert "GATES: PASA" in tabla


def test_resumen_gates_falla() -> None:
    medicion = MedicionTts(ttfa_caliente_p95_ms=450.0, vram_mib=2500.0)
    tabla = resumen_gates(evaluar_gates(medicion))
    assert "GATES: FALLA" in tabla


def test_resumen_gates_output_exacto() -> None:
    medicion = MedicionTts(ttfa_caliente_p95_ms=300.0, vram_mib=2500.0)
    tabla = resumen_gates(evaluar_gates(medicion))
    assert tabla == (
        "gate                      medido      límite  estado\n"
        "TTFA caliente p95          300.0    < 400 ms    PASA\n"
        "VRAM co-residente         2500.0    < 3.2 GB    PASA\n"
        "\n"
        "GATES: PASA"
    )


def test_resumen_gates_sin_medir() -> None:
    tabla = resumen_gates(evaluar_gates(MedicionTts()))
    assert tabla == (
        "gate                      medido      límite  estado\n"
        "TTFA caliente p95      sin medir    < 400 ms   FALLA\n"
        "VRAM co-residente      sin medir    < 3.2 GB   FALLA\n"
        "\n"
        "GATES: FALLA"
    )


def test_resumen_gates_vacio() -> None:
    tabla = resumen_gates([])
    assert tabla == "gate                      medido      límite  estado\n\n(sin gates)"
