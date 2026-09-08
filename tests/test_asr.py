"""Tests del benchmark ASR — WER, agregación y tabla comparativa.

PR #13 NO sustituye RealtimeSTT: mide faster-whisper vs moonshine (ES y EN)
para decidir con evidencia. Aquí se prueba lo puro (métricas y formato); los
motores reales corren en `scripts/benchmark_asr.py` en la máquina objetivo.
"""

import pytest

from traductor.asr.benchmark import resumen_asr, tabla_comparativa
from traductor.asr.resultado import ResultadoAsr
from traductor.asr.wer import distancia_edicion, wer


def test_wer_identicas_cero() -> None:
    assert wer("hola mundo", "hola mundo") == 0.0


def test_wer_borrado() -> None:
    assert wer("hola mundo", "hola") == 0.5


def test_wer_insercion() -> None:
    assert wer("hola mundo", "hola mundo feliz") == 0.5


def test_wer_sustitucion() -> None:
    assert wer("hola mundo", "hola tierra") == 0.5


def test_wer_normaliza_espacios() -> None:
    assert wer("hola   mundo", "hola mundo") == 0.0


def test_wer_vacio_vacio_cero() -> None:
    assert wer("", "") == 0.0


def test_wer_referencia_vacia_con_texto() -> None:
    assert wer("", "hola") == 1.0


def test_resultado_valido() -> None:
    r = ResultadoAsr(engine="faster-whisper", idioma="es", frase="hola", elapsed_ms=123.0)
    assert r.engine == "faster-whisper"
    assert r.idioma == "es"
    assert r.elapsed_ms == 123.0


def test_resultado_latencia_negativa_raise() -> None:
    with pytest.raises(ValueError, match="elapsed_ms"):
        ResultadoAsr(engine="faster-whisper", idioma="es", frase="hola", elapsed_ms=-1.0)


def test_resultado_idioma_vacio_raise() -> None:
    with pytest.raises(ValueError, match="idioma"):
        ResultadoAsr(engine="moonshine", idioma=" ", frase="hola", elapsed_ms=1.0)


def test_resumen_asr_agrupa_por_engine_idioma() -> None:
    resultados = [
        ResultadoAsr("faster-whisper", "es", "hola", 100.0, "hola"),
        ResultadoAsr("faster-whisper", "es", "hola", 300.0, "hola"),
        ResultadoAsr("moonshine", "es", "hola", 200.0, "hola"),
        ResultadoAsr("moonshine", "es", "hala", 200.0, "hola"),
    ]
    resumen = resumen_asr(resultados)
    assert "faster-whisper|es" in resumen
    assert "moonshine|es" in resumen
    assert resumen["faster-whisper|es"]["n"] == 2
    assert resumen["faster-whisper|es"]["p50_ms"] == 200.0
    assert resumen["faster-whisper|es"]["wer_mean"] == 0.0
    assert resumen["moonshine|es"]["wer_mean"] == 0.5


def test_resumen_asr_p95_none_con_pocas() -> None:
    r = ResultadoAsr("faster-whisper", "en", "hi", 100.0, "hi")
    resumen = resumen_asr([r])
    assert resumen["faster-whisper|en"]["p95_ms"] is None


def test_resumen_asr_sin_referencia_wer_none() -> None:
    r = ResultadoAsr("faster-whisper", "es", "hola", 100.0)
    resumen = resumen_asr([r])
    assert resumen["faster-whisper|es"]["wer_mean"] is None


def test_resumen_asr_vacio() -> None:
    assert resumen_asr([]) == {}


def test_tabla_comparativa_formato() -> None:
    r = ResultadoAsr("faster-whisper", "es", "hola", 100.0, "hola")
    tabla = tabla_comparativa(resumen_asr([r]))
    assert "engine" in tabla
    assert "faster-whisper" in tabla
    assert "es" in tabla


def test_tabla_comparativa_output_exacto() -> None:
    resumen = {
        "faster-whisper|es": {"n": 2.0, "p50_ms": 200.0, "p95_ms": None, "wer_mean": 0.0},
        "moonshine|es": {"n": 1.0, "p50_ms": 200.0, "p95_ms": None, "wer_mean": 0.5},
    }
    tabla = tabla_comparativa(resumen)
    assert tabla == (
        "engine          idioma   n   p50_ms   p95_ms     wer\n"
        "faster-whisper   es      2   200.0      -    0.00\n"
        "moonshine        es      1   200.0      -    0.50"
    )


def test_tabla_comparativa_vacia() -> None:
    tabla = tabla_comparativa({})
    assert tabla == "engine          idioma   n   p50_ms   p95_ms     wer\n\n(sin datos)"


def test_tabla_comparativa_n_cero() -> None:
    resumen = {"x|es": {"n": 0.0, "p50_ms": None, "p95_ms": None, "wer_mean": None}}
    tabla = tabla_comparativa(resumen)
    assert "es      0" in tabla


def test_tabla_comparativa_clave_con_varios_pipes() -> None:
    resumen = {"a|b|c": {"n": 1.0, "p50_ms": 10.0, "p95_ms": None, "wer_mean": None}}
    tabla = tabla_comparativa(resumen)
    assert "a                b|c" in tabla


def test_tabla_comparativa_p95_finito() -> None:
    """p95 real (n≥20) se pinta con número, no con el guion de None."""
    resumen: dict[str, dict[str, float | None]] = {
        "faster-whisper|es": {"n": 20.0, "p50_ms": 200.0, "p95_ms": 300.0, "wer_mean": 0.1}
    }
    tabla = tabla_comparativa(resumen)
    assert tabla == (
        "engine          idioma   n   p50_ms   p95_ms     wer\n"
        "faster-whisper   es     20   200.0   300.0    0.10"
    )


def test_tabla_comparativa_inf() -> None:
    """Valores no finitos se pintan en minúscula (inf, no INF): f vs F."""
    resumen: dict[str, dict[str, float | None]] = {
        "x|es": {"n": 1.0, "p50_ms": float("inf"), "p95_ms": float("inf"), "wer_mean": float("inf")}
    }
    tabla = tabla_comparativa(resumen)
    assert tabla == (
        "engine          idioma   n   p50_ms   p95_ms     wer\n"
        "x                es      1     inf     inf     inf"
    )


def test_distancia_edicion_referencia_vacia() -> None:
    assert distancia_edicion([], ["a"]) == 1


def test_distancia_edicion_hipotesis_vacia() -> None:
    assert distancia_edicion(["a", "b"], []) == 2
