"""Tests para medidor — lógica pura con reloj inyectable."""

import pytest

from latencia.medidor import (
    agregar_medicion,
    latencia_total_ms,
    medir_tiempo,
    resumen_estadisticas,
)


def _reloj_falso(valores: list[float]) -> object:
    """Devuelve callable que entrega valores secuenciales."""
    it = iter(valores)

    def clock() -> float:
        return next(it)

    return clock


def test_medir_tiempo_retorna_resultado_y_ms() -> None:
    """Clock 1.0 -> 1.15 = 150ms."""

    def suma(a: int, b: int) -> int:
        return a + b

    clock = _reloj_falso([1.0, 1.15])
    res, ms = medir_tiempo(suma, 2, 3, clock=clock)  # type: ignore[arg-type]
    assert res == 5
    assert ms == pytest.approx(150.0)


def test_medir_tiempo_con_kwargs() -> None:
    """Debe pasar kwargs al func."""

    def saluda(nombre: str, extra: str = "") -> str:
        return f"{nombre}{extra}"

    clock = _reloj_falso([0.0, 0.02])
    res, ms = medir_tiempo(saluda, "hi", extra="!", clock=clock)  # type: ignore[arg-type]
    assert res == "hi!"
    assert ms == pytest.approx(20.0)


def test_agregar_medicion_no_muta_original() -> None:
    registro: dict[str, list[float]] = {"asr": [100.0]}
    nuevo = agregar_medicion(registro, "asr", 120.0)
    assert registro["asr"] == [100.0]
    assert nuevo["asr"] == [100.0, 120.0]
    assert nuevo is not registro


def test_agregar_medicion_nueva_etapa() -> None:
    registro: dict[str, list[float]] = {}
    nuevo = agregar_medicion(registro, "tts", 300.0)
    assert nuevo["tts"] == [300.0]


def test_agregar_medicion_negativo_raise() -> None:
    with pytest.raises(ValueError, match="negativo"):
        agregar_medicion({}, "asr", -5.0)


def test_resumen_estadisticas_basico() -> None:
    registro = {"asr": [100.0, 200.0, 300.0, 400.0]}
    res = resumen_estadisticas(registro)
    assert res["asr"]["count"] == 4.0
    assert res["asr"]["mean"] == pytest.approx(250.0)
    assert res["asr"]["min"] == 100.0
    assert res["asr"]["max"] == 400.0
    # p50 ceil(0.5*4)=2 -> idx1 -> 200, p95 ceil(0.95*4)=4 -> idx3 ->400
    assert res["asr"]["p50"] == 200.0
    assert res["asr"]["p95"] == 400.0


def test_resumen_estadisticas_vacio() -> None:
    assert resumen_estadisticas({}) == {}
    assert resumen_estadisticas({"asr": []}) == {}


def test_resumen_estadisticas_orden_no_importa() -> None:
    registro = {"t": [300.0, 100.0, 200.0]}
    res = resumen_estadisticas(registro)
    assert res["t"]["min"] == 100.0
    assert res["t"]["max"] == 300.0
    assert res["t"]["p50"] == 200.0


def test_latencia_total_ms_suma() -> None:
    assert latencia_total_ms({"asr": 500.0, "traduccion": 150.0}) == 650.0
    assert latencia_total_ms({}) == 0.0
