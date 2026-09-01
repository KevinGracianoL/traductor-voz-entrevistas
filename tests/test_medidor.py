"""Tests para medidor — lógica pura con reloj inyectable."""

from collections.abc import Callable

import pytest

from traductor.latencia.medidor import (
    agregar_medicion,
    medir_tiempo,
    resumen_estadisticas,
)


def _reloj_falso(valores: list[float]) -> Callable[[], float]:
    """Devuelve callable que entrega valores secuenciales."""
    it = iter(valores)

    def clock() -> float:
        return next(it)

    return clock


def test_medir_tiempo_retorna_resultado_y_ms() -> None:
    """Clock 1.0 -> 1.15 = 150ms."""

    def suma() -> int:
        return 2 + 3

    clock = _reloj_falso([1.0, 1.15])
    res, ms = medir_tiempo(suma, clock=clock)
    assert res == 5
    assert ms == pytest.approx(150.0)


def test_medir_tiempo_lambda_con_args() -> None:
    """Call site usa lambda para evitar colisión de kwargs."""

    def saluda(nombre: str, extra: str = "") -> str:
        return f"{nombre}{extra}"

    clock = _reloj_falso([0.0, 0.02])
    res, ms = medir_tiempo(lambda: saluda("hi", extra="!"), clock=clock)
    assert res == "hi!"
    assert ms == pytest.approx(20.0)


def test_medir_tiempo_mide_aun_si_func_lanza() -> None:
    """Si func lanza, elapsed se adjunta a la excepción para diagnóstico."""

    def falla() -> None:
        raise RuntimeError("ASR se cayo")

    clock = _reloj_falso([1.0, 1.9])
    with pytest.raises(RuntimeError, match="ASR se cayo") as excinfo:
        medir_tiempo(falla, clock=clock)
    # El caller recupera duración aun con fallo: 0.9s = 900ms
    assert excinfo.value.elapsed_ms == pytest.approx(900.0)  # type: ignore[attr-defined]
    # Mata mutantes: elapsed None, *1000->/1000, -t0 -> +t0, *1000->*1001
    clock2 = _reloj_falso([2.0, 2.5])
    with pytest.raises(RuntimeError) as excinfo2:
        medir_tiempo(falla, clock=clock2)
    assert excinfo2.value.elapsed_ms == pytest.approx(500.0)  # type: ignore[attr-defined]


def test_medir_tiempo_no_colisiona_con_kwarg_clock() -> None:
    """Func con param clock no debe colisionar."""

    def sincroniza(texto: str, clock: str = "default") -> str:
        return f"{texto}:{clock}"

    # Antes habría colisionado: clock="monotonic" se pasaba al medidor
    # Ahora se usa lambda, no hay colisión
    fake = _reloj_falso([0.0, 0.01])
    res, _ = medir_tiempo(lambda: sincroniza("hola", clock="monotonic"), clock=fake)
    assert res == "hola:monotonic"


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


def test_agregar_medicion_acepta_cero() -> None:
    """0 ms es válido: relojes de baja resolución lo devuelven."""
    assert agregar_medicion({}, "asr", 0.0)["asr"] == [0.0]


def test_resumen_estadisticas_basico() -> None:
    registro = {"asr": [100.0, 200.0, 300.0, 400.0]}
    res = resumen_estadisticas(registro)
    assert res["asr"]["count"] == 4.0
    assert res["asr"]["mean"] == pytest.approx(250.0)
    assert res["asr"]["min"] == 100.0
    assert res["asr"]["max"] == 400.0
    # p50 ahora es median (250), p95 es None con n<20
    assert res["asr"]["p50"] == 250.0
    assert res["asr"]["p95"] is None


def test_resumen_estadisticas_vacio() -> None:
    assert resumen_estadisticas({}) == {}
    assert resumen_estadisticas({"asr": []}) == {}


def test_resumen_estadisticas_orden_no_importa() -> None:
    registro = {"t": [300.0, 100.0, 200.0]}
    res = resumen_estadisticas(registro)
    assert res["t"]["min"] == 100.0
    assert res["t"]["max"] == 300.0
    assert res["t"]["p50"] == 200.0
    assert res["t"]["p95"] is None


def test_resumen_estadisticas_p95_con_suficientes_muestras() -> None:
    # 20 muestras: p95 ya no es None
    registro = {"asr": [float(i) for i in range(1, 21)]}  # 1..20
    res = resumen_estadisticas(registro)
    assert res["asr"]["p95"] is not None
    assert res["asr"]["p95"] == 19.0  # ceil(0.95*20)=19 -> idx18 -> 19
    assert res["asr"]["p50"] == pytest.approx(10.5)  # median 1..20


def test_resumen_estadisticas_salta_etapa_vacia_y_sigue() -> None:
    """Una etapa sin datos no debe cortar el resumen de las siguientes."""
    res = resumen_estadisticas({"asr": [], "tts": [100.0, 200.0]})
    assert "asr" not in res
    assert res["tts"]["count"] == 2.0
