"""Medidor de latencia — instrumentación sin depender de torch ni audio.

Separa el cálculo (medir tiempo, agregar, resumir) del efecto (llamar a
argos/RealtimeSTT, leer reloj real). Todo lo testeable va aquí con reloj
inyectable; lo que toca hardware queda en benchmarks/.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def medir_tiempo(
    func: Callable[[], T],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[T, float]:
    """Ejecuta func y mide duración en ms con reloj inyectable.

    clock debe devolver segundos (como perf_counter). Se multiplica a ms.
    Mide aun si func lanza: el elapsed se adjunta a la excepción y se
    propaga (except/else), el caller recupera con e.elapsed_ms.
    """
    t0 = clock()
    try:
        resultado = func()
    except BaseException as exc:
        exc.elapsed_ms = (clock() - t0) * 1000.0  # type: ignore[attr-defined]
        raise
    else:
        elapsed_ms = (clock() - t0) * 1000.0
        return resultado, elapsed_ms


def agregar_medicion(
    registro: dict[str, list[float]],
    etapa: str,
    elapsed_ms: float,
) -> dict[str, list[float]]:
    """Añade una medición y devuelve nuevo registro (no muta el original)."""
    if elapsed_ms < 0:
        raise ValueError(f"elapsed_ms negativo: {elapsed_ms}")
    nuevo = {k: list(v) for k, v in registro.items()}
    if etapa not in nuevo:
        nuevo[etapa] = []
    nuevo[etapa].append(elapsed_ms)
    return nuevo


def resumen_estadisticas(
    registro: dict[str, list[float]],
) -> dict[str, dict[str, float | None]]:
    """Resumen por etapa: count, mean, min, max, p50, p95.

    p50 usa statistics.median (interpola en n par). p95 es None si n<20
    porque con pocas muestras es identico a max y engaña (ver ADR-003).
    Dict vacío -> dict vacío.
    """
    resumen: dict[str, dict[str, float | None]] = {}
    for etapa, valores in registro.items():
        if not valores:
            continue
        ordenados = sorted(valores)
        n = len(ordenados)
        mean = statistics.fmean(ordenados)
        p50 = statistics.median(ordenados)
        # p95 solo informativo con n>=20, si no None (con n<20 p95==max)
        p95: float | None
        if n >= 20:
            p95_idx = math.ceil(0.95 * n) - 1
            p95 = ordenados[p95_idx]
        else:
            p95 = None
        resumen[etapa] = {
            "count": float(n),
            "mean": mean,
            "min": ordenados[0],
            "max": ordenados[-1],
            "p50": p50,
            "p95": p95,
        }
    return resumen
