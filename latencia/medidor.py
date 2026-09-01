"""Medidor de latencia — instrumentación sin depender de torch ni audio.

Separa el cálculo (medir tiempo, agregar, resumir) del efecto (llamar a
argos/RealtimeSTT, leer reloj real). Todo lo testeable va aquí con reloj
inyectable; lo que toca hardware queda en benchmarks/.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def medir_tiempo(
    func: Callable[..., T],
    *args: object,
    clock: Callable[[], float] = time.perf_counter,
    **kwargs: object,
) -> tuple[T, float]:
    """Ejecuta func y mide duración en ms con reloj inyectable.

    clock debe devolver segundos (como perf_counter). Se multiplica a ms.
    """
    t0 = clock()
    resultado = func(*args, **kwargs)
    t1 = clock()
    elapsed_ms = (t1 - t0) * 1000.0
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
) -> dict[str, dict[str, float]]:
    """Resumen por etapa: count, mean, min, max, p50, p95.

    p50/p95 con método simple (índice ceil), suficiente para decidir modelos.
    Dict vacío -> dict vacío.
    """
    resumen: dict[str, dict[str, float]] = {}
    for etapa, valores in registro.items():
        if not valores:
            continue
        ordenados = sorted(valores)
        n = len(ordenados)
        mean = sum(ordenados) / n
        import math

        p50_idx = max(0, min(n - 1, math.ceil(0.5 * n) - 1))
        p95_idx = max(0, min(n - 1, math.ceil(0.95 * n) - 1))
        resumen[etapa] = {
            "count": float(n),
            "mean": mean,
            "min": ordenados[0],
            "max": ordenados[-1],
            "p50": ordenados[p50_idx],
            "p95": ordenados[p95_idx],
        }
    return resumen


def latencia_total_ms(mediciones_por_etapa: dict[str, float]) -> float:
    """Suma de latencias por etapa (mismo que sum(presupuesto) pero explícito para medidor)."""
    return sum(mediciones_por_etapa.values())
