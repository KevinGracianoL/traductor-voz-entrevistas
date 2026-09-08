"""Agregación y reporte del benchmark ASR — puro, sin motores.

Los motores reales (faster-whisper, moonshine) corren en
`scripts/benchmark_asr.py`; aquí solo se agregan resultados y se formatea la
tabla. Reusa el medidor de latencia (p50/p95 honesto, p95=None con n<20).
"""

from __future__ import annotations

import statistics

from traductor.asr.resultado import ResultadoAsr
from traductor.asr.wer import wer
from traductor.latencia.medidor import resumen_estadisticas


def _clave(r: ResultadoAsr) -> str:
    return f"{r.engine}|{r.idioma}"


def resumen_asr(resultados: list[ResultadoAsr]) -> dict[str, dict[str, float | None]]:
    """Resumen por `engine|idioma`: n, wer_n, p50_ms, p95_ms y wer_mean.

    `n` es el total de transcripciones (base de la latencia); `wer_n` es cuántas
    de ellas tenían referencia (base del WER). Si no coinciden, la celda no es
    fiable contra el criterio "n≥20 por celda".
    """
    agrupado: dict[str, list[ResultadoAsr]] = {}
    for r in resultados:
        agrupado.setdefault(_clave(r), []).append(r)
    if not agrupado:
        return {}
    registro = {clave: [r.elapsed_ms for r in rs] for clave, rs in agrupado.items()}
    latencia = resumen_estadisticas(registro)
    resumen: dict[str, dict[str, float | None]] = {}
    for clave, rs in agrupado.items():
        con_referencia = [r for r in rs if r.referencia]
        wers = [wer(r.referencia, r.frase) for r in con_referencia]
        stats = latencia[clave]
        resumen[clave] = {
            "n": stats["count"],
            "wer_n": float(len(con_referencia)),
            "p50_ms": stats["p50"],
            "p95_ms": stats["p95"],
            "wer_mean": statistics.fmean(wers) if wers else None,
        }
    return resumen


def _celda(v: float | None, formato: str) -> str:
    if v is None:
        return "     -"
    return formato.format(v)


def tabla_comparativa(resumen: dict[str, dict[str, float | None]]) -> str:
    """Tabla legible: una fila por `engine|idioma`, ordenada por clave."""
    cabecera = "engine          idioma   n  wer_n   p50_ms   p95_ms     wer"
    if not resumen:
        return cabecera + "\n\n(sin datos)"
    filas = [cabecera]
    for clave in sorted(resumen):
        engine, idioma = clave.split("|", 1)
        s = resumen[clave]
        filas.append(
            f"{engine:16s} {idioma:5s} {int(s['n'] or 0):>3d}"
            f" {int(s['wer_n'] or 0):>5d}"
            f" {_celda(s['p50_ms'], '{:>7.1f}')}"
            f" {_celda(s['p95_ms'], '{:>7.1f}')}"
            f" {_celda(s['wer_mean'], '{:>7.2f}')}"
        )
    return "\n".join(filas)
