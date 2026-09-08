"""Ejecución medida del benchmark ASR — lógica pura con motores inyectables.

Corre cada motor sobre cada muestra y produce `ResultadoAsr` con latencia
(reloj inyectable). Los motores reales se inyectan desde `scripts/benchmark_asr.py`;
aquí no se toca hardware ni modelos.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path

from traductor.asr.manifesto import Muestra
from traductor.asr.resultado import ResultadoAsr
from traductor.latencia.medidor import medir_tiempo


def medir_motores(
    motores: Mapping[str, Callable[[Path, str], str]],
    muestras: list[Muestra],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> list[ResultadoAsr]:
    """Transcribe cada muestra con cada motor y mide la latencia.

    `motores` mapea nombre → callable(ruta, idioma) -> texto. Si una muestra no
    existe, el transcribir falla y la excepción se propaga (con `elapsed_ms`
    adjunta por el medidor).
    """
    resultados: list[ResultadoAsr] = []
    for muestra in muestras:
        ruta = Path(muestra.ruta)
        for motor, transcribir in motores.items():
            frase, elapsed_ms = medir_tiempo(
                partial(transcribir, ruta, muestra.idioma),
                clock=clock,
            )
            resultados.append(
                ResultadoAsr(
                    engine=motor,
                    idioma=muestra.idioma,
                    frase=frase,
                    elapsed_ms=elapsed_ms,
                    referencia=muestra.referencia,
                )
            )
    return resultados
