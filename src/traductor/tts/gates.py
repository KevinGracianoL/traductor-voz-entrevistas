"""Gates de aceptación del motor TTS (ADR-014, Propuesto).

Criterios escritos para aceptar un candidato (ADR-011: sin motor elegido aún),
medidos en la GPU objetivo con Whisper co-residente:
- TTFA (time to first audio) caliente p95 < 400 ms (p95 honesto: necesita n≥20).
- VRAM total < 3.2 GB (3276.8 MiB).

Un gate con métrica sin medir (None) FALLA con motivo visible: un gate que no
se puede evaluar no debe pasar en silencio (misma regla que ADR-012).
"""

from __future__ import annotations

from dataclasses import dataclass

TTFA_MS_MAX = 400.0
VRAM_MIB_MAX = 3.2 * 1024


@dataclass(frozen=True)
class MedicionTts:
    """Métricas medidas de un motor candidato, en la máquina objetivo.

    `ttfa_caliente_p95_ms`: None si no hay n≥20 (p95 con pocas muestras es max).
    `vram_mib`: None si no se pudo medir (motor sin cargar, etc.).
    """

    ttfa_caliente_p95_ms: float | None = None
    vram_mib: float | None = None

    def __post_init__(self) -> None:
        for nombre, valor in (
            ("ttfa_caliente_p95_ms", self.ttfa_caliente_p95_ms),
            ("vram_mib", self.vram_mib),
        ):
            if valor is not None and valor < 0:
                raise ValueError(f"{nombre} negativo: {valor}")


@dataclass(frozen=True)
class GateResultado:
    """Resultado de un gate: nombre, pasa, valor medido y límite legible."""

    nombre: str
    pasa: bool
    medido: float | None
    limite: str


def evaluar_gates(medicion: MedicionTts) -> list[GateResultado]:
    """Evalúa los gates en orden estable: TTFA primero, VRAM después."""
    ttfa = medicion.ttfa_caliente_p95_ms
    if ttfa is None:
        resultado_ttfa = GateResultado("TTFA caliente p95", False, None, "< 400 ms")
    else:
        resultado_ttfa = GateResultado(
            "TTFA caliente p95", ttfa < TTFA_MS_MAX, ttfa, f"< {TTFA_MS_MAX:g} ms"
        )
    vram = medicion.vram_mib
    if vram is None:
        resultado_vram = GateResultado("VRAM co-residente", False, None, "< 3.2 GB")
    else:
        resultado_vram = GateResultado("VRAM co-residente", vram < VRAM_MIB_MAX, vram, "< 3.2 GB")
    return [resultado_ttfa, resultado_vram]


def cabe_en_gates(medicion: MedicionTts) -> bool:
    """True si todos los gates pasan. Los None cuentan como fallo."""
    return all(g.pasa for g in evaluar_gates(medicion))


def _medido(v: float | None) -> str:
    if v is None:
        return " sin medir"
    return f"{v:>10.1f}"


def resumen_gates(resultados: list[GateResultado]) -> str:
    """Tabla legible: gate, medido, límite, estado y veredicto final."""
    cabecera = f"{'gate':22s}{'medido':>10s}{'límite':>12s}{'estado':>8s}"
    if not resultados:
        return cabecera + "\n\n(sin gates)"
    filas = [cabecera]
    for g in resultados:
        estado = "PASA" if g.pasa else "FALLA"
        filas.append(f"{g.nombre:22s}{_medido(g.medido)}{g.limite:>12s}{estado:>8s}")
    veredicto = "PASA" if all(g.pasa for g in resultados) else "FALLA"
    filas.append("")
    filas.append(f"GATES: {veredicto}")
    return "\n".join(filas)
