"""Gates de aceptación del motor TTS (ADR-014, Propuesto).

Criterios escritos para aceptar un candidato (ADR-011: XTTS-v2 primario vía
fork coqui-tts; Supertonic+OpenVoice V2 como candidato B), medidos en la
máquina objetivo con el ASR co-residente:
- TTFA (time to first audio) caliente p95 < 400 ms.
- VRAM total < 3.2 GB (3276.8 MiB).
- RAM total < 18 GB (18432 MiB).
- Pipeline warm p95 < 2 s.
- Sin OOM, sin crecimiento sostenido de memoria, sin artefactos de palabras,
  voz reconocible en A/B, endurance de 90 minutos continuos.

Un gate con métrica sin medir (None) FALLA con motivo visible: un gate que no
se puede evaluar no debe pasar en silencio (misma regla que ADR-012/014).
"""

from __future__ import annotations

from dataclasses import dataclass

TTFA_MS_MAX = 400.0
VRAM_MIB_MAX = 3.2 * 1024
RAM_MIB_MAX = 18 * 1024
PIPELINE_MS_MAX = 2000.0


@dataclass(frozen=True)
class MedicionTts:
    """Métricas medidas de un motor candidato, en la máquina objetivo.

    Los numéricos usan None si no hay medición válida (p95 necesita n≥20).
    Los booleanos usan None si el criterio no se evaluó — y None FALLA.
    """

    ttfa_caliente_p95_ms: float | None = None
    vram_mib: float | None = None
    ram_mib: float | None = None
    pipeline_p95_ms: float | None = None
    oom: bool | None = None
    memoria_estable: bool | None = None
    artefactos: bool | None = None
    voz_reconocible_ab: bool | None = None
    endurance_90min: bool | None = None

    def __post_init__(self) -> None:
        for nombre, valor in (
            ("ttfa_caliente_p95_ms", self.ttfa_caliente_p95_ms),
            ("vram_mib", self.vram_mib),
            ("ram_mib", self.ram_mib),
            ("pipeline_p95_ms", self.pipeline_p95_ms),
        ):
            if valor is not None and valor < 0:
                raise ValueError(f"{nombre} negativo: {valor}")
        for nombre, valor in (
            ("oom", self.oom),
            ("memoria_estable", self.memoria_estable),
            ("artefactos", self.artefactos),
            ("voz_reconocible_ab", self.voz_reconocible_ab),
            ("endurance_90min", self.endurance_90min),
        ):
            if valor is not None and not isinstance(valor, bool):
                raise ValueError(f"{nombre} debe ser bool o None: {valor!r}")


@dataclass(frozen=True)
class GateResultado:
    """Resultado de un gate: nombre, pasa, valor medido y límite legible."""

    nombre: str
    pasa: bool
    medido: float | bool | None
    limite: str


def _gate_numerico(
    nombre: str,
    valor: float | None,
    maximo: float,
    unidad: str,
    divisor: float = 1.0,
) -> GateResultado:
    limite = f"< {maximo / divisor:g} {unidad}"
    if valor is None:
        return GateResultado(nombre, False, None, limite)
    return GateResultado(nombre, valor < maximo, valor, limite)


def _gate_booleano(
    nombre: str,
    valor: bool | None,
    pasa_si_true: bool,
    limite: str,
) -> GateResultado:
    if valor is None:
        return GateResultado(nombre, False, None, limite)
    # `==` y no `valor if pasa_si_true else not valor`: un pasa_si_true=None
    # (mutante) daría siempre FALLA y los tests lo cazan.
    return GateResultado(nombre, valor == pasa_si_true, valor, limite)


def evaluar_gates(medicion: MedicionTts) -> list[GateResultado]:
    """Evalúa los gates en orden estable, todos los criterios del ADR-014."""
    return [
        _gate_numerico("TTFA caliente p95", medicion.ttfa_caliente_p95_ms, TTFA_MS_MAX, "ms"),
        _gate_numerico("VRAM co-residente", medicion.vram_mib, VRAM_MIB_MAX, "GB", divisor=1024),
        _gate_numerico("RAM total", medicion.ram_mib, RAM_MIB_MAX, "GB", divisor=1024),
        _gate_numerico(
            "Pipeline warm p95", medicion.pipeline_p95_ms, PIPELINE_MS_MAX, "s", divisor=1000
        ),
        _gate_booleano("Sin OOM", medicion.oom, pasa_si_true=False, limite="sin OOM"),
        _gate_booleano(
            "Memoria estable",
            medicion.memoria_estable,
            pasa_si_true=True,
            limite="sin crecimiento sostenido",
        ),
        _gate_booleano(
            "Sin artefactos de palabras",
            medicion.artefactos,
            pasa_si_true=False,
            limite="sin añadir/omitir/repetir",
        ),
        _gate_booleano(
            "Voz reconocible A/B",
            medicion.voz_reconocible_ab,
            pasa_si_true=True,
            limite="prueba A/B",
        ),
        _gate_booleano(
            "Endurance 90 min",
            medicion.endurance_90min,
            pasa_si_true=True,
            limite="90 min continuos",
        ),
    ]


def cabe_en_gates(medicion: MedicionTts) -> bool:
    """True si todos los gates pasan. Los None cuentan como fallo."""
    return all(g.pasa for g in evaluar_gates(medicion))


def _medido(v: float | bool | None) -> str:
    if v is None:
        return " sin medir"
    if isinstance(v, bool):
        return f"{'sí' if v else 'no':>10s}"
    return f"{v:>10.1f}"


def resumen_gates(resultados: list[GateResultado]) -> str:
    """Tabla legible: gate, medido, límite, estado y veredicto final."""
    cabecera = f"{'gate':30s}{'medido':>10s}{'límite':>30s}{'estado':>8s}"
    if not resultados:
        return cabecera + "\n\n(sin gates)"
    filas = [cabecera]
    for g in resultados:
        estado = "PASA" if g.pasa else "FALLA"
        filas.append(f"{g.nombre:30s}{_medido(g.medido)}{g.limite:>30s}{estado:>8s}")
    veredicto = "PASA" if all(g.pasa for g in resultados) else "FALLA"
    filas.append("")
    filas.append(f"GATES: {veredicto}")
    return "\n".join(filas)
