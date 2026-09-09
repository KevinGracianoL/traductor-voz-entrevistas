"""Partes puras del harness del ADR-014: parser y composición de la medición.

Viven en el módulo (no en el script) para ser testeables y mutables en CI:
el harness es hardware (GPU + modelos), su lógica no. El script
`scripts/medir_gates_tts.py` queda como wrapper fino.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from traductor.tts.gates import MedicionTts


def _wav_existente(ruta: str) -> Path:
    """Validación de argparse: falla antes de tocar la GPU (r8 PR #15)."""
    p = Path(ruta)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"el WAV de warm-up no existe: {ruta}")
    return p


def parser_harness() -> argparse.ArgumentParser:
    """Parser del harness: medidos + gates de sesión (puro, testeable).

    Los gates de sesión (pipeline, OOM, memoria, artefactos, A/B, endurance)
    vienen de la corrida larga (ADR-019) y se pasan por flags: sin ellos
    quedan en None = FALLA — un "go" requiere la evidencia de sesión.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmup-audio",
        type=_wav_existente,
        required=True,
        help="WAV de voz real para el warm-up de Whisper (obligatorio: sin el "
        "decoder ejercitado la VRAM subestima y el harness hace raise)",
    )
    parser.add_argument(
        "--pipeline-p95",
        type=float,
        help="Pipeline warm p95 en ms, de la corrida de flujo (ADR-015)",
    )
    for nombre, ayuda in (
        ("oom", "la corrida larga registró OOM (True = FALLA)"),
        ("memoria_estable", "la memoria no creció sostenidamente (True = PASA)"),
        ("artefactos", "se detectaron palabras añadidas/omitidas/repetidas (True = FALLA)"),
        ("voz_reconocible_ab", "la voz fue reconocible en A/B (True = PASA)"),
        ("endurance_90min", "completó 90 minutos continuos (True = PASA)"),
    ):
        parser.add_argument(
            f"--{nombre.replace('_', '-')}",
            action=argparse.BooleanOptionalAction,
            help=ayuda,
        )
    return parser


def componer_medicion(
    ttfa: float | None,
    vram: float | None,
    ram: float | None,
    args: argparse.Namespace,
) -> MedicionTts:
    """Une lo medido (TTFA/VRAM/RAM) con lo reportado de sesión en la medición."""
    return MedicionTts(
        ttfa_caliente_p95_ms=ttfa,
        vram_mib=vram,
        ram_mib=ram,
        pipeline_p95_ms=args.pipeline_p95,
        oom=args.oom,
        memoria_estable=args.memoria_estable,
        artefactos=args.artefactos,
        voz_reconocible_ab=args.voz_reconocible_ab,
        endurance_90min=args.endurance_90min,
    )


def _bytes_a_mib(valor: float) -> float:
    """Bytes → MiB: unidad fijada (valor/1024², no valor/1024³ — el 1000x del #15)."""
    return valor / (1024 * 1024)


def medir_ram_mib() -> float | None:
    """RAM total en uso del sistema (MiB) vía psutil. None si no está instalado.

    Definición (ADR-014): `virtual_memory().used` es de TODA la máquina — el
    veredicto depende de qué más esté abierto; anotar al correr (como nvidia-smi).
    """
    try:
        import psutil
    except ImportError:
        return None
    return _bytes_a_mib(psutil.virtual_memory().used)  # pragma: no cover - máquina
