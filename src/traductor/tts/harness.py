"""Partes puras del harness del ADR-014: parser y composición de la medición.

Viven en el módulo (no en el script) para ser testeables y mutables en CI:
el harness es hardware (GPU + modelos), su lógica no. El script
`scripts/medir_gates_tts.py` queda como wrapper fino.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from traductor.tts.gates import MedicionTts
from traductor.tts.modelos import VoiceProfile

TEXTO_POR_DEFECTO = "hola, esto es una prueba del motor de voz"


def _wav_existente(nombre: str) -> Callable[[str], Path]:
    """Fábrica de validación argparse: el mensaje nombra al flag que falla."""

    def validar(ruta: str) -> Path:
        p = Path(ruta)
        if not p.is_file():
            raise argparse.ArgumentTypeError(f"el WAV de {nombre} no existe: {ruta}")
        return p

    return validar


def parser_harness() -> argparse.ArgumentParser:
    """Parser del harness: medidos + gates de sesión (puro, testeable).

    Los gates de sesión (pipeline, OOM, memoria, artefactos, A/B, endurance)
    vienen de la corrida larga (ADR-019) y se pasan por flags: sin ellos
    quedan en None = FALLA — un "go" requiere la evidencia de sesión.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmup-audio",
        type=_wav_existente("warm-up"),
        required=True,
        help="WAV de voz real para el warm-up de Whisper (obligatorio: sin el "
        "decoder ejercitado la VRAM subestima y el harness hace raise)",
    )
    parser.add_argument(
        "--referencia",
        type=_wav_existente("referencia"),
        nargs="+",
        help="Muestras de referencia (1 o más) para el perfil TTS (default: --warmup-audio)",
    )
    parser.add_argument(
        "--motor",
        choices=("xtts", "b"),
        default="xtts",
        help="Candidato a medir: xtts (primario ADR-011) o b (Supertonic 3 + OpenVoice V2)",
    )
    parser.add_argument(
        "--texto",
        default=TEXTO_POR_DEFECTO,
        help="Texto sintetizado en cada repetición (la salida del flujo ES→EN es inglés)",
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


def perfil_benchmark(args: argparse.Namespace) -> VoiceProfile:
    """Perfil del harness: `--referencia` (1+ muestras) o el warm-up si no se pasa."""
    referencia = args.referencia or [args.warmup_audio]
    return VoiceProfile(
        id="benchmark", nombre="Benchmark", muestras=tuple(str(m) for m in referencia)
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
