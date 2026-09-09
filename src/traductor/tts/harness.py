"""Partes puras del harness del ADR-014: parser y composición de la medición.

Viven en el módulo (no en el script) para ser testeables y mutables en CI:
el harness es hardware (GPU + modelos), su lógica no. El script
`scripts/medir_gates_tts.py` queda como wrapper fino.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

from traductor.tts.gates import MedicionTts
from traductor.tts.modelos import VoiceProfile

TEXTO_POR_DEFECTO = "hola, esto es una prueba del motor de voz"


class RegistroEtapas:
    """Marcas de tiempo por frontera con reloj inyectable (atribución del pipeline).

    Los deltas entre marcas consecutivas se miden con EL MISMO reloj, así que
    su suma es EXACTA por construcción (cierre del 100 %). `verificar_cierre`
    atrapa un hueco sin instrumentar (misma filosofía que los guards de ruteo:
    si no se puede atribuir, no se reporta como medido).
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._marcas: list[tuple[str, float]] = []

    def marcar(self, etapa: str) -> None:
        self._marcas.append((etapa, self._clock()))

    def desglose_ms(self) -> dict[str, float]:
        """Deltas por etapa en ms (la primera marca abre el contador)."""
        if not self._marcas:
            return {}
        desglose: dict[str, float] = {}
        for i in range(1, len(self._marcas)):
            etapa = self._marcas[i][0]
            desglose[etapa] = (self._marcas[i][1] - self._marcas[i - 1][1]) * 1000.0
        return desglose

    def total_ms(self) -> float:
        if len(self._marcas) < 2:
            return 0.0
        return (self._marcas[-1][1] - self._marcas[0][1]) * 1000.0

    def inicio_s(self) -> float:
        """Valor del reloj en la primera marca (para mediciones que parten de ahí)."""
        return self._marcas[0][1] if self._marcas else self._clock()

    def verificar_cierre(self, total_ms: float) -> None:
        """La suma de las etapas DEBE cerrar el total (100 % atribuido)."""
        suma = sum(self.desglose_ms().values())
        # pragma: no mutate - el mutante `>` → `>=` en el épsilon de cierre es
        # equivalente: la frontera exacta de 1e-6 ms no es alcanzable con
        # floats (la diferencia nunca es exactamente el épsilon)
        if abs(suma - total_ms) > 1e-6:  # pragma: no mutate - ver comentario
            raise RuntimeError(
                f"residuo sin atribuir: {total_ms - suma:.3f} ms (etapas "
                f"suman {suma:.3f} ms, total {total_ms:.3f} ms)"
            )


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
        "--asr-p95",
        type=float,
        help="ASR warm p95 en ms, MEDIDO en el hardware (presupuesto TTFA derivado)",
    )
    parser.add_argument(
        "--traduccion-p95",
        type=float,
        help="Traducción Argos p95 en ms, MEDIDA (presupuesto TTFA derivado)",
    )
    parser.add_argument(
        "--ruteo-p95",
        type=float,
        help="Ruteo a VB-CABLE p95 en ms, MEDIDO (presupuesto TTFA derivado)",
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
    *,
    ttfa: float | None,
    asr: float | None,
    traduccion: float | None,
    ruteo: float | None,
    pipeline: float | None,
    vram: float | None,
    ram: float | None,
    args: argparse.Namespace,
) -> MedicionTts:
    """Une lo medido (TTFA/etapas/pipeline/VRAM/RAM) con lo reportado de sesión.

    El pipeline MEDIDO gana sobre `--pipeline-p95` (el flag queda para la
    corrida de sesión del ADR-019); los flags de sesión van tal cual.
    """
    return MedicionTts(
        ttfa_caliente_p95_ms=ttfa,
        asr_p95_ms=asr,
        traduccion_p95_ms=traduccion,
        ruteo_p95_ms=ruteo,
        vram_mib=vram,
        ram_mib=ram,
        pipeline_p95_ms=args.pipeline_p95 if pipeline is None else pipeline,
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


def verificar_resolucion_audible(p95_ms: float | None, resolucion_ms: float) -> None:
    """Guard del detector de loopback: reportar por debajo de su resolución es
    una medición que no ocurrió (patrón del 0.1 ms vs 10 ms del PR #20).

    - p95 < resolucion / 2 -> raise (sub-resolución: el evento no se midió).
    - p95 None -> raise (sin medición).
    """
    if p95_ms is None:
        raise RuntimeError("p95 de la frontera audible: sin medir (instrumento roto)")
    if p95_ms < resolucion_ms / 2:
        raise RuntimeError(
            f"p95 de la frontera audible {p95_ms:g} ms < resolución del detector "
            f"({resolucion_ms:g} ms): medición sub-resolución, el evento audible "
            "no se midió (instrumento roto)"
        )


def verificar_ruteo_primer_sample(p95_ms: float | None, duracion_chunk_ms: float) -> None:
    """Auto-verificación del instrumento de ruteo (time-to-first-sample-audible).

    El ruteo mide desde que el chunk se entrega al dispositivo hasta que el
    PRIMER sample es reproducible en CABLE Input — NUNCA hasta que el chunk
    termina de sonar (la duración del audio no es latencia). El guard atrapa
    el error conocido (segunda métrica mal definida, ver ADR-014):
    - p95 >= duración del chunk -> la medición incluyó el drenado completo:
      mide "hasta que terminó de sonar", no "hasta que empezó" -> raise.
    - p95 <= 0 (o None) -> no se midió nada -> raise.
    """
    if p95_ms is None or p95_ms <= 0:
        raise RuntimeError(
            f"p95 de ruteo {p95_ms} ms: indistinguible de cero — no se midió "
            "nada (instrumento roto)"
        )
    if p95_ms >= duracion_chunk_ms:
        raise RuntimeError(
            f"p95 de ruteo {p95_ms:g} ms >= duración del chunk "
            f"{duracion_chunk_ms:g} ms: la medición incluyó el drenado "
            "completo, no el primer sample audible (error conocido, "
            "instrumento roto)"
        )
