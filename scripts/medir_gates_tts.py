"""Mide los gates de aceptación del motor TTS en la máquina objetivo (ADR-014).

Carga el motor candidato (inyectado; aún sin elegir, ADR-011) Y faster-whisper
co-residente (tiny int8; el "~1 GB" del ADR es un supuesto, la primera corrida
lo mide). Mide TTFA caliente p95 con el medidor honesto (n>=20, math.ceil,
None si falta) y la VRAM TOTAL a nivel driver (`vram_ocupada_mib`, incluye
CTranslate2). La foto de Whisper se toma justo tras su warm-up (delta aislado);
la de co-residencia, después de las síntesis TTFA.

Uso:
    $env:PYTHONPATH = "src"
    python scripts/medir_gates_tts.py --warmup-audio tu_voz.wav

El CI NO lo ejecuta: requiere GPU + modelo + Whisper. La salida se pega como
evidencia en el ADR-014.
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path
from typing import Any

from traductor.hardware.cuda import vram_ocupada_mib
from traductor.latencia.medidor import agregar_medicion, medir_tiempo, resumen_estadisticas
from traductor.tts.gates import MedicionTts, cabe_en_gates, evaluar_gates, resumen_gates
from traductor.tts.modelos import VoiceProfile

PERFIL = VoiceProfile(id="benchmark", nombre="Benchmark", muestras=("ref.wav",))
TEXTO = "hola, esto es una prueba del motor de voz"
N_REPETICIONES = 20
# Piso para la auto-verificación de Whisper. tiny int8 son decenas de MB de
# pesos; el bulto del contexto CUDA ya existe en vram_base. 50 es conservador,
# NO 500 (estimación de memoria citada sin verificar, r4).
DELTA_WHISPER_MIN_MIB = 50.0


def _cargar_motor() -> Any:
    """Carga el candidato primario (ADR-011): XTTS-v2 vía fork coqui-tts.

    Requiere coqui-tts instalado (venv propio del TTS) y el modelo descargado.
    Si la prueba rechaza XTTS, se cambia aquí por el candidato B
    (Supertonic 3 CPU + OpenVoice V2).
    """
    from traductor.tts.backend_xtts import BackendXtts

    return BackendXtts(idioma_salida="en")


def _cargar_whisper() -> Any:
    """faster-whisper tiny int8 en CUDA: carga el ASR co-residente (ADR-014)."""
    from faster_whisper import WhisperModel

    return WhisperModel("tiny", device="cuda", compute_type="int8_float16")


def _calentar_whisper_y_foto(whisper: Any, audio: Path) -> float | None:
    """Calienta la PRIMERA inferencia de Whisper y devuelve la VRAM tras ella.

    CTranslate2 reserva workspace/KV/beam en la primera `transcribe()`, no en
    `__init__` (r1 del PR #15). `transcribe()` devuelve un generador: se drena
    con `list()` o la inferencia no ocurre y el warm-up es decorativo.

    La foto se toma INMEDIATAMENTE después del warm-up (r4): así el delta
    contra la línea base aísla la VRAM de Whisper, sin mezclarla con las
    reservas perezosas que el motor haga en sus propias síntesis.

    Conteo de segmentos (r5/r6): el umbral de 50 detecta "Whisper ausente", no
    "presente pero frío" — los pesos se reservan en `__init__`, así que si
    `no_speech_threshold` corta el silencio, el delta igual pasa. 0 segmentos
    = el decoder no corrió y la VRAM subestimaría: `raise` con la salida
    (`--warmup-audio` con voz real), no un print que se pierde en la corrida.
    """
    import torch

    segmentos, _ = whisper.transcribe(str(audio), language="es")
    lista = list(segmentos)
    if not lista:
        raise RuntimeError(
            "el warm-up dio 0 segmentos: el decoder no se ejercitó y la VRAM "
            "subestimaría. El WAV pasado no produjo segmentos (¿silencio "
            "cortado por no_speech?). Usa --warmup-audio con un WAV de voz "
            "real; el instrumento no mide con warm-up frío."
        )
    print(f"warm-up: {len(lista)} segmentos")
    return vram_ocupada_mib(torch.cuda)


def _medir_ttfa_p95(motor: Any, n: int) -> float | None:
    """TTFA caliente p95, con el medidor honesto (math.ceil, None si n<20)."""
    motor.sintetizar(TEXTO, PERFIL)  # warm-up
    registro: dict[str, list[float]] = {}
    for _ in range(n):
        _, elapsed_ms = medir_tiempo(
            partial(motor.sintetizar, TEXTO, PERFIL),
            clock=time.perf_counter,
        )
        registro = agregar_medicion(registro, "ttfa", elapsed_ms)
    return resumen_estadisticas(registro)["ttfa"]["p95"]


def _wav_existente(ruta: str) -> Path:
    """Validación de argparse: falla antes de tocar la GPU (r8)."""
    p = Path(ruta)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"el WAV de warm-up no existe: {ruta}")
    return p


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup-audio",
        type=_wav_existente,
        required=True,
        help="WAV de voz real para el warm-up de Whisper (obligatorio: sin el "
        "decoder ejercitado la VRAM subestima y el harness hace raise)",
    )
    args = parser.parse_args()

    motor = _cargar_motor()
    vram_base = vram_ocupada_mib(torch.cuda)  # motor residente, Whisper aún no
    whisper = _cargar_whisper()
    # Orden deliberado: Whisper ya residente ANTES de medir TTFA — las síntesis
    # corren bajo presión de VRAM real (co-residencia, ADR-014). Si se invierte
    # el orden, el TTFA baja "gratis" y el gate miente.
    vram_tras_whisper = _calentar_whisper_y_foto(whisper, args.warmup_audio)
    if vram_base is not None and vram_tras_whisper is not None:
        delta = vram_tras_whisper - vram_base
        print(f"delta VRAM (Whisper, foto tras warm-up): {delta:.1f} MiB")
        if delta < DELTA_WHISPER_MIN_MIB:
            raise RuntimeError(
                f"Whisper no reservó VRAM (delta {delta:.1f} MiB < "
                f"{DELTA_WHISPER_MIN_MIB:g}): el warm-up no ejercitó CT2; "
                "instrumento no fiable."
            )
    ttfa = _medir_ttfa_p95(motor, N_REPETICIONES)
    # Foto de co-residencia DESPUÉS de las síntesis TTFA: Whisper + motor con
    # sus reservas reales. Es el número del gate.
    vram = vram_ocupada_mib(torch.cuda)
    del whisper, motor  # liberar DESPUÉS de la foto, no antes
    medicion = MedicionTts(ttfa_caliente_p95_ms=ttfa, vram_mib=vram)
    resultados = evaluar_gates(medicion)
    if ttfa is None:
        print(f"TTFA caliente p95: sin medir (n<{N_REPETICIONES})")
    else:
        print(f"TTFA caliente p95: {ttfa:.1f} ms")
    if vram is None:
        print("VRAM co-residente: sin medir (sin CUDA)")
    else:
        print(f"VRAM co-residente: {vram:.1f} MiB (compara con nvidia-smi ±100 MiB)")
    print()
    print(resumen_gates(resultados))
    print()
    print("cabe_en_gates:", cabe_en_gates(medicion))


if __name__ == "__main__":
    main()
