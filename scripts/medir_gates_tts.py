"""Mide los gates de aceptación del motor TTS en la máquina objetivo (ADR-014).

Carga el motor candidato (inyectado; aún sin elegir, ADR-011) Y faster-whisper
co-residente: el ~1 GB de Whisper es lo que da sentido al umbral de VRAM.
Mide TTFA caliente p95 con el medidor honesto (n>=20, math.ceil, None si falta)
y la VRAM TOTAL a nivel driver (`vram_ocupada_mib`, incluye CTranslate2).

Uso:
    $env:PYTHONPATH = "src"
    python scripts/medir_gates_tts.py

El CI NO lo ejecuta: requiere GPU + modelo + Whisper. La salida se pega como
evidencia en el ADR-014.
"""

from __future__ import annotations

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


def _cargar_motor() -> Any:
    """Carga el motor candidato. Cambiar cuando ADR-011 elija uno."""
    raise NotImplementedError("sin motor elegido (ADR-011): inyectar el candidato aquí")


def _cargar_whisper() -> Any:
    """faster-whisper tiny int8 en CUDA: carga el ASR co-residente (ADR-014)."""
    from faster_whisper import WhisperModel

    return WhisperModel("tiny", device="cuda", compute_type="int8_float16")


def _generar_wav_silencio(duracion_s: float = 1.0) -> Path:
    """WAV mono 16-bit de silencio: basta para forzar la primera inferencia."""
    import tempfile
    import wave

    sr = 16000
    ruta = Path(tempfile.gettempdir()) / "traductor_whisper_warmup.wav"
    with wave.open(str(ruta), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(b"\x00\x00" * int(sr * duracion_s))
    return ruta


def _medir_vram_con_whisper(whisper: Any, vram_base: float | None) -> float | None:
    """VRAM usada (MiB) con Whisper co-residente y la PRIMERA inferencia hecha.

    CTranslate2 reserva workspace/KV/beam en la primera `transcribe()`, no en
    `__init__` (r1 del PR #15). `transcribe()` devuelve un generador: se drena
    con `list()` o la inferencia no ocurre y el warm-up es decorativo.

    Auto-verificación del instrumento (r3): si el delta contra la línea base
    (< 500 MiB) dice que Whisper no cargó de verdad, `raise` — "no medido no
    pasa en silencio" (principio de gates.py) aplicado al propio harness.
    """
    import torch

    audio = _generar_wav_silencio()
    segmentos, _ = whisper.transcribe(str(audio), language="es")
    list(segmentos)
    vram = vram_ocupada_mib(torch.cuda)
    if vram_base is not None and vram is not None and vram - vram_base < 500:
        raise RuntimeError(
            f"Whisper no reservó VRAM (delta {vram - vram_base:.1f} MiB < 500): "
            "el warm-up no ejercitó CT2; instrumento no fiable."
        )
    return vram


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


def main() -> None:
    import torch

    motor = _cargar_motor()
    vram_base = vram_ocupada_mib(torch.cuda)  # motor residente, Whisper aún no
    whisper = _cargar_whisper()
    # Orden deliberado: Whisper ya residente ANTES de medir TTFA — las síntesis
    # corren bajo presión de VRAM real (co-residencia, ADR-014). Si se invierte
    # el orden, el TTFA baja "gratis" y el gate miente.
    ttfa = _medir_ttfa_p95(motor, N_REPETICIONES)
    vram = _medir_vram_con_whisper(whisper, vram_base)
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
