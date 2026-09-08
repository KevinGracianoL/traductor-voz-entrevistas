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


def _medir_vram_con_asr(whisper: Any) -> float | None:
    """VRAM usada (MiB) con Whisper y el motor co-residentes, a nivel driver.

    `whisper` se pasa para mantener el modelo vivo durante la medición: el
    gate mide la VRAM total del dispositivo (incluye lo que CTranslate2
    reserva fuera del allocator de torch).
    """
    return vram_ocupada_mib()


def main() -> None:
    motor = _cargar_motor()
    whisper = _cargar_whisper()
    ttfa = _medir_ttfa_p95(motor, N_REPETICIONES)
    vram = _medir_vram_con_asr(whisper)
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
