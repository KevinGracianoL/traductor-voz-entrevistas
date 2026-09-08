"""Mide los gates de aceptación del motor TTS en la máquina objetivo (ADR-014).

Carga el motor candidato (inyectado; aún sin elegir, ADR-011), lo calienta,
mide TTFA p95 sobre n≥20 síntesis y la VRAM con Whisper co-residente, y
evalúa los gates con `traductor.tts.gates`.

Uso:
    $env:PYTHONPATH = "src"
    python scripts/medir_gates_tts.py

El CI NO lo ejecuta: requiere GPU + modelo + Whisper. La salida se pega como
evidencia en el ADR-014.
"""

from __future__ import annotations

import time
from typing import Any

from traductor.latencia.medidor import medir_tiempo
from traductor.tts.gates import MedicionTts, cabe_en_gates, evaluar_gates, resumen_gates
from traductor.tts.modelos import VoiceProfile

PERFIL = VoiceProfile(id="benchmark", nombre="Benchmark", muestras=("ref.wav",))
TEXTO = "hola, esto es una prueba del motor de voz"
N_REPETICIONES = 20


def _cargar_motor() -> Any:
    """Carga el motor candidato. Cambiar cuando ADR-011 elija uno."""
    raise NotImplementedError("sin motor elegido (ADR-011): inyectar el candidato aquí")


def _medir_ttfa(motor: Any, n: int) -> float:
    """TTFA caliente: latencia de síntesis en estado caliente, p95 (n≥20)."""
    motor.sintetizar(TEXTO, PERFIL)  # warm-up
    latencias: list[float] = []
    for _ in range(n):
        _, elapsed_ms = medir_tiempo(
            lambda: motor.sintetizar(TEXTO, PERFIL),
            clock=time.perf_counter,
        )
        latencias.append(elapsed_ms)
    latencias.sort()
    return latencias[int(0.95 * n) - 1]


def _medir_vram_mib() -> float | None:
    """VRAM en uso (MiB) con Whisper + motor cargados (co-residencia)."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_reserved() / (1024 * 1024)


def main() -> None:
    motor = _cargar_motor()
    ttfa = _medir_ttfa(motor, N_REPETICIONES)
    vram = _medir_vram_mib()
    medicion = MedicionTts(ttfa_caliente_p95_ms=ttfa, vram_mib=vram)
    resultados = evaluar_gates(medicion)
    if vram is None:
        print(f"TTFA caliente p95: {ttfa:.1f} ms | VRAM: sin medir")
    else:
        print(f"TTFA caliente p95: {ttfa:.1f} ms | VRAM: {vram:.1f} MiB")
    print()
    print(resumen_gates(resultados))
    print()
    print("cabe_en_gates:", cabe_en_gates(medicion))


if __name__ == "__main__":
    main()
