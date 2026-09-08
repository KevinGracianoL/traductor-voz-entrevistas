"""Benchmark ASR bidireccional — faster-whisper vs moonshine, en la máquina objetivo.

Corre sobre WAVs de referencia en `scripts/audio/` (manifest: `scripts/audio/manifesto.json`)
y escribe la tabla comparativa por `engine|idioma`.

Uso:
    $env:PYTHONPATH = "src"
    python scripts/benchmark_asr.py

El CI NO lo ejecuta: requiere GPU + modelos + audios de referencia. Aquí se
deja listo para correr y pegar la evidencia en el ADR-012.
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import Any

from traductor.asr.benchmark import resumen_asr, tabla_comparativa
from traductor.asr.resultado import ResultadoAsr
from traductor.latencia.medidor import medir_tiempo

RAIZ_AUDIO = Path(__file__).parent / "audio"
MANIFESTO = RAIZ_AUDIO / "manifesto.json"


def _faster_whisper_tiny() -> object:
    """Modelo faster-whisper tiny int8 en CUDA, cargado una vez."""
    from faster_whisper import WhisperModel

    return WhisperModel("tiny", device="cuda", compute_type="int8_float16")


def _moonshine_tiny() -> object:
    """Modelo moonshine tiny. Import lento: no está en requirements aún."""
    import moonshine

    return moonshine.load_model("tiny")


MOTORES = {
    "faster-whisper": _faster_whisper_tiny,
    "moonshine": _moonshine_tiny,
}


def _transcribir(modelo: Any, motor: str, ruta: Path, idioma: str) -> str:
    """Transcribe un archivo con el motor dado.

    `modelo` es un objeto de runtime de una librería de terceros sin stubs
    (moonshine ni está en requirements): se trata como Any a propósito.
    """
    if motor == "faster-whisper":
        segmentos, _ = modelo.transcribe(str(ruta), language=idioma)
        return " ".join(s.text.strip() for s in segmentos).strip()
    if motor == "moonshine":
        import moonshine

        tokens = modelo.transcribe(str(ruta), language=idioma)
        return str(moonshine.decode_tokens(tokens)).strip()
    raise ValueError(f"motor desconocido: {motor}")


def _cargar_manifesto() -> list[dict[str, str]]:
    if not MANIFESTO.exists():
        print(f"Sin manifest en {MANIFESTO}: genera audios de referencia primero.")
        return []
    with MANIFESTO.open(encoding="utf-8") as fh:
        datos: list[dict[str, str]] = json.load(fh)
    return datos


def main() -> None:
    muestras = _cargar_manifesto()
    if not muestras:
        return

    modelos: dict[str, object] = {}
    for motor, cargar in MOTORES.items():
        try:
            modelos[motor] = cargar()
            print(f"{motor}: modelo cargado")
        except Exception as exc:  # ImportError / modelo no disponible
            print(f"{motor}: no disponible ({exc})")

    resultados: list[ResultadoAsr] = []
    for muestra in muestras:
        ruta = RAIZ_AUDIO / muestra["ruta"]
        idioma = muestra["idioma"]
        referencia = muestra.get("referencia", "")
        if not ruta.exists():
            print(f"falta audio: {ruta}")
            continue
        for motor, modelo in modelos.items():
            frase, elapsed_ms = medir_tiempo(
                partial(_transcribir, modelo, motor, ruta, idioma),
                clock=time.perf_counter,
            )
            resultados.append(
                ResultadoAsr(
                    engine=motor,
                    idioma=idioma,
                    frase=frase,
                    elapsed_ms=elapsed_ms,
                    referencia=referencia,
                )
            )

    print()
    print(tabla_comparativa(resumen_asr(resultados)))


if __name__ == "__main__":
    main()
