"""Benchmark ASR bidireccional — faster-whisper vs moonshine, en la máquina objetivo.

Wrapper fino: carga los motores reales y usa `traductor.asr` (manifesto,
medición, agregación). Corre sobre WAVs de `scripts/audio/` según
`scripts/audio/manifesto.json` y escribe la tabla comparativa.

Uso:
    $env:PYTHONPATH = "src"
    python scripts/benchmark_asr.py

El CI NO lo ejecuta: requiere GPU + modelos + audios de referencia. La salida
se pega como evidencia en el ADR-012.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from traductor.asr.benchmark import resumen_asr, tabla_comparativa
from traductor.asr.manifesto import Muestra, cargar_manifesto, muestras_existentes
from traductor.asr.medicion import medir_motores

RAIZ_AUDIO = Path(__file__).parent / "audio"
MANIFESTO = RAIZ_AUDIO / "manifesto.json"


def _faster_whisper() -> Any:
    """Modelo faster-whisper tiny int8 en CUDA (ya en requirements)."""
    from faster_whisper import WhisperModel

    return WhisperModel("tiny", device="cuda", compute_type="int8_float16")


def _moonshine() -> Any:
    """Modelo moonshine tiny. Import lento: no está en requirements aún."""
    import moonshine

    return moonshine.load_model("tiny")


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


def _cargar_modelos() -> dict[str, Any]:
    modelos: dict[str, Any] = {}
    for motor, cargar in (("faster-whisper", _faster_whisper), ("moonshine", _moonshine)):
        try:
            modelos[motor] = cargar()
            print(f"{motor}: modelo cargado")
        except Exception as exc:  # ImportError / modelo no disponible
            print(f"{motor}: no disponible ({exc})")
    return modelos


def _muestras() -> list[Muestra]:
    try:
        muestras = cargar_manifesto(MANIFESTO)
    except (FileNotFoundError, ValueError) as exc:
        print(f"manifest inválido: {exc}")
        return []
    existentes, faltantes = muestras_existentes(muestras)
    for nombre in faltantes:
        print(f"falta audio: {nombre}")
    return existentes


def _transcribir_con(modelo: Any, motor: str) -> Callable[[Path, str], str]:
    """Fija modelo+motor en un callable (ruta, idioma) -> texto, tipado."""

    def transcribir(ruta: Path, idioma: str) -> str:
        return _transcribir(modelo, motor, ruta, idioma)

    return transcribir


def main() -> None:
    muestras = _muestras()
    modelos = _cargar_modelos()
    if not muestras or not modelos:
        print("Sin muestras o sin motores disponibles.")
        return
    motores = {motor: _transcribir_con(modelo, motor) for motor, modelo in modelos.items()}
    resultados = medir_motores(motores, muestras)
    print()
    print(tabla_comparativa(resumen_asr(resultados)))


if __name__ == "__main__":
    main()
