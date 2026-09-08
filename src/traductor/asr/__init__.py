"""Benchmark ASR — medición de faster-whisper vs moonshine (ES y EN).

PR #13 NO sustituye RealtimeSTT: decide con evidencia qué motor usar para la
dirección que habla (ES→EN) y la que se escucha (EN→ES). La lógica pura
(WER, manifest, medición, agregación) vive aquí; los motores corren en
`scripts/benchmark_asr.py` en la máquina objetivo.
"""

from traductor.asr.benchmark import resumen_asr, tabla_comparativa
from traductor.asr.manifesto import Muestra, cargar_manifesto, muestras_existentes
from traductor.asr.medicion import medir_motores
from traductor.asr.resultado import ResultadoAsr
from traductor.asr.wer import normalizar_texto, wer

__all__ = [
    "Muestra",
    "ResultadoAsr",
    "cargar_manifesto",
    "medir_motores",
    "muestras_existentes",
    "normalizar_texto",
    "resumen_asr",
    "tabla_comparativa",
    "wer",
]
