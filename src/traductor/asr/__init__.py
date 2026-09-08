"""Benchmark ASR — medición de faster-whisper vs moonshine (ES y EN).

PR #13 NO sustituye RealtimeSTT: decide con evidencia qué motor usar para la
dirección que habla (ES→EN) y la que se escucha (EN→ES). La lógica pura
(WER, agregación, tabla) vive aquí; los motores corren en
`scripts/benchmark_asr.py` en la máquina objetivo.
"""

from traductor.asr.benchmark import resumen_asr, tabla_comparativa
from traductor.asr.resultado import ResultadoAsr
from traductor.asr.wer import wer

__all__ = ["ResultadoAsr", "resumen_asr", "tabla_comparativa", "wer"]
