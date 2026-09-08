"""Word Error Rate (WER) — métrica pura de precisión ASR.

Compara la transcripción del motor contra una referencia al nivel de palabra
(distancia de edición de Levenshtein normalizada por palabras de referencia).
Sin dependencias externas: solo stdlib.

Normalización: ambos lados se pasan por `normalizar_texto` (minúsculas, sin
puntuación, espacios colapsados) ANTES de comparar. Sin eso, un ASR que ponga
puntuación y mayúsculas (p. ej. faster-whisper) puntúa peor que uno que no,
aunque transcriba igual de bien — y el benchmark elegiría al motor equivocado.

Decisiones explícitas (no se normalizan, siguen contando como error):
- Números: "tres" ≠ "3".
- Tildes: "café" ≠ "cafe".
"""

from __future__ import annotations

import re

_PUNTUACION = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sin puntuación, espacios colapsados.

    Mantiene letras acentuadas (ñ, á) y dígitos: "café" y "3" NO cambian.
    """
    sin_puntuacion = _PUNTUACION.sub(" ", texto.lower())
    return " ".join(sin_puntuacion.split())


def distancia_edicion(a: list[str], b: list[str]) -> int:
    """Distancia de Levenshtein entre dos listas de palabras (row-based)."""
    fila_anterior = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        fila = [i]
        for j, cb in enumerate(b, start=1):
            costo = 0 if ca == cb else 1
            fila.append(
                min(
                    fila_anterior[j] + 1,  # borrado
                    fila[j - 1] + 1,  # inserción
                    fila_anterior[j - 1] + costo,  # sustitución / igual
                )
            )
        fila_anterior = fila
    return fila_anterior[-1]


def wer(referencia: str, hipotesis: str) -> float:
    """WER normalizado: ediciones / palabras de referencia.

    Ambos lados se normalizan con `normalizar_texto` antes de comparar.
    Con referencia vacía: 0.0 si la hipótesis también es vacía, si no 1.0
    (toda palabra emitida es un error; no hay denominador válido).
    """
    ref = normalizar_texto(referencia).split()
    hip = normalizar_texto(hipotesis).split()
    if not ref:
        return 0.0 if not hip else 1.0
    return distancia_edicion(ref, hip) / len(ref)
