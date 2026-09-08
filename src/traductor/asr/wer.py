"""Word Error Rate (WER) — métrica pura de precisión ASR.

Compara la transcripción del motor contra una referencia al nivel de palabra
(distancia de edición de Levenshtein normalizada por palabras de referencia).
Sin dependencias externas: solo stdlib.
"""

from __future__ import annotations


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

    Con referencia vacía: 0.0 si la hipótesis también es vacía, si no 1.0
    (toda palabra emitida es un error; no hay denominador válido).
    """
    ref = referencia.split()
    hip = hipotesis.split()
    if not ref:
        return 0.0 if not hip else 1.0
    return distancia_edicion(ref, hip) / len(ref)
