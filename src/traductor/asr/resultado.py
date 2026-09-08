"""Resultado de una transcripción ASR — dominio del benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultadoAsr:
    """Una transcripción medida: motor, idioma, texto y latencia en ms.

    `referencia` es el texto esperado (para WER); si viene vacía, no se
    calcula WER para este resultado (transcripción en vivo sin ground truth).
    """

    engine: str
    idioma: str
    frase: str
    elapsed_ms: float
    referencia: str = ""

    def __post_init__(self) -> None:
        idioma = self.idioma.strip()
        if not idioma:
            raise ValueError("idioma vacío: se necesita un valor")
        if self.elapsed_ms < 0:
            raise ValueError(f"elapsed_ms negativo: {self.elapsed_ms}")
        # dataclass frozen: requiere object.__setattr__ para reescribir
        object.__setattr__(self, "idioma", idioma)
