"""TTS con Chatterbox Multilingual — lazy import.

Hipótesis sin medir: ~2-3 GB VRAM y 400-700 ms primer chunk en GTX 1650 Ti.
Pendiente: instalación limpia + inferencia real + VRAM pico + latencia cold/warm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ChatterboxModel(Protocol):
    """Contrato mínimo que usa sintetizar (evita object + type: ignore)."""

    sr: int

    def generate(
        self,
        *,
        text: str,
        audio_prompt_path: str,
        exaggeration: float,
        language_id: str,
    ) -> object: ...


def cargar_modelo(device: str = "cuda") -> ChatterboxModel:
    """Carga Chatterbox Multilingual. Lazy import para no exigir deps en CI/tests."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    modelo: ChatterboxModel = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return modelo


def sintetizar(
    texto: str,
    ref_audio: str | Path,
    modelo: ChatterboxModel,
    exaggeration: float = 0.5,
    language_id: str = "en",
) -> tuple[object, int]:
    """Texto -> (wav tensor, sample_rate). Necesita clip de voz de ~10s.

    `modelo` es obligatorio y lo posee el caller (cargado una vez vía
    `cargar_modelo` o por el worker): la API hace imposible recargar
    pesos por frase, no solo lo desaconseja.
    """
    if not texto or not str(texto).strip():
        raise ValueError("texto vacío")
    ref = Path(ref_audio)
    if not ref.is_file():
        raise FileNotFoundError(f"ref_audio no existe: {ref}")
    if not 0.0 <= exaggeration <= 1.0:
        raise ValueError(f"exaggeration fuera de rango [0,1]: {exaggeration}")

    wav = modelo.generate(
        text=texto,
        audio_prompt_path=str(ref),
        exaggeration=exaggeration,
        language_id=language_id,
    )
    return wav, modelo.sr
