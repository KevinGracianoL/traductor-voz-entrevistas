"""TTS con Chatterbox Multilingual — lazy import.

Hipótesis sin medir: ~2-3 GB VRAM y 400-700 ms primer chunk en GTX 1650 Ti.
Pendiente: instalación limpia + inferencia real + VRAM pico + latencia cold/warm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast


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

    return cast(ChatterboxModel, ChatterboxMultilingualTTS.from_pretrained(device=device))


def sintetizar(
    texto: str,
    ref_audio: str | Path,
    device: str = "cuda",
    exaggeration: float = 0.5,
    language_id: str = "en",
    modelo: ChatterboxModel | None = None,
) -> tuple[object, int]:
    """Texto -> (wav tensor, sample_rate). Necesita clip de voz de ~10s.

    Si se pasa `modelo`, se reutiliza (evita recargar por frase). Si no, se
    carga uno nuevo vía `cargar_modelo`.
    """
    if not texto or not str(texto).strip():
        raise ValueError("texto vacío")
    ref = Path(ref_audio)
    if not ref.is_file():
        raise FileNotFoundError(f"ref_audio no existe: {ref}")
    if not 0.0 <= exaggeration <= 1.0:
        raise ValueError(f"exaggeration fuera de rango [0,1]: {exaggeration}")

    m = modelo if modelo is not None else cargar_modelo(device=device)
    wav = m.generate(
        text=texto,
        audio_prompt_path=str(ref),
        exaggeration=exaggeration,
        language_id=language_id,
    )
    return wav, m.sr
