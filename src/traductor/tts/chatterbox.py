"""TTS con Chatterbox Multilingual — lazy import, 2-3 GB VRAM.

Mitigación VRAM: Whisper int8 ~1 GB + Chatterbox ~2-3 GB = ~3-4 GB en 4 GB totales.
Medir con medidor.py antes de fijar ADR de latencia (estimado 400-700 ms primer chunk).
"""

from __future__ import annotations

from pathlib import Path


def cargar_modelo(device: str = "cuda", multilingue: bool = True):  # pragma: no cover
    """Carga Chatterbox. Lazy import para no exigir deps en CI/tests."""
    if multilingue:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        return ChatterboxMultilingualTTS.from_pretrained(device=device)
    else:
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(device=device)


def sintetizar(
    texto: str,
    ref_audio: str | Path,
    device: str = "cuda",
    exaggeration: float = 0.5,
    language_id: str = "en",
) -> tuple[object, int]:
    """Texto -> (wav tensor, sample_rate). Necesita clip de voz de ~10s."""
    if not texto or not str(texto).strip():
        raise ValueError("texto vacío")
    ref = Path(ref_audio)
    if not ref.exists():
        raise FileNotFoundError(f"ref_audio no existe: {ref}")
    if not 0.0 <= exaggeration <= 1.0:
        raise ValueError(f"exaggeration fuera de rango [0,1]: {exaggeration}")

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    modelo = ChatterboxMultilingualTTS.from_pretrained(device=device)
    wav = modelo.generate(
        text=texto,
        audio_prompt_path=str(ref),
        exaggeration=exaggeration,
        language_id=language_id,
    )
    return wav, modelo.sr
