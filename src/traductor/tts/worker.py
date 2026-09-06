"""Worker TTS entre procesos (stdin/stdout, JSON lines).

Frontera obligada: el pipeline principal (torch 2.13+cu132, numpy 2) y
Chatterbox (torch 2.6, numpy<2) viven en venvs incompatibles, así que no se
importan entre sí. Este worker corre DENTRO de venv-tts, carga el modelo UNA
vez y atiende una frase por línea:

  entrada:  {"texto": "hola", "ref_audio": "ref.wav", "exaggeration": 0.5, "language_id": "es"}
  salida:   {"ok": true, "wav": "out/0000.wav", "sr": 24000, "ms": 512.3}

Alcance de este PR: adapter standalone + contrato probado. La integración
con el pipeline en vivo va en otro PR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any, TextIO

from traductor.tts.chatterbox import ChatterboxModel, cargar_modelo, sintetizar


def parse_request(line: str) -> tuple[str, str, float, str]:
    """Valida una línea JSON -> (texto, ref_audio, exaggeration, language_id)."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"línea no es JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("request debe ser un objeto JSON")
    texto = data.get("texto", "")
    ref_audio = data.get("ref_audio", "")
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("campo 'texto' requerido y no vacío")
    if not isinstance(ref_audio, str) or not ref_audio:
        raise ValueError("campo 'ref_audio' requerido")
    try:
        exaggeration = float(data.get("exaggeration", 0.5))
    except (TypeError, ValueError) as e:
        raise ValueError("campo 'exaggeration' debe ser número") from e
    language_id = data.get("language_id", "en")
    if not isinstance(language_id, str) or not language_id:
        raise ValueError("campo 'language_id' debe ser string no vacío")
    return texto, ref_audio, exaggeration, language_id


def guardar_wav(ruta: Path, wav: Any, sr: int) -> None:
    """Guarda wav (tensor torch o array numpy) como PCM16 mono. Solo stdlib+numpy."""
    import numpy as np

    arr = np.asarray(wav.cpu().numpy() if hasattr(wav, "cpu") else wav, dtype=np.float32)
    arr = np.atleast_1d(arr.reshape(-1))
    pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(ruta), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def run(
    entrada: TextIO,
    salida: TextIO,
    modelo: ChatterboxModel,
    out_dir: Path,
) -> int:
    """Atiende líneas hasta EOF. Devuelve frases procesadas (errores no paran el loop)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in entrada:
        line = line.strip()
        if not line:
            continue
        try:
            texto, ref_audio, exaggeration, language_id = parse_request(line)
            t0 = time.perf_counter()
            wav, sr = sintetizar(
                texto,
                ref_audio,
                modelo,
                exaggeration=exaggeration,
                language_id=language_id,
            )
            ms = (time.perf_counter() - t0) * 1000.0
            destino = out_dir / f"{n:04d}.wav"
            guardar_wav(destino, wav, sr)
            salida.write(
                json.dumps({"ok": True, "wav": str(destino), "sr": sr, "ms": round(ms, 1)}) + "\n"
            )
        except (ValueError, FileNotFoundError, OSError) as e:
            salida.write(json.dumps({"ok": False, "error": str(e)}) + "\n")
        n += 1
    salida.flush()
    return n


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: carga el modelo una vez y atiende stdin. Para venv-tts."""
    parser = argparse.ArgumentParser(description="Worker TTS Chatterbox (stdin/stdout JSON)")
    parser.add_argument("--out-dir", default="out_tts", help="carpeta para los .wav")
    parser.add_argument("--device", default="cuda", help="cuda o cpu")
    args = parser.parse_args(argv)

    modelo = cargar_modelo(device=args.device)
    total = run(sys.stdin, sys.stdout, modelo, Path(args.out_dir))
    print(f"frases: {total}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point, cubierto por test_main_*
    raise SystemExit(main())
