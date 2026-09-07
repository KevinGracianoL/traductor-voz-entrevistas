"""Servidor TTS persistente — wrapper fino. Corre con el venv principal.

Uso:
  $env:PYTHONPATH = "src"
  python scripts/tts_servidor.py
Lee líneas stdin como JSON {"texto": ..., "idioma": "es"} y guarda out/NNNN.wav.
"""

from traductor.tts.servidor import ServidorTTS

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    srv = ServidorTTS.cargar()
    out = Path("out")
    out.mkdir(exist_ok=True)
    for i, linea in enumerate(sys.stdin):
        linea = linea.strip()
        if not linea:
            continue
        data = json.loads(linea)
        pcm, sr = srv.atender(data["texto"], data.get("idioma", "es"))
        destino = out / f"{i:04d}.wav"
        import wave

        with wave.open(str(destino), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(pcm)
        print(str(destino), flush=True)
