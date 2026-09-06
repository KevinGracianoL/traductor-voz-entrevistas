"""Worker TTS — wrapper fino. Corre DENTRO de venv-tts (ver requirements-tts.txt).

Uso:
  $env:PYTHONPATH = "src"
  Get-Content frases.jsonl | python scripts/tts_worker.py --out-dir out_tts
"""

from traductor.tts.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
