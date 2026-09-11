"""Arranque del worker TTS real (ADR-013): BackendXtts + tienda JSON + main.

El worker del ADR-013 es un proceso aparte; este boot es lo que ese proceso
corre en la máquina objetivo (venv-tts, donde vive coqui-tts). El flujo
outgoing lo lanza vía `TtsWorkerCliente`.

Uso: python worker_tts_boot.py <directorio_salida>
"""

from __future__ import annotations

import sys
from pathlib import Path

from traductor.tts import worker
from traductor.tts.backend_xtts import BackendXtts
from traductor.tts.tienda_json import TiendaPerfilesJson


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - máquina
    args = list(sys.argv[1:] if argv is None else argv)
    directorio_salida = Path(args[0])
    tienda = TiendaPerfilesJson(directorio=Path(__file__).resolve().parents[1] / "perfiles")
    worker.main(
        BackendXtts(idioma_salida="en"),
        tienda,
        directorio_salida,
    )


if __name__ == "__main__":
    main()
