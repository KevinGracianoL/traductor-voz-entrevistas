"""Adaptadores del flujo outgoing_es_to_en — hardware/máquina (ADR-015).

El core (`outgoing.py`) es puro y testeado; aquí viven las piezas que tocan
hardware: micrófono (RealtimeSTT), worker TTS (proceso aparte, ADR-013),
salida a VB-CABLE y teleprompter HTTP. Son delgadas y documentadas; las
líneas que tocan hardware van `# pragma: no cover` (se ejercitan en la
máquina objetivo, como el harness del ADR-014).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from traductor.flujo.outgoing import FlujoOutgoing


class AsrRealtime:  # pragma: no cover - requiere micrófono + RealtimeSTT
    """Micrófono → segmentos: los PARCIALES van a pantalla y los FINALES al flujo.

    RealtimeSTT ya integra VAD/endpointing: `text` se dispara con el segmento
    final (traducción y TTS reciben únicamente segmentos finales, regla del
    ADR-015) y `on_realtime_transcription_update` con los parciales.
    """

    def __init__(self, flujo: FlujoOutgoing) -> None:
        self._flujo = flujo

    def _parcial(self, texto: str) -> None:
        self._flujo.parcial(texto)

    def _final(self, texto: str) -> None:
        self._flujo.segmento_final(texto)

    def correr(self) -> None:
        from RealtimeSTT import AudioToTextRecorder

        grabador = AudioToTextRecorder(
            model="tiny",
            language="es",
            device="cuda",
            compute_type="int8",
            on_realtime_transcription_update=self._parcial,
        )
        print("Flujo outgoing listo: habla en español. Ctrl+C para salir.")
        while True:
            grabador.text(self._final)


class TtsWorkerCliente:  # pragma: no cover - requiere venv-tts + modelo
    """Cliente del worker TTS (ADR-013): proceso aparte, jobs JSON-line.

    Lanza `worker_tts_boot.py` con el python del venv del TTS (que es donde
    vive coqui-tts, ADR-011), envía un job y lee el WAV de salida.
    """

    def __init__(self, *, python: Path, directorio_salida: Path, perfil_id: str) -> None:
        self._python = str(python)
        self._directorio_salida = directorio_salida
        self._perfil_id = perfil_id
        self._proceso: Any | None = None

    def iniciar(self) -> None:
        boot = Path(__file__).resolve().parents[2] / "scripts" / "worker_tts_boot.py"
        # el python viene del venv de la máquina, el script es del repo y el
        # directorio es un tempdir propio: ninguna entrada de red ni no
        # confiable llega a subprocess (falso positivo de S603).
        self._proceso = subprocess.Popen(  # noqa: S603 - ver comentario
            [self._python, str(boot), str(self._directorio_salida)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def sintetizar(self, texto_en: str) -> tuple[bytes, float, str] | None:
        """Job → WAV. None si el worker falló (escalera del flujo)."""
        if self._proceso is None or self._proceso.stdin is None:
            return None
        nombre = "turno.wav"
        job = {"texto": texto_en, "perfil_id": self._perfil_id, "salida": nombre}
        self._proceso.stdin.write(json.dumps(job) + "\n")
        self._proceso.stdin.flush()
        linea = self._proceso.stdout.readline()
        if not linea:
            return None
        resultado = json.loads(linea)
        if not resultado["ok"]:
            return None
        ruta = self._directorio_salida / resultado["salida"]
        import wave

        with wave.open(str(ruta), "rb") as w:
            duracion_s = w.getnframes() / w.getframerate()
        return ruta.read_bytes(), duracion_s, "xtts-kevin"

    def cerrar(self) -> None:
        if self._proceso is not None:
            self._proceso.terminate()
            self._proceso = None


class SalidaCable:  # pragma: no cover - requiere VB-CABLE
    """Escribe el WAV del TTS a CABLE Input (el audio sintetizado NO vuelve
    al micrófono físico: la ruta de salida es explícitamente el cable)."""

    def __init__(self, rate_cable: int = 48000) -> None:
        self._rate_cable = rate_cable

    def reproducir(self, audio: bytes, duracion_s: float, nombre: str) -> None:
        import io

        import numpy as np
        import pyaudio
        import soundfile as sf

        datos, sr = sf.read(io.BytesIO(audio), dtype="float32")
        mono = np.repeat(np.asarray(datos, dtype=np.float32), self._rate_cable // sr)
        stereo = np.repeat(mono, 2)
        pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        pa = pyaudio.PyAudio()
        indice = next(
            i
            for i in range(pa.get_device_count())
            if "CABLE Input" in str(pa.get_device_info_by_index(i)["name"])
            and pa.get_device_info_by_index(i)["maxOutputChannels"] == 2
        )
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=self._rate_cable,
            output=True,
            output_device_index=indice,
        )
        stream.write(pcm)
        stream.stop_stream()
        stream.close()
        pa.terminate()


class TeleprompterHttp:  # pragma: no cover - requiere el UI corriendo
    """Manda el texto al teleprompter local (FastAPI, POST /api/transcripcion)."""

    def __init__(self, url: str = "http://localhost:8000/api/transcripcion") -> None:
        self._url = url

    def mostrar(self, es: str, en: str) -> None:
        self._post({"es": es, "en": en})

    def parcial(self, es: str) -> None:
        self._post({"es": es, "en": ""})

    def _post(self, payload: dict[str, str]) -> None:
        from contextlib import suppress

        import httpx

        with suppress(Exception):  # teleprompter caído no corta el flujo (ADR-008)
            httpx.post(self._url, json=payload, timeout=2.0)


def validar_arranque_real() -> bool:
    """Validación offline BLOQUEANTE (ADR-014/015): la traducción es→en debe
    funcionar sin red antes de aceptar una llamada (precarga del mwt)."""
    from traductor.flujo.outgoing import validar_arranque
    from traductor.traduccion.argos import traducir

    salud = validar_arranque(lambda es: traducir(es, "es", "en"))
    if not salud.disponible:
        print(f"Arranque BLOQUEADO: {salud.detalle}")
        return False
    print("Arranque offline de la traducción: OK (mwt precargado)")
    return True


def python_venv_tts() -> Path:  # pragma: no cover - ruta de máquina
    """El python del venv del TTS (coqui-tts vive ahí, ADR-011)."""
    raiz = Path(__file__).resolve().parents[2]
    venv_tts = raiz / "venv-tts" / "Scripts" / "python.exe"
    if venv_tts.is_file():
        return venv_tts
    return Path(sys.executable)


def perfil_por_defecto() -> str:
    """Perfil del usuario por defecto (tienda JSON, ADR-013): el id del
    enrolamiento previo o 'kevin' si no hay ninguno."""
    id_ = os.environ.get("TRADUCTOR_PERFIL_ID", "kevin")
    return id_
