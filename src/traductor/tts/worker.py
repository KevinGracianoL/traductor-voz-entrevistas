"""Worker TTS aislado: corre un `TTSBackend` en su propio proceso.

Protocolo: jobs JSON-line por la entrada (`texto`, `perfil_id`, `salida`) y
resultados JSON-line por la salida (`ok`, más `salida`/`elapsed_ms` o `error`).
Un worker NO muere por un job malo: los fallos esperados vuelven como valor.
El motor real se inyecta (ADR-011: sin motor elegido aún); el proceso aislado
permite cargar/descargar el modelo sin rezar con la VRAM de Whisper.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, TextIO, TypedDict

from traductor.latencia.medidor import medir_tiempo
from traductor.tts.backend import TTSBackend
from traductor.tts.tienda import VoiceProfileStore


@dataclass(frozen=True)
class Job:
    """Una petición de síntesis: texto, perfil y ruta de salida."""

    texto: str
    perfil_id: str
    salida: str


class ResultadoOk(TypedDict):
    ok: Literal[True]
    salida: str
    elapsed_ms: float


class ResultadoError(TypedDict):
    ok: Literal[False]
    error: str


Resultado = ResultadoOk | ResultadoError


def procesar_job(
    job: Job,
    backend: TTSBackend,
    tienda: VoiceProfileStore,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> Resultado:
    """Sintetiza `job.texto` con el perfil de `job.perfil_id` y escribe audio.

    Nunca lanza por un fallo esperado (perfil ausente, síntesis que falla):
    devuelve `ResultadoError`. El audio se escribe como bytes.
    """
    perfil = tienda.obtener(job.perfil_id)
    if perfil is None:
        return {"ok": False, "error": f"perfil no encontrado: {job.perfil_id}"}
    try:
        audio, elapsed_ms = medir_tiempo(
            partial(backend.sintetizar, job.texto, perfil),
            clock=clock,
        )
    except Exception as exc:
        return {"ok": False, "error": f"síntesis falló: {exc}"}
    ruta = Path(job.salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(audio.datos)
    return {"ok": True, "salida": job.salida, "elapsed_ms": elapsed_ms}


def _parsear_job(linea: str) -> Job:
    datos = json.loads(linea)
    return Job(
        texto=str(datos["texto"]),
        perfil_id=str(datos["perfil_id"]),
        salida=str(datos["salida"]),
    )


def main(
    backend: TTSBackend,
    tienda: VoiceProfileStore,
    *,
    entrada: TextIO = sys.stdin,
    salida: TextIO = sys.stdout,
) -> None:
    """Lee jobs JSON-line de `entrada` y escribe resultados JSON-line.

    Un job inválido devuelve `{"ok": False, "error": ...}` y el bucle sigue.
    """
    for linea in entrada:
        if not linea.strip():
            continue
        try:
            job = _parsear_job(linea)
        except (ValueError, KeyError, TypeError) as exc:
            salida.write(json.dumps({"ok": False, "error": f"job inválido: {exc}"}) + "\n")
            salida.flush()
            continue
        salida.write(json.dumps(procesar_job(job, backend, tienda)) + "\n")
        salida.flush()
