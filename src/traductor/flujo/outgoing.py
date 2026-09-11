"""Flujo outgoing_es_to_en (ADR-015): mic → VAD → ASR es → Argos → teleprompter → TTS → VB-CABLE.

El core es PURO y testeable: las etapas se inyectan (traducción, TTS primario
y de fallback, teleprompter, salida de audio) y los adaptadores reales
(micrófono, worker TTS, VB-CABLE, teleprompter HTTP) viven en
`traductor.flujo.adaptadores` y se cablean en `scripts/flujo_outgoing.py`.

Reglas del ADR-015 implementadas aquí:
- los PARCIALES solo llegan a pantalla (nunca a traducción/TTS);
- cola de tamaño 1: una respuesta atrasada se descarta;
- cancelación: una solicitud nueva cancela la antigua (el audio de un turno
  superado nunca se enruta);
- timestamps por etapa con cierre exacto (`RegistroEtapas`);
- escalera de presupuesto: clonado → voz genérica → solo subtítulos (nunca
  reproducir audio sospechoso).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from traductor.tts.harness import RegistroEtapas
from traductor.tts.modelos import Salud

NIVEL_CLONADO = 2
NIVEL_VOZ_GENERICA = 3
NIVEL_SUBTITULOS = 4

_TEXTO_PROBE_ARRANQUE = "Mi experiencia mas fuerte es con sistemas distribuidos."
DETALLE_BASURA = (
    "traducción es→en produce basura al arrancar: precarga del modelo "
    "spacy mwt no efectiva (el worker de traducción NO entra al flujo)"
)


class Teleprompter(Protocol):
    def mostrar(self, es: str, en: str) -> None: ...
    def parcial(self, es: str) -> None: ...


class SalidaAudio(Protocol):
    def reproducir(self, audio: bytes, duracion_s: float, nombre: str) -> None: ...


class EtapaTts(Protocol):
    """Etapa de síntesis. None = el backend falló (escalera).

    `audio` es WAV; `nombre` identifica la voz (para logs y la escalera).
    """

    def sintetizar(self, texto_en: str) -> tuple[bytes, float, str] | None: ...


@dataclass
class FlujoOutgoing:
    """Orquesta un turno: segmento final → traducción → TTS (escalera) → ruteo.

    Contrato de hilos (corregido en la revisión del PR #22): RealtimeSTT
    entrega parciales y finales desde SUS propios hilos, y cada `segmento_final`
    corre en un hilo worker del adaptador. Por eso el core es seguro bajo
    concurrencia:
    - el contador de turnos y el turno activo se mutan bajo `_lock`;
    - la cancelación (`cancelar_turno_activo`, invocada por el callback de
      parciales cuando el usuario vuelve a hablar) y el arranque de un turno
      nuevo invalidan el turno en vuelo: el check `turno != self._turno_activo`
      (lectura atómica) descarta su audio y su texto según el momento;
    - la ventana de carrera restante (cancelar justo entre el check y el ruteo)
      es la duración de una escritura al cable: se documenta, no se promete
      cero.
    """

    traducir: Callable[[str], str]
    tts_primario: EtapaTts
    teleprompter: Teleprompter
    salida_audio: SalidaAudio
    tts_fallback: EtapaTts | None = None
    reloj: Callable[[], float] = field(default_factory=lambda: __import__("time").perf_counter)
    ultimo_turno_etapas: dict[str, float] = field(default_factory=dict, init=False)
    ultimo_turno_total_ms: float = field(default=0.0, init=False)
    _numero_turno: int = field(default=0, init=False)
    _turno_activo: int | None = field(default=None, init=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def parcial(self, texto_es: str) -> None:
        """Parcial del ASR: SOLO a pantalla (regla del ADR-015)."""
        self.teleprompter.parcial(texto_es)

    def cancelar_turno_activo(self) -> None:
        """Cancela el turno en curso. Lo invoca el callback de PARCIALES del
        micrófono cuando el usuario vuelve a hablar: la síntesis del turno
        anterior queda invalidada y su audio no se enruta."""
        with self._lock:
            self._turno_activo = None

    def segmento_final(self, texto_es: str) -> int | None:
        """Procesa un segmento final; devuelve el nivel de la escalera usado
        (None si el turno fue cancelado antes de enrutar)."""
        with self._lock:
            self._numero_turno += 1
            turno = self._numero_turno
            self._turno_activo = turno
        etapas = RegistroEtapas(clock=self.reloj)
        etapas.marcar("entrada")
        texto_en = self.traducir(texto_es)
        etapas.marcar("traduccion")
        if turno != self._turno_activo:
            return None  # cancelado: el texto de un turno superado no se muestra
        self.teleprompter.mostrar(texto_es, texto_en)
        escalera: list[tuple[str, EtapaTts]] = [("clonado", self.tts_primario)]
        if self.tts_fallback is not None:
            escalera.append(("generica", self.tts_fallback))
        for nombre, tts in escalera:
            salida = tts.sintetizar(texto_en)
            if salida is None:
                continue  # escalera: siguiente nivel
            audio, duracion_s, _nombre = salida
            if turno != self._turno_activo:
                return None  # cancelado: el audio de un turno superado no se enruta
            etapas.marcar("tts")
            self.salida_audio.reproducir(audio, duracion_s, nombre)
            etapas.marcar("ruteo")
            self.ultimo_turno_etapas = etapas.desglose_ms()
            self.ultimo_turno_total_ms = etapas.total_ms()
            return NIVEL_CLONADO if nombre == "clonado" else NIVEL_VOZ_GENERICA
        self.ultimo_turno_etapas = etapas.desglose_ms()
        self.ultimo_turno_total_ms = etapas.total_ms()
        return NIVEL_SUBTITULOS  # ambos TTS fallaron: solo subtítulos, sin audio


def _normalizar(texto: str) -> str:
    return " ".join("".join(c for c in texto.lower() if c.isalnum() or c.isspace()).split())


def validar_arranque(
    traducir: Callable[[str], str],
    *,
    probe: str = _TEXTO_PROBE_ARRANQUE,
) -> Salud:
    """Valida el arranque del flujo de forma BLOQUEANTE (ADR-014/015).

    La traducción es→en debe funcionar SIN red antes de aceptar una llamada:
    la precarga del modelo spacy `mwt` (que argos descarga on-demand y mató 1
    de 20 corridas del harness) se comprueba traduciendo una frase de prueba y
    verificando que la salida sea sana (mismo sanity laxo del harness: frase
    clave presente, longitud acotada, sin repetición patológica).
    """
    try:
        salida = _normalizar(traducir(probe))
    except Exception as exc:
        return Salud(disponible=False, detalle=f"traducción no disponible al arrancar: {exc}")
    palabras = salida.split()
    if not palabras or "distributed systems" not in salida or len(salida) > 80:
        return Salud(disponible=False, detalle=DETALLE_BASURA)
    if max(palabras.count(p) for p in set(palabras)) > 3:
        return Salud(disponible=False, detalle=DETALLE_BASURA)
    return Salud(disponible=True)
