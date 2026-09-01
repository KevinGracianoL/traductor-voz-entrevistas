"""Audio virtual en Windows — plomería sin depender de hardware real.

ADR-002: exponer micrófono/altavoz virtuales a nivel SO para funcionar con
cualquier plataforma (Zoom/Teams/Meet) sin API.

Todo lo testeable aquí es puro: clasificar nombres, elegir ruta, validar.
Lo que toca WASAPI/PyAudio queda detrás de callables inyectables.
"""

from __future__ import annotations

from collections.abc import Mapping


def es_virtual(nombre: str) -> bool:
    """True si el nombre indica cable virtual (VB-CABLE, CABLE, Virtual)."""
    lower = nombre.lower()
    return "cable" in lower or "virtual" in lower or "vb-audio" in lower


def clasificar_dispositivos(
    nombres: list[str],
) -> dict[str, list[str]]:
    """Separa lista de nombres en virtuales vs físicos."""
    virtuales = [n for n in nombres if es_virtual(n)]
    fisicos = [n for n in nombres if not es_virtual(n)]
    return {"virtuales": virtuales, "fisicos": fisicos}


def seleccionar_ruta(
    dispositivos: list[str],
    patron_virtual: str = "cable",
) -> dict[str, str | None]:
    """Elige ruta determinista por nombre (ADR-009).

    - Entrevistador (EN->ES): primer dispositivo que matchea patron_virtual (case-insensitive)
    - Usuario (ES->EN): primer físico (no virtual)
    Si no hay virtual/físico, devuelve None para esa clave (caller decide fallback).
    """
    patron = patron_virtual.lower()
    virtual = next((d for d in dispositivos if patron in d.lower()), None)
    # Físico: primer no-virtual
    fisico = next((d for d in dispositivos if not es_virtual(d)), None)
    return {"entrevistador": virtual, "usuario": fisico}


def validar_ruta(ruta: Mapping[str, str | None]) -> None:
    """Valida que al menos un lado tenga dispositivo.

    Raises:
        ValueError: si ambos son None o si entrevistador es None (modo principal).
    """
    if ruta.get("entrevistador") is None and ruta.get("usuario") is None:
        raise ValueError("ruta vacía: no hay dispositivos físicos ni virtuales")
    if ruta.get("entrevistador") is None:
        raise ValueError("sin dispositivo virtual: instala VB-CABLE para entrevistador")


def describir_ruta(ruta: Mapping[str, str | None]) -> str:
    """Descripción humana para logs/teleprompter."""
    ent = ruta.get("entrevistador") or "— (sin virtual)"
    usr = ruta.get("usuario") or "— (sin mic)"
    return f"entrevistador->{ent} | usuario->{usr}"
