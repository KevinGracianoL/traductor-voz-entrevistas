"""Traducción local bidireccional con argos-translate (ADR-005).

Offline, CPU, sin red. Extraído de paso2_traducir.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("ARGOS_COMPUTE_TYPE", "default")

import argostranslate.package
import argostranslate.translate


def instalar_idioma(origen: str, destino: str) -> None:
    """Baja e instala el paquete origen->destino. Idempotente."""
    argostranslate.package.update_package_index()
    disponibles = argostranslate.package.get_available_packages()

    paquete = next(
        (p for p in disponibles if p.from_code == origen and p.to_code == destino),
        None,
    )

    if paquete is None:
        raise RuntimeError(f"No hay paquete {origen}->{destino}")

    argostranslate.package.install_from_path(paquete.download())

    palabra = "hello" if origen == "en" else "hola"
    prueba = traducir(palabra, origen, destino)

    if not (prueba and prueba != palabra and len(prueba) < 40):
        raise RuntimeError(f"modelo {origen}->{destino} corrupto: {prueba!r}")


def traducir(texto: str, origen: str, destino: str) -> str:
    """Traduce origen -> destino."""
    return str(argostranslate.translate.translate(texto, origen, destino))
