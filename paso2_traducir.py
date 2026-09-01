"""Paso 2 — Traducción local bidireccional con argos-translate.

Toma el texto en inglés que suelta Whisper (paso 1) y lo pasa a español,
TODO en local: sin red, sin API, sin costo. En una entrevista no quieres
depender de que el internet aguante justo en el peor momento.

CÓMO USAR ESTE ARCHIVO
----------------------
Igual que el paso 1: está incompleto A PROPÓSITO. Los `TODO` los llenas tú.
En la revisión te voy a preguntar por qué está cada línea.

    python paso2_traducir.py

y me mandas la salida COMPLETA, funcione o truene.

DECISIÓN (ADR-005): argos-translate, no DeepL/Google.
- Offline y gratis: corre en CPU. La GPU ya la ocupa Whisper (paso 1).
- Calidad menor que un servicio de nube, pero suficiente para LEER el sentido
  en vivo. La velocidad y el no-depender-de-red pesan más aquí."""

import os

os.environ.setdefault(
    "ARGOS_COMPUTE_TYPE", "default"
)  # "auto" produce basura en es->en en este hardware — ver diagnóstico

import argostranslate.package
import argostranslate.translate


def instalar_idioma(origen: str, destino: str) -> None:
    """Baja e instala el paquete origen->destino. Idempotente.

    Se llama dos veces: una para (en, es) y otra para (es, en).
    """
    argostranslate.package.update_package_index()
    disponibles = argostranslate.package.get_available_packages()

    # TODO(1): igual que antes, pero comparando contra origen/destino
    #          (los parámetros), no contra unas constantes globales.
    paquete = next((p for p in disponibles if p.from_code == origen and p.to_code == destino), None)

    if paquete is None:
        raise RuntimeError(f"No hay paquete {origen}->{destino}")

    # TODO(2): instálalo (lo mismo que ya sabías hacer).
    argostranslate.package.install_from_path(paquete.download())

    palabra = "hello" if origen == "en" else "hola"
    prueba = traducir(palabra, origen, destino)

    # Un modelo roto devuelve la entrada intacta o basura larga y repetida.
    if not (prueba and prueba != palabra and len(prueba) < 40):
        raise RuntimeError(f"modelo {origen}->{destino} corrupto: {prueba!r}")


def traducir(texto: str, origen: str, destino: str) -> str:
    """origen -> destino."""
    # TODO(3): argostranslate.translate.translate(texto, origen, destino)
    return argostranslate.translate.translate(texto, origen, destino)  # type: ignore[no-any-return]


if __name__ == "__main__":
    for o, d in [("en", "es"), ("es", "en")]:
        instalar_idioma(o, d)
    print(traducir("Tell me about a hard bug you fixed.", "en", "es"))
    print(traducir("Háblame de un bug difícil que resolviste.", "es", "en"))
