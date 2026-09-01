"""Tests para traduccion/argos — validación sin red ni GPU.

argos importa argostranslate de forma perezosa (dentro de las funciones), y CI
NO instala argostranslate (arrastra torch+ctranslate2). Por eso NO usamos
patch("argostranslate...."): mock.patch importa el módulo para resolver el path
y revienta en CI. En su lugar inyectamos un módulo falso en sys.modules, que es
lo que el import perezoso encontrará.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from traductor.traduccion.argos import instalar_idioma, traducir


def _fake_pkg(origen: str = "en", destino: str = "es") -> MagicMock:
    m = MagicMock()
    m.from_code = origen
    m.to_code = destino
    m.download.return_value = "fake_path"  # noqa: S108
    return m


def _inject(traducir_ret: str, paquetes: list[MagicMock]) -> Iterator[MagicMock]:
    """Inyecta un argostranslate falso en sys.modules; yield-ea el mock translate."""
    pkg = types.ModuleType("argostranslate")
    package = types.ModuleType("argostranslate.package")
    translate = types.ModuleType("argostranslate.translate")
    package.update_package_index = MagicMock()  # type: ignore[attr-defined]
    package.install_from_path = MagicMock()  # type: ignore[attr-defined]
    package.get_available_packages = MagicMock(return_value=paquetes)  # type: ignore[attr-defined]
    translate_fn = MagicMock(return_value=traducir_ret)
    translate.translate = translate_fn  # type: ignore[attr-defined]
    pkg.package = package  # type: ignore[attr-defined]
    pkg.translate = translate  # type: ignore[attr-defined]

    claves = ("argostranslate", "argostranslate.package", "argostranslate.translate")
    originales = {k: sys.modules.get(k) for k in claves}
    sys.modules["argostranslate"] = pkg
    sys.modules["argostranslate.package"] = package
    sys.modules["argostranslate.translate"] = translate
    try:
        yield translate_fn
    finally:
        for k, v in originales.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_traducir_pasa_por_argos() -> None:
    """traducir delega a argostranslate, fuerza str, y pasa los args correctos."""
    for translate_fn in _inject("hola", [_fake_pkg()]):
        res = traducir("hello", "en", "es")
        assert res == "hola"
        assert isinstance(res, str)
        translate_fn.assert_called_once_with("hello", "en", "es")


def test_traducir_mutacion_none_no_pasa() -> None:
    """Mata mutmut_2/4: los argumentos deben ser los originales, no None."""
    for translate_fn in _inject("hola", [_fake_pkg()]):
        traducir("hello", "en", "es")
        args, _ = translate_fn.call_args
        assert args == ("hello", "en", "es")


def test_instalar_idioma_ok_no_lanza() -> None:
    """Traducción sana (distinta, corta, no vacía) -> no lanza.

    Verifica también los args internos (mata mutmut_11/22/23): install_from_path
    recibe la ruta descargada, y la prueba se traduce con origen/destino reales.
    """
    pkg = _fake_pkg("es", "en")
    for translate_fn in _inject("hi", [pkg]):
        import argostranslate.package as ap  # noqa: PLC0415

        instalar_idioma("es", "en")
        ap.install_from_path.assert_called_once_with("fake_path")
        translate_fn.assert_called_once_with("hola", "es", "en")


def test_instalar_idioma_frontera_longitud() -> None:
    """Mata mutmut_31/32: el umbral es estricto (< 40), ambos lados.

    len == 39 pasa; len == 40 se considera corrupto y lanza.
    """
    for _ in _inject("x" * 39, [_fake_pkg("es", "en")]):
        instalar_idioma("es", "en")  # 39 < 40 -> ok
    for _ in _inject("x" * 40, [_fake_pkg("es", "en")]):
        with pytest.raises(RuntimeError, match="corrupto"):
            instalar_idioma("es", "en")  # 40 no es < 40 -> lanza


def test_instalar_idioma_es_en_usa_hola() -> None:
    """Mata mutmut_18: la rama else usa 'hola' como palabra de prueba (es->en)."""
    for translate_fn in _inject("hi", [_fake_pkg("es", "en")]):
        instalar_idioma("es", "en")
        # la prueba interna traduce 'hola' (rama else), no 'hello'
        assert translate_fn.call_args[0][0] == "hola"


def test_instalar_idioma_and_no_or() -> None:
    """Mata mutmut_6: match de paquete es AND, no OR.

    Un paquete que sólo coincide en from_code no debe ser aceptado.
    """
    solo_from = _fake_pkg("en", "de")  # from_code coincide, to_code no
    for _ in _inject("hallo", [solo_from]):
        with pytest.raises(RuntimeError, match="No hay paquete"):
            instalar_idioma("en", "es")


def test_instalar_idioma_sin_paquete_lanza() -> None:
    """No hay paquete origen->destino -> RuntimeError."""
    for _ in _inject("hola", []):
        with pytest.raises(RuntimeError, match="No hay paquete"):
            instalar_idioma("en", "es")


def test_instalar_idioma_validacion_vacia_lanza() -> None:
    """Traducción vacía -> RuntimeError corrupto."""
    for _ in _inject("", [_fake_pkg()]):
        with pytest.raises(RuntimeError, match="corrupto"):
            instalar_idioma("en", "es")


def test_instalar_idioma_validacion_identica_lanza() -> None:
    """Traducción idéntica al input -> RuntimeError corrupto."""
    for _ in _inject("hello", [_fake_pkg()]):
        with pytest.raises(RuntimeError, match="corrupto"):
            instalar_idioma("en", "es")


def test_instalar_idioma_validacion_larga_lanza() -> None:
    """Traducción de 40+ chars -> RuntimeError corrupto."""
    for _ in _inject("x" * 45, [_fake_pkg()]):
        with pytest.raises(RuntimeError, match="corrupto"):
            instalar_idioma("en", "es")
