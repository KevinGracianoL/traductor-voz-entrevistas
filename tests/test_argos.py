"""Tests para traduccion/argos — validación sin red ni GPU."""

from unittest.mock import MagicMock, patch

import pytest

from traductor.traduccion.argos import instalar_idioma, traducir


def _pkg(en: str = "en", es: str = "es") -> MagicMock:
    m = MagicMock()
    m.from_code = en
    m.to_code = es
    m.download.return_value = "fake_path"  # noqa: S108
    return m


def test_traducir_pasa_por_argos() -> None:
    """traducir delega a argostranslate y fuerza str."""
    with patch(
        "argostranslate.translate.translate",
        return_value="hola",
        create=True,
    ) as mock:
        res = traducir("hello", "en", "es")
        mock.assert_called_once_with("hello", "en", "es")
        assert res == "hola"
        assert isinstance(res, str)


def test_traducir_mutacion_none_no_pasa() -> None:
    """Si argos devuelve None, traducir lo convierte a str (mata translate(None))."""
    with patch("argostranslate.translate.translate", return_value=None, create=True):
        res = traducir("hello", "en", "es")
        assert res == "None"


def test_instalar_idioma_validacion_vacia_lanza() -> None:
    """Traducción vacía -> RuntimeError."""
    fake = _pkg()
    with (
        patch("argostranslate.package.update_package_index", create=True),
        patch(
            "argostranslate.package.get_available_packages",
            return_value=[fake],
            create=True,
        ),
        patch("argostranslate.package.install_from_path", create=True),
        patch("traductor.traduccion.argos.traducir", return_value=""),
        pytest.raises(RuntimeError, match="corrupto"),
    ):
        instalar_idioma("en", "es")


def test_instalar_idioma_validacion_identica_lanza() -> None:
    """Traducción idéntica al input -> RuntimeError."""
    fake = _pkg()
    with (
        patch("argostranslate.package.update_package_index", create=True),
        patch(
            "argostranslate.package.get_available_packages",
            return_value=[fake],
            create=True,
        ),
        patch("argostranslate.package.install_from_path", create=True),
        patch("traductor.traduccion.argos.traducir", return_value="hello"),
        pytest.raises(RuntimeError, match="corrupto"),
    ):
        instalar_idioma("en", "es")


def test_instalar_idioma_validacion_larga_lanza() -> None:
    """Traducción muy larga -> RuntimeError."""
    fake = _pkg()
    with (
        patch("argostranslate.package.update_package_index", create=True),
        patch(
            "argostranslate.package.get_available_packages",
            return_value=[fake],
            create=True,
        ),
        patch("argostranslate.package.install_from_path", create=True),
        patch("traductor.traduccion.argos.traducir", return_value="x" * 50),
        pytest.raises(RuntimeError, match="corrupto"),
    ):
        instalar_idioma("en", "es")


def test_instalar_idioma_ok_no_lanza() -> None:
    """Traducción válida no lanza."""
    fake = _pkg()
    with (
        patch("argostranslate.package.update_package_index", create=True),
        patch(
            "argostranslate.package.get_available_packages",
            return_value=[fake],
            create=True,
        ),
        patch("argostranslate.package.install_from_path", create=True),
        patch("traductor.traduccion.argos.traducir", return_value="hola"),
    ):
        instalar_idioma("en", "es")  # no raise


def test_instalar_idioma_and_no_or() -> None:
    """and → or sobrevive si no hay test con solo un lado igual."""
    solo_origen = _pkg(en="en", es="fr")
    with (
        patch("argostranslate.package.update_package_index", create=True),
        patch(
            "argostranslate.package.get_available_packages",
            return_value=[solo_origen],
            create=True,
        ),
        patch("argostranslate.package.install_from_path", create=True),
        pytest.raises(RuntimeError, match="No hay paquete"),
    ):
        instalar_idioma("en", "es")


def test_instalar_idioma_es_en_usa_hola() -> None:
    """Rama else 'hola' para es->en."""
    fake = _pkg(en="es", es="en")
    with (
        patch("argostranslate.package.update_package_index", create=True),
        patch(
            "argostranslate.package.get_available_packages",
            return_value=[fake],
            create=True,
        ),
        patch("argostranslate.package.install_from_path", create=True),
        patch("traductor.traduccion.argos.traducir", return_value="hello"),
    ):
        instalar_idioma("es", "en")

    with (
        patch("argostranslate.package.update_package_index", create=True),
        patch(
            "argostranslate.package.get_available_packages",
            return_value=[fake],
            create=True,
        ),
        patch("argostranslate.package.install_from_path", create=True),
        patch("traductor.traduccion.argos.traducir", return_value="hola"),
        pytest.raises(RuntimeError, match="corrupto"),
    ):
        instalar_idioma("es", "en")
