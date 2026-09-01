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
        "traductor.traduccion.argos.argostranslate.translate.translate",
        return_value="hola",
    ) as mock:
        res = traducir("hello", "en", "es")
        mock.assert_called_once_with("hello", "en", "es")
        assert res == "hola"
        assert isinstance(res, str)


def test_instalar_idioma_validacion_vacia_lanza() -> None:
    """Traducción vacía -> RuntimeError."""
    fake = _pkg()
    with (
        patch("traductor.traduccion.argos.argostranslate.package.update_package_index"),
        patch(
            "traductor.traduccion.argos.argostranslate.package.get_available_packages",
            return_value=[fake],
        ),
        patch("traductor.traduccion.argos.argostranslate.package.install_from_path"),
        patch("traductor.traduccion.argos.traducir", return_value=""),
        pytest.raises(RuntimeError, match="corrupto"),
    ):
        instalar_idioma("en", "es")


def test_instalar_idioma_validacion_identica_lanza() -> None:
    """Traducción idéntica al input -> RuntimeError."""
    fake = _pkg()
    with (
        patch("traductor.traduccion.argos.argostranslate.package.update_package_index"),
        patch(
            "traductor.traduccion.argos.argostranslate.package.get_available_packages",
            return_value=[fake],
        ),
        patch("traductor.traduccion.argos.argostranslate.package.install_from_path"),
        patch("traductor.traduccion.argos.traducir", return_value="hello"),
        pytest.raises(RuntimeError, match="corrupto"),
    ):
        instalar_idioma("en", "es")


def test_instalar_idioma_validacion_larga_lanza() -> None:
    """Traducción muy larga -> RuntimeError."""
    fake = _pkg()
    with (
        patch("traductor.traduccion.argos.argostranslate.package.update_package_index"),
        patch(
            "traductor.traduccion.argos.argostranslate.package.get_available_packages",
            return_value=[fake],
        ),
        patch("traductor.traduccion.argos.argostranslate.package.install_from_path"),
        patch("traductor.traduccion.argos.traducir", return_value="x" * 50),
        pytest.raises(RuntimeError, match="corrupto"),
    ):
        instalar_idioma("en", "es")


def test_instalar_idioma_ok_no_lanza() -> None:
    """Traducción válida no lanza."""
    fake = _pkg()
    with (
        patch("traductor.traduccion.argos.argostranslate.package.update_package_index"),
        patch(
            "traductor.traduccion.argos.argostranslate.package.get_available_packages",
            return_value=[fake],
        ),
        patch("traductor.traduccion.argos.argostranslate.package.install_from_path"),
        patch("traductor.traduccion.argos.traducir", return_value="hola"),
    ):
        instalar_idioma("en", "es")  # no raise
