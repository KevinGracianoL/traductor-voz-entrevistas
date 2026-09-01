"""Tests para audio virtual — pura clasificación y selección."""

import pytest

from audio.virtual import (
    clasificar_dispositivos,
    describir_ruta,
    es_virtual,
    seleccionar_ruta,
    validar_ruta,
)


def test_es_virtual_detecta_cable() -> None:
    assert es_virtual("CABLE Input (VB-Audio Virtual Cable)") is True
    assert es_virtual("Virtual Mic") is True
    assert es_virtual("Microfono Realtek") is False
    assert es_virtual("cable output") is True


def test_es_virtual_detecta_voicemeeter_por_vb_audio() -> None:
    """VoiceMeeter no dice 'cable' ni 'virtual': sólo lo salva la cláusula vb-audio."""
    assert es_virtual("VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)") is True


def test_clasificar_separa() -> None:
    nombres = ["CABLE Input", "Realtek Mic", "Virtual Speaker", "Auriculares"]
    res = clasificar_dispositivos(nombres)
    assert res["virtuales"] == ["CABLE Input", "Virtual Speaker"]
    assert res["fisicos"] == ["Realtek Mic", "Auriculares"]


def test_clasificar_vacio() -> None:
    assert clasificar_dispositivos([]) == {"virtuales": [], "fisicos": []}


def test_seleccionar_ruta_virtual_y_fisico() -> None:
    disps = ["Realtek Mic", "CABLE Output", "Auriculares"]
    ruta = seleccionar_ruta(disps, patron_virtual="cable")
    assert ruta["entrevistador"] == "CABLE Output"
    assert ruta["usuario"] == "Realtek Mic"


def test_seleccionar_ruta_sin_virtual() -> None:
    disps = ["Realtek Mic", "Auriculares"]
    ruta = seleccionar_ruta(disps)
    assert ruta["entrevistador"] is None
    assert ruta["usuario"] == "Realtek Mic"


def test_seleccionar_ruta_sin_fisico() -> None:
    disps = ["CABLE Input", "CABLE Output"]
    ruta = seleccionar_ruta(disps)
    assert ruta["entrevistador"] == "CABLE Input"
    assert ruta["usuario"] is None


def test_seleccionar_ruta_vacia() -> None:
    ruta = seleccionar_ruta([])
    assert ruta["entrevistador"] is None
    assert ruta["usuario"] is None


def test_validar_ruta_ok() -> None:
    validar_ruta({"entrevistador": "CABLE", "usuario": "Mic"})


def test_validar_ruta_sin_virtual_raise() -> None:
    with pytest.raises(ValueError, match="sin dispositivo virtual"):
        validar_ruta({"entrevistador": None, "usuario": "Mic"})


def test_validar_ruta_vacia_raise() -> None:
    with pytest.raises(ValueError, match="ruta vacía"):
        validar_ruta({"entrevistador": None, "usuario": None})


def test_validar_ruta_virtual_sin_mic_no_lanza() -> None:
    """Virtual presente y mic ausente es válido: el modo principal funciona."""
    validar_ruta({"entrevistador": "CABLE Output", "usuario": None})


def test_describir_ruta_formato() -> None:
    ruta = {"entrevistador": "CABLE", "usuario": "Mic"}
    assert describir_ruta(ruta) == "entrevistador->CABLE | usuario->Mic"
    ruta2 = {"entrevistador": None, "usuario": None}
    assert "sin virtual" in describir_ruta(ruta2)
    assert "sin mic" in describir_ruta(ruta2)
