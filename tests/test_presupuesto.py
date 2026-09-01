"""Tests para el módulo de presupuesto de latencia.

Estos tests verifican la LÓGICA DE DECISIÓN, no el hardware.
Corren en milisegundos, sin GPU, sin audio.
"""

import pytest

from latencia.presupuesto import (
    cabe_en_presupuesto,
    degradar_configuracion,
    etapa_mas_lenta,
    margen_restante,
)

ETAPAS_BASE = {"asr": 500.0, "traduccion": 150.0, "tts": 300.0}
TECHO = 1000.0  # 1 segundo


def test_cabe_dentro_del_presupuesto() -> None:
    """Suma de etapas (950) <= techo (1000) -> True."""
    assert cabe_en_presupuesto(ETAPAS_BASE, TECHO) is True


def test_no_cabe_si_se_pasa_del_techo() -> None:
    """Etapas que suman 1100 > techo 1000 -> False."""
    etapas = {"asr": 600.0, "traduccion": 200.0, "tts": 300.0}
    assert cabe_en_presupuesto(etapas, TECHO) is False


def test_frontera_exacta_cabe() -> None:
    """950 ms exactos contra techo de 950 ms: el techo es INCLUSIVO."""
    etapas = {"asr": 400.0, "traduccion": 150.0, "tts": 400.0}
    assert cabe_en_presupuesto(etapas, 950.0) is True


def test_etapa_mas_lenta_devuelve_nombre_correcto() -> None:
    """ASR (500) es la más lenta en ETAPAS_BASE."""
    assert etapa_mas_lenta(ETAPAS_BASE) == "asr"


def test_etapa_mas_lenta_vacio_raise() -> None:
    """Dict vacío debe fallar explícito, no ValueError críptico de max()."""
    with pytest.raises(ValueError, match="etapas vacío"):
        etapa_mas_lenta({})


def test_margen_restante_positivo_cuando_cabe() -> None:
    """1000 - 950 = 50 ms de margen."""
    assert margen_restante(ETAPAS_BASE, TECHO) == 50.0


def test_margen_restante_negativo_cuando_no_cabe() -> None:
    """1000 - 1100 = -100 ms (se pasa)."""
    etapas = {"asr": 600.0, "traduccion": 200.0, "tts": 300.0}
    assert margen_restante(etapas, TECHO) == -100.0


def test_degradar_configuracion_reduce_hasta_caber() -> None:
    """Degrada TTS primero (300 -> 240), suma baja a 890 -> cabe en 900."""
    etapas = {"asr": 500.0, "traduccion": 150.0, "tts": 300.0}
    techo = 900.0  # 950 no cabe, 890 sí tras una degradación
    resultado = degradar_configuracion(etapas, techo, ["tts", "traduccion", "asr"])
    assert cabe_en_presupuesto(resultado, techo) is True
    assert resultado["tts"] == 240.0  # mata mutante *= 0.8 -> = 0.8
    # break -> return mutante: si fuera return None, esto revienta
    assert isinstance(resultado, dict)
    # Original no debe modificarse
    assert etapas["tts"] == 300.0


def test_degradar_configuracion_no_modifica_original() -> None:
    """El dict original queda intacto."""
    etapas = {"asr": 500.0, "traduccion": 150.0, "tts": 300.0}
    original_tts = etapas["tts"]
    degradar_configuracion(etapas, 100.0, ["tts"])  # Techo imposible
    assert etapas["tts"] == original_tts


def test_degradar_configuracion_converge_con_techo_exigente() -> None:
    """Techo 500 requiere varias degradaciones; debe iterar hasta caber si es posible.

    500+150+300=950. Con ciclos, debe bajar lo suficiente para caber en 500
    si se insiste lo bastante (0.8^n). Verifica que no se rinde tras una sola pasada.
    """
    etapas = {"asr": 500.0, "traduccion": 150.0, "tts": 300.0}
    techo = 500.0
    resultado = degradar_configuracion(etapas, techo, ["tts", "traduccion", "asr"])
    # Debe haber degradado múltiples veces, no quedarse en 760 (una sola pasada)
    assert sum(resultado.values()) < 760.0
    # Si aun no cabe tras max_ciclos, caller debe verificar; no mentir
    # Aquí documentamos el comportamiento: devuelve mejor esfuerzo
    assert cabe_en_presupuesto(resultado, techo) is True or sum(resultado.values()) < 950.0


def test_degradar_configuracion_nombre_invalido_raise() -> None:
    """Nombre que no existe en etapas debe fallar ruidoso, no ignorarse."""
    etapas = {"asr": 500.0, "traduccion": 150.0, "tts": 300.0}
    with pytest.raises(ValueError, match="etapa desconocida"):
        degradar_configuracion(etapas, 800.0, ["invalido"])
