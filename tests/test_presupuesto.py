"""Tests para el módulo de presupuesto de latencia.

Estos tests verifican la LÓGICA DE DECISIÓN, no el hardware.
Corren en milisegundos, sin GPU, sin audio.
"""

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


def test_margen_restante_positivo_cuando_cabe() -> None:
    """1000 - 950 = 50 ms de margen."""
    assert margen_restante(ETAPAS_BASE, TECHO) == 50.0


def test_margen_restante_negativo_cuando_no_cabe() -> None:
    """1000 - 1100 = -100 ms (se pasa)."""
    etapas = {"asr": 600.0, "traduccion": 200.0, "tts": 300.0}
    assert margen_restante(etapas, TECHO) == -100.0


def test_degradar_configuracion_reduce_hasta_caber() -> None:
    """Degrada TTS primero (300 -> 240), suma baja a 890 <= 1000."""
    etapas = {"asr": 500.0, "traduccion": 150.0, "tts": 300.0}
    techo = 800.0  # 950 no cabe, 890 sí
    resultado = degradar_configuracion(etapas, techo, ["tts", "traduccion", "asr"])
    assert cabe_en_presupuesto(resultado, techo) is True
    # Original no debe modificarse
    assert etapas["tts"] == 300.0


def test_degradar_configuracion_no_modifica_original() -> None:
    """El dict original queda intacto."""
    etapas = {"asr": 500.0, "traduccion": 150.0, "tts": 300.0}
    original_tts = etapas["tts"]
    degradar_configuracion(etapas, 100.0, ["tts"])  # Techo imposible
    assert etapas["tts"] == original_tts
