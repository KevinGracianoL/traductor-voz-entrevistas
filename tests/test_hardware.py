"""Tests de traductor.hardware.cuda — vram_ocupada_mib con cuda inyectado.

El CI no instala torch: el cálculo se prueba con un fake de `torch.cuda`.
Un bug de unidades ahí (p. ej. 1024**3 en vez de 1024**2) pasaría el gate de
VRAM por ~1000x sin que nada lo note — este test fija la unidad (MiB).
"""

import pytest

from traductor.hardware.cuda import vram_ocupada_mib


class _CudaFake:
    def __init__(self, disponible: bool) -> None:
        self._disponible = disponible

    def is_available(self) -> bool:
        return self._disponible

    def mem_get_info(self) -> tuple[int, int]:
        return (1 * 1024**3, 4 * 1024**3)


def test_vram_sin_cuda_none() -> None:
    assert vram_ocupada_mib(_CudaFake(disponible=False)) is None


def test_vram_total_menos_libre_en_mib() -> None:
    """4 GiB total − 1 GiB libre = 3 GiB = 3072 MiB (no ~3, no 3*1024**2*...)."""
    assert vram_ocupada_mib(_CudaFake(disponible=True)) == pytest.approx(3 * 1024)
