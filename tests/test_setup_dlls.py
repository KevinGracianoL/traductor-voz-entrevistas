"""Tests del keeper DLL — solo corren donde setup_dlls.py ya se ejecutó."""

import subprocess
import sys
from pathlib import Path

import pytest

zz = pytest.importorskip("zz_nvidia_dlls")


def test_pth_importa_keeper() -> None:
    import site

    pth = Path(site.getsitepackages()[0]) / "zz_nvidia_dlls.pth"
    if not pth.exists():
        # virtualenv responde otro dir: buscar junto al keeper real
        pth = Path(zz.__file__).with_name("zz_nvidia_dlls.pth")
    assert pth.read_text(encoding="utf-8").strip() == "import zz_nvidia_dlls"


def test_handles_viven_y_permanecen() -> None:
    """Los handles deben seguir abiertos tras el import (si no, las rutas se caen)."""
    assert len(zz._HANDLES) >= 4
    for h in zz._HANDLES:
        assert h is not None
    # Permanencia en proceso fresco: importar dos veces no las cierra
    r = subprocess.run(
        [sys.executable, "-c", "import zz_nvidia_dlls; print(len(zz_nvidia_dlls._HANDLES))"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert int(r.stdout.strip()) >= 4
