"""Tests de setup_dlls — corren siempre (tmp_path, sin venv real)."""

import runpy
from pathlib import Path

from setup_dlls import dirs_cuda, escribir_keeper, instalar, plan_dlls


def test_dirs_cuda_ordena_destino_primero(tmp_path: Path) -> None:
    import os

    bins = dirs_cuda("/nvidia", str(tmp_path))
    assert bins[0] == str(tmp_path)
    esperados = [os.path.join("/nvidia", sub, "bin") for sub in ("cublas", "cudnn", "cuda_runtime")]
    assert bins[1:] == esperados


def test_plan_dlls_destino_y_solo_3_a_ct2(tmp_path: Path) -> None:
    archivos = {"cublas": ["a.dll", "x.txt"], "cudnn": ["b.dll"], "cuda_runtime": ["c.dll"]}
    plan = plan_dlls("/nvidia", str(tmp_path), str(tmp_path / "ct2"), archivos)
    a_dest = sorted(Path(src).name for src, dst in plan if dst == str(tmp_path))
    assert a_dest == ["a.dll", "b.dll", "c.dll"]
    a_ct2 = sorted(Path(src).name for src, dst in plan if dst == str(tmp_path / "ct2"))
    assert a_ct2 == ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]


def test_escribir_keeper_contenido(tmp_path: Path) -> None:
    keeper, pth = escribir_keeper(str(tmp_path), ["D1", "D2"])
    texto = Path(keeper).read_text(encoding="utf-8")
    compile(texto, keeper, "exec")
    assert "_HANDLES = [os.add_dll_directory(p) for p in _DIRS]" in texto
    assert "r'D1'" in texto and "r'D2'" in texto
    assert Path(pth).read_text(encoding="utf-8").strip() == "import zz_nvidia_dlls"


def test_keeper_ejecuta_y_conserva_handles(tmp_path: Path) -> None:
    """El keeper generado registra cada dir y conserva los handles."""
    sub = tmp_path / "bin"
    sub.mkdir()
    keeper, _ = escribir_keeper(str(tmp_path), [str(sub)])
    ns = runpy.run_path(str(keeper))
    assert "_HANDLES" in ns
    assert len(ns["_HANDLES"]) == 1


def test_instalar_copia_y_registra_en_arbol_falso(tmp_path: Path) -> None:
    """instalar() end-to-end contra tmp: copias reales + keeper ejecutable."""
    nvidia = tmp_path / "nvidia"
    for sub, files in {
        "cublas": ["a.dll", "cublas64_12.dll", "cublasLt64_12.dll", "x.txt"],
        "cudnn": ["b.dll"],
        "cuda_runtime": ["c.dll", "cudart64_12.dll"],
    }.items():
        bindir = nvidia / sub / "bin"
        bindir.mkdir(parents=True)
        for f in files:
            (bindir / f).write_bytes(b"fake-dll")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    keeper, pth = instalar(str(nvidia), str(dest), str(site_packages))
    assert (dest / "a.dll").exists()
    assert not (dest / "x.txt").exists()
    ct2 = site_packages / "ctranslate2"
    assert (ct2 / "cublas64_12.dll").exists()
    assert not (ct2 / "b.dll").exists()
    assert Path(pth).read_text(encoding="utf-8").strip() == "import zz_nvidia_dlls"
    assert Path(keeper).exists()
