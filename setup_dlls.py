"""Copia DLLs CUDA al venv y registra su búsqueda vía .pth.

Copiar a venv\\Scripts ya no alcanza: con torch 2.13 (CUDA 13) el dir del
exe no entra al search path de este Python y ctranslate2 no encuentra
cublas64_12. El .pth añade los dirs con add_dll_directory en cada arranque.
"""

from __future__ import annotations

import os
import shutil
import sys

SUBS = ("cublas", "cudnn", "cuda_runtime")

# Solo estos 3 van junto a ctranslate2.dll (su loader los busca ahí).
# NO copiar el set cudnn: envenena la resolución de torch (WinError 127).
SOLO_CT2 = {
    "cublas64_12.dll": "cublas",
    "cublasLt64_12.dll": "cublas",
    "cudart64_12.dll": "cuda_runtime",
}

KEEPER_TXT = """# Registra dirs CUDA. Los handles VIVEN en _HANDLES: si se descartan,
# CPython puede cerrar las rutas y los imports posteriores fallan.
# Generado por setup_dlls.py, no editar a mano.
import os
_DIRS = [
{dirs}]
_HANDLES = [os.add_dll_directory(p) for p in _DIRS]
"""


def dirs_cuda(nvidia_path: str, dest: str) -> list[str]:
    """Dirs a registrar: destino + bins de nvidia."""
    return [dest] + [os.path.join(nvidia_path, sub, "bin") for sub in SUBS]


def plan_dlls(
    nvidia_path: str, dest: str, ct2_dir: str, archivos_por_sub: dict[str, list[str]]
) -> list[tuple[str, str]]:
    """Plan puro [(origen, destino)]: todo .dll a dest + las 3 de ctranslate2."""
    plan: list[tuple[str, str]] = []
    for sub in SUBS:
        src = os.path.join(nvidia_path, sub, "bin")
        for f in archivos_por_sub.get(sub, []):
            if f.endswith(".dll"):
                plan.append((os.path.join(src, f), dest))
    for f, sub in SOLO_CT2.items():
        plan.append((os.path.join(nvidia_path, sub, "bin", f), ct2_dir))
    return plan


def copiar_dlls(nvidia_path: str, dest: str, ct2_dir: str) -> None:
    """Ejecuta el plan contra el FS real."""
    archivos = {sub: os.listdir(os.path.join(nvidia_path, sub, "bin")) for sub in SUBS}
    for src, dst in plan_dlls(nvidia_path, dest, ct2_dir, archivos):
        shutil.copy2(src, dst)


def escribir_keeper(site_packages: str, bins: list[str]) -> tuple[str, str]:
    """Escribe keeper .py + .pth. Devuelve ambas rutas."""
    keeper = os.path.join(site_packages, "zz_nvidia_dlls.py")
    rendered_dirs = "".join(f"    r'{b}',\n" for b in bins)
    with open(keeper, "w", encoding="utf-8") as fh:
        fh.write(KEEPER_TXT.format(dirs=rendered_dirs))
    pth = os.path.join(site_packages, "zz_nvidia_dlls.pth")
    with open(pth, "w", encoding="utf-8") as fh:
        fh.write("import zz_nvidia_dlls\n")
    return keeper, pth


def instalar(nvidia_path: str, dest: str, site_packages: str) -> tuple[str, str]:
    """Pipeline completo contra dirs dados (testeable con árbol falso)."""
    ct2_dir = os.path.join(site_packages, "ctranslate2")
    os.makedirs(ct2_dir, exist_ok=True)
    copiar_dlls(nvidia_path, dest, ct2_dir)
    print("DLLs copiadas a", dest, "y", ct2_dir)
    keeper, pth = escribir_keeper(site_packages, dirs_cuda(nvidia_path, dest))
    print("keeper+pth en", keeper, "y", pth)
    return keeper, pth


def main() -> None:
    import nvidia

    dest = os.path.dirname(sys.executable)  # venv\Scripts (dir de la app)
    # site-packages real (site.getsitepackages miente en virtualenv: devuelve venv\)
    site_packages = os.path.dirname(nvidia.__path__[0])
    instalar(nvidia.__path__[0], dest, site_packages)


__all__ = ["dirs_cuda", "plan_dlls", "copiar_dlls", "escribir_keeper", "instalar", "main"]


if __name__ == "__main__":
    main()
