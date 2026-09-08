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
        for f in archivos_por_sub[sub]:
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
    """Escribe keeper .py + .pth en UTF-8 explícito (hay usernames no-ASCII)."""
    keeper = os.path.join(site_packages, "zz_nvidia_dlls.py")
    rendered_dirs = "".join(f"    {b!r},\n" for b in bins)
    with open(keeper, "wb") as fh:
        fh.write(KEEPER_TXT.format(dirs=rendered_dirs).encode())
    pth = os.path.join(site_packages, "zz_nvidia_dlls.pth")
    with open(pth, "wb") as fh:
        fh.write(b"import zz_nvidia_dlls\n")
    return keeper, pth


def dir_ct2() -> str:
    """Dir del paquete ctranslate2 real. Falla claro si no está instalado."""
    import ctranslate2

    archivo: str | None = ctranslate2.__file__
    if archivo is None:
        raise RuntimeError("ctranslate2 sin __file__: instalación rota")
    return os.path.dirname(archivo)


def instalar(
    nvidia_path: str, dest: str, site_packages: str, ct2_dir: str | None = None
) -> tuple[str, str]:
    """Pipeline completo contra dirs dados (testeable con árbol falso)."""
    destino_ct2 = ct2_dir if ct2_dir is not None else dir_ct2()
    copiar_dlls(nvidia_path, dest, destino_ct2)
    return escribir_keeper(site_packages, dirs_cuda(nvidia_path, dest))


def main(
    nvidia_path: str | None = None,
    dest: str | None = None,
    site_packages: str | None = None,
) -> None:
    import nvidia

    real_nvidia = nvidia_path if nvidia_path is not None else nvidia.__path__[0]
    real_dest = dest if dest is not None else os.path.dirname(sys.executable)
    # site-packages real (site.getsitepackages miente en virtualenv: devuelve venv\)
    real_sp = site_packages if site_packages is not None else os.path.dirname(nvidia.__path__[0])
    keeper, pth = instalar(real_nvidia, real_dest, real_sp)
    print(f"keeper+pth en {keeper} y {pth}")


__all__ = ["dirs_cuda", "plan_dlls", "copiar_dlls", "escribir_keeper", "instalar", "main"]


if __name__ == "__main__":
    main()
