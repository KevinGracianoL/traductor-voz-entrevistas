"""Copia DLLs CUDA al venv y registra su búsqueda vía .pth.

Copiar a venv\\Scripts ya no alcanza: con torch 2.13 (CUDA 13) el dir del
exe no entra al search path de este Python y ctranslate2 no encuentra
cublas64_12. El .pth añade los dirs con add_dll_directory en cada arranque.
"""

import os
import shutil
import sys

import nvidia

dest = os.path.dirname(sys.executable)  # venv\Scripts (dir de la app)
site_packages = os.path.dirname(nvidia.__path__[0])
ct2_dir = os.path.join(site_packages, "ctranslate2")
bins = [dest]
for sub in ("cublas", "cudnn", "cuda_runtime"):
    src = os.path.join(nvidia.__path__[0], sub, "bin")
    bins.append(src)
    for f in os.listdir(src):
        if f.endswith(".dll"):
            shutil.copy2(os.path.join(src, f), dest)  # copy2 = re-copia si cambió la versión
# Solo estos 3 van junto a ctranslate2.dll (su loader los busca ahí).
# NO copiar el set cudnn: envenena la resolución de torch (WinError 127).
solo_ct2 = {
    "cublas64_12.dll": "cublas",
    "cublasLt64_12.dll": "cublas",
    "cudart64_12.dll": "cuda_runtime",
}
for f, sub in solo_ct2.items():
    shutil.copy2(os.path.join(nvidia.__path__[0], sub, "bin", f), ct2_dir)
print("DLLs copiadas a", dest, "y", ct2_dir)

# site-packages real (site.getsitepackages miente en virtualenv: devuelve venv\)
site_packages = os.path.dirname(nvidia.__path__[0])
keeper = os.path.join(site_packages, "zz_nvidia_dlls.py")
with open(keeper, "w", encoding="utf-8") as fh:
    fh.write(
        "# Registra dirs CUDA. Los handles VIVEN en _HANDLES: si se descartan,\n"
        "# CPython puede cerrar las rutas y los imports posteriores fallan.\n"
        "# Generado por setup_dlls.py, no editar a mano.\n"
        "import os\n"
        "_DIRS = [\n" + "".join(f"    r'{b}',\n" for b in bins) + "]\n"
        "_HANDLES = [os.add_dll_directory(p) for p in _DIRS]\n"
    )
pth = os.path.join(site_packages, "zz_nvidia_dlls.pth")
with open(pth, "w", encoding="utf-8") as fh:
    fh.write("import zz_nvidia_dlls\n")
print("keeper+pth en", keeper, "y", pth)
