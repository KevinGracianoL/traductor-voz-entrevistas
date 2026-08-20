import os
import shutil
import sys

import nvidia

dest = os.path.dirname(sys.executable)  # venv\Scripts (dir de la app)
for sub in ("cublas", "cudnn"):
    src = os.path.join(nvidia.__path__[0], sub, "bin")
    for f in os.listdir(src):
        if f.endswith(".dll"):
            shutil.copy2(os.path.join(src, f), dest)  # copy2 = re-copia si cambió la versión
print("DLLs copiadas a", dest)
