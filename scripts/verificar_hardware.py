"""Verifica hardware — wrapper fino para src/traductor/hardware/cuda.py."""

from traductor.audio.captura import probar_microfono
from traductor.hardware.cuda import verificar_gpu

if __name__ == "__main__":
    if verificar_gpu():
        probar_microfono()
