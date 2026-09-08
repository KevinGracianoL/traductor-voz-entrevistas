"""Verificación de hardware — CUDA y VRAM.

torch se importa lazy (CI no lo instala): las funciones de hardware quedan
`# pragma: no cover` y el cálculo puro se inyecta y se prueba sin GPU.
"""

from __future__ import annotations

from typing import Any


def _torch_cuda() -> Any:  # pragma: no cover
    """torch.cuda, importado lazy: solo existe en la máquina objetivo."""
    import torch

    return torch.cuda


def verificar_gpu() -> bool:  # pragma: no cover
    """Comprueba que PyTorch tenga CUDA y reporta la GPU.

    Devuelve True solo si podemos seguir. Imprime diagnóstico para el usuario.
    """
    cuda = _torch_cuda()
    disponible = cuda.is_available()
    print(f"CUDA disponible: {disponible}")

    if not disponible:
        print("\n  Sin CUDA no seguimos. Causa más probable:")
        print("  instalaste torch sin la variante de GPU.")
        print("  Revisa pytorch.org y reinstala con el --index-url correcto.")
        return False

    print(f"GPU: {cuda.get_device_name(0)}")

    total_bytes = cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / 1024**3
    print(f"VRAM total: {total_gb:.2f} GB")

    libre_bytes, _ = cuda.mem_get_info()
    libre_gb = libre_bytes / 1024**3
    print(f"VRAM libre: {libre_gb:.2f} GB")
    return True


def vram_ocupada_mib(cuda: Any) -> float | None:
    """VRAM total en uso (MiB) a nivel de driver, para `cuda` inyectado.

    Usa `mem_get_info` (total del dispositivo menos libre): incluye lo que
    reserva CTranslate2/faster-whisper, que queda FUERA del allocator de
    torch. None si no hay CUDA. Cierra el gate "VRAM co-residente" (ADR-014).
    `cuda` se inyecta para que el cálculo sea testeable sin GPU.
    """
    if not cuda.is_available():
        return None
    total_bytes = cuda.get_device_properties(0).total_memory
    libre_bytes, _ = cuda.mem_get_info()
    return float((total_bytes - libre_bytes) / (1024 * 1024))
