"""Verificación de hardware — CUDA y VRAM."""

from __future__ import annotations

import torch


def verificar_gpu() -> bool:  # pragma: no cover
    """Comprueba que PyTorch tenga CUDA y reporta la GPU.

    Devuelve True solo si podemos seguir. Imprime diagnóstico para el usuario.
    """
    disponible = torch.cuda.is_available()
    print(f"CUDA disponible: {disponible}")

    if not disponible:
        print("\n  Sin CUDA no seguimos. Causa más probable:")
        print("  instalaste torch sin la variante de GPU.")
        print("  Revisa pytorch.org y reinstala con el --index-url correcto.")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / 1024**3
    print(f"VRAM total: {total_gb:.2f} GB")

    libre_bytes, _ = torch.cuda.mem_get_info()
    libre_gb = libre_bytes / 1024**3
    print(f"VRAM libre: {libre_gb:.2f} GB")
    return True


def vram_ocupada_mib() -> float | None:
    """VRAM total en uso (MiB) a nivel de driver.

    Usa `mem_get_info` (total del dispositivo menos libre): incluye lo que
    reserva CTranslate2/faster-whisper, que queda FUERA del allocator de
    torch. None si no hay CUDA. Cierra el gate "VRAM co-residente" (ADR-014).
    """
    if not torch.cuda.is_available():
        return None
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    libre_bytes, _ = torch.cuda.mem_get_info()
    return float((total_bytes - libre_bytes) / (1024 * 1024))
