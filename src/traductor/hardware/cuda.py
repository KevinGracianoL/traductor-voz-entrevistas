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
