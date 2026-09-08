"""Contrato neutral de almacenamiento de perfiles de voz.

`VoiceProfileStore` separa el ciclo de vida de los perfiles (enrolamiento) del
backend que los usa para sintetizar. La implementación real (JSON, SQLite, lo
que decida un ADR futuro) queda fuera de este PR.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from traductor.tts.modelos import VoiceProfile


@runtime_checkable
class VoiceProfileStore(Protocol):
    """Almacén de `VoiceProfile` identificados por `id`."""

    def listar(self) -> list[VoiceProfile]:
        """Todos los perfiles, sin orden garantizado."""
        ...

    def obtener(self, perfil_id: str) -> VoiceProfile | None:
        """Perfil por id, o None si no existe."""
        ...

    def guardar(self, perfil: VoiceProfile) -> None:
        """Crea o reemplaza el perfil con `perfil.id`."""
        ...

    def eliminar(self, perfil_id: str) -> bool:
        """Borra el perfil. True si existía."""
        ...
