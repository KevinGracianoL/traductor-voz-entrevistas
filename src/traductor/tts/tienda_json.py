"""Persistencia real de perfiles — un JSON por perfil en un directorio.

Implementa `VoiceProfileStore` (contrato de `tienda.py`). Lectura/escritura
UTF-8 explícita por bytes (mismo criterio que el manifest del ASR: sin literal
`encoding` mutable). Un archivo corrupto NO se silencia: falla claro.
"""

from __future__ import annotations

import json
from pathlib import Path

from traductor.tts.modelos import VoiceProfile


class TiendaPerfilesJson:
    """VoiceProfileStore sobre `directorio`, un `{id}.json` por perfil."""

    def __init__(self, directorio: Path) -> None:
        self._directorio = directorio

    def _ruta(self, perfil_id: str) -> Path:
        return self._directorio / f"{perfil_id}.json"

    @staticmethod
    def _serializar(perfil: VoiceProfile) -> dict[str, object]:
        return {"id": perfil.id, "nombre": perfil.nombre, "muestras": list(perfil.muestras)}

    @staticmethod
    def _deserializar(datos: dict[str, object]) -> VoiceProfile:
        id_perfil = datos.get("id")
        nombre = datos.get("nombre")
        muestras = datos.get("muestras")
        if not isinstance(id_perfil, str) or not isinstance(nombre, str):
            raise ValueError("perfil inválido: id y nombre deben ser texto")
        if not isinstance(muestras, list):
            raise ValueError("perfil inválido: muestras debe ser una lista")
        return VoiceProfile(
            id=id_perfil,
            nombre=nombre,
            muestras=tuple(str(m) for m in muestras),
        )

    def _leer(self, ruta: Path) -> VoiceProfile:
        datos: dict[str, object] = json.loads(ruta.read_bytes().decode())
        return self._deserializar(datos)

    def listar(self) -> list[VoiceProfile]:
        if not self._directorio.exists():
            return []
        perfiles: list[VoiceProfile] = []
        for archivo in sorted(self._directorio.glob("*.json")):
            perfiles.append(self._leer(archivo))
        return perfiles

    def obtener(self, perfil_id: str) -> VoiceProfile | None:
        ruta = self._ruta(perfil_id)
        if not ruta.exists():
            return None
        return self._leer(ruta)

    def guardar(self, perfil: VoiceProfile) -> None:
        self._directorio.mkdir(parents=True, exist_ok=True)
        # json.dumps con default (ASCII-escaped): bytes válidos UTF-8 en cualquier
        # plataforma. Sin literal ensure_ascii que mute a un equivalente (None==False).
        datos = json.dumps(self._serializar(perfil))
        self._ruta(perfil.id).write_bytes(datos.encode())
        return None

    def eliminar(self, perfil_id: str) -> bool:
        ruta = self._ruta(perfil_id)
        if not ruta.exists():
            return False
        ruta.unlink()
        return True
