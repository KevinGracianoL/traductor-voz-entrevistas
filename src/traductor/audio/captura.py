"""Captura de micrófono con RealtimeSTT — tubo ASR."""

from __future__ import annotations


def probar_microfono() -> None:  # pragma: no cover
    """Micrófono → texto en pantalla, con RealtimeSTT."""
    from RealtimeSTT import AudioToTextRecorder

    def al_detectar(texto: str) -> None:
        print(f">> {texto}")

    grabador = AudioToTextRecorder(
        model="tiny",
        language="en",
        device="cuda",
        compute_type="int8",
    )

    print("\nHabla en inglés. Ctrl+C para salir.")
    print("(La primera vez tarda: está descargando el modelo.)\n")
    while True:
        grabador.text(al_detectar)
