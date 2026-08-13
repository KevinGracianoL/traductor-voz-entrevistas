"""
Paso 1 — Prueba de hardware.

OBJETIVO: descubrir HOY si tu máquina puede con este proyecto, en vez de
descubrirlo en la semana 3. Esto no es un "hola mundo": es la verificación que
decide si toda la arquitectura que discutimos se sostiene en tu laptop.

CÓMO USAR ESTE ARCHIVO
----------------------
Está incompleto A PROPÓSITO. Los `TODO` los escribes tú. Copiar código que no
entiendes no te enseña nada, y en la revisión te voy a preguntar por qué está
cada línea.

Cuando lo tengas corriendo:

    python paso1_verificar.py

y me mandas la salida COMPLETA, tal cual, funcione o truene.

QUÉ ESTAMOS VERIFICANDO
-----------------------
1. Que PyTorch vea tu GPU (el error silencioso más común del ecosistema:
   `pip install torch` a secas instala la versión SIN GPU y todo "funciona",
   20 veces más lento, sin avisar).
2. Que sea la GPU correcta y sepamos cuánta VRAM hay libre de verdad.
3. Que el micrófono llegue hasta el modelo y salga texto.
"""

# TODO(1): importa torch.
#          Pista: una línea. Si esto ya truena, el problema es la instalación,
#          no tu código — y saberlo ya es información útil.

import torch

def verificar_gpu() -> bool:
    """Comprueba que PyTorch tenga CUDA y reporta la GPU.

    Devuelve True solo si podemos seguir. Devolver un bool en vez de imprimir
    y ya, permite que quien llame DECIDA qué hacer — un chequeo que solo
    imprime no sirve para nada automatizable.
    """
    # TODO(2): guarda en `disponible` si CUDA está disponible.
    #          Pista: torch.cuda.is_available()
    disponible = torch.cuda.is_available()  # <- reemplaza esto

    print(f"CUDA disponible: {disponible}")

    if not disponible:
        # Fallar temprano y ruidosamente. Lo contrario — seguir a medias y
        # tronar 200 líneas después con un error que no dice nada — es cómo se
        # pierden tardes enteras.
        print("\n  Sin CUDA no seguimos. Causa más probable:")
        print("  instalaste torch sin la variante de GPU.")
        print("  Revisa pytorch.org y reinstala con el --index-url correcto.")
        return False

    # TODO(3): imprime el NOMBRE de la GPU.
    #          Pista: torch.cuda.get_device_name(0)
    #          El 0 es el índice del dispositivo: tu laptop tiene una sola GPU.
    #          Criterio de éxito: debe decir "GTX 1650 Ti".
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # TODO(4): imprime la VRAM TOTAL en GB.
    #          Pista: torch.cuda.get_device_properties(0).total_memory
    #                 devuelve BYTES. Divide entre 1024**3.
    #          Formatea con f"{x:.2f}" para no imprimir 20 decimales.
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / 1024**3
    print(f"VRAM total: {total_gb:.2f} GB")

    # TODO(5): imprime la VRAM LIBRE en GB.
    #          Pista: torch.cuda.mem_get_info() devuelve una TUPLA
    #                 (libre, total), en bytes.
    #
    #          POR QUÉ IMPORTA MÁS QUE EL TOTAL: tienes 4 GB en papel, pero
    #          Windows y el escritorio ya se comen una parte. El presupuesto
    #          real para el modelo es lo LIBRE, no lo total. Este número
    #          decide qué modelo cabe.
    libre_bytes, _ = torch.cuda.mem_get_info()
    libre_gb = libre_bytes / 1024**3
    print(f"VRAM libre: {libre_gb:.2f} GB")
    return True


def probar_microfono() -> None:
    """Micrófono → texto en pantalla, con RealtimeSTT.

    Modelo deliberadamente malo y chiquito: hoy NO medimos calidad, medimos
    que la tubería exista. La calidad se mide en el paso 2, con cronómetro.
    """
    from RealtimeSTT import AudioToTextRecorder

    def al_detectar(texto: str) -> None:
        print(f">> {texto}")

    grabador = AudioToTextRecorder(
        model="tiny",       # el más chico y más impreciso que hay. A propósito.
        language="en",
        device="cuda",
        # TODO(6): pon el compute_type que corresponde a TU hardware.
        #          Está decidido en el ADR-004 del README de este repo.
        #          En la revisión te voy a preguntar por qué ese y no float16.
        compute_type="int8",
    )

    print("\nHabla en inglés. Ctrl+C para salir.")
    print("(La primera vez tarda: está descargando el modelo.)\n")
    while True:
        grabador.text(al_detectar)


if __name__ == "__main__":
    # Este `if` NO es opcional en Windows.
    #
    # RealtimeSTT lanza procesos en paralelo. En Windows los procesos hijos
    # arrancan RE-IMPORTANDO este archivo; sin esta guarda, cada hijo volvería
    # a ejecutar el código de arranque y lanzaría más hijos: bucle infinito de
    # procesos hasta que se te congela la máquina.
    #
    # En Linux no pasa (usa fork, no re-importa). Es exactamente el tipo de
    # detalle específico de plataforma que te cuesta una tarde si nadie te lo
    # advierte.
    if verificar_gpu():
        probar_microfono()
