# Traductor de voz — Paso 1

## Setup
1. `python -m venv venv`
2. `.\venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132`
4. `python setup_dlls.py`
5. `python paso1_verificar.py`
6. `python paso2_traducir.py`

## Decisiones de arquitectura (ADR)

### ADR-009 — Dirección de traducción fijada por fuente de audio, no por detección de idioma
**Decisión:** la dirección de traducción (EN→ES o ES→EN) la determina de qué canal viene
el audio, no un modelo detectando el idioma en tiempo real. Micrófono (yo hablando) =
siempre ES→EN. Audio virtual del entrevistador (Paso 4) = siempre EN→ES. Fijo, sin
adivinar.

**Por qué:** detectar el idioma en vivo cuesta latencia extra (un paso más antes de
poder traducir) y falla justo donde más duele: en el *code-switching* — si sueltas un
"okay" o un nombre propio en inglés a media frase en español, un detector puede titubear
o voltear la dirección en el peor momento, a mitad de una entrevista. Enrutar por la
fuente del audio en vez de por el contenido es determinista y no le agrega ni un
milisegundo al pipeline. Es la misma idea del ADR-002 (enrutar por dispositivo de audio
a nivel de SO, no por una API que interpreta) aplicada a la dirección de traducción.