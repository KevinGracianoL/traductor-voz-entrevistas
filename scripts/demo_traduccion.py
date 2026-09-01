"""Demo traducción offline — wrapper fino para src/traductor/traduccion/argos.py."""

from traductor.traduccion.argos import instalar_idioma, traducir

if __name__ == "__main__":
    for o, d in [("en", "es"), ("es", "en")]:
        instalar_idioma(o, d)
    print(traducir("Tell me about a hard bug you fixed.", "en", "es"))
    print(traducir("Háblame de un bug difícil que resolviste.", "es", "en"))
