import spacy
import os

# Verificar si el modelo ya está descargado
try:
    spacy.load("en_core_web_sm")
except OSError:
    # Descargar el modelo si no está disponible
    os.system("python -m spacy download en_core_web_sm")
