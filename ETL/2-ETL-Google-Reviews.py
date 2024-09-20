import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import numpy as np
import ast
from datetime import datetime
from ast import literal_eval

# Cargar el archivo JSON línea por línea
with open('Datasets/Google/review-estados/review-Florida/19.json', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# Crear DataFrame
carga_json = pd.DataFrame(data)

# Función para convertir listas a strings
def list_to_string(x):
    if isinstance(x, list):
        return json.dumps(x)
    return x

# Aplicar la función a todas las columnas
for col in carga_json.columns:
    carga_json[col] = carga_json[col].apply(list_to_string)

# Eliminar columnas
columnas_a_eliminar = ['pics', 'resp']
carga_json = carga_json.drop(columns=columnas_a_eliminar)

# 1. Verificar y eliminar duplicados
carga_json = carga_json.drop_duplicates()

# 2. Transformar la columna 'time' a datetime
carga_json['date'] = pd.to_datetime(carga_json['time'], unit='ms')

# 3. Crear una columna 'Year'
carga_json['Year'] = carga_json['date'].dt.year

# 4. Eliminar la columna 'time' original si ya no es necesaria
carga_json = carga_json.drop('time', axis=1)

# 5. Resetear el índice después de eliminar duplicados
carga_json = carga_json.reset_index(drop=True)

# 6. Convertir 'rating' a tipo int (asumiendo que siempre son números enteros)
carga_json['rating'] = carga_json['rating'].astype(int)

# 7. Rellenar valores nulos en 'text' con un valor por defecto
carga_json['text'] = carga_json['text'].fillna('No text')

# 9. Mostrar las primeras filas del DataFrame
carga_json.head()

carga_json.to_csv('data_reviews.csv')
