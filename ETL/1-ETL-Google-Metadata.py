import pandas as pd
import re
import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq
import json
import numpy as np
import ast

def procesar_json_metadata(ruta_archivo):
    # Cargar el archivo JSON línea por línea
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    
    # Crear DataFrame
    carga_json = pd.DataFrame(data)
    
    # Eliminar columnas especificadas
    columnas_a_eliminar = ['description', 'MISC', 'state', 'relative_results', 'url', 'price', 'hours']
    carga_json = carga_json.drop(columns=[col for col in columnas_a_eliminar if col in carga_json.columns])
    
    # Extraer el estado de la dirección
    carga_json['estado'] = carga_json['address'].str.extract(r',\s*([A-Z]{2})\s*\d{5}')
    
    # Lista de estados de la costa este
    estados_deseados = ['ME', 'NH', 'MA', 'RI', 'CT', 'NY', 'NJ', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL']
    
    # Filtrar por estados deseados y crear una copia
    df_metadata_ce = carga_json[carga_json['estado'].isin(estados_deseados)].copy()
    
    # Mapeo de abreviaturas a nombres completos de estados
    mapa_estados = {
        'ME': 'Maine', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'RI': 'Rhode Island',
        'CT': 'Connecticut', 'NY': 'New York', 'NJ': 'New Jersey', 'DE': 'Delaware',
        'MD': 'Maryland', 'VA': 'Virginia', 'NC': 'North Carolina', 'SC': 'South Carolina',
        'GA': 'Georgia', 'FL': 'Florida'
    }
    
    # Crear columna con nombre completo del estado
    df_metadata_ce.loc[:, 'nombre_estado'] = df_metadata_ce['estado'].map(mapa_estados)
    
    # Extraer la ciudad de la dirección
    df_metadata_ce.loc[:, 'city'] = df_metadata_ce['address'].str.extract(r',\s*([^,]+),\s*[A-Z]{2}\s*\d{5}')
    
    # Eliminamos duplicados del DF columna gmap_id
    df_metadata_ce = df_metadata_ce.drop_duplicates(subset=['gmap_id'])

    # Se Eliminan las categorias nulas
    df_metadata_ce = df_metadata_ce.dropna(subset=['category'])

    # Convertimos las cadenas a listas (si los datos son cadenas que parecen listas)
    df_metadata_ce['category'] = df_metadata_ce['category'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # Definimos las palabras clave para filtrar
    keywords = ['Chinese restaurant', 'Delivery Chinese restaurant', 'Chinese noodle restaurant']

    # Filtramos las filas que contienen al menos una de las palabras clave en la lista de categorías
    df_metadata_ce = df_metadata_ce[df_metadata_ce['category'].apply(lambda x: any(keyword.lower() in category.lower() for keyword in keywords for category in x))]

    # Eliminar la columna 'category'
    df_metadata_ce = df_metadata_ce.drop(columns=['category'])

    return df_metadata_ce

# Ejemplo de uso:
df_procesado = procesar_json_metadata('Datasets/Google/metadata-sitios/11.json')
df_procesado.to_csv('procesado_11.csv', index=False)
print(f"Número de filas en el CSV: {len(df_procesado)}")
print(f"Columnas en el CSV: {df_procesado.columns.tolist()}")




