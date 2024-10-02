import streamlit as st
import spacy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import folium
import json
from google.oauth2 import service_account
from google.cloud import storage
import pyarrow.parquet as pq
import io
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
import seaborn as sns
from wordcloud import WordCloud
import re 
import os
from dotenv import load_dotenv
import subprocess


#darle un nombre a la pestaña de la aplicación
st.set_page_config(page_title="China Garden ML", layout="wide")

##*Subir datos y hacer pre-procesamiento. Aplicar modelo,análisis de sentimientos, adjetivos y adverbios*

#####*********************************subir env credenciales a github**********************************

#subimos las variables de entorno para poder cargar las credenciales de nuestro google cloud de manera segura

# Cargar las variables de entorno desde el archivo .env
load_dotenv(dotenv_path='googlecloudkeys.env')


# Acceder a las variables de entorno necesarias
type = os.getenv("GOOGLE_CLOUD_type")
project_id = os.getenv("GOOGLE_CLOUD_project_id")
private_key_id = os.getenv("GOOGLE_CLOUD_private_key_id")
private_key = os.getenv("GOOGLE_CLOUD_private_key")  # Reemplazar \n con saltos de línea
client_email = os.getenv("GOOGLE_CLOUD_client_email")
client_id = os.getenv("GOOGLE_CLOUD_client_id")
auth_uri = os.getenv("GOOGLE_CLOUD_auth_uri")
token_uri = os.getenv("GOOGLE_CLOUD_token_uri")
auth_provider_x509_cert_url = os.getenv("GOOGLE_CLOUD_auth_provider_x509_cert_url")
client_x509_cert_url = os.getenv("GOOGLE_CLOUD_client_x509_cert_url")
universe_domain = os.getenv("GOOGLE_CLOUD_universe_domain")


# Crear las credenciales usando un diccionario en lugar de un archivo
credentials_dict = {
    "type": type,
    "project_id": project_id,
    "private_key_id": private_key_id,
    "private_key": private_key,
    "client_email": client_email,
    "client_id": client_id,
    "auth_uri": auth_uri,
    "token_uri": token_uri,
    "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
    "client_x509_cert_url": client_x509_cert_url,
    "universe_domain": universe_domain}


def get_bigquery_client():
    try:
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        return bigquery.Client(credentials=credentials, project=credentials.project_id)
    except Exception as e:
        print(f"Error al crear el cliente de BigQuery: {e}")
        raise

client = get_bigquery_client()

# Función para cargar datos
#@st.cache_data
def load_data(_client, table_name, limit=1000): #Cambien limit a la cantidad de filas que quieren cargar.
    query = f"""
    SELECT *
    FROM `china-garden-435200.Metadatasitios.{table_name}`
    WHERE city IN ('Orlando', 'Kissimmee', 'Davenport', 'Myrtle Beach', 'New York')
    LIMIT {limit}
    """
    return _client.query(query).to_dataframe()

# Cargar datos
df = load_data(client, 'Reviews-google_ML')


#***********************subir el modelo ML al github****************************************
@st.cache_resource
def cargar_modelo():
    return spacy.load('modelo_ner_comida_experiencia')

# Cargar el modelo usando la función cacheada
modelo_ner = cargar_modelo()

#eliminamos filas con valores nulos.
df = df.dropna(subset=['text']) 

#nos aseguramos que el resto de filas sean strings 
df['text'] = df['text'].astype(str)

#añadimos el componente 'sentencizer' al pipeline
if 'sentencizer' not in modelo_ner.pipe_names:
    modelo_ner.add_pipe('sentencizer', before='ner')

#creamos una función que nos divida por etiquetas y las guarde en un diccionario
@st.cache_data
def process_review(review):
    doc = modelo_ner(review)
    results = {
        'COMIDA': [],
        'LOCACIÓN': [],
        'PRECIO': [],
        'SERVICIO': []}    
    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ in results:
                #agregamos la oración completa que contiene la entidad
                results[ent.label_].append(sent.text.strip())    
    #convertimos las listas a strings, manteniendo oraciones únicas
    for key in results:
        results[key] = ' | '.join(set(results[key])) if results[key] else ''
    
    return results

#aplicamos la función a la columna de "text"
processed_reviews = df['text'].apply(process_review)

#creamos las columnas en el dataFrame
for category in ['COMIDA', 'LOCACIÓN', 'PRECIO', 'SERVICIO']:
    df[category] = processed_reviews.apply(lambda x: x[category])

#Implementamos el análisis de sentimientos

#implementamos la librería Vader 
sia = SentimentIntensityAnalyzer()

@st.cache_data
def analyze_sentiment(text):
    if pd.isna(text) or text == '':
        return float(0.0)
    return sia.polarity_scores(text)['compound']

#aplicamos el análisis de sentimiento a cada columna
for category in ['COMIDA', 'LOCACIÓN', 'PRECIO', 'SERVICIO']:
    df[f'{category}_SENTIMENT'] = df[category].apply(analyze_sentiment)

# Ejecutar el script de instalación del modelo
subprocess.call(["python", "setup.py"])


#implementamos el modelo preentrenado 'en_core_web_sm'
nlp1 = spacy.load("en_core_web_sm")

#creamos la función para extraer los adverbios y adjetivos
@st.cache_data
def adj_adv(text):       
        doc = nlp1(text)
        adjectives_adverbs = [token.text for token in doc if token.pos_ in ["ADJ", "ADV"]]
        return adjectives_adverbs

# Creamos una nueva columna de adjetivos y adverbios para cada columna que queremos analizar
df['adj_adv_COMIDA'] = df['COMIDA'].apply(adj_adv)
df['adj_adv_PRECIO'] = df['PRECIO'].apply(adj_adv)
df['adj_adv_LOCACIÓN'] = df['LOCACIÓN'].apply(adj_adv)
df['adj_adv_SERVICIO'] = df['SERVICIO'].apply(adj_adv)


##*Página principal*

#título principal
st.markdown("<h1 style='text-align: center;'>Análisis de Sentimientos por Ciudad y Entidades de Restaurantes Chinos \U0001F3D9 \U0001F30D</h1>",
             unsafe_allow_html=True)

#Dataframe Resturantes
st.subheader('Dataframe Restaurantes Chinos\U0001F962:')    

#mostramos el DataFrame de forma expansiva
with st.expander('Expandir/Contraer'):
    st.dataframe(df)

#comienza el proceso de elegir el estado, la ciudad y el restuarante chino con el que se hará el análisis
st.subheader('Elección Estado, Ciudad para Análisis\U0001F9E0')

#**************subir el Json a geojason a github*******************************************+
#cargar el archivo GeoJSON localmente. Este archivo tiene las coordenadas que delimitan a cada uno de los estados.
with open('us-states.json') as f:
    geojson_data = json.load(f)

#diccionario con coordenadas y con los colores personalizados con los que se dibujaran los estados del Este en el mapa
estados_este = {
    'Florida': (27.994914, -81.760254, 'orange'),
    'Georgia': (32.165622, -82.900075, 'green'),
    'South Carolina': (33.856892, -80.945007, 'blue'),
    'North Carolina': (35.630066, -80.842224, 'purple'),
    'Virginia': (37.431573, -78.656894, 'red'),
    'Maryland': (39.045754, -76.641273, 'darkblue'),
    'Delaware': (39.145252, -75.428119, 'darkgreen'),
    'Pennsylvania': (41.203322, -77.194525, 'darkred'),
    'New Jersey': (40.298904, -74.521011, 'lightred'),
    'New York': (43.299428, -74.217933, 'cadetblue'),
    'Connecticut': (41.603221, -73.087749, 'lightblue'),
    'Rhode Island': (41.580095, -71.477429, 'pink'),
    'Massachusetts': (42.407211, -71.382437, 'lightgreen'),
    'Vermont': (44.558204, -72.577841, 'orange'),
    'New Hampshire': (43.193852, -71.572395, 'gray'),
    'Maine': (45.367584, -68.972168, 'black'),
}

#crear un formulario para la selección del estado
with st.form("formulario_seleccion_estado", clear_on_submit=False):
    # Crear la lista de opciones, asegurando que "Todos los estados del este" esté al principio
    opciones = ['Todos los Estados del Este'] + list(estados_este.keys())
    #poner como estado predeterminado en el form 'Todos los Estados del Este'
    if 'estado_seleccionado' not in st.session_state:
        st.session_state.estado_seleccionado = 'Todos los Estados del Este'  # Opción por defecto
    Estado_seleccionado = st.selectbox(
        "Selecciona un Estado:",
        opciones,
        key="estado_seleccionado",
        index=opciones.index('Todos los Estados del Este') if 'estado_seleccionado' not in st.session_state else 
        opciones.index(st.session_state.estado_seleccionado))
    submit_button = st.form_submit_button("Confirmar Estado")
    # actualizar el estado seleccionado si se confirma
    if submit_button:
          st.session_state.ciudad_seleccionada = f'Ciudades con Restaurantes Chinos en {Estado_seleccionado}'


#crear el mapa centrado en la costa este de EE. UU.
m = folium.Map(
    location=[35.9, -75.0],#Centrar el mapa en el medio de los 14 estados
    zoom_start=3.5)

#desactivar el desplazamiento del mapa
m.options['dragging'] = False

#desactivar el zoom del mapa
m.options['scrollWheelZoom'] = False  #desactivar zoom con la rueda del ratón
m.options['zoomControl'] = False  #desactivar el control de zoom

#función para obtener el color basado en el estado seleccionado
def get_color(state_name):
    if st.session_state.estado_seleccionado == 'Todos los Estados del Este':
        return estados_este.get(state_name, ('', '', 'gray'))[2]  # Si el estado del form es 'Todos los Estados del Este', pintar todos los
        #estados del color asigndo en estados_este  
    if state_name == st.session_state.estado_seleccionado:
        return estados_este.get(state_name, ('', '', 'gray'))[2] # Si el estado del form es un estado en específico, pintar ese estado del
        #color asigndo en estados_este y el resto de gris
    return 'gray'  # Color por defecto para otros estados

# Añadir los límites de los estados usando GeoJSON, pero solo para los 14 estados del este
for feature in geojson_data['features']:
    state_name = feature['properties']['name']
    
    # Filtrar solo los estados del este
    if state_name in estados_este: #procesar sólo los 14 estados del este
        folium.GeoJson(#crear un mapa con folium y crear la capa de delimitación con GeoJson
            data=feature,
            style_function=lambda x, state_name=state_name: {
                'fillColor': get_color(state_name),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            highlight_function=lambda x: {
                'weight': 3,
                'color': 'blue',
                'fillOpacity': 0.9
            },
            tooltip=state_name,
        ).add_to(m)#esta función recorre cada estado que está en estados_este y utiliza la función get_color(state_name),según sea el estado
                    #del form. También aplica los límites de cada estado y la función que, cuando uno pase por sobre el estado el curso,
                    # se resalte de azul.


 
        
            

#crear tres columnas, la del centro más grande, para centrar el mapa
col1, col2, col3 = st.columns([1,3,1])

#mostrar el mapa en Streamlit, limitado a los 14 estados y  en la columna del centro
with col2:
    st.components.v1.html(m._repr_html_(), width=600, height=350)





#una vez seleccionado el estado, se carga un formulario que me diga cuáles son las ciudades de ese estado que tienen restaurantes chinos
#y cuales son los restaurantes de comida china en esa ciudad
#empezamos creando el formulario para que seleccione el usuario la ciudad
if Estado_seleccionado != 'Todos los Estados del Este':
    with st.form("formulario_seleccion_ciudad", clear_on_submit=True):
        df_filtrado = df[df['nombre_estado'].isin([Estado_seleccionado])]#hacemos un filtrado de las ciudades en el estado seleccionado
        lista_ciudades = df_filtrado['city'].unique().tolist()       
        #agregar la opción predeterminada a la lista del form
        opciones = [f'Ciudades con Restaurantes Chinos en {Estado_seleccionado}'] + lista_ciudades
        #poner como estado predeterminado en el form 'Ciudades con Resturantes Chinos en {Estado_seleccionado}
        if 'ciudad_seleccionada' not in st.session_state:
            st.session_state.ciudad_seleccionada = f'Ciudades con Restaurantes Chinos en {Estado_seleccionado}'  
        selected_city = st.selectbox(
            "Selecciona una Ciudad:",
            opciones,
            key="ciudad_seleccionada",
            index=opciones.index(f'Ciudades con Restaurantes Chinos en {Estado_seleccionado}') if "ciudad_seleccionada"
            not in st.session_state else opciones.index(st.session_state.ciudad_seleccionada))
        submit_button_ciudad = st.form_submit_button("Confirmar Ciudad")
        # actualizar el estado seleccionado si se confirma
        if submit_button_ciudad:
            st.session_state.restaurante_chino_seleccionado= f'Restaurantes Chinos en {selected_city}'

 
        
#una vez seleccionada la ciudad, creamos un formulario para seleccionar algún restuarante chino de la ciudad seleccionada
    if selected_city  != f'Ciudades con Restaurantes Chinos en {Estado_seleccionado}':
        with st.form("formulario_seleccion_resturante", clear_on_submit=True):  
            df_filtrado_ciudad = df_filtrado[df['city'].isin([selected_city])]          
            lista_resturantes = df_filtrado_ciudad ['name'].unique().tolist()  
            opciones_ciudades = [f'Restaurantes Chinos en {selected_city}'] + lista_resturantes
            #poner como estado predeterminado en el form 'Ciudades con Resturantes Chinos en {Estado_seleccionado}'
            if 'restaurante_chino_seleccionado' not in st.session_state:
                st.session_state.restaurante_chino_seleccionado = f'Restaurantes Chinos en {selected_city}'  
            selected_restaurant = st.selectbox(
                "Selecciona un Restaurante Chino:",
                opciones_ciudades,
                key="restaurante_chino_seleccionado",
                index=opciones_ciudades.index(f'Restaurantes Chinos en {selected_city}') if "restaurante_chino_seleccionado"
                not in st.session_state else opciones_ciudades.index(st.session_state.restaurante_chino_seleccionado))
            submit_button_restaurant= st.form_submit_button("Confirmar Restaurante Chino")     
                        
        #creamos las dos columnas que nos dividen los análisis general de la ciudad y por local en la ciudad. Añadimos los gráficos
        #propuestos según la columna.
        if submit_button_restaurant:
            col4, col5 = st.columns(2)                 
            #dentro de la columnas 4
            with col4:
                st.markdown(f"<h1 style='text-align: center;'>General de Restaurantes Chinos en {selected_city} \U0001F4DA</h1>", 
                unsafe_allow_html=True)
                               
                #crear gráfico de radar por ciudad y categorías
                def plot_city_sentiment_radar(df, city):
                    # Filtrar el dataframe por ciudad
                    city_df = df[df['city'] == city]
                    
                    # Calcular promedios de sentimientos positivos y negativos
                    categories = ['COMIDA', 'PRECIO', 'LOCACIÓN', 'SERVICIO']
                    pos_sentiments = []
                    neg_sentiments = []
                    total_pos = 0
                    total_neg = 0
                    
                    for cat in categories:
                        sentiment_col = f'{cat}_SENTIMENT'
                        pos = city_df[city_df[sentiment_col] > 0][sentiment_col].mean()
                        neg = abs(city_df[city_df[sentiment_col] < 0][sentiment_col].mean())
                        pos_sentiments.append(pos)
                        neg_sentiments.append(neg)
                        
                        total_pos += (city_df[sentiment_col] > 0).sum()
                        total_neg += (city_df[sentiment_col] < 0).sum()
                    
                    # Configurar el gráfico de radar
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
                    pos_sentiments = np.concatenate((pos_sentiments, [pos_sentiments[0]]))
                    neg_sentiments = np.concatenate((neg_sentiments, [neg_sentiments[0]]))
                    angles = np.concatenate((angles, [angles[0]]))
                    
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                    
                    # Dibujar los polígonos
                    ax.plot(angles, pos_sentiments, 'o-', linewidth=2, color='green', label='Positivo')
                    ax.fill(angles, pos_sentiments, alpha=0.25, color='green')
                    ax.plot(angles, neg_sentiments, 'o-', linewidth=2, color='red', label='Negativo')
                    ax.fill(angles, neg_sentiments, alpha=0.25, color='red')
                    
                    # Configurar el gráfico
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_ylim(0, 1)
                    ax.set_title(f'Promedio de análisis de sentimiento en {city}')
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

                    # Ajustar las posiciones de las etiquetas (ángulos)
                    for label, angle, category in zip(ax.get_xticklabels(), angles, categories):
                        if category == 'PRECIO':  # Etiqueta arriba
                            label.set_horizontalalignment('center')
                            label.set_rotation(0)
                        elif category == 'SERVICIO':  # Etiqueta abajo
                            label.set_horizontalalignment('center')
                            label.set_rotation(0)
                        elif category == 'LOCACIÓN':  # Etiqueta a la izquierda (rotada verticalmente)
                            label.set_horizontalalignment('right')
                            label.set_rotation(90)
                        elif category == 'COMIDA':  # Etiqueta a la derecha (rotada verticalmente)
                            label.set_horizontalalignment('left')
                            label.set_rotation(-90)
                    
                    # Añadir texto con el total de comentarios
                    plt.text(1.35, 0.9, f'Total comentarios positivos: {total_pos}', transform=ax.transAxes)
                    plt.text(1.35, 0.85, f'Total comentarios negativos: {total_neg}', transform=ax.transAxes)
                    
                    return fig
                #gráficar radar puntaje general de sentimiento por categorías en una ciudad                
                st.subheader(f'Gráfico de Radar Puntaje General de Sentimiento por Categorías en {selected_city}:') 
                fig = plot_city_sentiment_radar(df, selected_city)
                st.pyplot(fig)

                #crear gráfico de barras puntaje general de sentimiento top 5 restaurnates chinos en ciudad
                def plot_top_restaurants_sentiment(df, city, n=5):
                    # Filtrar el dataframe por ciudad
                    city_df = df[df['city'] == city]
                    
                    # Agrupar por restaurante y calcular promedios
                    restaurant_stats = city_df.groupby('name').agg({
                        'avg_rating': 'first',
                        'num_of_reviews': 'first',
                        'COMIDA_SENTIMENT': lambda x: (x > 0).sum(),
                        'PRECIO_SENTIMENT': lambda x: (x > 0).sum(),
                        'LOCACIÓN_SENTIMENT': lambda x: (x > 0).sum(),
                        'SERVICIO_SENTIMENT': lambda x: (x > 0).sum()
                    }).reset_index()
                    
                    # Calcular total de sentimientos positivos y ordenar
                    restaurant_stats['total_positive'] = restaurant_stats[['COMIDA_SENTIMENT', 'PRECIO_SENTIMENT', 'LOCACIÓN_SENTIMENT', 'SERVICIO_SENTIMENT']].sum(axis=1)
                    top_restaurants = restaurant_stats.nlargest(n, 'total_positive')
                    
                    # Preparar datos para el gráfico
                    restaurants = top_restaurants['name']
                    positive_sentiments = top_restaurants['total_positive']
                    negative_sentiments = top_restaurants['num_of_reviews'] - top_restaurants['total_positive']
                    avg_ratings = top_restaurants['avg_rating']
                    
                    # Crear el gráfico
                    fig, ax1 = plt.subplots(figsize=(10, 10))
                    
                    # Barras apiladas
                    bar_width = 0.6
                    bars_pos = ax1.bar(restaurants, positive_sentiments, bar_width, label='Positivo', color='green', alpha=0.7)
                    bars_neg = ax1.bar(restaurants, negative_sentiments, bar_width, bottom=positive_sentiments, label='Negativo', color='red', alpha=0.7)
                    
                    # Añadir etiquetas con el número de reviews
                    for i, (pos, neg) in enumerate(zip(bars_pos, bars_neg)):
                        total = pos.get_height() + neg.get_height()
                        ax1.text(pos.get_x() + pos.get_width()/2, total/2, f'{int(total)}', 
                                ha='center', va='center', color='white', fontweight='bold')
                    
                    # Configurar el eje y primario
                    ax1.set_ylabel('Número de reviews')
                    ax1.set_ylim(0, max(positive_sentiments + negative_sentiments) * 1.1)
                    
                    # Añadir línea de tendencia (puntaje promedio)
                    ax2 = ax1.twinx()
                    ax2.plot(restaurants, avg_ratings, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8, label='Puntaje promedio')
                    ax2.set_ylabel('Puntaje promedio')
                    ax2.set_ylim(0, 5)
                    
                    # Configuración general del gráfico
                    plt.title(f'Top {n} Restaurantes Chinos en {city} - Sentimientos y Puntaje Promedio')
                    ax1.set_xticks(range(len(restaurants)))
                    ax1.set_xticklabels(restaurants, rotation=45, ha='right')
                    
                    # Leyendas
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.20, 1), borderaxespad=0.)
                    
                    plt.tight_layout()

                    return fig
                
                st.subheader(f'Gráfico de Barras Puntaje General de Sentimiento Top 5 Restaurnates Chinos en {selected_city}:')
                #Gráficar barras puntaje general de sentimiento top 5 restaurnates chinos en ciudad 
                fig3 = plot_top_restaurants_sentiment(df, selected_city, n=5)
                st.pyplot(fig3)

                #creamos el gráfico nube de palabas con adjetivos y adverbios asociados
                def limpiar_texto(texto):                    
                    # Elimina comillas y caracteres no alfanuméricos, manteniendo espacios
                    texto_limpio = re.sub(r'[^\w\s]', '', texto)
                    return texto_limpio

                # Función para eliminar duplicados, manteniendo solo una aparición de cada palabra
                def eliminar_duplicados(texto):                    
                    palabras = texto.split()
                    palabras_unicas = list(dict.fromkeys(palabras))  # Elimina duplicados preservando el orden
                    return ' '.join(palabras_unicas)

                # Función simplificada para generar nubes de palabras por ciudad
                def generar_nube_palabras_por_ciudad(df, city, max_words=100):                    
                    # Filtrar el DataFrame por la ciudad seleccionada
                    df_ciudad = df[df['city'] == city]
                    
                    # Concatenar todas las palabras de las columnas de adjetivos/adverbios
                    columnas_adj_adv = ['adj_adv_COMIDA', 'adj_adv_PRECIO', 'adj_adv_LOCACIÓN', 'adj_adv_SERVICIO']
                    text = ''
                    for columna in columnas_adj_adv:
                        text += ' '.join([str(item) if isinstance(item, list) else item for item in df_ciudad[columna].dropna()]) + ' '

                    # Limpiar el texto eliminando comillas y caracteres no deseados
                    text = limpiar_texto(text)
                    
                    # Eliminar palabras duplicadas
                    text = eliminar_duplicados(text)

                    # Crear la nube de palabras
                    wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(text)

                    # Mostrar la nube de palabras                    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')  # Desactiva los ejes
                    ax.set_title(f'Nube de Palabras para {city}', fontsize=16)
                    return fig                    


                st.subheader(f'Nube de Palabras de Adverbios y Adjetivos en {selected_city}:')
                #creamos el gráfico nube de palabas con adjetivos y adverbios asociados
                fig5 = generar_nube_palabras_por_ciudad(df, selected_city)
                st.pyplot(fig5)
                
                                              
            with col5:
                st.markdown(f"<h1 style='text-align: center;'>Estado de {selected_restaurant} en la ciudad de {selected_city} \U0001F37D</h1>", 
                unsafe_allow_html=True)
                                
                #crear gráfico de radar por restaurante y categorías en un aciudad
                def plot_restaurant_sentiment_radar(df, city, restaurant):
                    # Filtrar el dataframe por ciudad    
                    restaurant_df = df[(df['name'] == restaurant) & (df['city'] == city)]
                    
                    # Calcular promedios de sentimientos positivos y negativos
                    categories = ['COMIDA', 'PRECIO', 'LOCACIÓN', 'SERVICIO']
                    pos_sentiments = []
                    neg_sentiments = []
                    total_pos = 0
                    total_neg = 0
                    
                    for cat in categories:
                        sentiment_col = f'{cat}_SENTIMENT'
                        pos = restaurant_df[restaurant_df[sentiment_col] > 0][sentiment_col].mean()
                        neg = abs(restaurant_df[restaurant_df[sentiment_col] < 0][sentiment_col].mean())
                        pos_sentiments.append(pos)
                        neg_sentiments.append(neg)
                        
                        total_pos += (restaurant_df[sentiment_col] > 0).sum()
                        total_neg += (restaurant_df[sentiment_col] < 0).sum()
                    
                    # Configurar el gráfico de radar
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
                    pos_sentiments = np.concatenate((pos_sentiments, [pos_sentiments[0]]))
                    neg_sentiments = np.concatenate((neg_sentiments, [neg_sentiments[0]]))
                    angles = np.concatenate((angles, [angles[0]]))
                    
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                    
                    # Dibujar los polígonos
                    ax.plot(angles, pos_sentiments, 'o-', linewidth=2, color='green', label='Positivo')
                    ax.fill(angles, pos_sentiments, alpha=0.25, color='green')
                    ax.plot(angles, neg_sentiments, 'o-', linewidth=2, color='red', label='Negativo')
                    ax.fill(angles, neg_sentiments, alpha=0.25, color='red')
                    
                    # Configurar el gráfico
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_ylim(0, 1)
                    ax.set_title(f'Promedio del análisis de sentimientos en {restaurant}')
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

                    # Ajustar las posiciones de las etiquetas (ángulos)
                    for label, angle, category in zip(ax.get_xticklabels(), angles, categories):
                        if category == 'PRECIO':  # Etiqueta arriba
                            label.set_horizontalalignment('center')
                            label.set_rotation(0)
                        elif category == 'SERVICIO':  # Etiqueta abajo
                            label.set_horizontalalignment('center')
                            label.set_rotation(0)
                        elif category == 'LOCACIÓN':  # Etiqueta a la izquierda (rotada verticalmente)
                            label.set_horizontalalignment('right')
                            label.set_rotation(90)
                        elif category == 'COMIDA':  # Etiqueta a la derecha (rotada verticalmente)
                            label.set_horizontalalignment('left')
                            label.set_rotation(-90)
                    
                    # Añadir texto con el total de comentarios
                    plt.text(1.35, 0.9, f'Total comentarios positivos: {total_pos}', transform=ax.transAxes)
                    plt.text(1.35, 0.85, f'Total comentarios negativos: {total_neg}', transform=ax.transAxes)
                    
                    return fig   
                #Gráfico de radar por restaurante y categorías en un aciudad
                st.subheader(f'Gráfico de Radar Puntaje de Sentimiento {selected_restaurant} por Categorías en {selected_city}:') 
                fig2 = plot_restaurant_sentiment_radar(df, selected_city, selected_restaurant)
                st.pyplot(fig2) 
                
                #creamos un gráfico para evaluar la evolución de lo sentimiento a lo largo de los años
                def plot_restaurant_chain_sentiment_trend(df, city, chain_name, start_year=2020, end_year=2021):
                    # Filtrar el dataframe por la cadena de restaurantes, ciudad y el rango de años
                    chain_df = df[(df['name'] == chain_name) & (df['city'] == city) & (df['Year'].between(start_year, end_year))].copy()
                    
                    # Convertir la columna 'date' a datetime si no lo está ya
                    chain_df['date'] = pd.to_datetime(chain_df['date'])
                    
                    # Crear una columna de año-mes para agrupar por mes
                    chain_df['year_month'] = chain_df['date'].dt.to_period('M')
                    
                    # Agrupar por mes y calcular promedios de sentimientos
                    sentiment_trends = chain_df.groupby('year_month').agg({
                        'COMIDA_SENTIMENT': 'mean',
                        'PRECIO_SENTIMENT': 'mean',
                        'LOCACIÓN_SENTIMENT': 'mean',
                        'SERVICIO_SENTIMENT': 'mean'
                    }).reset_index()
                    
                    # Convertir year_month a datetime para el gráfico
                    sentiment_trends['date'] = sentiment_trends['year_month'].dt.to_timestamp()
                    
                    # Calcular sentimiento promedio general
                    sentiment_columns = ['COMIDA_SENTIMENT', 'PRECIO_SENTIMENT', 'LOCACIÓN_SENTIMENT', 'SERVICIO_SENTIMENT']
                    sentiment_trends['avg_sentiment'] = sentiment_trends[sentiment_columns].mean(axis=1)
                    
                    # Crear la figura con dos subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1])
                    
                    # Gráfico de líneas para la evolución de sentimientos
                    sns.lineplot(data=sentiment_trends, x='date', y='avg_sentiment', ax=ax1, color='blue', label='Sentimiento Promedio')
                    ax1.axhline(y=0, color='r', linestyle='--')
                    ax1.set_title(f'Evolución de Sentimientos para {chain_name} en {city} ({start_year}-{end_year})')
                    ax1.set_xlabel('Fecha')
                    ax1.set_ylabel('Puntaje de Sentimiento')
                    ax1.legend()
                    
                    # Rellenar áreas positivas y negativas
                    ax1.fill_between(sentiment_trends['date'], sentiment_trends['avg_sentiment'], 0, 
                                    where=(sentiment_trends['avg_sentiment'] >= 0), interpolate=True, color='green', alpha=0.3)
                    ax1.fill_between(sentiment_trends['date'], sentiment_trends['avg_sentiment'], 0, 
                                    where=(sentiment_trends['avg_sentiment'] <= 0), interpolate=True, color='red', alpha=0.3)
                    
                    # Gráfico de barras apiladas para el número de reviews
                    reviews_count = chain_df.groupby('Year').agg({
                        'COMIDA_SENTIMENT': lambda x: (x > 0).sum(),
                        'PRECIO_SENTIMENT': lambda x: (x > 0).sum(),
                        'LOCACIÓN_SENTIMENT': lambda x: (x > 0).sum(),
                        'SERVICIO_SENTIMENT': lambda x: (x > 0).sum()
                    }).reset_index()
                    
                    reviews_count['positive'] = reviews_count[sentiment_columns].sum(axis=1)
                    reviews_count['total'] = chain_df.groupby('Year').size().values
                    reviews_count['negative'] = reviews_count['total'] - reviews_count['positive']
                    
                    # Crear el gráfico de barras apiladas
                    ax2.bar(reviews_count['Year'], reviews_count['positive'], color='green', label='Positivas')
                    ax2.bar(reviews_count['Year'], reviews_count['negative'], bottom=reviews_count['positive'], color='red', label='Negativas')
                    
                    # Añadir etiquetas con el número de reviews positivas y negativas
                    for i, (pos, neg) in enumerate(zip(reviews_count['positive'], reviews_count['negative'])):
                        ax2.text(reviews_count['Year'][i], pos/2, f'{pos}', ha='center', va='center', color='white')
                        ax2.text(reviews_count['Year'][i], pos + neg/2, f'{neg}', ha='center', va='center', color='white')
                    
                    ax2.set_title(f'Número de Reviews por Año para {chain_name} en {city} ({start_year}-{end_year})')
                    ax2.set_xlabel('Año')
                    ax2.set_ylabel('Número de Reviews')
                    ax2.legend()
                    
                    # Ajustar el rango del eje x para ambos gráficos
                    ax1.set_xlim(pd.Timestamp(f'{start_year}-01-01'), pd.Timestamp(f'{end_year}-12-31'))
                    ax2.set_xlim(start_year - 0.5, end_year + 0.5)
                    
                    # Rotar las etiquetas del eje x para mejor legibilidad
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                    
                    plt.tight_layout()
                    return fig
                #gráfico para evaluar la evolución de lo sentimiento a lo largo de los años
                st.subheader(f'Gráfico de Barra Variación Puntaje de Sentimiento {selected_restaurant} en {selected_city} en el Tiempo:') 
                fig4 = plot_restaurant_chain_sentiment_trend(df, selected_city, selected_restaurant)                
                st.pyplot(fig4)

                def generar_nube_palabras_por_restaurante(df, ciudad, restaurante, max_words=100):                    
                    # Filtrar el DataFrame por la ciudad seleccionada
                    df_filtrado = df[(df['city'] == ciudad) & (df['name'] == restaurante)]
                    
                    # Concatenar todas las palabras de las columnas de adjetivos/adverbios
                    columnas_adj_adv = ['adj_adv_COMIDA', 'adj_adv_PRECIO', 'adj_adv_LOCACIÓN', 'adj_adv_SERVICIO']
                    text = ''
                    for columna in columnas_adj_adv:
                        text += ' '.join([str(item) if isinstance(item, list) else item for item in df_filtrado[columna].dropna()]) + ' '

                    # Limpiar el texto eliminando comillas y caracteres no deseados
                    text = limpiar_texto(text)
                    
                    # Eliminar palabras duplicadas
                    text = eliminar_duplicados(text)

                    # Crear la nube de palabras
                    wordcloud = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(text)

                    # Mostrar la nube de palabras                    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')  # Desactiva los ejes
                    plt.title(f'Nube de Palabras para {restaurante} en {ciudad}', fontsize=16)
                    return fig
                
                st.subheader(f'Nube de Palabras de Adverbios y Adjetivos en {selected_city}:')
                #creamos el gráfico nube de palabas con adjetivos y adverbios asociados
                fig6 = generar_nube_palabras_por_restaurante(df, selected_city, selected_restaurant, max_words=100)
                st.pyplot(fig6)



##*Sidebar*

#creamos un sidebar donde vamos a a poder realizar un testeo del modelo y le ponemos un título
st.sidebar.markdown("<h1 style='text-align: center;'>Ejemplo de Funcionamiento Machine Learning para Obtención de Etiquetas \U0001F916</h1>",
             unsafe_allow_html=True)

#creamos la caja de texto donde el usuario podrá poner el comentario 
texto = st.sidebar.text_area('Agregar el review relacionado con restaurantes chinos al cual se le quiere sacar etiquetas \U0001F363')

#creamos un botón para que se ejecute el etiquetado 

# Definir una función para asignar el color según la etiqueta

@st.cache_data
def obtener_color_etiqueta(label):
    if label == "COMIDA":
        return "#ffcccb"  # Color para COMIDA (rojo claro)
    elif label == "SERVICIO":
        return "#add8e6"  # Color para SERVICIO (azul claro)
    elif label == "PRECIO":
        return "#90ee90"  # Color para PRECIO (verde claro)
    elif label == "LOCACIÓN":
        return "#f0e68c"  # Color para LOCACIÓN (amarillo claro)
    return "#d3d3d3"  # Color por defecto (gris claro)

# Tu lógica para mostrar el botón y procesar el texto
if st.sidebar.button("Etiquetas") or texto:
    # Procesar el texto con el modelo entrenado
    doc = modelo_ner(texto)
        
    # Mostrar las entidades reconocidas
    st.sidebar.subheader(f"Entidades encontradas: {len(doc.ents)}")
    for ent in doc.ents:
         # Obtener el color para la etiqueta
        color = obtener_color_etiqueta(ent.label_)
        # Crear el estilo HTML para el cuadro relleno
        st.sidebar.markdown(
            f'{ent.text} <span style="background-color:{color}; padding:4px; border-radius:5px;">{ent.label_}</span>',
            unsafe_allow_html=True)
                        
    #Hacemos el análisis de sentimiento
    st.sidebar.subheader('Análisis de sentimiento')

    sia = SentimentIntensityAnalyzer()
    
    @st.cache_data
    def analyze_sentiment(text):
        if pd.isna(text) or text == '':
            return float(0.0)
        return sia.polarity_scores(text)['compound']
        
    for ent in doc.ents:
         # Obtener el color para la etiqueta
        color = obtener_color_etiqueta(ent.label_)
        # Crear el estilo HTML para el cuadro relleno
        st.sidebar.markdown(
            f'{analyze_sentiment(ent.text)} <span style="background-color:{color}; padding:4px; border-radius:5px;">{ent.label_}</span>',
            unsafe_allow_html=True)
    
    #Sacamos los adverbios y adjetivos relacionados 
    st.sidebar.subheader('Adjetivos y Adverbios Relacionados a la Etiqueta')

    # Implementamos el modelo preentrenado 'en_core_web_sm'
    nlp1 = spacy.load("en_core_web_sm")

    # Creamos la función para extraer los adverbios y adjetivos
    @st.cache_data
    def adj_adv(text):       
        doc = nlp1(text)
        adjectives_adverbs = [token.text for token in doc if token.pos_ in ["ADJ", "ADV"]]

        return adjectives_adverbs
    
    for ent in doc.ents:
         # Obtener el color para la etiqueta
        color = obtener_color_etiqueta(ent.label_)
        # Crear el estilo HTML para el cuadro relleno
        st.sidebar.markdown(
            f'{adj_adv(ent.text)} <span style="background-color:{color}; padding:4px; border-radius:5px;">{ent.label_}</span>',
            unsafe_allow_html=True)
