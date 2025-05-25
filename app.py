import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk
import os

# Descargar stopwords de NLTK
nltk.download('stopwords', quiet=True)

# --- Configuración esencial para Render ---
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Análisis de Sentimientos", layout="wide")

# --- Carga de datos optimizada y robusta ---
@st.cache_data
def cargar_datos():
    try:
        # Verificar si la carpeta data existe
        if not os.path.exists("data"):
            os.makedirs("data")
            st.warning("Se creó la carpeta 'data' porque no existía")
        
        # Verificar si el archivo existe
        csv_path = os.path.join("opiniones_clientes.csv")
        if not os.path.exists(csv_path):
            st.error(f"Archivo no encontrado en: {csv_path}")
            st.info("Por favor sube el archivo CSV a la carpeta 'data'")
            st.stop()
        
        # Leer el archivo línea por línea para evitar problemas de formato
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Eliminar encabezado si existe
        if lines[0].lower().startswith('opinion'):
            lines = lines[1:]
            
        # Crear DataFrame con las opiniones
        df = pd.DataFrame(lines, columns=['Opinion'])
        df = df[df['Opinion'].str.len() > 0]  # Filtrar líneas vacías
        
        return df.sample(min(100, len(df)))  # Limitar a 100 opiniones para pruebas
    
    except Exception as e:
        st.error(f"Error crítico al cargar datos: {str(e)}")
        st.stop()

df = cargar_datos()

# --- Carga eficiente del modelo ---
@st.cache_resource(show_spinner="Cargando modelo de análisis...")
def cargar_modelo():
    try:
        modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
        return pipeline(
            "sentiment-analysis", 
            model=modelo,
            tokenizer=modelo,
            device_map="auto",
            truncation=True
        )
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

clasificador = cargar_modelo()

# --- Lógica de procesamiento mejorada ---
def interpretar(texto):
    try:
        # Limitar a 512 caracteres (límite de BERT) y forzar string
        texto = str(texto)[:512]
        resultado = clasificador(texto)[0]
        estrellas = int(resultado['label'][0])
        return "⭐ Positivo" if estrellas >= 4 else "🔄 Neutro" if estrellas == 3 else "⚠️ Negativo"
    except Exception as e:
        st.warning(f"Error al analizar: {str(e)}")
        return "❓ Error"

# --- Interfaz de usuario mejorada ---
st.title("📊 Análisis de Opiniones en Tiempo Real")

with st.expander("📝 Instrucciones"):
    st.write("""
    1. Asegúrate de tener el archivo 'opiniones_clientes.csv' en la carpeta 'data'
    2. Cada línea del CSV debe contener una opinión completa
    3. Las opiniones deben estar entre comillas si contienen comas
    """)

if st.button("🔁 Actualizar análisis"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Analizando sentimientos..."):
    df['Sentimiento'] = df['Opinion'].progress_apply(interpretar)

# Mostrar resultados
st.dataframe(
    df[['Opinion', 'Sentimiento']],
    use_container_width=True,
    height=600,
    hide_index=True,
    column_config={
        "Opinion": st.column_config.TextColumn(
            "Comentario",
            width="large"
        ),
        "Sentimiento": st.column_config.SelectboxColumn(
            "Clasificación",
            options=["⭐ Positivo", "🔄 Neutro", "⚠️ Negativo", "❓ Error"],
            width="medium"
        )
    }
)

# --- Configuración final para Render ---
if __name__ == "__main__":
    os.system(f"streamlit run {__file__} --server.port {PORT} --server.address 0.0.0.0")
