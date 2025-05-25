import streamlit as st
import pandas as pd
import os
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuración básica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración para Render
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Análisis de Sentimientos Lite", layout="wide")

# --- Optimización de Memoria ---
@st.cache_resource
def load_lightweight_model():
    try:
        logger.info("Cargando modelo ligero...")
        
        # Modelo más pequeño y optimizado para CPU
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # Fuerza CPU
            truncation=True,
            max_length=128  # Reduce el tamaño máximo de entrada
        )
    except Exception as e:
        logger.error(f"Error al cargar modelo: {str(e)}")
        st.error("Error al iniciar el servicio. Por favor intenta más tarde.")
        st.stop()

# Cargar modelo al inicio (versión ligera)
classifier = load_lightweight_model()

# --- Manejo de Datos Optimizado ---
@st.cache_data
def load_data():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "opiniones_clientes.csv")
        if not os.path.exists(csv_path):
            st.error("Archivo no encontrado. Verifica que opinions_clientes.csv esté en la raíz.")
            st.stop()
            
        # Leer solo las columnas necesarias
        return pd.read_csv(
            csv_path,
            usecols=['Opinion'],
            encoding='utf-8',
            on_bad_lines='skip'
        ).dropna()
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.stop()

# --- Interfaz ---
st.title("📊 Analizador Lite de Opiniones")

if st.button("Comenzar Análisis"):
    with st.spinner("Procesando (modo ligero)..."):
        try:
            df = load_data()
            st.info(f"Analizando {len(df)} opiniones...")
            
            # Procesar en lotes pequeños
            batch_size = 10
            results = []
            for i in range(0, len(df), batch_size):
                batch = df['Opinion'].iloc[i:i+batch_size].astype(str).tolist()
                results.extend(classifier(batch))
            
            df['Sentimiento'] = [r['label'] for r in results]
            
            # Mostrar resultados parciales
            st.dataframe(df.head(20))
            st.success("¡Análisis completado!")
            
        except Exception as e:
            st.error(f"Error durante el análisis: {str(e)}")

if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port={PORT} --server.address=0.0.0.0 --server.headless=true")

