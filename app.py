import streamlit as st
import pandas as pd
import os
import time
from transformers import pipeline, AutoTokenizer

# Configuración especial para Render
PORT = int(os.environ.get("PORT", 8501))
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Inicialización robusta
@st.cache_resource
def init_model():
    if DEBUG:
        st.warning("⚠️ Modo DEBUG activado (modelo ligero)")
        return pipeline(
            "text-classification",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=-1,
            truncation=True,
            max_length=64
        )
    else:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased",
            device=-1,
            truncation=True,
            max_length=128
        )

# Interfaz minimalista
st.set_page_config(
    page_title="Analizador",
    page_icon="📊",
    layout="centered"
)

st.title("Análisis de Sentimientos")
model = init_model()

# Carga de datos optimizada
def load_data():
    try:
        return pd.read_csv(
            "opiniones_clientes.csv",
            usecols=['Opinion'],
            nrows=50  # Limita a 50 registros
        ).dropna()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

if st.button("Analizar"):
    with st.spinner("Procesando..."):
        try:
            df = load_data()
            start_time = time.time()
            
            # Procesamiento por lotes pequeños
            results = []
            for i, text in enumerate(df['Opinion'].astype(str)):
                results.append(model(text[:128])[0])
                if DEBUG and i >= 2:  # Solo 3 ejemplos en debug
                    break
            
            st.success(f"Análisis completado en {time.time()-start_time:.2f}s")
            st.json(results[:3] if DEBUG else results)
            
        except Exception as e:
            st.error(f"Fallo crítico: {str(e)}")
            st.stop()

if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port={PORT} --server.address=0.0.0.0 --server.headless=true --server.enableXsrfProtection=false")
