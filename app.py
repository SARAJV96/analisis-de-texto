import streamlit as st
import pandas as pd
import os
from transformers import pipeline

# Configuración especial para Render
PORT = int(os.environ.get("PORT", 8501))
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", False)

# Solución para el error 404
if RENDER_EXTERNAL_URL:
    st.set_page_config(
        page_title="Analizador de Sentimientos",
        layout="wide",
        page_icon="📊"
    )
else:
    st.set_page_config(
        page_title="Analizador de Sentimientos",
        layout="wide"
    )

# Cargar modelo optimizado
@st.cache_resource
def load_model():
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased",
            device=-1,  # Fuerza CPU
            truncation=True,
            max_length=128  # Reduce memoria
        )
    except Exception as e:
        st.error(f"Error al cargar modelo: {str(e)}")
        st.stop()

# Cargar datos
@st.cache_data
def load_data():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "opiniones_clientes.csv")
        return pd.read_csv(csv_path, usecols=['Opinion']).dropna().head(50)  # Limita a 50 registros
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.stop()

# --- Interfaz ---
model = load_model()
st.title("📊 Analizador Optimizado para Render")

if st.button("Iniciar Análisis"):
    df = load_data()
    with st.spinner("Analizando (puede tomar unos segundos)..."):
        results = []
        for text in df['Opinion'].astype(str):
            results.append(model(text[:128])[0]['label'])
        df['Sentimiento'] = results
    
    st.dataframe(df)
    st.success("¡Análisis completado!")

# Configuración crítica para Render
if __name__ == "__main__":
    if RENDER_EXTERNAL_URL:
        os.system(f"streamlit run app.py --server.port={PORT} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false")
    else:
        os.system(f"streamlit run app.py --server.port={PORT} --server.address=0.0.0.0")
