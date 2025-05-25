import streamlit as st
import pandas as pd
import os
from transformers import pipeline

# Configuración para Render
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(
    page_title="Analizador de Sentimientos",
    layout="centered",
    page_icon="📊"
)

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased",
        device=-1,
        truncation=True,
        max_length=128
    )

def load_data():
    try:
        return pd.read_csv(
            "opiniones_clientes.csv",
            usecols=['Opinion'],
            nrows=50  # Limita a 50 registros para memoria
        ).dropna()
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.stop()

# Interfaz minimalista
st.title("📊 Analizador Optimizado")
model = load_model()

if st.button("Analizar Opiniones"):
    with st.spinner("Procesando (puede tomar unos segundos)..."):
        try:
            df = load_data()
            df['Resultado'] = df['Opinion'].apply(
                lambda x: model(str(x)[:128])[0]['label']
            )
            st.dataframe(df)
            st.success("¡Análisis completado!")
        except Exception as e:
            st.error(f"Error durante el análisis: {str(e)}")

if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port={PORT} --server.address=0.0.0.0 --server.headless=true")
