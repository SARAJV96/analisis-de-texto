import streamlit as st
import pandas as pd
import os
from transformers import pipeline

# Configuración para Render
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Análisis Lite", layout="wide")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased",
        device=-1,  # Fuerza CPU
        truncation=True,
        max_length=128
    )

model = load_model()

def load_data():
    try:
        df = pd.read_csv("opiniones_clientes.csv", usecols=['Opinion'])
        return df.dropna().head(100)  # Limita a 100 registros
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

st.title("📊 Analizador Optimizado")
if st.button("Analizar"):
    df = load_data()
    with st.spinner("Procesando..."):
        df['Resultado'] = df['Opinion'].apply(lambda x: model(str(x)[:128])[0]['label'])
    st.dataframe(df)

if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port={PORT} --server.address=0.0.0.0")
