import streamlit as st
import pandas as pd
import os
from transformers import pipeline

# Configuración para Render
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Análisis de Sentimientos", layout="wide")

# Solución definitiva para cargar datos desde la raíz
@st.cache_data
def cargar_datos():
    try:
        # Opción 1: Buscar en la raíz (donde está app.py)
        csv_path = "opiniones_clientes.csv"
        
        # Leer el archivo con pandas
        df = pd.read_csv(
            csv_path,
            quotechar='"',
            on_bad_lines='skip',
            names=['Opinion']
        )
        
        # Si está vacío, intentar leer manualmente
        if df.empty:
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines[0].lower().startswith('opinion'):
                lines = lines[1:]
            df = pd.DataFrame({'Opinion': lines})
        
        return df.dropna()
    
    except Exception as e:
        st.error(f"ERROR: No se pudo leer el archivo CSV. Detalle: {str(e)}")
        st.markdown("""
        ### 🔍 Solución:
        1. Verifica que el archivo se llame **exactamente** `opiniones_clientes.csv`
        2. Asegúrate de que esté en la **raíz** de tu repositorio
        3. Revisa que tenga este formato:
           ```csv
           Opinion
           "Texto de opinión entre comillas"
           "Otra opinión, con comas si es necesario"
           ```
        """)
        st.stop()

# Cargar datos
df = cargar_datos()
st.success(f"✅ Archivo CSV cargado correctamente ({len(df)} opiniones)")

# Modelo optimizado para Render (ligero)
@st.cache_resource
def cargar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased",
        truncation=True
    )

# Interfaz mejorada
st.title("📊 Analizador de Opiniones")
st.write("---")

with st.spinner("🔍 Analizando sentimientos..."):
    try:
        classifier = cargar_modelo()
        df['Sentimiento'] = df['Opinion'].apply(
            lambda x: classifier(str(x)[:512])[0]['label']  # Truncar a 512 caracteres
        )
    except Exception as e:
        st.error(f"Error en el análisis: {str(e)}")
        st.stop()

# Mostrar resultados
st.dataframe(
    df,
    column_config={
        "Opinion": "Comentario del cliente",
        "Sentimiento": st.column_config.SelectboxColumn(
            "Resultado",
            options=["POSITIVE", "NEGATIVE", "NEUTRAL"],
            required=True
        )
    },
    hide_index=True,
    use_container_width=True
)

if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port {PORT} --server.address 0.0.0.0")
