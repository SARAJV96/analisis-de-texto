import streamlit as st
import pandas as pd
import os
from transformers import pipeline
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración para Render
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Análisis de Sentimientos", layout="wide")

# Solución mejorada para cargar datos
@st.cache_data
def cargar_datos():
    try:
        # Ruta absoluta para mayor confiabilidad
        csv_path = os.path.join(os.path.dirname(__file__), "opiniones_clientes.csv")
        logger.info(f"Buscando archivo CSV en: {csv_path}")
        
        if not os.path.exists(csv_path):
            st.error(f"Archivo no encontrado en: {csv_path}")
            st.stop()

        # Leer el archivo con pandas
        df = pd.read_csv(
            csv_path,
            quotechar='"',
            on_bad_lines='skip',
            names=['Opinion'],
            encoding='utf-8'
        )
        
        # Si está vacío, intentar leer manualmente
        if df.empty:
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines and lines[0].lower().startswith('opinion'):
                lines = lines[1:]
            df = pd.DataFrame({'Opinion': lines})
        
        return df.dropna()
    
    except Exception as e:
        logger.error(f"Error al cargar CSV: {str(e)}")
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

# Modelo optimizado para Render
@st.cache_resource
def cargar_modelo():
    try:
        logger.info("Cargando modelo de análisis de sentimientos...")
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased",
            truncation=True,
            device=-1  # Fuerza uso de CPU
        )
    except Exception as e:
        logger.error(f"Error al cargar modelo: {str(e)}")
        st.error(f"Error al cargar el modelo: {str(e)}")
        st.stop()

# ---------------------------
# Interfaz de la aplicación
# ---------------------------

# Precarga el modelo al iniciar (evita timeout en Render)
classifier = cargar_modelo()

# Cargar datos
try:
    df = cargar_datos()
    st.success(f"✅ Archivo CSV cargado correctamente ({len(df)} opiniones)")
except Exception as e:
    st.error(f"Error al cargar datos: {str(e)}")
    st.stop()

st.title("📊 Analizador de Opiniones")
st.write("---")

# Análisis por lotes para mayor eficiencia
with st.spinner("🔍 Analizando sentimientos..."):
    try:
        texts = df['Opinion'].astype(str).str[:512].tolist()
        results = classifier(texts)
        df['Sentimiento'] = [r['label'] for r in results]
        logger.info("Análisis completado exitosamente")
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")
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

# Estadísticas adicionales
st.write("---")
st.subheader("📊 Resumen de sentimientos")
sentiment_counts = df['Sentimiento'].value_counts()
st.bar_chart(sentiment_counts)

# Configuración final para Render
if __name__ == "__main__":
    os.system(f"streamlit run app.py --server.port={PORT} --server.address=0.0.0.0 --server.headless=true")


