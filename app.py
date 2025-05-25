import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from transformers import pipeline
import nltk
import os

# Configuración inicial
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('spanish'))  # Cambia a 'english' si es necesario

# --- Configuración para Render ---
PORT = int(os.environ.get("PORT", 8501))
st.set_page_config(page_title="Análisis Completo", layout="wide")

# --- Carga optimizada de datos ---
@st.cache_data
def cargar_datos():
    try:
        csv_path = os.path.join("data", "opiniones_clientes.csv")
        
        # Crear carpeta data si no existe
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists(csv_path):
            st.error(f"Archivo no encontrado: {os.path.abspath(csv_path)}")
            st.stop()
            
        df = pd.read_csv(
            csv_path,
            usecols=['Opinion'],
            encoding='utf-8',
            on_bad_lines='skip'
        ).dropna()
        
        return df.sample(min(100, len(df)))  # Limitar a 100 registros
    
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        st.stop()

# --- Modelo ligero para Render ---
@st.cache_resource
def cargar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis",
        device=-1,  # Forzar CPU
        truncation=True,
        max_length=128
    )

# --- Procesamiento de texto ---
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = ' '.join([word for word in texto.split() if word not in stop_words])
    return texto

# --- Visualizaciones ---
def mostrar_graficos(df):
    # 1. Distribución de sentimientos
    with st.container():
        st.subheader("📈 Distribución de Sentimientos")
        fig1, ax1 = plt.subplots()
        df['Sentimiento'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'], ax=ax1)
        st.pyplot(fig1)
        plt.close(fig1)
    
    # 2. WordCloud
    with st.container():
        st.subheader("☁️ Nube de Palabras (Opiniones)")
        all_text = ' '.join(df['Opinion'].astype(str))
        cleaned_text = limpiar_texto(all_text)
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stop_words
        ).generate(cleaned_text)
        
        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close(fig2)
    
    # 3. Palabras más frecuentes
    with st.container():
        st.subheader("🔠 Palabras Más Frecuentes")
        words = cleaned_text.split()
        word_counts = Counter(words).most_common(15)
        
        fig3, ax3 = plt.subplots()
        pd.DataFrame(word_counts, columns=['Palabra', 'Frecuencia']).plot(
            kind='barh', 
            x='Palabra',
            color='skyblue',
            ax=ax3
        )
        st.pyplot(fig3)
        plt.close(fig3)

# --- Interfaz principal ---
def main():
    st.title("📊 Análisis Completo de Opiniones")
    
    # Carga de datos
    with st.spinner("Cargando datos..."):
        df = cargar_datos()
    
    # Carga del modelo
    with st.spinner("Cargando modelo de análisis..."):
        clasificador = cargar_modelo()
    
    # Análisis de sentimientos
    with st.spinner("Analizando opiniones..."):
        df['Sentimiento'] = df['Opinion'].apply(
            lambda x: clasificador(str(x)[:128])[0]['label']
        )
        df['Sentimiento'] = df['Sentimiento'].map(
            {'POS': '⭐ Positivo', 'NEU': '🔄 Neutro', 'NEG': '⚠️ Negativo'}
        )
    
    # Mostrar datos
    with st.expander("🗃️ Datos completos", expanded=False):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    # Mostrar gráficos
    mostrar_graficos(df)
    
    # Botón de actualización
    if st.button("🔄 Actualizar Análisis"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- Configuración para Render ---
if __name__ == "__main__":
    os.system(f"streamlit run {__file__} --server.port={PORT} --server.address=0.0.0.0 --server.headless=true")
