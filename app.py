import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from collections import Counter
from transformers import pipeline
import nltk
import requests
from io import StringIO

# Configuración inicial
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('spanish'))

# --- Configuración de la página ---
st.set_page_config(
    page_title="📊 Analizador de Sentimientos",
    page_icon="📈",
    layout="wide"
)

# --- Carga desde GitHub ---
@st.cache_data
def cargar_datos():
    try:
        # URL raw de GitHub (usa el enlace directo al archivo raw)
        github_url = "https://raw.githubusercontent.com/SARAJV96/analisis-de-texto/main/opiniones_clientes.csv"
        
        # Descargar el archivo
        response = requests.get(github_url)
        response.raise_for_status()  # Verifica errores HTTP
        
        # Leer CSV directamente desde la respuesta
        df = pd.read_csv(
            StringIO(response.text),
            usecols=['Opinion'],
            encoding='utf-8',
            on_bad_lines='skip'
        ).dropna()
        
        return df.sample(min(100, len(df)))  # Limitar a 100 registros para pruebas
    
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")
        st.info("ℹ️ Verifica que la URL del CSV en GitHub sea correcta y pública")
        return None

# --- Resto del código permanece igual ---
@st.cache_resource
def cargar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis",
        device=-1,
        truncation=True,
        max_length=128
    )

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = ' '.join([word for word in texto.split() if word not in stop_words])
    return texto

def mostrar_graficos(df):
    # 1. Distribución de sentimientos
    st.subheader("📊 Distribución de Sentimientos")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df['Sentimiento'].value_counts().plot(
        kind='bar',
        color=['#4CAF50', '#2196F3', '#F44336'],
        ax=ax1
    )
    plt.xticks(rotation=0)
    st.pyplot(fig1)
    plt.close(fig1)

    # 2. Nube de palabras
    st.subheader("☁️ Nube de Palabras")
    texto_completo = ' '.join(df['Opinion'].astype(str))
    texto_limpio = limpiar_texto(texto_completo)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stop_words
    ).generate(texto_limpio)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)
    plt.close(fig2)

    # 3. Top 15 palabras más frecuentes
    st.subheader("🔠 Palabras Clave Más Usadas")
    palabras = texto_limpio.split()
    contador = Counter(palabras).most_common(15)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    pd.DataFrame(contador, columns=['Palabra', 'Frecuencia']).plot(
        kind='barh',
        x='Palabra',
        color='#2196F3',
        ax=ax3
    )
    plt.xlabel('Frecuencia')
    st.pyplot(fig3)
    plt.close(fig3)

def main():
    st.title("📊 Analizador de Sentimientos en Tiempo Real")
    st.markdown("---")

    # Carga de datos
    df = cargar_datos()
    if df is None:
        return

    # Carga del modelo
    modelo = cargar_modelo()

    # Análisis de sentimientos
    with st.spinner("🔍 Analizando opiniones..."):
        df['Sentimiento'] = df['Opinion'].apply(
            lambda x: modelo(str(x)[:128])[0]['label']
        )
        df['Sentimiento'] = df['Sentimiento'].map({
            'POS': '⭐ Positivo',
            'NEU': '🔄 Neutro',
            'NEG': '⚠️ Negativo'
        })

    # Mostrar resultados
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)
    mostrar_graficos(df)

    if st.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

if __name__ == "__main__":
    main()
