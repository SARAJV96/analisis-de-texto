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
stop_words = set(stopwords.words('spanish'))  # Cambiar a 'english' si es necesario

# --- Configuración de la página ---
st.set_page_config(
    page_title="📊 Analizador de Sentimientos",
    page_icon="📈",
    layout="wide"
)

# --- Carga optimizada de datos ---
@st.cache_data
def cargar_datos():
    try:
        if not os.path.exists("data/opiniones_clientes.csv"):
            st.error("❌ Archivo no encontrado en: `data/opiniones_clientes.csv`")
            st.info("📌 Por favor, sube un archivo CSV con una columna llamada **'Opinion'**")
            return None
        
        df = pd.read_csv(
            "data/opiniones_clientes.csv",
            usecols=['Opinion'],
            encoding='utf-8',
            on_bad_lines='skip'
        ).dropna()
        
        return df.sample(min(100, len(df)))  # Limitar a 100 registros para pruebas
    
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")
        return None

# --- Carga del modelo (optimizado para CPU) ---
@st.cache_resource
def cargar_modelo():
    return pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis",  # Modelo ligero
        device=-1,  # Forzar CPU
        truncation=True,
        max_length=128  # Reducir carga
    )

# --- Limpieza de texto para WordCloud ---
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar puntuación
    texto = ' '.join([word for word in texto.split() if word not in stop_words])  # Quitar stopwords
    return texto

# --- Visualización de gráficos ---
def mostrar_graficos(df):
    # 1. Distribución de sentimientos
    st.subheader("📊 Distribución de Sentimientos")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df['Sentimiento'].value_counts().plot(
        kind='bar',
        color=['#4CAF50', '#2196F3', '#F44336'],  # Verde, Azul, Rojo
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

# --- Interfaz principal ---
def main():
    st.title("📊 Analizador de Sentimientos en Tiempo Real")
    st.markdown("---")

    # Carga de datos
    df = cargar_datos()
    if df is None:
        return  # Detener ejecución si no hay datos

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

    # Mostrar tabla de resultados
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # Mostrar gráficos
    mostrar_graficos(df)

    # Botón para reiniciar análisis
    if st.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- Ejecutar la app ---
if __name__ == "__main__":
    main()
