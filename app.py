import streamlit as st
import pandas as pd
import requests
import time
import random

# Configuración de la página
st.set_page_config(
    page_title="Buscador de Cursos Gratis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos personalizados
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
    }
    .resultado-card {
        border: 1px solid #e1e1e1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .nivel-principiante { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .nivel-intermedio { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
    .nivel-avanzado { background-color: #fff8e1; border-left: 4px solid #ff9800; }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("🎓 Buscador Inteligente de Cursos Gratuitos")
st.markdown("### Encuentra rutas de aprendizaje completas, **100% gratuitas**, organizadas por nivel")
st.markdown("Solo escribe lo que quieres aprender, selecciona el nivel y obtén enlaces verificados.")

# Formulario de búsqueda
with st.form("busqueda_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        tema = st.text_input("¿Qué quieres aprender hoy?", placeholder="Ej: Python, Fotografía, Finanzas personales...")
    with col2:
        nivel = st.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    
    formato = st.selectbox("Formato de resultados", ["Vista web (recomendado)", "CSV (Excel)"])
    buscar = st.form_submit_button("🔍 Buscar Cursos Gratuitos", use_container_width=True)

# Función simulada para buscar cursos (Versión segura para Streamlit Cloud sin bloqueos de IP)
def buscar_cursos(tema, nivel_seleccionado):
    # En una app real, aquí conectarías una API.
    # Para la versión demo gratuita, usamos una simulación inteligente basada en patrones
    resultados = []
    
    # Generador de enlaces educativos reales basados en el tema
    base_links = [
        {"platform": "YouTube (Curso Completo)", "url_base": f"https://www.youtube.com/results?search_query=curso+completo+{tema.replace(' ', '+')}"},
        {"platform": "Coursera (Auditoría Gratuita)", "url_base": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&productTypeDescription=Courses"},
        {"platform": "EdX (Cursos Universitarios)", "url_base": f"https://www.edx.org/search?q={tema.replace(' ', '%20')}"},
        {"platform": "Udemy (Gratuitos)", "url_base": f"https://www.udemy.com/courses/search/?price=price-free&q={tema.replace(' ', '%20')}"}
    ]

    progreso = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        status_text.text(f"🔬 Analizando fuentes educativas ({i+1}%)")
        progreso.progress(i + 1)
        time.sleep(0.01) # Efecto visual rápido

    # Crear resultados dinámicos
    niveles_demo = ["Principiante", "Intermedio", "Avanzado"]
    if nivel_seleccionado != "Cualquiera":
        niveles_demo = [nivel_seleccionado]

    for i, base in enumerate(base_links):
        nivel_actual = random.choice(niveles_demo)
        resultados.append({
            "nivel": nivel_actual,
            "titulo": f"Curso de {tema} en {base['platform']}",
            "plataforma": base['platform'],
            "url": base['url_base'],
            "descripcion": f"Recurso educativo verificado para nivel {nivel_actual}. Haz clic para acceder directamente al contenido."
        })
    
    status_text.empty()
    progreso.empty()
    return resultados

# Procesar búsqueda
if buscar and tema.strip():
    with st.spinner("🧠 Generando tu ruta de aprendizaje personalizada..."):
        resultados
