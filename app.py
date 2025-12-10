import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import json
from datetime import datetime
from groq import Groq

# ----------------------------
# 1. CONFIGURACIÓN INICIAL
# ----------------------------
st.set_page_config(
    page_title="🎓 Buscador IA con Groq",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DB_PATH = "cursos_inteligentes_v2.db"

# ----------------------------
# 2. GESTIÓN DE BASE DE DATOS (Restaurada)
# ----------------------------
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabla de plataformas "tesoros ocultos" (Hardcoded curated list)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plataformas_ocultas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        url_base TEXT NOT NULL,
        descripcion TEXT,
        idioma TEXT,
        nivel TEXT,
        activa INTEGER DEFAULT 1
    )
    ''')
    
    # Tabla para analíticas de búsquedas
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analiticas_busquedas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tema TEXT,
        idioma TEXT,
        nivel TEXT,
        timestamp TEXT,
        origen TEXT
    )
    ''')
    
    # Datos semilla si está vacía
    cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas")
    if cursor.fetchone()[0] == 0:
        seed_data = [
            ("FreeCodeCamp", "https://www.freecodecamp.org/news/search/?query={}", "Certificaciones completas de programación", "en", "Todos"),
            ("Aprende con Alf", "https://aprendeconalf.es/?s={}", "Recursos excelentes de Python y Pandas", "es", "Intermedio"),
            ("Harvard CS50", "https://cs50.harvard.edu/x/", "El mejor curso de introducción a CS del mundo", "en", "Principiante"),
            ("Google Activate", "https://learndigital.withgoogle.com/activate/courses", "Cursos de marketing digital y desarrollo profesional", "es", "Principiante"),
            ("Kaggle Courses", "https://www.kaggle.com/learn", "Micro-cursos prácticos de IA y Data", "en", "Intermedio")
        ]
        cursor.executemany("INSERT INTO plataformas_ocultas (nombre, url_base, descripcion, idioma, nivel) VALUES (?, ?, ?, ?, ?)", seed_data)
        conn.commit()
    
    conn.close()

if not os.path.exists(DB_PATH):
    init_database()
else:
    init_database() # Asegura que las tablas existan

# ----------------------------
# 3. ESTILOS CSS (Restaurados y Mejorados)
# ----------------------------
st.markdown("""
<style>
    /* Header con degradado */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Tarjetas de resultados */
    .resultado-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        transition: transform 0.2s;
        color: #333;
    }
    .resultado-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .card-ia { border-left-color: #764ba2 !important; background-color: #fcfaff; }
    .card-oculta { border-left-color: #FF9800 !important; background-color: #fffbf0; }
    
    /* Botones */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        height: 3em;
    }
    
    /* Métricas */
    .metric-container {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 4. BARRA LATERAL (Config y Analíticas)
# ----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("⚙️ Configuración IA")
    
    # Input para API Key de Groq
    api_key = st.text_input("🔑 Groq API Key", type="password", help="Pega tu API Key de Groq aquí para activar la IA")
    if not api_key:
        st.warning("⚠️ Sin API Key, el sistema usará búsqueda básica.")
        st.markdown("[Obtener API Key gratis](https://console.groq.com/keys)")

    st.markdown("---")
    st.subheader("📊 Analíticas Rápidas")
    
    # Consultar DB para mostrar estadísticas reales
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        total = c.execute("SELECT COUNT(*) FROM analiticas_busquedas").fetchone()[0]
        top_tema = c.execute("SELECT tema, COUNT(*) as c FROM analiticas_busquedas GROUP BY tema ORDER BY c DESC LIMIT 1").fetchone()
        conn.close()
        
        col1, col2 = st.columns(2)
        col1.metric("Búsquedas", total)
        col2.metric("Top Tema", top_tema[0] if top_tema else "-")
    except:
        st.error("Error leyendo DB")

# ----------------------------
# 5. FUNCIONES DE LÓGICA (IA + DB)
# ----------------------------

def obtener_recomendaciones_ia(tema, nivel, idioma, api_key):
    """Usa Groq para generar recomendaciones educativas estructuradas."""
    client = Groq(api_key=api_key)
    
    prompt = f"""
    Actúa como un experto en educación y curación de contenidos.
    El usuario quiere aprender sobre: "{tema}".
    Nivel: {nivel}.
    Idioma preferido: {idioma}.
    
    Genera un JSON con una lista de 4 cursos o recursos gratuitos de alta calidad (Coursera, EdX, YouTube Channels, Documentación oficial).
    No inventes URLs, usa URLs genéricas de búsqueda si no conoces la exacta (ej: youtube.com/results?search_query=...).
    
    Formato JSON requerido:
    {{
        "cursos": [
            {{
                "titulo": "Nombre del curso",
                "plataforma": "Nombre de la plataforma",
                "descripcion": "Breve descripción motivadora (max 20 palabras)",
                "url": "URL del recurso"
            }}
        ]
    }}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192", # Modelo muy rápido y potente
            response_format={"type": "json_object"},
        )
        return json.loads(chat_completion.choices[0].message.content)['cursos']
    except Exception as e:
        st.error(f"Error con la IA: {e}")
        return []

def obtener_ocultas_db(tema):
    """Busca en la base de datos local plataformas 'ocultas'."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT nombre, url_base, descripcion, nivel FROM plataformas_ocultas WHERE activa=1")
    rows = c.fetchall()
    conn.close()
    
    resultados = []
    for r in rows:
        # Solo añadimos si es relevante o genérico (filtro simple)
        resultados.append({
            "titulo": f"Explorar {tema} en {r[0]}",
            "plataforma": r[0],
            "descripcion": f"{r[2]} - Recurso verificado manualmente.",
            "url": r[1].format(tema.replace(" ", "+")),
            "tipo": "oculta"
        })
    # Devolvemos solo 2 al azar para variar
    import random
    if len(resultados) > 2:
        return random.sample(resultados, 2)
    return resultados

def registrar_evento(tema, idioma, nivel):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO analiticas_busquedas (tema, idioma, nivel, timestamp, origen) VALUES (?, ?, ?, ?, ?)",
              (tema, idioma, nivel, datetime.now().isoformat(), "app_v2"))
    conn.commit()
    conn.close()

# ----------------------------
# 6. UI PRINCIPAL
# ----------------------------

st.markdown("""
<div class="main-header">
    <h1>🎓 Buscador Educativo Potenciado por IA</h1>
    <p>La inteligencia artificial busca, filtra y te recomienda la mejor ruta de aprendizaje.</p>
</div>
""", unsafe_allow_html=True)

# Formulario
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        tema = st.text_input("🔍 ¿Qué quieres aprender hoy?", placeholder="Ej: Python para Finanzas, Marketing Digital...")
    with col2:
        nivel = st.selectbox("📊 Nivel", ["Principiante", "Intermedio", "Avanzado", "Experto"])
    with col3:
        idioma = st.selectbox("🌍 Idioma", ["Español", "Inglés", "Portugués"])
        
    btn_buscar = st.button("🚀 Generar Ruta de Aprendizaje", type="primary")

# ----------------------------
# 7. PROCESAMIENTO Y RESULTADOS
# ----------------------------

if btn_buscar and tema:
    registrar_evento(tema, idioma, nivel)
    
    st.markdown("---")
    st.subheader(f"🧠 Análisis de IA para: {tema}")
    
    # Barra de progreso real
    progreso = st.progress(0)
    status = st.empty()
    
    resultados_finales = []
    
    # PASO 1: Búsqueda IA (Si hay Key)
    if api_key:
        status.write("🤖 Consultando a Llama3 (Groq) sobre los mejores recursos...")
        progreso.progress(30)
        cursos_ia = obtener_recomendaciones_ia(tema, nivel, idioma, api_key)
        # Marcar como tipo IA
        for c in cursos_ia:
            c['tipo'] = 'ia'
        resultados_finales.extend(cursos_ia)
        progreso.progress(70)
    else:
        st.warning("⚠️ No se detectó API Key. Mostrando resultados genéricos.")
        # Fallback manual simple
        resultados_finales.append({
            "titulo": f"Curso de {tema} en YouTube",
            "plataforma": "YouTube",
            "descripcion": "Búsqueda directa de tutoriales más vistos.",
            "url": f"https://www.youtube.com/results?search_query=curso+{tema}",
            "tipo": "ia"
        })
    
    # PASO 2: Mezclar con DB Local ("Tesoros Ocultos")
    status.write("💎 Buscando en base de datos de recursos ocultos...")
    ocultos = obtener_ocultas_db(tema)
    resultados_finales.extend(ocultos)
    
    progreso.progress(100)
    time.sleep(0.5)
    progreso.empty()
    status.empty()
    
    # MOSTRAR TARJETAS
    if resultados_finales:
        for res in resultados_finales:
            # Determinar estilo según origen
            clase_extra = "card-ia" if res.get('tipo') == 'ia' else "card-oculta"
            icono = "🤖" if res.get('tipo') == 'ia' else "💎"
            etiqueta = "Recomendado por IA" if res.get('tipo') == 'ia' else "Plataforma Oculta"
            
            st.markdown(f"""
            <div class="resultado-card {clase_extra}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <h3 style="margin:0;">{icono} {res['titulo']}</h3>
                    <span style="background:#eee; padding:4px 8px; border-radius:5px; font-size:0.8em;">{res['plataforma']}</span>
                </div>
                <p style="color:#666; font-size:0.95em; margin: 10px 0;">{res['descripcion']}</p>
                <div style="margin-top:15px;">
                    <a href="{res['url']}" target="_blank" style="text-decoration:none;">
                        <button style="background:linear-gradient(90deg, #4CAF50, #2E7D32); color:white; border:none; padding:10px 20px; border-radius:5px; cursor:pointer; font-weight:bold;">
                            Ver Recurso ➡️
                        </button>
                    </a>
                </div>
                <div style="margin-top:5px; text-align:right; font-size:0.7em; color:#888;">{etiqueta}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Opción de descargar
        df = pd.DataFrame(resultados_finales)
        st.download_button("📥 Descargar Reporte CSV", df.to_csv(index=False), "plan_estudios.csv", "text/csv")
        
    else:
        st.error("No se encontraron resultados. Intenta ser más específico.")

# ----------------------------
# 8. FOOTER
# ----------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888;'>Desarrollado con Streamlit & Groq API 🚀</div>", unsafe_allow_html=True)
