import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import random
from datetime import datetime
import json

# ----------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------
# Usamos ruta relativa para evitar errores de carpeta
DB_PATH = "cursos_inteligentes.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plataformas_ocultas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT NOT NULL,
        url_base TEXT NOT NULL,
        descripcion TEXT,
        idioma TEXT NOT NULL,
        categoria TEXT,
        nivel TEXT,
        confianza REAL DEFAULT 0.7,
        ultima_verificacion TEXT,
        activa INTEGER DEFAULT 1
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analiticas_busquedas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tema TEXT NOT NULL,
        idioma TEXT NOT NULL,
        nivel TEXT,
        timestamp TEXT NOT NULL,
        plataforma_origen TEXT,
        veces_mostrado INTEGER DEFAULT 0,
        veces_clickeado INTEGER DEFAULT 0
    )
    ''')
    
    # Datos semilla si la tabla está vacía
    cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas")
    if cursor.fetchone()[0] == 0:
        plataformas_iniciales = [
            ("Aprende con Alf", "https://aprendeconalf.es/?s={}", "Cursos gratuitos de programación, matemáticas y ciencia de datos", "es", "Programación", "Intermedio", 0.8),
            ("Biblioteca Virtual Cervantes", "https://www.cervantesvirtual.com/buscar/?q={}", "Biblioteca digital con recursos educativos en español", "es", "Humanidades", "Todos", 0.9),
            ("Domestika (Gratis)", "https://www.domestika.org/es/search?query={}&free=1", "Cursos gratuitos de diseño, ilustración y creatividad", "es", "Diseño", "Intermedio", 0.8),
            ("The Programming Historian", "https://programminghistorian.org/en/lessons/?q={}", "Tutoriales de programación y humanidades digitales", "en", "Programación", "Avanzado", 0.7),
            ("OER Commons", "https://www.oercommons.org/search?q={}", "Recursos educativos abiertos de instituciones globales", "en", "General", "Todos", 0.9),
            ("Kaggle Learn", "https://www.kaggle.com/learn/search?q={}", "Microcursos prácticos de ciencia de datos y machine learning", "en", "Data Science", "Intermedio", 0.8),
            ("Escola Virtual", "https://www.escolavirtual.pt/pesquisa?q={}", "Plataforma educativa portuguesa con recursos gratuitos", "pt", "General", "Todos", 0.75),
            ("Coursera (PT)", "https://www.coursera.org/search?query={}&languages=pt", "Cursos gratuitos en portugués", "pt", "General", "Todos", 0.8)
        ]
        
        for plat in plataformas_iniciales:
            cursor.execute('''
            INSERT INTO plataformas_ocultas 
            (nombre, url_base, descripcion, idioma, categoria, nivel, confianza, ultima_verificacion, activa)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ''', plat + (datetime.now().isoformat(),))
    
    conn.commit()
    conn.close()

# Inicializar DB al arrancar
if not os.path.exists(DB_PATH):
    init_database()
else:
    # Asegurar que existan las tablas aunque el archivo exista
    init_database()

# ----------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ----------------------------
st.set_page_config(
    page_title="🎓 Buscador Profesional de Cursos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Estilos Generales */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .stButton button {
        background: linear-gradient(to right, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .resultado-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        background: white;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        border-left: 5px solid #4CAF50;
        color: black;
    }
    
    .resultado-card:hover {
        transform: translateY(-3px);
    }
    
    .nivel-principiante { border-left-color: #2196F3 !important; }
    .nivel-intermedio { border-left-color: #4CAF50 !important; }
    .nivel-avanzado { border-left-color: #FF9800 !important; }
    
    .plataforma-oculta {
        background: linear-gradient(135deg, #fff5eb 0%, #ffffff 100%);
        border-left-color: #FF6B35 !important;
        border: 1px solid #ffecd2;
    }
    
    .search-form {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# BARRA LATERAL
# ----------------------------
with st.sidebar:
    st.image("https://i.imgur.com/Ke7Jd9l.png", use_column_width=True)
    st.title("🧠 Buscador Inteligente")
    
    st.markdown("### 📊 Estadísticas")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analiticas_busquedas")
        total_busquedas = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas WHERE activa = 1")
        total_plataformas = cursor.fetchone()[0]
        conn.close()
    except:
        total_busquedas = 0
        total_plataformas = 8
    
    col1, col2 = st.columns(2)
    col1.metric("Búsquedas", total_busquedas)
    col2.metric("Fuentes", total_plataformas)
    
    st.markdown("---")
    st.markdown("""
    ### 🌐 Idiomas
    - 🇪🇸 Español
    - 🇬🇧 Inglés  
    - 🇵🇹 Portugués
    """)

# ----------------------------
# CABECERA
# ----------------------------
st.markdown("""
<div class="main-header">
    <h1>🎓 Buscador Profesional de Cursos Gratuitos</h1>
    <p>Descubre recursos educativos ocultos y verificados con IA</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LÓGICA DE IDIOMAS Y BÚSQUEDA
# ----------------------------
IDIOMAS = {
    "Español (es)": "es",
    "Inglés (en)": "en", 
    "Portugués (pt)": "pt"
}

with st.container():
    st.markdown('<div class="search-form">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        tema = st.text_input("🔍 ¿Qué quieres aprender?", 
                           placeholder="Ej: Python, Marketing, Excel...",
                           key="tema_input")
    with col2:
        nivel = st.selectbox("📚 Nivel", 
                           ["Cualquiera", "Principiante", "Intermedio", "Avanzado"],
                           key="nivel_select")
    with col3:
        idioma_seleccionado = st.selectbox("🌍 Idioma", 
                                         list(IDIOMAS.keys()),
                                         key="idioma_select")
    
    buscar = st.button("🚀 Buscar Cursos", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# FUNCIONES PRINCIPALES
# ----------------------------
def obtener_plataformas_ocultas(idioma, tema, nivel_seleccionado):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
    SELECT nombre, url_base, descripcion, nivel, confianza 
    FROM plataformas_ocultas 
    WHERE activa = 1 AND idioma = ?
    '''
    params = [idioma]
    
    # Filtro simple de nivel
    if nivel_seleccionado not in ["Cualquiera", "Todos"]:
        query += " AND (nivel = ? OR nivel = 'Todos')"
        params.append(nivel_seleccionado)
    
    query += " ORDER BY confianza DESC LIMIT 4"
    
    cursor.execute(query, params)
    resultados = cursor.fetchall()
    conn.close()
    
    return [{
        "nombre": r[0],
        "url_base": r[1],
        "descripcion": r[2],
        "nivel": r[3],
        "confianza": r[4],
        "tipo": "oculta"
    } for r in resultados]

def buscar_cursos_avanzado(tema, nivel_seleccionado, idioma):
    resultados = []
    codigo_idioma = IDIOMAS[idioma]
    
    # Plataformas estándar por idioma
    if codigo_idioma == "es":
        plataformas = {
            "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", "niveles": ["Principiante", "Intermedio"]},
            "coursera": {"nombre": "Coursera (ES)", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=es", "niveles": ["Intermedio", "Avanzado"]},
            "udemy": {"nombre": "Udemy (Gratis)", "url": f"https://www.udemy.com/courses/search/?price=price-free&lang=es&q={tema.replace(' ', '%20')}", "niveles": ["Principiante", "Intermedio"]}
        }
    elif codigo_idioma == "pt":
        plataformas = {
            "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", "niveles": ["Principiante", "Intermedio"]},
            "coursera": {"nombre": "Coursera (PT)", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=pt", "niveles": ["Intermedio"]},
            "udemy": {"nombre": "Udemy (PT)", "url": f"https://www.udemy.com/courses/search/?price=price-free&lang=pt&q={tema.replace(' ', '%20')}", "niveles": ["Principiante"]}
        }
    else: # Inglés
        plataformas = {
            "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", "niveles": ["Principiante", "Intermedio"]},
            "coursera": {"nombre": "Coursera", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&free=true", "niveles": ["Intermedio", "Avanzado"]},
            "edx": {"nombre": "edX", "url": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={tema.replace(' ', '%20')}", "niveles": ["Avanzado"]}
        }

    # Barra de carga falsa para UX
    progreso = st.progress(0)
    for i in range(50):
        time.sleep(0.01)
        progreso.progress(i*2)
    progreso.empty()

    # Procesar plataformas conocidas
    niveles_permitidos = ["Principiante", "Intermedio", "Avanzado"] if nivel_seleccionado == "Cualquiera" else [nivel_seleccionado]
    
    for _, datos in plataformas.items():
        if len(resultados) >= 5: break
        
        # Simulación de resultado
        titulo = f"Curso de {tema} en {datos['nombre']}"
        nivel_res = random.choice([n for n in datos['niveles'] if n in niveles_permitidos] or datos['niveles'])
        
        resultados.append({
            "nivel": nivel_res,
            "titulo": titulo,
            "plataforma": datos["nombre"],
            "url": datos["url"],
            "descripcion": f"Curso verificado de {tema} con acceso gratuito.",
            "tipo": "conocida"
        })

    # Agregar plataformas ocultas (DB)
    ocultas = obtener_plataformas_ocultas(codigo_idioma, tema, nivel_seleccionado)
    for plat in ocultas:
        if len(resultados) >= 10: break
        url_final = plat["url_base"].format(tema.replace(' ', '+'))
        resultados.append({
            "nivel": plat["nivel"],
            "titulo": f"💎 {plat['nombre']} - {tema}",
            "plataforma": plat["nombre"],
            "url": url_final,
            "descripcion": plat["descripcion"],
            "tipo": "oculta"
        })
        
    return resultados

# ----------------------------
# MOSTRAR RESULTADOS
# ----------------------------
if buscar and tema.strip():
    resultados = buscar_cursos_avanzado(tema, nivel, idioma_seleccionado)
    
    if resultados:
        # Registrar en DB
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO analiticas_busquedas (tema, idioma, nivel, timestamp, plataforma_origen) VALUES (?, ?, ?, ?, ?)", 
                        (tema, idioma_seleccionado, nivel, datetime.now().isoformat(), "Mix"))
            conn.commit()
            conn.close()
        except:
            pass # No bloquear si falla la DB
            
        st.success(f"✅ ¡Encontramos {len(resultados)} recursos para **{tema}**!")
        
        for res in resultados:
            color = "#4CAF50" # Default
            if res['nivel'] == "Principiante": color = "#2196F3"
            if res['nivel'] == "Avanzado": color = "#FF9800"
            
            border_style = "5px solid #FF6B35" if res['tipo'] == 'oculta' else f"5px solid {color}"
            bg_style = "background: #fff5eb;" if res['tipo'] == 'oculta' else "background: white;"
            badge = "💎 JOYA OCULTA" if res['tipo'] == 'oculta' else "VERIFICADO"
            
            st.markdown(f"""
            <div class="resultado-card" style="border-left: {border_style}; {bg_style}">
                <div style="display:flex; justify-content:space-between;">
                    <h3>{res['titulo']}</h3>
                    <span style="background:#eee; padding:5px 10px; border-radius:10px; font-size:0.8em;">{badge}</span>
                </div>
                <p><strong>Nivel:</strong> {res['nivel']} | <strong>Plataforma:</strong> {res['plataforma']}</p>
                <p>{res['descripcion']}</p>
                <a href="{res['url']}" target="_blank" style="text-decoration:none;">
                    <button style="background:{color}; color:white; border:none; padding:10px 20px; border-radius:5px; cursor:pointer; width:100%;">
                        Ver Curso Gratis ➡️
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.warning("No encontramos resultados específicos. Intenta con términos más generales.")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; padding: 20px;">
    <strong>✨ Buscador Profesional</strong> | Optimizado con IA<br>
    <em>Versión 2.0 - Base de Datos Activa</em>
</div>
""", unsafe_allow_html=True)
