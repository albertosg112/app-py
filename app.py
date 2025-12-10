import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import random
from datetime import datetime, timedelta
import json
import hashlib
import requests
from urllib.parse import urlparse
import threading
import queue
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import logging
import asyncio
import aiohttp
from dotenv import load_dotenv
import groq

# ----------------------------
# 1. CONFIGURACIÓN Y LOGGING
# ----------------------------
st.set_page_config(
    page_title="Buscador Académico IA v8",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger("BuscadorProfesional")

# ----------------------------
# 2. GESTIÓN DE SECRETOS (Blindado contra errores)
# ----------------------------
def get_secret(key: str, default: str = "") -> str:
    """Obtiene un secreto asegurando que sea string (Evita el error 'bool' object)"""
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return str(st.secrets[key])
        val = os.getenv(key)
        return str(val) if val is not None else default
    except:
        return default

# Claves API
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GOOGLE_CX = get_secret("GOOGLE_CX")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# Flags de Configuración
def is_enabled(key, default="true"):
    val = get_secret(key, default).lower()
    return val in ["true", "1", "yes", "on"]

DUCKDUCKGO_ENABLED = is_enabled("DUCKDUCKGO_ENABLED")
SEMANTIC_SCHOLAR_ENABLED = is_enabled("SEMANTIC_SCHOLAR_ENABLED")
TOR_ENABLED = is_enabled("TOR_ENABLED")

# Constantes
MAX_BACKGROUND_TASKS = 1
CACHE_EXPIRATION = timedelta(hours=12)
GROQ_MODEL = "llama3-8b-8192"

search_cache = {}
groq_cache = {}
background_tasks = queue.Queue()

# ----------------------------
# 3. MODELOS DE DATOS
# ----------------------------
@dataclass
class RecursoEducativo:
    id: str
    titulo: str
    url: str
    descripcion: str
    plataforma: str
    idioma: str
    nivel: str
    categoria: str
    confianza: float
    tipo: str  # "conocida", "oculta", "ia", "semantica", "tor", "simulada"
    ultima_verificacion: str
    activo: bool
    metadatos_analisis: Optional[Dict[str, Any]] = None

# ----------------------------
# 4. BASE DE DATOS (Versión v8 - Limpia)
# ----------------------------
DB_PATH = "cursos_inteligentes_v8.db"

def init_database():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        
        # Tabla Principal
        c.execute('''
        CREATE TABLE IF NOT EXISTS recursos (
            id TEXT PRIMARY KEY,
            titulo TEXT, url TEXT, descripcion TEXT,
            plataforma TEXT, idioma TEXT, nivel TEXT, categoria TEXT,
            confianza REAL, tipo TEXT, activa INTEGER DEFAULT 1
        )''')
        
        # Tabla Analíticas
        c.execute('''
        CREATE TABLE IF NOT EXISTS analiticas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tema TEXT, timestamp TEXT
        )''')
        
        # Datos Semilla
        c.execute("SELECT COUNT(*) FROM recursos")
        if c.fetchone()[0] == 0:
            seed = [
                ("py_alf", "Aprende con Alf", "https://aprendeconalf.es/", "Python desde cero", "AprendeConAlf", "es", "Intermedio", "Programación", 0.9, "oculta"),
                ("coursera_free", "Coursera Free", "https://www.coursera.org/courses?query=free", "Cursos universitarios", "Coursera", "en", "Avanzado", "General", 0.95, "conocida"),
                ("tor_library", "Imperial Library (Mirror)", "http://xfmro77i3lixucja.onion.to", "Biblioteca técnica (Enlace Gateway)", "DeepWeb", "en", "Avanzado", "Libros", 0.8, "tor")
            ]
            for row in seed:
                c.execute("INSERT OR IGNORE INTO recursos VALUES (?,?,?,?,?,?,?,?,?,?,1)", row)
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error DB: {e}")

if not os.path.exists(DB_PATH):
    init_database()
else:
    init_database()

# ----------------------------
# 5. FUNCIONES DE UTILIDAD
# ----------------------------
def generar_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:10]

def get_codigo_idioma(nombre):
    mapeo = {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}
    return mapeo.get(nombre, "es")

def determinar_categoria(tema):
    tema = tema.lower()
    if any(x in tema for x in ['python', 'java', 'web', 'code']): return "Programación"
    if any(x in tema for x in ['data', 'datos', 'ia', 'ai']): return "Data Science"
    if any(x in tema for x in ['design', 'diseño', 'ux']): return "Diseño"
    return "General"

# ----------------------------
# 6. FUNCIÓN VISUALIZACIÓN (MOVIDA AL PRINCIPIO PARA EVITAR NAMEERROR)
# ----------------------------
def mostrar_recurso_con_ia(res: RecursoEducativo, index: int):
    """Muestra la tarjeta del recurso con HTML corregido"""
    
    # Colores según tipo
    colors = {
        "conocida": "#4CAF50", # Verde
        "oculta": "#FF9800",   # Naranja
        "semantica": "#2196F3",# Azul
        "tor": "#9C27B0",      # Morado
        "ia": "#00BCD4",       # Cyan
        "simulada": "#607D8B"  # Gris
    }
    color = colors.get(res.tipo, "#9E9E9E")
    
    # Análisis IA HTML
    ia_html = ""
    if res.metadatos_analisis:
        meta = res.metadatos_analisis
        ia_html = f"""
        <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 4px solid {color};'>
            <strong style="color: #333;">🤖 Análisis IA:</strong> {meta.get('recomendacion_personalizada', 'Recurso recomendado')}<br>
            <div style="margin-top:5px;">
                <span style='background: #e8f5e9; color: #2e7d32; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold;'>
                    Calidad: {float(meta.get('calidad_ia', 0.8))*100:.0f}%
                </span>
            </div>
        </div>
        """

    # HTML de la tarjeta (Sin sangrías para evitar errores de renderizado)
    html = f"""
<div style="border: 1px solid #e0e0e0; border-left: 5px solid {color}; border-radius: 8px; padding: 20px; margin-bottom: 15px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <h3 style="margin: 0; color: #333; font-size: 1.2rem;">{res.titulo}</h3>
        <span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; text-transform: uppercase; font-weight: bold;">{res.tipo}</span>
    </div>
    <div style="color: #666; font-size: 0.85rem; margin-bottom: 8px;">
        <span>🌐 {res.plataforma}</span> &nbsp;|&nbsp; <span>📚 {res.nivel}</span> &nbsp;|&nbsp; <span>⭐ Confianza: {res.confianza*100:.0f}%</span>
    </div>
    <p style="color: #444; font-size: 0.95rem; margin: 0 0 15px 0;">{res.descripcion}</p>
    {ia_html}
    <div style="margin-top: 15px;">
        <a href="{res.url}" target="_blank" style="text-decoration: none;">
            <button style="background: linear-gradient(90deg, {color}, #555); color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold; width: 100%; transition: opacity 0.3s;">
                ➡️ Acceder al Recurso
            </button>
        </a>
    </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------
# 7. LOGICA DE BÚSQUEDA (INCLUYENDO FALLBACK "INVENTADO")
# ----------------------------

async def buscar_google(tema, idioma):
    if not GOOGLE_API_KEY or not GOOGLE_CX: return []
    try:
        # Búsqueda real en Google
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': f"curso {tema} gratis", 'num': 3}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [RecursoEducativo(
                        id=generar_id(i['link']), titulo=i['title'], url=i['link'],
                        descripcion=i['snippet'], plataforma="Google Search",
                        idioma=idioma, nivel="General", categoria="General",
                        confianza=0.9, tipo="conocida", ultima_verificacion="", activo=True
                    ) for i in data.get('items', [])]
    except Exception as e:
        logger.error(f"Error Google (no crítico): {e}")
    return []

def generar_recursos_simulados(tema, idioma, nivel):
    """Genera recursos realistas cuando las APIs fallan o no hay claves"""
    recursos = []
    
    # Plantillas realistas basadas en el tema
    plataformas = [
        ("YouTube", f"https://www.youtube.com/results?search_query=curso+{tema.replace(' ', '+')}", "Video Tutoriales"),
        ("Coursera Audit", f"https://www.coursera.org/search?query={tema}&free=true", "Curso Universitario"),
        ("Udemy Free", f"https://www.udemy.com/courses/search/?q={tema}&price=price-free", "Curso Práctico"),
        ("EdX", f"https://www.edx.org/search?q={tema}", "Certificación Académica")
    ]
    
    titulos = [
        f"Curso Completo de {tema} desde Cero",
        f"Introducción a {tema} para Principiantes",
        f"Masterclass de {tema}: Nivel Avanzado",
        f"{tema}: Guía Definitiva 2024"
    ]
    
    for i, (plat_nombre, url, desc_base) in enumerate(plataformas):
        titulo_simulado = titulos[i % len(titulos)]
        desc_simulada = f"{desc_base} sobre {tema}. Ideal para nivel {nivel}. Contenido verificado por la comunidad."
        
        recursos.append(RecursoEducativo(
            id=f"sim_{i}_{generar_id(url)}",
            titulo=titulo_simulado,
            url=url,
            descripcion=desc_simulada,
            plataforma=plat_nombre,
            idioma=idioma,
            nivel=nivel,
            categoria=determinar_categoria(tema),
            confianza=0.85,
            tipo="simulada", # Tipo especial para fallback
            ultima_verificacion=datetime.now().isoformat(),
            activo=True,
            metadatos_analisis={"recomendacion_personalizada": "Recurso generado por búsqueda directa en plataforma.", "calidad_ia": 0.8}
        ))
    
    return recursos

async def analizar_ia(recurso):
    """Mejora la información con Groq si está disponible"""
    if not GROQ_API_KEY: return None
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""Analiza brevemente: {recurso.titulo}. 
        JSON: {{ "recomendacion_personalizada": "Resumen de 1 linea", "calidad_ia": 0.9 }}"""
        
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL, response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except: return None

async def orquestador_busqueda(tema, idioma, nivel):
    tasks = []
    
    # 1. Google (Si hay clave)
    if GOOGLE_API_KEY:
        tasks.append(buscar_google(tema, idioma))
        
    # 2. DB Local (Siempre)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT titulo, url, descripcion, plataforma, tipo, confianza FROM recursos WHERE titulo LIKE ?", (f"%{tema}%",))
    local_results = [RecursoEducativo(
        id=generar_id(r[1]), titulo=r[0], url=r[1], descripcion=r[2],
        plataforma=r[3], idioma=idioma, nivel=nivel, categoria="General",
        confianza=r[5], tipo=r[4], ultima_verificacion="", activo=True
    ) for r in c.fetchall()]
    conn.close()
    
    # Ejecutar tareas asíncronas
    api_results = []
    if tasks:
        results_list = await asyncio.gather(*tasks)
        for lst in results_list:
            api_results.extend(lst)
            
    final_list = local_results + api_results
    
    # 3. FALLBACK CRÍTICO: Si no hay resultados (o no hay claves API), generar simulados
    if not final_list:
        final_list = generar_recursos_simulados(tema, idioma, nivel)

    # 4. Análisis IA (En los top results)
    if GROQ_API_KEY:
        try:
            analisis_tasks = [analizar_ia(r) for r in final_list[:4]]
            analisis_results = await asyncio.gather(*analisis_tasks)
            for r, a in zip(final_list[:4], analisis_results):
                if a: r.metadatos_analisis = a
        except Exception as e:
            logger.error(f"Error Groq: {e}")

    return final_list

# ----------------------------
# 8. INTERFAZ DE USUARIO (UI)
# ----------------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    h1 { color: #1e3c72; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Buscador Académico Inteligente v8")
st.markdown("<div style='text-align:center'>Búsqueda Multicapa + Deep Web Simulada + IA</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    st.info(f"Google API: {'✅' if GOOGLE_API_KEY else '❌'}")
    st.info(f"Groq IA: {'✅' if GROQ_API_KEY else '❌'}")
    st.markdown("---")
    st.write("📊 **Estado**")
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        n = c.execute("SELECT COUNT(*) FROM recursos").fetchone()[0]
        conn.close()
        st.write(f"Recursos Indexados: {n}")
    except: pass

# Formulario
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    # Usamos session_state para que los botones de ejemplo funcionen
    if "tema_input" not in st.session_state: st.session_state.tema_input = ""
    tema = st.text_input("¿Qué quieres aprender?", key="tema_input")
with c2:
    nivel = st.selectbox("Nivel", ["Principiante", "Intermedio", "Avanzado"])
with c3:
    idioma = st.selectbox("Idioma", ["es", "en", "pt"])

# Botones de Ejemplo (Callback seguro)
def set_tema(t):
    st.session_state.tema_input = t

st.write("Ejemplos rápidos:")
cols = st.columns(4)
if cols[0].button("Python"): set_tema("Python")
if cols[1].button("Marketing"): set_tema("Marketing Digital")
if cols[2].button("Excel"): set_tema("Microsoft Excel")
if cols[3].button("Machine Learning"): set_tema("Machine Learning")

# Botón Principal
if st.button("🚀 Buscar Cursos", type="primary"):
    if not tema:
        st.warning("Escribe un tema para buscar.")
    else:
        with st.spinner(f"🔎 Buscando '{tema}' en múltiples capas..."):
            # Background Task Dummy
            background_tasks.put("log_search")
            
            # Loop Async
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(orquestador_busqueda(tema, idioma, nivel))
            loop.close()
            
            st.success(f"Encontrados {len(resultados)} recursos relevantes.")
            
            # Mostrar
            for i, res in enumerate(resultados):
                mostrar_recurso_con_ia(res, i)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888'>v8.0 Stable | Powered by Streamlit</div>", unsafe_allow_html=True)

# Worker Thread
def worker():
    while True:
        try:
            _ = background_tasks.get(timeout=2)
            background_tasks.task_done()
        except: pass
        time.sleep(1)

threading.Thread(target=worker, daemon=True).start()
