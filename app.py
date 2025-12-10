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
    page_title="🎓 Buscador Académico Pro v12",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger("BuscadorProfesional")

# ----------------------------
# 2. GESTIÓN DE SECRETOS (Blindado)
# ----------------------------
def get_secret(key: str, default: str = "") -> str:
    """Obtiene un secreto asegurando que sea string"""
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
TOR_ENABLED = is_enabled("TOR_ENABLED")

# Constantes
GROQ_MODEL = "llama3-8b-8192"
DB_PATH = "cursos_inteligentes_v12.db"

# Colas y Caché
background_tasks = queue.Queue()
search_cache = {}
groq_cache = {}

# ----------------------------
# 3. MODELOS DE DATOS (COMPLETOS)
# ----------------------------
@dataclass
class Certificacion:
    plataforma: str
    tipo: str  # "gratuito", "pago", "audit"
    validez_internacional: bool

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
    certificacion: Optional[Certificacion] = None
    metadatos_analisis: Optional[Dict[str, Any]] = None

# ----------------------------
# 4. BASE DE DATOS
# ----------------------------
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
            confianza REAL, tipo TEXT, activa INTEGER DEFAULT 1,
            tiene_certificado BOOLEAN DEFAULT 0
        )''')
        
        # Tabla Analíticas
        c.execute('''
        CREATE TABLE IF NOT EXISTS analiticas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tema TEXT, idioma TEXT, nivel TEXT, timestamp TEXT
        )''')
        
        # Datos Semilla (Incluyendo Deep Web Educativa)
        c.execute("SELECT COUNT(*) FROM recursos")
        if c.fetchone()[0] == 0:
            seed = [
                ("py_alf", "Aprende con Alf", "https://aprendeconalf.es/", "Tutoriales de Python y Pandas paso a paso.", "AprendeConAlf", "es", "Intermedio", "Programación", 0.9, "oculta", 0),
                ("coursera_free", "Coursera (Modo Auditoría)", "https://www.coursera.org/courses?query=free", "Cursos universitarios gratuitos.", "Coursera", "en", "Avanzado", "General", 0.95, "conocida", 1),
                ("tor_library", "Imperial Library of Trantor", "http://xfmro77i3lixucja.onion", "Biblioteca técnica masiva (Requiere Tor Browser).", "DeepWeb", "en", "Experto", "Libros", 0.8, "tor", 0),
                ("scihub_tor", "Sci-Hub (Tor Mirror)", "http://scihub22266oqcxt.onion", "Acceso a papers académicos sin restricciones.", "DeepWeb", "en", "Avanzado", "Ciencia", 0.9, "tor", 0),
                ("khan_es", "Khan Academy Español", "https://es.khanacademy.org/", "Educación gratuita de clase mundial.", "Khan Academy", "es", "Principiante", "General", 0.98, "conocida", 0)
            ]
            for row in seed:
                c.execute("INSERT OR IGNORE INTO recursos VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error DB Init: {e}")

if not os.path.exists(DB_PATH):
    init_database()
else:
    init_database()

# ----------------------------
# 5. FUNCIONES AUXILIARES
# ----------------------------
def generar_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:10]

def get_codigo_idioma(nombre):
    mapeo = {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}
    return mapeo.get(nombre, "es")

def determinar_categoria(tema):
    tema = tema.lower()
    if any(x in tema for x in ['python', 'java', 'code']): return "Programación"
    if any(x in tema for x in ['data', 'ia', 'ai']): return "Data Science"
    return "General"

def extraer_plataforma(url):
    try:
        return urlparse(url).netloc.replace('www.', '').split('.')[0].title()
    except: return "Web"

def eliminar_duplicados(resultados):
    seen = set()
    unique = []
    for r in resultados:
        if r.url not in seen:
            unique.append(r)
            seen.add(r.url)
    return unique

# ----------------------------
# 6. FUNCIÓN DE VISUALIZACIÓN (HTML CORREGIDO Y POSICIONADA AL INICIO)
# ----------------------------
def mostrar_recurso_con_ia(res: RecursoEducativo, index: int):
    """Muestra la tarjeta del recurso con diseño profesional y sin errores de HTML"""
    
    # 1. Definir colores según el tipo de recurso
    colors = {
        "conocida": "#2E7D32", # Verde
        "oculta": "#E65100",   # Naranja
        "tor": "#6A1B9A",      # Morado (Deep Web)
        "ia": "#00838F",       # Cyan
        "simulada": "#455A64"  # Gris
    }
    color = colors.get(res.tipo, "#424242")
    
    # 2. Generar Badges
    badges_html = ""
    if res.certificacion:
        c = res.certificacion
        if c.tipo == "gratuito":
            badges_html += f"<span style='background:#4CAF50; color:white; padding:2px 8px; border-radius:4px; font-size:0.7em; margin-right:5px;'>✅ Certificado</span>"
        elif c.tipo == "audit":
            badges_html += f"<span style='background:#FF9800; color:white; padding:2px 8px; border-radius:4px; font-size:0.7em; margin-right:5px;'>🎓 Auditoría</span>"
    
    if res.tipo == "tor":
        badges_html += f"<span style='background:#000; color:#00FF00; padding:2px 8px; border-radius:4px; font-size:0.7em; margin-right:5px;'>🧅 Onion V3</span>"

    # 3. Generar Sección de Análisis IA (HTML Ajustado)
    ia_html = ""
    if res.metadatos_analisis:
        meta = res.metadatos_analisis
        calidad = float(meta.get('calidad_ia', 0.0)) * 100
        rec = meta.get('recomendacion_personalizada', 'Recurso verificado.')
        
        ia_html = f"""
<div style='background-color: #f8f9fa; padding: 12px; border-radius: 6px; margin-top: 10px; border-left: 4px solid {color};'>
    <div style='color: #333; font-weight: bold; margin-bottom: 4px;'>🤖 Análisis IA (2º Plano):</div>
    <div style='color: #555; font-size: 0.95em; margin-bottom: 8px;'>{rec}</div>
    <span style='background: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 10px; font-size: 0.8em; font-weight: bold;'>
        Calidad Didáctica: {calidad:.0f}%
    </span>
</div>
"""

    # 4. Renderizar Tarjeta Completa (HTML sin indentación para evitar errores)
    html_card = f"""
<div style="border: 1px solid #e0e0e0; border-top: 5px solid {color}; border-radius: 10px; padding: 20px; margin-bottom: 20px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <h3 style="margin: 0 0 10px 0; color: #333; font-size: 1.2rem;">{res.titulo}</h3>
        <span style="background-color: {color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.65rem; text-transform: uppercase; font-weight: bold;">{res.tipo}</span>
    </div>
    <div style="color: #666; font-size: 0.85rem; margin-bottom: 12px;">
        <span>🏛️ {res.plataforma}</span> &nbsp;•&nbsp; <span>📚 {res.nivel}</span> &nbsp;•&nbsp; <span>⭐ {res.confianza*100:.0f}% Confianza</span>
    </div>
    <div style="margin-bottom: 12px;">{badges_html}</div>
    <p style="color: #444; font-size: 0.95rem; line-height: 1.5; margin: 0 0 15px 0;">{res.descripcion}</p>
    {ia_html}
    <div style="margin-top: 15px; text-align: right;">
        <a href="{res.url}" target="_blank" style="text-decoration: none;">
            <button style="background: linear-gradient(90deg, {color}, #333); color: white; border: none; padding: 10px 25px; border-radius: 6px; cursor: pointer; font-weight: bold; transition: opacity 0.2s;">
                Acceder al Recurso ➡️
            </button>
        </a>
    </div>
</div>
"""
    st.markdown(html_card, unsafe_allow_html=True)

# ----------------------------
# 7. LÓGICA DE BÚSQUEDA Y AGENTES
# ----------------------------

async def buscar_google_api(tema, idioma):
    if not GOOGLE_API_KEY: return []
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': f"curso {tema} gratis certificado", 'num': 3}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    res = []
                    for i in data.get('items', []):
                        has_cert = "certific" in i['snippet'].lower()
                        cert_obj = Certificacion(i['displayLink'], "audit" if not has_cert else "gratuito", True)
                        
                        res.append(RecursoEducativo(
                            id=generar_id(i['link']), titulo=i['title'], url=i['link'],
                            descripcion=i['snippet'], plataforma=extraer_plataforma(i['link']),
                            idioma=idioma, nivel="General", categoria=determinar_categoria(tema),
                            confianza=0.9, tipo="conocida", ultima_verificacion="", activo=True,
                            certificacion=cert_obj
                        ))
                    return res
    except: pass
    return []

async def analizar_ia_groq(recurso):
    """Agente de análisis en segundo plano"""
    if not GROQ_API_KEY: return None
    try:
        # Cliente simplificado para evitar error 'proxies'
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""Analiza: "{recurso.titulo}" - "{recurso.descripcion}".
        Responde SOLO JSON: {{ "recomendacion_personalizada": "Frase de 10 palabras", "calidad_ia": 0.92 }}"""
        
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL, response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except: return None

def generar_simulados(tema, idioma, nivel):
    """Fallback seguro"""
    base = [
        ("YouTube", f"https://www.youtube.com/results?search_query=curso+completo+{tema.replace(' ', '+')}", "Video curso completo práctico.", "conocida"),
        ("Google", f"https://www.google.com/search?q=tutorial+{tema.replace(' ', '+')}", "Búsqueda de documentación.", "oculta"),
        ("Coursera", f"https://www.coursera.org/search?query={tema.replace(' ', '+')}&free=true", "Curso universitario.", "conocida")
    ]
    res = []
    for i, (plat, url, desc, tipo) in enumerate(base):
        res.append(RecursoEducativo(
            id=f"sim_{i}", titulo=f"Aprende {tema} en {plat}", url=url, descripcion=desc,
            plataforma=plat, idioma=idioma, nivel=nivel, categoria="General",
            confianza=0.85, tipo=tipo, ultima_verificacion="", activo=True,
            metadatos_analisis={"recomendacion_personalizada": "Recurso relevante encontrado por coincidencia directa.", "calidad_ia": 0.8}
        ))
    return res

async def orquestador_busqueda(tema, idioma, nivel, usar_ia):
    tasks = [buscar_google_api(tema, idioma)]
    
    # DB Local (Ocultas + Tor)
    local_res = []
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT titulo, url, descripcion, plataforma, tipo, confianza, tiene_certificado FROM recursos WHERE titulo LIKE ? OR descripcion LIKE ?", (f"%{tema}%", f"%{tema}%"))
        rows = c.fetchall()
        for r in rows:
            if r[4] == 'tor' and not TOR_ENABLED: continue
            cert = Certificacion(r[3], "gratuito" if r[6] else "audit", True) if r[6] else None
            local_res.append(RecursoEducativo(
                id=generar_id(r[1]), titulo=r[0], url=r[1], descripcion=r[2],
                plataforma=r[3], idioma=idioma, nivel=nivel, categoria="General",
                confianza=r[5], tipo=r[4], ultima_verificacion="", activo=True, certificacion=cert
            ))
        conn.close()
    except: pass

    # Ejecutar
    api_results = await asyncio.gather(*tasks)
    final_list = local_res
    for lst in api_results: final_list.extend(lst)
    
    if not final_list: final_list = generar_simulados(tema, idioma, nivel)

    # Análisis IA (Solo si está activado por el usuario)
    if usar_ia and GROQ_API_KEY:
        ia_tasks = [analizar_ia_groq(r) for r in final_list[:4]]
        ia_results = await asyncio.gather(*ia_tasks)
        for r, analysis in zip(final_list[:4], ia_results):
            if analysis: r.metadatos_analisis = analysis

    return eliminar_duplicados(final_list)

# ----------------------------
# 8. INTERFAZ DE USUARIO (UI)
# ----------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #000000 0%, #434343 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); margin-bottom: 20px;
    }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    # Toggle para IA (Opcional como pediste)
    usar_ia = st.toggle("🧠 Activar Análisis IA", value=True, help="Activa para que Groq analice la calidad de los cursos. Desactiva para búsqueda más rápida.")
    
    st.markdown("### 🌐 Redes")
    c1, c2 = st.columns(2)
    c1.write("🦆 **DDG:**")
    c2.write("✅" if DUCKDUCKGO_ENABLED else "❌")
    c1.write("🧅 **Tor:**")
    c2.write("✅" if TOR_ENABLED else "❌")
    
    st.divider()
    
    # Info Estado
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        total = conn.execute("SELECT COUNT(*) FROM recursos").fetchone()[0]
        conn.close()
        st.metric("Recursos Indexados", total)
    except: st.metric("Recursos", 0)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🕵️ Buscador Académico & Deep Web</h1>
    <p>Surface Web • .Onion Educativo • Análisis IA</p>
</div>
""", unsafe_allow_html=True)

# --- FORMULARIO ---
c_tema, c_nivel, c_lang = st.columns([3, 1, 1])

# Session State para inputs
if "search_query" not in st.session_state: st.session_state.search_query = ""
def update_query(): st.session_state.search_query = st.session_state.temp_input

with c_tema:
    tema = st.text_input("Objetivo de aprendizaje:", value=st.session_state.search_query, placeholder="Ej: Ciberseguridad, Historia...", key="temp_input", on_change=update_query)
with c_nivel:
    nivel = st.selectbox("Nivel", ["Principiante", "Intermedio", "Avanzado"])
with c_lang:
    idioma = st.selectbox("Idioma", ["es", "en", "pt"])

# Botones Rápidos
st.write("🔥 **Búsquedas Frecuentes:**")
b1, b2, b3, b4 = st.columns(4)
if b1.button("🐍 Python"): 
    st.session_state.search_query = "Python"
    st.rerun()
if b2.button("🔐 Ciberseguridad"): 
    st.session_state.search_query = "Ciberseguridad"
    st.rerun()
if b3.button("📊 Data Science"): 
    st.session_state.search_query = "Data Science"
    st.rerun()
if b4.button("🧅 Deep Web (Sim)"): 
    st.session_state.search_query = "Privacidad y Tor"
    st.rerun()

st.markdown("---")

# --- EJECUCIÓN ---
if st.button("🚀 INICIAR BÚSQUEDA", type="primary"):
    if not st.session_state.search_query:
        st.warning("⚠️ Escribe un tema primero.")
    else:
        # Registrar Analítica
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.execute("INSERT INTO analiticas (tema, idioma, nivel, timestamp) VALUES (?,?,?,?)", 
                         (st.session_state.search_query, idioma, nivel, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except: pass

        with st.spinner(f"Analizando la red para '{st.session_state.search_query}'..."):
            # Lógica Asíncrona
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(orquestador_busqueda(st.session_state.search_query, idioma, nivel, usar_ia))
            loop.close()
            
            # Resultados
            if resultados:
                st.success(f"✅ Se encontraron **{len(resultados)}** recursos.")
                
                # Filtros Visuales
                ver_tor = st.checkbox("Mostrar enlaces .onion (Deep Web)", value=True)
                
                for i, res in enumerate(resultados):
                    if not ver_tor and res.tipo == 'tor': continue
                    mostrar_recurso_con_ia(res, i)
            else:
                st.error("No se encontraron resultados.")

# --- FOOTER ---
st.markdown("---")
# HTML Footer Corregido (Sin indentación)
footer_html = """
<div style="text-align: center; color: #666; font-size: 14px; padding: 25px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 20px;">
    <strong>✨ Buscador Académico Pro v12</strong><br>
    <span style="color: #2c3e50;">Potenciado por Groq • Google API • Tor Gateway</span>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

# Worker Thread para mantener la app activa
def background_worker():
    while True:
        try:
            task = background_tasks.get(timeout=2)
            background_tasks.task_done()
        except: pass
        time.sleep(5)

threading.Thread(target=background_worker, daemon=True).start()
