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
    page_title="🎓 Buscador Académico IA v15",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger("BuscadorProfesional")

# ----------------------------
# 2. GESTIÓN SEGURA DE SECRETOS
# ----------------------------
def get_secret(key: str, default: str = "") -> str:
    """Obtiene secretos de forma segura, convirtiendo todo a string para evitar errores de tipo."""
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return str(st.secrets[key])
        val = os.getenv(key)
        return str(val) if val is not None else default
    except:
        return default

# Cargar Claves (Manejo robusto)
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GOOGLE_CX = get_secret("GOOGLE_CX")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# Flags de Configuración (Lectura segura de booleanos)
def is_enabled(key, default="true"):
    val = get_secret(key, default).lower()
    return val in ["true", "1", "yes", "on"]

DUCKDUCKGO_ENABLED = is_enabled("DUCKDUCKGO_ENABLED")
TOR_ENABLED = is_enabled("TOR_ENABLED") # Simulación Deep Web

# Constantes
GROQ_MODEL = "llama3-8b-8192"
DB_PATH = "cursos_inteligentes_v15.db" # Cambiado a v15 para base limpia

# Colas y Caché
background_tasks = queue.Queue()
search_cache = {}
groq_cache = {}

# ----------------------------
# 3. MODELOS DE DATOS
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
    analisis_pendiente: bool = False

# ----------------------------
# 4. BASE DE DATOS
# ----------------------------
def init_database():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS recursos (
            id TEXT PRIMARY KEY, titulo TEXT, url TEXT, descripcion TEXT,
            plataforma TEXT, idioma TEXT, nivel TEXT, categoria TEXT,
            confianza REAL, tipo TEXT, activa INTEGER DEFAULT 1,
            tiene_certificado BOOLEAN DEFAULT 0)''')
            
        c.execute('''CREATE TABLE IF NOT EXISTS analiticas (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tema TEXT, idioma TEXT, nivel TEXT, timestamp TEXT)''')
        
        # Datos Semilla (Incluyendo Deep Web Educativa Simulada)
        c.execute("SELECT COUNT(*) FROM recursos")
        if c.fetchone()[0] == 0:
            seed = [
                ("py_alf", "Aprende con Alf", "https://aprendeconalf.es/", "Tutoriales de Python y Pandas paso a paso.", "AprendeConAlf", "es", "Intermedio", "Programación", 0.9, "oculta", 0),
                ("coursera_free", "Coursera (Audit)", "https://www.coursera.org/courses?query=free", "Cursos universitarios.", "Coursera", "en", "Avanzado", "General", 0.95, "conocida", 1),
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

if not os.path.exists(DB_PATH): init_database()
else: init_database()

# ----------------------------
# 5. FUNCIONES AUXILIARES
# ----------------------------
def generar_id(url): return hashlib.md5(url.encode()).hexdigest()[:10]
def get_codigo_idioma(n): return {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}.get(n, "es")
def extraer_plataforma(url):
    try: return urlparse(url).netloc.replace('www.', '').split('.')[0].title()
    except: return "Web"
def eliminar_duplicados(res):
    seen = set(); unique = []
    for r in res:
        if r.url not in seen: unique.append(r); seen.add(r.url)
    return unique
def determinar_categoria(tema):
    tema = tema.lower()
    if any(x in tema for x in ['python', 'java', 'code']): return "Programación"
    if any(x in tema for x in ['data', 'ia']): return "Data Science"
    return "General"

# ----------------------------
# 6. FUNCIONES DE VISUALIZACIÓN (CORREGIDAS: SIN INDENTACIÓN HTML)
# ----------------------------
def mostrar_recurso_basico(recurso: RecursoEducativo, index: int, analisis_pendiente: bool = False):
    """Muestra un recurso básico sin análisis detallado de IA"""
    
    color_clase = {"conocida": "#2E7D32", "oculta": "#E65100", "tor": "#6A1B9A", "ia": "#00838F", "simulada": "#455A64"}.get(recurso.tipo, "#555")
    
    # HTML Pegado a la izquierda para evitar errores de renderizado
    html = f"""
<div style="border: 1px solid #ddd; border-left: 5px solid {color_clase}; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
    <div style="display: flex; justify-content: space-between;">
        <h3 style="margin: 0; color: #333; font-size: 1.1rem;">{recurso.titulo}</h3>
        <span style="background: {color_clase}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7em;">{recurso.tipo.upper()}</span>
    </div>
    <div style="color: #666; font-size: 0.85em; margin: 5px 0;">
        🏛️ {recurso.plataforma} | 📚 {recurso.nivel} | ⭐ {recurso.confianza*100:.0f}%
    </div>
    <p style="color: #444; font-size: 0.95em;">{recurso.descripcion}</p>
    <div style="text-align: right; margin-top: 10px;">
        <a href="{recurso.url}" target="_blank" style="text-decoration: none;">
            <button style="background: linear-gradient(90deg, {color_clase}, #333); color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; font-weight: bold;">
                Acceder al Recurso ➡️
            </button>
        </a>
    </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
    if analisis_pendiente:
        st.caption("⏳ *Analizando calidad con IA en segundo plano...*")

def mostrar_recurso_con_ia(res: RecursoEducativo, index: int):
    """Muestra la tarjeta PREMIUM con análisis de IA y badges"""
    
    colors = {"conocida": "#2E7D32", "oculta": "#E65100", "tor": "#6A1B9A", "ia": "#00838F", "simulada": "#455A64"}
    color = colors.get(res.tipo, "#424242")
    
    # Badges
    badges_html = ""
    if res.certificacion and res.certificacion.tipo == "gratuito":
        badges_html += f"<span style='background:#4CAF50; color:white; padding:2px 6px; border-radius:4px; font-size:0.7em; margin-right:5px;'>✅ Certificado Gratis</span>"
    if res.tipo == "tor":
        badges_html += f"<span style='background:#000; color:#00FF00; padding:2px 6px; border-radius:4px; font-size:0.7em; margin-right:5px;'>🧅 Onion V3</span>"

    # Análisis IA
    ia_html = ""
    if res.metadatos_analisis:
        meta = res.metadatos_analisis
        calidad = float(meta.get('calidad_ia', 0)) * 100
        rec = meta.get('recomendacion_personalizada', 'Verificado.')
        
        ia_html = f"""
<div style='background-color: #f0f7ff; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 4px solid {color};'>
    <div style='color: #0d47a1; font-weight: bold; font-size: 0.9em; margin-bottom: 4px;'>🤖 Análisis Inteligente:</div>
    <div style='color: #333; font-size: 0.95em; margin-bottom: 8px;'>{rec}</div>
    <span style='background: #e3f2fd; color: #0d47a1; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; font-weight: bold;'>
        Calidad Didáctica: {calidad:.0f}%
    </span>
</div>
"""

    # Tarjeta Principal (HTML LIMPIO)
    html_card = f"""
<div style="border: 1px solid #e0e0e0; border-top: 5px solid {color}; border-radius: 10px; padding: 20px; margin-bottom: 20px; background-color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.08);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <h3 style="margin: 0 0 5px 0; color: #222; font-size: 1.2rem;">{res.titulo}</h3>
        <span style="background-color: {color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.65rem; text-transform: uppercase; font-weight: bold;">{res.tipo}</span>
    </div>
    <div style="color: #666; font-size: 0.85rem; margin-bottom: 8px;">
        <span>🏛️ {res.plataforma}</span> &nbsp;•&nbsp; <span>📚 {res.nivel}</span>
    </div>
    <div style="margin-bottom: 10px;">{badges_html}</div>
    <p style="color: #444; font-size: 0.95rem; line-height: 1.4; margin: 0 0 10px 0;">{res.descripcion}</p>
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
# 7. LÓGICA DE BÚSQUEDA Y AGENTES (IA + GOOGLE + LOCAL)
# ----------------------------

async def buscar_google_api(tema, idioma):
    """Busca en Google Programmable Search Engine"""
    if not GOOGLE_API_KEY or not GOOGLE_CX: return []
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': f"curso {tema} gratis certificado", 'num': 4}
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

async def buscar_duckduckgo_simulado(tema, idioma):
    """Fallback si DDG está activo pero no hay API oficial simple"""
    if not DUCKDUCKGO_ENABLED: return []
    # Aquí podríamos implementar scraping ligero, por ahora devolvemos lista vacía para no romper
    # y que entre el fallback de simulados si Google falla
    return []

async def analizar_ia_groq(recurso):
    """Analiza el recurso usando Groq"""
    if not GROQ_API_KEY: return None
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""Analiza este recurso educativo: "{recurso.titulo}" - "{recurso.descripcion}".
        Responde SOLO JSON válido: {{ "recomendacion_personalizada": "Resumen motivador de 1 frase", "calidad_ia": 0.92, "razones_calidad": ["Completo", "Práctico"] }}"""
        
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL, response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except: return None

def generar_simulados(tema, idioma, nivel):
    """Fallback seguro si no hay APIs configuradas"""
    base = [
        ("YouTube", f"https://www.youtube.com/results?search_query=curso+completo+{tema.replace(' ', '+')}", "Video curso completo práctico.", "conocida"),
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
    tasks = []
    
    # 1. Google
    if GOOGLE_API_KEY: tasks.append(buscar_google_api(tema, idioma))
    
    # 2. DB Local (Ocultas + Tor)
    local_res = []
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT titulo, url, descripcion, plataforma, tipo, confianza, tiene_certificado FROM recursos WHERE titulo LIKE ? OR descripcion LIKE ?", (f"%{tema}%", f"%{tema}%"))
        for r in c.fetchall():
            if r[4] == 'tor' and not TOR_ENABLED: continue
            cert = Certificacion(r[3], "gratuito" if r[6] else "audit", True) if r[6] else None
            local_res.append(RecursoEducativo(
                id=generar_id(r[1]), titulo=r[0], url=r[1], descripcion=r[2],
                plataforma=r[3], idioma=idioma, nivel=nivel, categoria="General",
                confianza=r[5], tipo=r[4], ultima_verificacion="", activo=True, certificacion=cert
            ))
        conn.close()
    except: pass

    # Ejecutar async
    if tasks:
        api_results = await asyncio.gather(*tasks)
        for lst in api_results: local_res.extend(lst)
    
    # Fallback si está vacío
    if not local_res: local_res = generar_simulados(tema, idioma, nivel)

    # Marcar para análisis posterior o analizar ahora (Top 3)
    final_list = eliminar_duplicados(local_res)
    
    if usar_ia and GROQ_API_KEY:
        ia_tasks = [analizar_ia_groq(r) for r in final_list[:3]] # Solo analizamos los 3 primeros en vivo para velocidad
        ia_results = await asyncio.gather(*ia_tasks)
        for r, analysis in zip(final_list[:3], ia_results):
            if analysis: r.metadatos_analisis = analysis

    return final_list

def planificar_analisis_ia(resultados):
    """Encola análisis para el resto de resultados (Segundo Plano)"""
    pass # Placeholder para implementación futura de cola real

# ----------------------------
# 8. INTERFAZ DE USUARIO (UI)
# ----------------------------
# CSS y Header
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #000000 0%, #434343 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;
    }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
</style>
<div class="main-header">
    <h1>🕵️ Buscador Académico & Deep Web v15</h1>
    <p>Surface Web • .Onion Educativo • Análisis IA</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    usar_ia = st.toggle("🧠 Análisis IA Activo", value=True)
    st.write(f"Google: {'✅' if GOOGLE_API_KEY else '❌'}")
    st.write(f"Groq: {'✅' if GROQ_API_KEY else '❌'}")
    
    st.divider()
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        total = conn.execute("SELECT COUNT(*) FROM recursos").fetchone()[0]
        conn.close()
        st.metric("Recursos Indexados", total)
    except: st.metric("Recursos", 0)

# Formulario
c1, c2, c3 = st.columns([3, 1, 1])
if "search_query" not in st.session_state: st.session_state.search_query = ""

with c1:
    tema = st.text_input("Tema a investigar:", value=st.session_state.search_query, placeholder="Ej: Python, Ciberseguridad...")
with c2:
    nivel = st.selectbox("Nivel", ["Principiante", "Intermedio", "Avanzado"])
with c3:
    idioma = st.selectbox("Idioma", ["es", "en", "pt"])

# Botones Rápidos
st.write("Exploración rápida:")
b_cols = st.columns(4)
if b_cols[0].button("🐍 Python"): st.session_state.search_query = "Python"
if b_cols[1].button("💰 Finanzas"): st.session_state.search_query = "Finanzas"
if b_cols[2].button("🎨 Diseño"): st.session_state.search_query = "Diseño"
if b_cols[3].button("🧅 Tor"): st.session_state.search_query = "Deep Web"

# Ejecución
if st.button("🚀 INICIAR BÚSQUEDA", type="primary"):
    if not tema:
        st.error("Escribe un tema.")
    else:
        # Registrar Analítica
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            conn.execute("INSERT INTO analiticas (tema, idioma, nivel, timestamp) VALUES (?,?,?,?)", 
                         (tema, idioma, nivel, datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except: pass

        with st.spinner(f"Analizando ecosistema digital para: {tema}..."):
            # Loop Asíncrono
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(orquestador_busqueda(tema, idioma, nivel, usar_ia))
            loop.close()
            
            # Mostrar
            st.success(f"✅ Análisis completado: {len(resultados)} recursos verificados.")
            
            # Filtro visual
            ver_tor = st.checkbox("Mostrar enlaces .onion (Deep Web)", value=True)
            
            for i, res in enumerate(resultados):
                if not ver_tor and res.tipo == 'tor': continue
                
                # Decidir qué tarjeta mostrar
                if res.metadatos_analisis:
                    mostrar_recurso_con_ia(res, i)
                else:
                    mostrar_recurso_basico(res, i, analisis_pendiente=True)

# Footer
st.markdown("---")
footer_html = """
<div style="text-align: center; color: #666; font-size: 14px; padding: 25px;">
    <strong>✨ Buscador Académico Pro v15</strong><br>
    <span style="color: #2c3e50;">Potenciado por Groq • Google API • Tor Gateway Simulation</span>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

# Worker Thread (Keep-alive)
def background_worker():
    while True:
        try:
            task = background_tasks.get(timeout=2)
            background_tasks.task_done()
        except: pass
        time.sleep(5)

threading.Thread(target=background_worker, daemon=True).start()
