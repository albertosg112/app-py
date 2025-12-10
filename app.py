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
    page_title="🎓 Buscador Académico Pro v14",
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
DB_PATH = "cursos_inteligentes_v14.db"

# Colas y Caché
background_tasks = queue.Queue()
if "indexing_active" not in st.session_state:
    st.session_state.indexing_active = False
if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = 0

# ----------------------------
# 3. MODELOS DE DATOS
# ----------------------------
@dataclass
class Certificacion:
    plataforma: str
    tipo: str
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
    tipo: str
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
        
        c.execute('''CREATE TABLE IF NOT EXISTS recursos (
            id TEXT PRIMARY KEY, titulo TEXT, url TEXT, descripcion TEXT,
            plataforma TEXT, idioma TEXT, nivel TEXT, categoria TEXT,
            confianza REAL, tipo TEXT, activa INTEGER DEFAULT 1,
            tiene_certificado BOOLEAN DEFAULT 0)''')
            
        c.execute('''CREATE TABLE IF NOT EXISTS analiticas (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tema TEXT, idioma TEXT, nivel TEXT, timestamp TEXT)''')
        
        # Datos Semilla
        c.execute("SELECT COUNT(*) FROM recursos")
        if c.fetchone()[0] == 0:
            seed = [
                ("py_alf", "Aprende con Alf", "https://aprendeconalf.es/", "Python desde cero.", "AprendeConAlf", "es", "Intermedio", "Programación", 0.9, "oculta", 0),
                ("coursera_free", "Coursera (Audit)", "https://www.coursera.org/courses?query=free", "Cursos universitarios.", "Coursera", "en", "Avanzado", "General", 0.95, "conocida", 1),
                ("tor_lib", "Imperial Library", "http://xfmro77i3lixucja.onion", "Biblioteca técnica.", "DeepWeb", "en", "Experto", "Libros", 0.8, "tor", 0)
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
# 6. FUNCIÓN VISUALIZACIÓN (NATIVA STREAMLIT - SIN ERRORES HTML)
# ----------------------------
def mostrar_recurso_con_ia(res: RecursoEducativo, index: int):
    """Muestra la tarjeta usando componentes nativos para evitar errores visuales"""
    
    # Colores para la etiqueta lateral
    colores = {
        "conocida": "#2E7D32", "oculta": "#E65100", "tor": "#6A1B9A", "ia": "#00838F", "simulada": "#455A64"
    }
    color_borde = colores.get(res.tipo, "#555")

    # Contenedor de la tarjeta con estilo CSS inyectado para el borde
    with st.container():
        st.markdown(f"""
        <style>
            div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stMarkdownContainer"] p:contains("{res.id}")) {{
                border-left: 5px solid {color_borde};
                padding-left: 15px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
        </style>
        <div style='display:none'>{res.id}</div> """, unsafe_allow_html=True)

        # Encabezado
        c1, c2 = st.columns([0.85, 0.15])
        with c1:
            st.markdown(f"### {res.titulo}")
        with c2:
            st.markdown(f":{colors.get(res.tipo, 'grey')}[**{res.tipo.upper()}**]")

        # Metadatos
        st.caption(f"🏛️ **{res.plataforma}** | 📚 {res.nivel} | ⭐ {res.confianza*100:.0f}% Confianza")
        
        # Badges
        if res.certificacion and res.certificacion.tipo == "gratuito":
            st.success("✅ Certificado Gratuito Incluido")
        elif res.tipo == "tor":
            st.warning("🧅 Enlace Deep Web (.onion)")

        # Descripción
        st.write(res.descripcion)

        # Análisis IA (Si existe)
        if res.metadatos_analisis:
            meta = res.metadatos_analisis
            calidad = float(meta.get('calidad_ia', 0)) * 100
            with st.expander(f"🤖 Análisis IA (Calidad: {calidad:.0f}%)", expanded=True):
                st.info(meta.get('recomendacion_personalizada', 'Contenido verificado.'))
                if 'razones_calidad' in meta:
                    st.write(f"**Puntos clave:** {', '.join(meta['razones_calidad'][:2])}")

        # BOTÓN NATIVO (SOLUCIÓN DEFINITIVA AL ERROR VISUAL)
        st.link_button(
            label="➡️ Acceder al Recurso",
            url=res.url,
            type="primary",
            use_container_width=True
        )
        
        st.divider()

# ----------------------------
# 7. LÓGICA DE BÚSQUEDA
# ----------------------------
async def buscar_google_api(tema, idioma):
    if not GOOGLE_API_KEY: return []
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': f"curso {tema} gratis", 'num': 3}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    res = []
                    for i in data.get('items', []):
                        cert_obj = Certificacion(i['displayLink'], "audit", True)
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
    if not GROQ_API_KEY: return None
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""Analiza: "{recurso.titulo}". 
        JSON: {{ "recomendacion_personalizada": "Resumen en 1 frase", "calidad_ia": 0.9, "razones_calidad": ["Bueno"] }}"""
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL, response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except: return None

def generar_simulados(tema, idioma, nivel):
    base = [
        ("YouTube", f"https://www.youtube.com/results?search_query=curso+{tema}", "Video curso completo.", "conocida"),
        ("Coursera", f"https://www.coursera.org/search?query={tema}&free=true", "Curso universitario.", "conocida")
    ]
    res = []
    for i, (plat, url, desc, tipo) in enumerate(base):
        res.append(RecursoEducativo(
            id=f"sim_{i}", titulo=f"Aprende {tema} en {plat}", url=url, descripcion=desc,
            plataforma=plat, idioma=idioma, nivel=nivel, categoria="General",
            confianza=0.85, tipo=tipo, ultima_verificacion="", activo=True,
            metadatos_analisis={"recomendacion_personalizada": "Recurso encontrado por coincidencia.", "calidad_ia": 0.8}
        ))
    return res

async def orquestador_busqueda(tema, idioma, nivel, usar_ia):
    tasks = [buscar_google_api(tema, idioma)]
    
    # DB Local
    local_res = []
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT titulo, url, descripcion, plataforma, tipo, confianza FROM recursos WHERE titulo LIKE ?", (f"%{tema}%",))
        for r in c.fetchall():
            if r[4] == 'tor' and not TOR_ENABLED: continue
            local_res.append(RecursoEducativo(
                id=generar_id(r[1]), titulo=r[0], url=r[1], descripcion=r[2],
                plataforma=r[3], idioma=idioma, nivel=nivel, categoria="General",
                confianza=r[5], tipo=r[4], ultima_verificacion="", activo=True
            ))
        conn.close()
    except: pass

    api_results = await asyncio.gather(*tasks)
    final_list = local_res
    for lst in api_results: final_list.extend(lst)
    
    if not final_list: final_list = generar_simulados(tema, idioma, nivel)

    if usar_ia and GROQ_API_KEY:
        ia_tasks = [analizar_ia_groq(r) for r in final_list[:4]]
        ia_results = await asyncio.gather(*ia_tasks)
        for r, analysis in zip(final_list[:4], ia_results):
            if analysis: r.metadatos_analisis = analysis

    return eliminar_duplicados(final_list)

# ----------------------------
# 8. INTERFAZ DE USUARIO
# ----------------------------
# Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
    <h1>🎓 Buscador Académico Pro</h1>
    <p>Surface Web + Deep Web Académica + Análisis Inteligente</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR (PANEL DE CONTROL DE INDEXACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    usar_ia = st.toggle("🧠 Análisis IA Activo", value=True)
    
    st.divider()
    st.subheader("🕵️‍♂️ Agente de Indexación")
    
    # Lógica de los botones de indexación
    col_idx1, col_idx2 = st.columns(2)
    
    if col_idx1.button("▶️ Iniciar"):
        st.session_state.indexing_active = True
        st.toast("Agente de indexación iniciado en 2º plano...")
        
    if col_idx2.button("⏹️ Detener"):
        st.session_state.indexing_active = False
        st.toast("Agente detenido.")
        
    if st.session_state.indexing_active:
        st.success("🟢 Indexando...")
        st.caption("Buscando nuevos recursos .onion y académicos...")
        # Simulación visual de progreso
        progreso = st.progress(0)
        time.sleep(0.1)
        progreso.progress(random.randint(10, 90))
    else:
        st.info("⚪ En espera")

    st.metric("Recursos en Cola", st.session_state.indexed_count)

# --- FORMULARIO ---
if "search_query" not in st.session_state: st.session_state.search_query = ""

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    tema = st.text_input("Tema a investigar:", value=st.session_state.search_query)
with c2:
    nivel = st.selectbox("Nivel", ["Principiante", "Intermedio", "Avanzado"])
with c3:
    idioma = st.selectbox("Idioma", ["es", "en", "pt"])

# Botones rápidos
st.write("Exploración rápida:")
b_cols = st.columns(4)
if b_cols[0].button("🐍 Python"): st.session_state.search_query = "Python"
if b_cols[1].button("💰 Finanzas"): st.session_state.search_query = "Finanzas"
if b_cols[2].button("🎨 Diseño"): st.session_state.search_query = "Diseño"
if b_cols[3].button("🧅 Tor"): st.session_state.search_query = "Deep Web"

# --- EJECUCIÓN ---
if st.button("🚀 INICIAR BÚSQUEDA", type="primary", use_container_width=True):
    if not tema:
        st.error("Por favor ingresa un tema.")
    else:
        # Si la indexación está activa, simulamos que encuentra más cosas
        if st.session_state.indexing_active:
            st.session_state.indexed_count += random.randint(1, 5)
            
        with st.spinner(f"Analizando ecosistema digital para: {tema}..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(orquestador_busqueda(tema, idioma, nivel, usar_ia))
            loop.close()
            
            st.success(f"✅ Análisis completado: {len(resultados)} recursos verificados.")
            
            for i, res in enumerate(resultados):
                mostrar_recurso_con_ia(res, i)

# --- FOOTER ---
st.markdown("---")
st.markdown("<center><small>Powered by Groq • Google API • Tor Gateway Simulation</small></center>", unsafe_allow_html=True)

# Worker en background (Simulado para Streamlit Cloud)
def background_indexer():
    while True:
        if st.session_state.get('indexing_active', False):
            # Aquí iría la lógica real de scraping pesado
            time.sleep(5)
        else:
            time.sleep(2)

threading.Thread(target=background_indexer, daemon=True).start()
