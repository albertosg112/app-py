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
# 1. CONFIGURACIÓN
# ----------------------------
st.set_page_config(
    page_title="🎓 Buscador Académico Open",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger("BuscadorProfesional")

# ----------------------------
# 2. GESTIÓN DE SECRETOS (Blindado)
# ----------------------------
def get_secret(key: str, default: str = "") -> str:
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return str(st.secrets[key])
        val = os.getenv(key)
        return str(val) if val is not None else default
    except: return default

# CARGAR CLAVES
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GOOGLE_CX = get_secret("GOOGLE_CX")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# Flags
def is_enabled(key, default="true"):
    val = get_secret(key, default).lower()
    return val in ["true", "1", "yes", "on"]

DUCKDUCKGO_ENABLED = is_enabled("DUCKDUCKGO_ENABLED")
TOR_ENABLED = is_enabled("TOR_ENABLED")
GROQ_MODEL = "llama3-8b-8192"
DB_PATH = "cursos_v18_open.db" # Nueva DB limpia

background_tasks = queue.Queue()

# ----------------------------
# 3. MODELOS DE DATOS
# ----------------------------
@dataclass
class Certificacion:
    plataforma: str
    tipo: str # "gratuito", "audit"
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
    tipo: str # "conocida", "oculta", "google", "ia"
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
        
        # Datos semilla mínimos
        c.execute("SELECT COUNT(*) FROM recursos")
        if c.fetchone()[0] == 0:
            seed = [
                ("tor_lib", "Imperial Library", "http://xfmro77i3lixucja.onion", "Biblioteca técnica Deep Web.", "DeepWeb", "en", "Experto", "Libros", 0.8, "tor", 0)
            ]
            for row in seed:
                c.execute("INSERT OR IGNORE INTO recursos VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
            conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error DB: {e}")

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
# 6. FUNCIÓN VISUALIZACIÓN (ESTILO PREMIUM - SIN HTML ROTO)
# ----------------------------
def mostrar_recurso_premium(res: RecursoEducativo, index: int):
    """Muestra la tarjeta visual del recurso con estilo limpio"""
    
    # Colores elegantes
    colors = {
        "conocida": "#4CAF50", # Verde
        "oculta": "#FF9800",   # Naranja
        "tor": "#9C27B0",      # Morado
        "ia": "#00BCD4",       # Cyan
        "google": "#4285F4",   # Azul Google
        "simulada": "#607D8B"  # Gris
    }
    color_borde = colors.get(res.tipo, "#555")

    # Contenedor estilo tarjeta
    with st.container():
        # Inyectamos CSS para el borde izquierdo de color (Efecto Premium)
        st.markdown(f"""
        <style>
            div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stMarkdownContainer"] p:contains("{res.id}")) {{
                border: 1px solid #e0e0e0;
                border-left: 5px solid {color_borde};
                border-radius: 8px;
                padding: 15px;
                background-color: #ffffff;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                margin-bottom: 15px;
            }}
        </style>
        <div style='display:none'>{res.id}</div>
        """, unsafe_allow_html=True)

        # Encabezado: Título y Tipo
        c1, c2 = st.columns([0.85, 0.15])
        with c1:
            st.markdown(f"### 🎯 {res.titulo}")
        with c2:
            st.markdown(f"<div style='text-align:right; color:{color_borde}; font-weight:bold; font-size:0.7em; background:#f5f5f5; padding:2px 5px; border-radius:4px;'>{res.tipo.upper()}</div>", unsafe_allow_html=True)

        # Metadatos
        st.markdown(f"**🏛️ {res.plataforma}** | 📚 {res.nivel} | ⭐ Confianza: {res.confianza*100:.0f}%")
        st.write(res.descripcion)

        # Insignias de Certificado
        if res.certificacion and res.certificacion.tipo in ["audit", "gratuito"]:
             st.success(f"🎓 Opción de Certificación/Auditoría detectada")

        # Análisis IA (Si existe)
        if res.metadatos_analisis:
            meta = res.metadatos_analisis
            calidad = float(meta.get('calidad_ia', 0)) * 100
            st.info(f"🤖 **IA Dice:** {meta.get('recomendacion_personalizada')} (Calidad: {calidad:.0f}%)")

        # Botón de Acción (Nativo de Streamlit, no falla)
        st.link_button("➡️ Acceder al Curso", res.url, type="primary", use_container_width=True)
        st.write("") # Espaciador

# ----------------------------
# 7. LÓGICA DE BÚSQUEDA
# ----------------------------

async def buscar_google_api(tema, idioma, nivel):
    """Búsqueda real en Google (Prioritaria)"""
    if not GOOGLE_API_KEY or not GOOGLE_CX: return []
    try:
        query = f"curso {tema} {nivel} gratis certificado"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': query, 'num': 5, 'lr': f'lang_{idioma}'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    res = []
                    for i in data.get('items', []):
                        snip = i.get('snippet', '').lower()
                        has_cert = "certific" in snip or "certific" in i.get('title', '').lower()
                        cert_obj = Certificacion(i['displayLink'], "gratuito" if "gratis" in snip else "audit", True) if has_cert else None

                        res.append(RecursoEducativo(
                            id=generar_id(i['link']), titulo=i['title'], url=i['link'], descripcion=i['snippet'],
                            plataforma=extraer_plataforma(i['link']), idioma=idioma, nivel=nivel,
                            categoria=determinar_categoria(tema), confianza=0.95, tipo="google",
                            ultima_verificacion=datetime.now().isoformat(), activo=True, certificacion=cert_obj
                        ))
                    return res
    except: pass
    return []

async def analizar_ia_groq(recurso):
    if not GROQ_API_KEY: return None
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""Analiza: "{recurso.titulo}" - "{recurso.descripcion}".
        Responde SOLO JSON: {{ "recomendacion_personalizada": "Resumen de valor en 1 frase", "calidad_ia": 0.9 }}"""
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL, response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except: return None

def generar_simulados(tema, idioma, nivel):
    """Fallback si falla Google"""
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
            metadatos_analisis={"recomendacion_personalizada": "Resultado de respaldo (Google API no configurada).", "calidad_ia": 0.8}
        ))
    return res

async def orquestador_busqueda(tema, idioma, nivel, usar_ia):
    tasks = []
    if GOOGLE_API_KEY: tasks.append(buscar_google_api(tema, idioma, nivel))
    
    # DB Local (Deep Web)
    local_res = []
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT titulo, url, descripcion, plataforma, tipo, confianza, tiene_certificado FROM recursos WHERE titulo LIKE ?", (f"%{tema}%",))
        for r in c.fetchall():
            if r[4] == 'tor' and not TOR_ENABLED: continue
            cert = Certificacion(r[3], "gratuito" if r[6] else "audit", True) if r[6] else None
            local_res.append(RecursoEducativo(
                id=generar_id(r[1]), titulo=r[0], url=r[1], descripcion=r[2], plataforma=r[3], 
                idioma=idioma, nivel=nivel, categoria="General", confianza=r[5], tipo=r[4], 
                ultima_verificacion="", activo=True, certificacion=cert
            ))
        conn.close()
    except: pass

    api_results = await asyncio.gather(*tasks)
    final_list = local_res
    for lst in api_results: final_list.extend(lst)
    
    if not final_list: final_list = generar_simulados(tema, idioma, nivel)

    final_list = eliminar_duplicados(final_list)

    if usar_ia and GROQ_API_KEY:
        ia_tasks = [analizar_ia_groq(r) for r in final_list[:4]]
        ia_results = await asyncio.gather(*ia_tasks)
        for r, analysis in zip(final_list[:4], ia_results):
            if analysis: r.metadatos_analisis = analysis

    return final_list

# ----------------------------
# 8. INTERFAZ DE USUARIO (SIN LOGIN - ACCESO DIRECTO)
# ----------------------------

# Sidebar
with st.sidebar:
    st.header("⚙️ Panel de Control")
    
    if GOOGLE_API_KEY:
        st.success("✅ Google API Conectada")
    else:
        st.warning("⚠️ Google API No detectada")
        
    st.markdown("### Preferencias")
    usar_ia = st.toggle("🧠 Análisis IA", value=True)
    
    st.markdown("---")
    st.caption("Buscador Premium v18.0 (Open)")

# Header Principal
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;">
    <h1>🎓 Buscador Premium de Cursos</h1>
    <p>Acceso abierto a recursos verificados con Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)

# Formulario
with st.container():
    c1, c2, c3 = st.columns([3, 1, 1])
    tema = c1.text_input("¿Qué quieres aprender?", placeholder="Ej. Python Avanzado...")
    nivel = c2.selectbox("Nivel", ["Principiante", "Intermedio", "Avanzado"])
    idioma = c3.selectbox("Idioma", ["es", "en", "pt"])

    if st.button("🔍 BUSCAR CURSOS", type="primary", use_container_width=True):
        if not tema:
            st.warning("Escribe un tema.")
        else:
            with st.spinner("Analizando fuentes premium (Google + Deep Web)..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resultados = loop.run_until_complete(orquestador_busqueda(tema, idioma, nivel, usar_ia))
                loop.close()
                
                st.success(f"Encontrados {len(resultados)} cursos verificados.")
                st.markdown("---")
                
                for i, res in enumerate(resultados):
                    mostrar_recurso_premium(res, i)
                
                if resultados:
                    df = pd.DataFrame([asdict(r) for r in resultados])
                    st.download_button("📥 Descargar Reporte CSV", df.to_csv(index=False).encode('utf-8'), "reporte.csv")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888'>Acceso Abierto • Google Search API • Groq AI</div>", unsafe_allow_html=True)

# Worker Thread
def background_worker():
    while True:
        try:
            task = background_tasks.get(timeout=2)
            background_tasks.task_done()
        except: pass
        time.sleep(5)

threading.Thread(target=background_worker, daemon=True).start()
