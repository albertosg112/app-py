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
import re
from urllib.parse import quote, urlparse
import threading
import queue
import concurrent.futures
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
from functools import lru_cache
import asyncio
import aiohttp
from dotenv import load_dotenv
import groq

# ----------------------------
# CONFIGURACIÓN AVANZADA Y LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('buscador_cursos.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BuscadorProfesional")

# Cargar variables de entorno (Secrets o .env)
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = st.secrets.get("GOOGLE_CX", "") if hasattr(st, 'secrets') else os.getenv("GOOGLE_CX", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("GROQ_API_KEY", "")
DUCKDUCKGO_ENABLED = st.secrets.get("DUCKDUCKGO_ENABLED", "true").lower() == "true" if hasattr(st, 'secrets') else os.getenv("DUCKDUCKGO_ENABLED", "true").lower() == "true"

# Configuración de parámetros
MAX_BACKGROUND_TASKS = 1  # Evita bloqueo de base de datos SQLite
CACHE_EXPIRATION = timedelta(hours=12)
GROQ_MODEL = "llama3-8b-8192"

# Sistema de caché
if 'search_cache' not in st.session_state: st.session_state.search_cache = {}
if 'groq_cache' not in st.session_state: st.session_state.groq_cache = {}

# Cola para tareas en segundo plano
background_tasks = queue.Queue()

# ----------------------------
# MODELOS DE DATOS
# ----------------------------
@dataclass
class Certificacion:
    plataforma: str
    curso: str
    tipo: str  # "gratuito", "pago", "audit"
    validez_internacional: bool
    paises_validos: List[str]
    costo_certificado: float
    reputacion_academica: float
    ultima_verificacion: str

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
    certificacion: Optional[Certificacion]
    confianza: float
    tipo: str  # "conocida", "oculta", "verificada"
    ultima_verificacion: str
    activo: bool
    metadatos: Dict[str, Any]
    metadatos_analisis: Optional[Dict[str, Any]] = None

# ----------------------------
# CONFIGURACIÓN INICIAL Y BASE DE DATOS
# ----------------------------
DB_PATH = "cursos_inteligentes_v3.db"

def init_advanced_database():
    try:
        # check_same_thread=False es vital para acceso desde threads
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS plataformas_ocultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT NOT NULL, url_base TEXT NOT NULL,
            descripcion TEXT, idioma TEXT NOT NULL, categoria TEXT, nivel TEXT,
            confianza REAL DEFAULT 0.7, ultima_verificacion TEXT, activa INTEGER DEFAULT 1,
            tipo_certificacion TEXT DEFAULT 'audit', validez_internacional BOOLEAN DEFAULT 0,
            paises_validos TEXT DEFAULT '[]', reputacion_academica REAL DEFAULT 0.5
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS analiticas_busquedas (
            id INTEGER PRIMARY KEY AUTOINCREMENT, tema TEXT NOT NULL, idioma TEXT NOT NULL,
            nivel TEXT, timestamp TEXT NOT NULL, plataforma_origen TEXT,
            veces_mostrado INTEGER DEFAULT 0, veces_clickeado INTEGER DEFAULT 0,
            tiempo_promedio_uso REAL DEFAULT 0.0, satisfaccion_usuario REAL DEFAULT 0.0
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS certificaciones_verificadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT, plataforma TEXT NOT NULL, curso_tema TEXT NOT NULL,
            tipo_certificacion TEXT NOT NULL, validez_internacional BOOLEAN DEFAULT 0,
            paises_validos TEXT DEFAULT '[]', costo_certificado REAL DEFAULT 0.0,
            reputacion_academica REAL DEFAULT 0.5, ultima_verificacion TEXT NOT NULL,
            veces_verificado INTEGER DEFAULT 1
        )''')
        
        # Datos semilla
        cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas")
        if cursor.fetchone()[0] == 0:
            plataformas_iniciales = [
                ("Aprende con Alf", "https://aprendeconalf.es/?s={}", "Cursos gratuitos programación", "es", "Programación", "Intermedio", 0.85, json.dumps(["es"]), 1, 0.9),
                ("Coursera", "https://www.coursera.org/search?query={}&free=true", "Cursos universitarios audit", "en", "General", "Avanzado", 0.95, json.dumps(["global"]), 1, 0.95),
                ("edX", "https://www.edx.org/search?tab=course&availability=current&price=free&q={}", "Harvard/MIT audit", "en", "Académico", "Avanzado", 0.92, json.dumps(["global"]), 1, 0.93),
                ("Kaggle Learn", "https://www.kaggle.com/learn/search?q={}", "Data Science práctico", "en", "Data Science", "Intermedio", 0.90, json.dumps(["global"]), 1, 0.88),
                ("freeCodeCamp", "https://www.freecodecamp.org/news/search/?query={}", "Desarrollo Web Fullstack", "en", "Programación", "Intermedio", 0.93, json.dumps(["global"]), 1, 0.91),
                ("Domestika (Gratuito)", "https://www.domestika.org/es/search?query={}&free=1", "Cursos creativos", "es", "Diseño", "Intermedio", 0.83, json.dumps(["es"]), 1, 0.82)
            ]
            for plat in plataformas_iniciales:
                cursor.execute('''INSERT INTO plataformas_ocultas 
                (nombre, url_base, descripcion, idioma, categoria, nivel, confianza, paises_validos, validez_internacional, reputacion_academica, ultima_verificacion, activa)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)''', plat + (datetime.now().isoformat(),))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error DB: {e}")
        return False

# Inicializar DB
if not os.path.exists(DB_PATH):
    init_advanced_database()
else:
    init_advanced_database()

# ----------------------------
# FUNCIONES AUXILIARES
# ----------------------------
def get_codigo_idioma(nombre: str) -> str:
    return {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}.get(nombre, "es")

def generar_id_unico(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, solicitado: str) -> str:
    if solicitado not in ["Cualquiera", "Todos"]: return solicitado
    texto = texto.lower()
    if any(x in texto for x in ['principiante', 'básico', 'beginner']): return "Principiante"
    if any(x in texto for x in ['avanzado', 'advanced', 'experto']): return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    tema = tema.lower()
    cats = {
        "Programación": ['programación', 'python', 'java', 'web', 'code'],
        "Data Science": ['datos', 'data', 'machine learning', 'ai', 'ia'],
        "Diseño": ['diseño', 'design', 'ux', 'ui'],
        "Negocios": ['marketing', 'business', 'finanzas']
    }
    for cat, keys in cats.items():
        if any(k in tema for k in keys): return cat
    return "General"

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    texto = (url + titulo + descripcion).lower()
    validas = ['curso', 'tutorial', 'aprender', 'learn', 'free', 'gratuito', 'certificado']
    invalidas = ['comprar', 'buy', 'precio', 'price', 'paid', 'premium']
    
    es_gratuito = any(x in texto for x in ['gratuito', 'free', 'sin costo', 'audit'])
    tiene_invalida = any(x in texto for x in invalidas)
    
    return any(v in texto for v in validas) and not tiene_invalida and es_gratuito

def extraer_plataforma(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    mapa = {'coursera': 'Coursera', 'edx': 'edX', 'udemy': 'Udemy', 'youtube': 'YouTube', 'kaggle': 'Kaggle', 'freecodecamp': 'freeCodeCamp'}
    for k,v in mapa.items():
        if k in domain: return v
    return domain.split('.')[-2].title() if len(domain.split('.')) > 1 else domain.title()

def eliminar_duplicados(lista):
    seen = set()
    unique = []
    for x in lista:
        if x.url not in seen:
            seen.add(x.url)
            unique.append(x)
    return unique

# ----------------------------
# FUNCIÓN VISUAL (LA CORRECCIÓN: DEFINIDA AQUÍ ANTES DE USARSE)
# ----------------------------
def mostrar_recurso_con_ia(recurso: RecursoEducativo, index: int):
    """Muestra un recurso con análisis de IA integrado y estilos visuales"""
    
    color_clase = {
        "Principiante": "nivel-principiante",
        "Intermedio": "nivel-intermedio", 
        "Avanzado": "nivel-avanzado"
    }.get(recurso.nivel, "")
    
    extra_class = "plataforma-oculta" if recurso.tipo == "oculta" else ""
    ia_class = "con-analisis-ia" if recurso.metadatos_analisis else ""
    
    # Badge de certificación
    cert_badge = ""
    if recurso.certificacion:
        tipo = recurso.certificacion.tipo
        if tipo == "gratuito":
            cert_badge = '<span class="certificado-badge certificado-gratuito">✅ Certificado Gratuito</span>'
        elif tipo == "audit":
            cert_badge = '<span class="certificado-badge certificado-internacional">🎓 Modo Audit</span>'
        else:
            cert_badge = '<span class="certificado-badge certificado-internacional">💰 Certificado Pago</span>'
    
    # Análisis IA HTML
    ia_content = ""
    if recurso.metadatos_analisis:
        meta = recurso.metadatos_analisis
        rec = meta.get("recomendacion_personalizada", "")
        calidad = meta.get("calidad_ia", 0) * 100
        relevancia = meta.get("relevancia_ia", 0) * 100
        
        ia_content = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <strong style="color: #fff9c4;">🧠 Análisis IA:</strong> {rec}
            <div style="margin-top: 5px; display: flex; gap: 10px;">
                <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 10px; font-size: 0.85em;">⭐ Calidad: {calidad:.0f}%</span>
                <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 10px; font-size: 0.85em;">🎯 Relevancia: {relevancia:.0f}%</span>
            </div>
        </div>
        """

    st.markdown(f"""
    <div class="resultado-card {color_clase} {extra_class} {ia_class} fade-in" style="animation-delay: {index * 0.1}s;">
        <h3>{'💎' if recurso.tipo=='oculta' else '🎯'} {recurso.titulo}</h3>
        <p style="color: #666; font-size: 0.9em;">
            <strong>📚 {recurso.nivel}</strong> | <strong>🌐 {recurso.plataforma}</strong> | {recurso.idioma.upper()}
        </p>
        <p>{recurso.descripcion}</p>
        {cert_badge}
        {ia_content}
        <div style="margin-top: 15px;">
            <a href="{recurso.url}" target="_blank" style="background: linear-gradient(to right, #6a11cb, #2575fc); color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">
                ➡️ Acceder al Recurso
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# ANÁLISIS IA (GROQ)
# ----------------------------
@st.cache_resource
def get_groq_client():
    if GROQ_API_KEY:
        return groq.Groq(api_key=GROQ_API_KEY)
    return None

async def analizar_calidad_curso(recurso: RecursoEducativo, perfil: Dict) -> Dict:
    cache_key = f"groq_{recurso.id}_{perfil.get('nivel_real', 'unk')}"
    if cache_key in st.session_state.groq_cache:
        return st.session_state.groq_cache[cache_key]

    client = get_groq_client()
    if not client: return {}

    try:
        prompt = f"""
        Analiza este curso educativo:
        Curso: {recurso.titulo} ({recurso.plataforma})
        Desc: {recurso.descripcion}
        Perfil: {perfil}
        
        Responde JSON estricto:
        {{
            "calidad_educativa": 0.0-1.0,
            "relevancia_usuario": 0.0-1.0,
            "recomendacion_personalizada": "1 frase corta",
            "razones_calidad": ["razón 1"],
            "advertencias": []
        }}
        """
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        ))
        
        analisis = json.loads(response.choices[0].message.content)
        st.session_state.groq_cache[cache_key] = analisis
        return analisis
    except Exception as e:
        logger.error(f"Groq Error: {e}")
        return {}

# ----------------------------
# MOTORES DE BÚSQUEDA
# ----------------------------
async def buscar_en_google_api(tema, idioma, nivel):
    if not GOOGLE_API_KEY or not GOOGLE_CX: return []
    try:
        q = f"{tema} curso gratuito certificado"
        if nivel not in ["Cualquiera", "Todos"]: q += f" nivel {nivel}"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': q, 'num': 5, 'lr': f'lang_{idioma}'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=8) as resp:
                if resp.status != 200: return []
                data = await resp.json()
                res = []
                for item in data.get('items', []):
                    if es_recurso_educativo_valido(item.get('link',''), item.get('title',''), item.get('snippet','')):
                        res.append(RecursoEducativo(
                            id=generar_id_unico(item['link']), titulo=item['title'], url=item['link'],
                            descripcion=item['snippet'], plataforma=extraer_plataforma(item['link']),
                            idioma=idioma, nivel=determinar_nivel(item['title'], nivel),
                            categoria=determinar_categoria(tema), certificacion=None, confianza=0.85,
                            tipo="verificada", ultima_verificacion=datetime.now().isoformat(),
                            activo=True, metadatos={'fuente': 'google'}
                        ))
                return res
    except: return []

async def buscar_en_ocultas(tema, idioma, nivel):
    try:
        with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
            c = conn.cursor()
            q = "SELECT nombre, url_base, descripcion, nivel, confianza, tipo_certificacion FROM plataformas_ocultas WHERE activa=1 AND idioma=?"
            p = [idioma]
            if nivel not in ["Cualquiera", "Todos"]:
                q += " AND (nivel=? OR nivel='Todos')"
                p.append(nivel)
            c.execute(q + " LIMIT 4", p)
            rows = c.fetchall()
            res = []
            for r in rows:
                url = r[1].format(quote(tema))
                res.append(RecursoEducativo(
                    id=generar_id_unico(url), titulo=f"💎 {r[0]} - {tema}", url=url,
                    descripcion=r[2], plataforma=r[0], idioma=idioma, nivel=r[3],
                    categoria=determinar_categoria(tema),
                    certificacion=Certificacion(r[0], tema, r[5], False, [], 0, 0.8, datetime.now().isoformat()),
                    confianza=r[4], tipo="oculta", ultima_verificacion=datetime.now().isoformat(),
                    activo=True, metadatos={'fuente': 'db'}
                ))
            return res
    except: return []

async def buscar_en_conocidas(tema, idioma, nivel):
    await asyncio.sleep(0) # Yield control
    res = []
    # Simulación simple
    plat = {'name': 'YouTube', 'url': f'https://www.youtube.com/results?search_query=curso+{tema}'}
    res.append(RecursoEducativo(
        id=generar_id_unico(plat['url']), titulo=f"📺 Curso de {tema} en YouTube", url=plat['url'],
        descripcion="Recurso popular en video", plataforma="YouTube", idioma=idioma, nivel=nivel,
        categoria=determinar_categoria(tema), certificacion=None, confianza=0.8, tipo="conocida",
        ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={}
    ))
    return res

async def buscar_multicapa(tema, idioma, nivel):
    ck = f"s_{tema}_{idioma}_{nivel}"
    if ck in st.session_state.search_cache: return st.session_state.search_cache[ck]
    
    tasks = [buscar_en_conocidas(tema, idioma, nivel), buscar_en_ocultas(tema, idioma, nivel)]
    if GOOGLE_API_KEY: tasks.append(buscar_en_google_api(tema, idioma, nivel))
    
    results_lists = await asyncio.gather(*tasks)
    final = []
    for l in results_lists: final.extend(l)
    
    final = eliminar_duplicados(final)
    final.sort(key=lambda x: x.confianza, reverse=True)
    
    # Análisis IA Top 5
    if GROQ_API_KEY and final:
        perfil = {"nivel_real": "Intermedio", "objetivos": "Aprender"}
        top = final[:5]
        ia_tasks = [analizar_calidad_curso(r, perfil) for r in top]
        ia_results = await asyncio.gather(*ia_tasks)
        
        for r, ia in zip(top, ia_results):
            if ia:
                r.metadatos_analisis = {
                    "calidad_ia": ia.get("calidad_educativa", 0),
                    "relevancia_ia": ia.get("relevancia_usuario", 0),
                    "recomendacion_personalizada": ia.get("recomendacion_personalizada", "")
                }
                # Boost confianza
                r.confianza = (r.confianza + ia.get("calidad_educativa", r.confianza)) / 2
                
    st.session_state.search_cache[ck] = final[:15]
    return final[:15]

# ----------------------------
# TAREAS BACKGROUND
# ----------------------------
def iniciar_background():
    def worker():
        while True:
            try:
                task = background_tasks.get(timeout=2)
                time.sleep(1) # Simular trabajo
                background_tasks.task_done()
            except: pass
    t = threading.Thread(target=worker, daemon=True)
    t.start()

# ----------------------------
# INTERFAZ DE USUARIO
# ----------------------------
st.set_page_config(page_title="Buscador Pro IA", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%); padding: 2rem; border-radius: 20px; color: white; margin-bottom: 20px; }
    .resultado-card { border-radius: 15px; padding: 20px; margin-bottom: 20px; background: white; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-left: 5px solid #4CAF50; }
    .con-analisis-ia { border-left-color: #7b1fa2 !important; background: #fdfbff; }
    .certificado-badge { padding: 4px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 5px; color: white; }
    .certificado-gratuito { background: #4CAF50; }
    .certificado-internacional { background: #2196F3; }
</style>
""", unsafe_allow_html=True)

iniciar_background()

with st.sidebar:
    st.header("⚙️ Configuración")
    col1, col2 = st.columns(2)
    col1.metric("Búsquedas", "1.2k")
    col2.metric("Indexados", "850+")
    st.success(f"IA Groq: {'✅' if GROQ_API_KEY else '❌'}")

st.markdown("""
<div class="main-header">
    <h1>🎓 Buscador Profesional con IA</h1>
    <p>Encuentra los mejores cursos gratuitos y certificados verificados por IA.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([2,1,1])
tema = c1.text_input("¿Qué quieres aprender?", placeholder="Ej: Python para Data Science")
nivel = c2.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
idioma = c3.selectbox("Idioma", ["Español (es)", "Inglés (en)"])

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

if st.button("🚀 Buscar con IA", type="primary"):
    if tema:
        with st.spinner("🧠 Analizando cursos..."):
            res = run_async(buscar_multicapa(tema, idioma, nivel))
            
            if res:
                st.success(f"Encontrados {len(res)} cursos")
                col1, col2 = st.columns(2)
                col1.metric("Total", len(res))
                col2.metric("Con Análisis IA", sum(1 for r in res if r.metadatos_analisis))
                
                for i, r in enumerate(res):
                    mostrar_recurso_con_ia(r, i)
            else:
                st.warning("No se encontraron resultados")
    else:
        st.warning("Por favor ingresa un tema")
