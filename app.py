# app.py — Versión Fusión "Maestra" (SG1 + Code B Improvements)
# Arquitectura de Alto Rendimiento (Async/Threads) + Lógica de Búsqueda Mejorada + Estilo Visual Premium

import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import random
from datetime import datetime, timedelta
import json
import hashlib
import re
from urllib.parse import urlparse
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
import asyncio
import aiohttp
import contextlib
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# 1. LOGGING & CONFIGURACIÓN
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('buscador_cursos.log'), logging.StreamHandler()]
)
logger = logging.getLogger("BuscadorProfesional")

# --- GESTIÓN DE CREDENCIALES (Mejorada con lógica del Código B) ---
def obtener_credenciales_seguras():
    """Obtiene credenciales priorizando Secrets y luego Variables de Entorno."""
    try:
        # Intenta obtener de st.secrets, si falla busca en os.environ
        g_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        g_cx = st.secrets.get("GOOGLE_CX", os.getenv("GOOGLE_CX", ""))
        groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        return g_key, g_cx, groq_key
    except (AttributeError, FileNotFoundError):
        # Fallback para entorno local sin .streamlit/secrets.toml
        return os.getenv("GOOGLE_API_KEY", ""), os.getenv("GOOGLE_CX", ""), os.getenv("GROQ_API_KEY", "")

GOOGLE_API_KEY, GOOGLE_CX, GROQ_API_KEY = obtener_credenciales_seguras()

MAX_BACKGROUND_TASKS = 2
GROQ_MODEL = "llama-3.1-70b-versatile"

def validate_api_key(key: str, key_type: str) -> bool:
    if not key or len(key) < 10: return False
    if key_type == "google" and not key.startswith(("AIza", "AIz")): return False
    return True

GROQ_AVAILABLE = False
try:
    import groq
    if validate_api_key(GROQ_API_KEY, "groq"):
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API disponible y validada")
    else:
        logger.warning("⚠️ Groq API Key inválida o ausente")
except ImportError:
    logger.warning("⚠️ Biblioteca 'groq' no instalada")

# ----------------------------
# 2. GESTIÓN DE CACHÉ Y CONCURRENCIA
# ----------------------------
class ExpiringCache:
    """Caché con TTL y limpieza lazy (Arquitectura A)."""
    def __init__(self, ttl_seconds=43200):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = (value, time.time())

search_cache = ExpiringCache(ttl_seconds=43200)
executor = ThreadPoolExecutor(max_workers=MAX_BACKGROUND_TASKS)

# ----------------------------
# 3. MODELOS DE DATOS & UTILIDADES JSON
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
    tipo: str
    ultima_verificacion: str
    activo: bool
    metadatos: Dict[str, Any]
    metadatos_analisis: Optional[Dict[str, Any]] = None
    analisis_pendiente: bool = False

def safe_json_dumps(obj: Dict) -> str:
    try: return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception: return "{}"

def safe_json_loads(text: str, default_value: Dict = None) -> Dict:
    """Deserialización robusta (Del Código B)."""
    if default_value is None: default_value = {}
    try: return json.loads(text)
    except Exception: return default_value

# ----------------------------
# 4. BASE DE DATOS (Context Manager Seguro)
# ----------------------------
DB_PATH = "cursos_inteligentes_v3.db"

@contextlib.contextmanager
def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try: yield conn
    except sqlite3.Error as e:
        logger.error(f"Error BD: {e}")
        conn.rollback()
        raise e
    finally: conn.close()

def init_advanced_database() -> bool:
    """Inicializa DB con datos semilla enriquecidos (Combinación A+B)."""
    try:
        with get_db_connection(DB_PATH) as conn:
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
                activa INTEGER DEFAULT 1,
                tipo_certificacion TEXT DEFAULT 'audit',
                validez_internacional INTEGER DEFAULT 0,
                paises_validos TEXT DEFAULT '[]',
                reputacion_academica REAL DEFAULT 0.5
            )
            ''')
            
            cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas")
            if cursor.fetchone()[0] == 0:
                # Datos semilla combinados y enriquecidos
                plataformas_iniciales = [
                    {"nombre": "Aprende con Alf", "url_base": "https://aprendeconalf.es/?s={}", "descripcion": "Tutoriales Python/Data", "idioma": "es", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.85, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["es"], "reputacion_academica": 0.90},
                    {"nombre": "Coursera", "url_base": "https://www.coursera.org/search?query={}&free=true", "descripcion": "Cursos universitarios audit", "idioma": "en", "categoria": "General", "nivel": "Avanzado", "confianza": 0.95, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.95},
                    {"nombre": "Kaggle Learn", "url_base": "https://www.kaggle.com/learn/search?q={}", "descripcion": "Microcursos Data Science", "idioma": "en", "categoria": "Data Science", "nivel": "Intermedio", "confianza": 0.90, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.88},
                    {"nombre": "freeCodeCamp", "url_base": "https://www.freecodecamp.org/news/search/?query={}", "descripcion": "Certificados Web/Dev", "idioma": "en", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.93, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.91},
                    {"nombre": "edX", "url_base": "https://www.edx.org/search?tab=course&availability=current&price=free&q={}", "descripcion": "Cursos Harvard/MIT", "idioma": "en", "categoria": "Académico", "nivel": "Avanzado", "confianza": 0.92, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.93}
                ]
                for p in plataformas_iniciales:
                    cursor.execute(
                        '''INSERT INTO plataformas_ocultas 
                           (nombre, url_base, descripcion, idioma, categoria, nivel, confianza, ultima_verificacion, activa, tipo_certificacion, validez_internacional, paises_validos, reputacion_academica)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (p["nombre"], p["url_base"], p["descripcion"], p["idioma"], p["categoria"], p["nivel"], p["confianza"], datetime.now().isoformat(), 1, p["tipo_certificacion"], p["validez_internacional"], safe_json_dumps(p["paises_validos"]), p["reputacion_academica"])
                    )
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error Init DB: {e}")
        return False

init_advanced_database()

# ----------------------------
# 5. UTILIDADES
# ----------------------------
def get_codigo_idioma(nombre_idioma: str) -> str:
    return {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}.get(nombre_idioma, "es")

def generar_id_unico(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    if nivel_solicitado not in ("Cualquiera", "Todos"): return nivel_solicitado
    texto = texto.lower()
    if any(x in texto for x in ['principiante', 'básico', 'beginner', 'cero']): return "Principiante"
    if any(x in texto for x in ['avanzado', 'advanced', 'expert']): return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    tema = tema.lower()
    if any(x in tema for x in ['python', 'java', 'web', 'code']): return "Programación"
    if any(x in tema for x in ['data', 'datos', 'ia', 'ai']): return "Data Science"
    if any(x in tema for x in ['design', 'diseño']): return "Diseño"
    if any(x in tema for x in ['marketing', 'negocios']): return "Negocios"
    return "General"

def extraer_plataforma(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        if 'youtube' in domain: return 'YouTube'
        if 'coursera' in domain: return 'Coursera'
        if 'udemy' in domain: return 'Udemy'
        if 'edx' in domain: return 'edX'
        if not domain: return "Web"
        parts = domain.split('.')
        return parts[-2].title() if len(parts) >= 2 else domain.title()
    except: return "Web"

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    texto = (url + titulo + descripcion).lower()
    invalidas = ['comprar', 'buy', 'precio', 'price', 'premium', 'paid']
    validas = ['curso', 'tutorial', 'aprender', 'learn', 'gratis', 'free', 'class']
    dominios = ['.edu', 'coursera', 'edx', 'khanacademy', 'udemy', 'youtube']
    
    if any(i in texto for i in invalidas): return False
    return (any(v in texto for v in validas) or any(d in url.lower() for d in dominios))

# ----------------------------
# 6. LÓGICA DE BÚSQUEDA (MEJORADA)
# ----------------------------

# A. Google API (Async - Del código A, insuperable en velocidad)
async def buscar_en_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not validate_api_key(GOOGLE_API_KEY, "google") or not GOOGLE_CX: return []
    query = f"{tema} curso gratuito certificado"
    if nivel != "Cualquiera": query += f" nivel {nivel}"
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': query, 'num': 5, 'lr': f'lang_{idioma}'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=5) as response:
                if response.status != 200: return []
                data = await response.json()
                res = []
                for item in data.get('items', []):
                    link, title, snippet = item.get('link'), item.get('title'), item.get('snippet', '')
                    if not es_recurso_educativo_valido(link, title, snippet): continue
                    res.append(RecursoEducativo(
                        id=generar_id_unico(link), titulo=title, url=link, descripcion=snippet,
                        plataforma=extraer_plataforma(link), idioma=idioma, nivel=determinar_nivel(title+snippet, nivel),
                        categoria=determinar_categoria(tema), certificacion=None, confianza=0.85, tipo="verificada",
                        ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={'fuente': 'google'}
                    ))
                return res
    except Exception: return []

# B. Plataformas Ocultas (DB Interna - Del código A)
def buscar_en_plataformas_ocultas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            query = "SELECT nombre, url_base, descripcion, nivel, confianza, tipo_certificacion, validez_internacional FROM plataformas_ocultas WHERE activa = 1 AND idioma = ?"
            params = [idioma]
            if nivel not in ("Cualquiera", "Todos"):
                query += " AND (nivel = ? OR nivel = 'Todos')"
                params.append(nivel)
            cursor.execute(query, params)
            filas = cursor.fetchall()
            res = []
            for r in filas:
                nom, url_b, desc, niv, conf, tip_c, val_i = r
                url_f = url_b.format(tema.replace(' ', '+'))
                cert = Certificacion(plataforma=nom, curso=tema, tipo=tip_c, validez_internacional=bool(val_i), paises_validos=["global"], costo_certificado=0, reputacion_academica=0.8, ultima_verificacion=datetime.now().isoformat()) if tip_c != 'none' else None
                res.append(RecursoEducativo(id=generar_id_unico(url_f), titulo=f"💎 {nom} — {tema}", url=url_f, descripcion=desc, plataforma=nom, idioma=idioma, nivel=niv if nivel == "Cualquiera" else nivel, categoria=determinar_categoria(tema), certificacion=cert, confianza=conf, tipo="oculta", ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={"fuente": "db"}))
            return res
    except Exception: return []

# C. Plataformas Conocidas (MEJORADO CON CÓDIGO B - Diccionario explícito)
def buscar_en_plataformas_conocidas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Fallback robusto usando lógica del Código B."""
    recursos = []
    # Diccionario del Código B
    plataformas = {
        "es": [
            {"nombre": "YouTube Educativo", "url": f"https://www.youtube.com/results?search_query=curso+gratis+{tema.replace(' ', '+')}"},
            {"nombre": "Coursera (ES)", "url": f"https://www.coursera.org/search?query={tema}&languages=es&free=true"},
            {"nombre": "Udemy (Gratis)", "url": f"https://www.udemy.com/courses/search/?q={tema}&price=price-free&lang=es"}
        ],
        "en": [
            {"nombre": "YouTube Education", "url": f"https://www.youtube.com/results?search_query=free+course+{tema.replace(' ', '+')}"},
            {"nombre": "Khan Academy", "url": f"https://www.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}"},
            {"nombre": "Coursera", "url": f"https://www.coursera.org/search?query={tema}&free=true"},
        ],
        "pt": [
             {"nombre": "YouTube BR", "url": f"https://www.youtube.com/results?search_query=curso+gratuito+{tema.replace(' ', '+')}"},
             {"nombre": "Coursera (PT)", "url": f"https://www.coursera.org/search?query={tema}&languages=pt&free=true"}
        ]
    }
    
    lista = plataformas.get(idioma, plataformas["en"])
    for plat in lista:
        recursos.append(RecursoEducativo(
            id=generar_id_unico(plat["url"]),
            titulo=f"🎯 {plat['nombre']} — {tema}",
            url=plat["url"],
            descripcion=f"Búsqueda directa en {plat['nombre']}",
            plataforma=plat["nombre"],
            idioma=idioma,
            nivel=nivel if nivel != "Cualquiera" else "Intermedio",
            categoria=determinar_categoria(tema),
            certificacion=None,
            confianza=0.85,
            tipo="conocida",
            ultima_verificacion=datetime.now().isoformat(),
            activo=True,
            metadatos={"fuente": "plataformas_conocidas"}
        ))
    return recursos

async def buscar_recursos_multicapa(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    cache_key = f"{tema}|{idioma}|{nivel}"
    cached = search_cache.get(cache_key)
    if cached: return cached

    res = []
    # 1. DB Interna
    res.extend(buscar_en_plataformas_ocultas(tema, get_codigo_idioma(idioma), nivel))
    # 2. Google API (Async)
    res.extend(await buscar_en_google_api(tema, get_codigo_idioma(idioma), nivel))
    # 3. Plataformas Conocidas (Ahora con la lógica mejorada de B)
    if len(res) < 5:
        res.extend(buscar_en_plataformas_conocidas(tema, get_codigo_idioma(idioma), nivel))
    
    # Deduplicar
    seen, final = set(), []
    for r in res:
        if r.url not in seen:
            seen.add(r.url)
            final.append(r)
    
    final.sort(key=lambda x: x.confianza, reverse=True)
    final = final[:12]
    
    if GROQ_AVAILABLE:
        for r in final[:4]: r.analisis_pendiente = True
        
    search_cache.set(cache_key, final)
    return final

# ----------------------------
# 7. ANÁLISIS IA (Worker Thread)
# ----------------------------
def analizar_recurso_groq_sync(recurso: RecursoEducativo, perfil: Dict):
    """Worker para ThreadPool. Usa la extracción robusta de JSON (Regex) del Código B."""
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        # Prompt mejorado del Código B
        prompt = f"""
        Evalúa este curso. Devuelve SOLO JSON válido.
        TÍTULO: {recurso.titulo}
        DESCRIPCIÓN: {recurso.descripcion}
        NIVEL: {recurso.nivel}
        
        Responde con este JSON exacto:
        {{
            "calidad_educativa": 0.85,
            "relevancia_usuario": 0.90,
            "razones_calidad": ["razon1", "razon2"],
            "recomendacion_personalizada": "Tu conclusión breve"
        }}
        """
        
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL, temperature=0.3, max_tokens=600
        )
        contenido = resp.choices[0].message.content.strip()
        
        # Extracción robusta con Regex (Mejora del código B)
        json_match = re.search(r'\{.*\}', contenido, re.DOTALL)
        data = {}
        if json_match:
            data = safe_json_loads(json_match.group())
            
        recurso.metadatos_analisis = {
            "calidad_ia": data.get("calidad_educativa", 0.7),
            "relevancia_ia": data.get("relevancia_usuario", 0.7),
            "recomendacion_personalizada": data.get("recomendacion_personalizada", "Curso verificado."),
            "razones_calidad": data.get("razones_calidad", [])
        }
    except Exception as e:
        logger.error(f"Error Groq Worker: {e}")

def ejecutar_analisis_background(resultados: List[RecursoEducativo]):
    pendientes = [r for r in resultados if r.analisis_pendiente]
    if not pendientes: return
    for r in pendientes:
        executor.submit(analizar_recurso_groq_sync, r, {})

# ----------------------------
# 8. UI Y VISUALIZACIÓN (Estilo Código B + Fix HTML)
# ----------------------------
st.set_page_config(page_title="Buscador de Cursos PRO", layout="wide", page_icon="🎓")

# CSS Mejorado (Del código B - Gradientes y tarjetas limpias)
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
    color: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.main-header h1 { margin: 0; font-size: 2.2rem; }
.resultado-card {
    border-radius: 15px; padding: 20px; margin-bottom: 20px; background: white;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08); border-left: 6px solid #4CAF50;
    transition: transform 0.2s;
}
.resultado-card:hover { transform: translateY(-3px); }
.nivel-principiante { border-left-color: #2196F3 !important; }
.nivel-avanzado { border-left-color: #FF9800 !important; }
.plataforma-oculta { border-left-color: #FF6B35 !important; background: #fff5f0; }
.certificado-badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; background: #e8f5e9; color: #2e7d32; margin-right: 5px; }
a { text-decoration: none !important; }
</style>
""", unsafe_allow_html=True)

def link_button(url: str, label: str) -> str:
    # Botón estilo gradiente del código B
    return f'''<a href="{url}" target="_blank" style="display: inline-block; background: linear-gradient(to right, #6a11cb, #2575fc); color: white; padding: 10px 20px; border-radius: 8px; font-weight: bold;">{label}</a>'''

def badge_certificacion(cert: Optional[Certificacion]) -> str:
    if not cert: return ""
    html = ""
    if cert.tipo == "gratuito": html += '<span class="certificado-badge">✅ Gratis</span>'
    elif cert.tipo == "audit": html += '<span class="certificado-badge" style="background:#e3f2fd;color:#1565c0;">🎓 Audit</span>'
    return html

def mostrar_recurso(r: RecursoEducativo, idx: int):
    # Fusión visual: Estructura HTML limpia (sin indentación) + Estilos CSS del código B
    extra_class = "plataforma-oculta" if r.tipo == "oculta" else ""
    nivel_class = f"nivel-{r.nivel.lower()}"
    cert_html = badge_certificacion(r.certificacion)
    
    ia_block = ""
    if r.metadatos_analisis:
        data = r.metadatos_analisis
        cal = int(data.get('calidad_ia', 0)*100)
        ia_block = f"""
        <div style="background:#f3e5f5;padding:12px;border-radius:8px;margin:12px 0;border-left:4px solid #9c27b0;">
            <strong>🧠 Análisis IA:</strong> Calidad {cal}% • {data.get('recomendacion_personalizada')}
        </div>"""
    elif r.analisis_pendiente:
        ia_block = "<div style='color:#9c27b0;font-size:0.9em;margin:5px 0;'>⏳ Analizando...</div>"

    st.markdown(f"""
<div class="resultado-card {nivel_class} {extra_class}" style="animation-delay: {idx * 0.08}s;">
    <h3 style="margin-top:0;">{r.titulo}</h3>
    <p><strong>📚 {r.nivel}</strong> | 🌐 {r.plataforma} | 🏷️ {r.categoria}</p>
    <p style="color:#555;">{r.descripcion}</p>
    <div style="margin-bottom:10px;">{cert_html}</div>
    {ia_block}
    <div style="margin-top:15px;">
        {link_button(r.url, "➡️ Acceder al Recurso")}
    </div>
    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.8rem; color: #888;">
        Confianza: {r.confianza*100:.0f}% | Verificado: {datetime.fromisoformat(r.ultima_verificacion).strftime('%d/%m/%Y')}
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# 9. MAIN APP
# ----------------------------
def main():
    # Header Visual del Código B
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Buscador Profesional de Cursos</h1>
        <p>Descubre recursos educativos verificados con búsqueda inmediata y análisis IA</p>
        <div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.2);padding:4px 10px;border-radius:15px;font-size:0.8rem;">✅ Sistema Activo</span>
            <span style="background:rgba(255,255,255,0.2);padding:4px 10px;border-radius:15px;font-size:0.8rem;">⚡ AsyncIO Core</span>
            <span style="background:rgba(255,255,255,0.2);padding:4px 10px;border-radius:15px;font-size:0.8rem;">🌐 Multilingüe</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    tema = col1.text_input("¿Qué quieres aprender?", placeholder="Ej: Python, Machine Learning...")
    nivel = col2.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    idioma = col3.selectbox("Idioma", ["Español (es)", "Inglés (en)", "Portugués (pt)"])

    if st.button("🚀 Buscar Cursos", type="primary", use_container_width=True):
        if not tema.strip():
            st.warning("Por favor ingresa un tema.")
            return

        with st.spinner("🔍 Buscando en múltiples fuentes (Async)..."):
            # Búsqueda Asíncrona (Arquitectura A)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(buscar_recursos_multicapa(tema.strip(), idioma, nivel))
            loop.close()
            
            if resultados:
                st.success(f"✅ Se encontraron {len(resultados)} recursos de alta calidad.")
                
                # Ejecutar análisis IA en background (ThreadPool)
                if GROQ_AVAILABLE:
                    ejecutar_analisis_background(resultados)
                    time.sleep(0.5) # Pequeña pausa para permitir que algunos threads terminen rápido

                # Grid de resultados
                for i, r in enumerate(resultados):
                    mostrar_recurso(r, i)
                    
                # Exportar CSV (Utilidad del Código B)
                df = pd.DataFrame([{'Título': r.titulo, 'URL': r.url, 'Plataforma': r.plataforma, 'Confianza': r.confianza} for r in resultados])
                st.download_button("📥 Descargar CSV", df.to_csv(index=False).encode('utf-8'), "cursos.csv", "text/csv")
            else:
                st.warning("No se encontraron resultados. Intenta con términos más generales.")

if __name__ == "__main__":
    main()
