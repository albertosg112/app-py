# app.py — Consolidado Definitivo Ultra-Robust PRO (SG1 + Async + UI/Chat + Parches + Módulos Extra)
# Objetivo: 1200+ líneas de código robusto y creativo, uniendo lo mejor de tus versiones, sin quitar funcionalidades.
# Incluye:
# - Búsqueda multicapa (Google API, plataformas conocidas, plataformas ocultas en DB)
# - Caché expirable, deduplicación y orden por confianza
# - Análisis IA con Groq en background (fallback seguro si no está disponible)
# - Chat IA con limpieza de HTML/JSON visible
# - UI moderna con badges, métricas, CSV export, favoritos y notas del usuario
# - Base de datos con context manager, semilla restaurada y migraciones
# - Panel de configuración avanzada, banderas de características, temas y accesibilidad
# - Trazabilidad, auditoría, telemetría opt-out, perfilado liviano, y depuración
# - Módulos creativos: marcadores, calificaciones, feedback del usuario, historial de sesiones
# - Utilidades para limpieza, validación, normalización, test integrado básico
# - Persistencia de estado en session_state y sincronización con la DB
# - Extensiones (DDG opcional), import/export de búsquedas, modo offline con caché
import streamlit as st
import pandas as pd
import sqlite3
import os
import sys
import time
import random
from datetime import datetime, timedelta
import json
import hashlib
import re
from urllib.parse import urlparse, quote_plus
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Callable
import logging
import asyncio
import aiohttp
import contextlib
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# ============================================================
# 0. PERFILADO LIGERO Y DECORADORES DE UTILIDAD
# ============================================================
def profile(func: Callable):
    """Decorador simple para medir tiempo de ejecución y loggear demoras sospechosas."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt > 0.3:
            logging.getLogger("Perf").info(f"⏱️ {func.__name__} tardó {dt:.3f}s")
        return out
    return wrapper
def async_profile(func: Callable):
    """Decorador para funciones async, reporta tiempos."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = await func(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt > 0.3:
            logging.getLogger("Perf").info(f"⏱️ {func.__name__} tardó {dt:.3f}s (async)")
        return out
    return wrapper

# ============================================================
# 1. LOGGING & CONFIGURACIÓN
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('buscador_cursos.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BuscadorProfesional")

def obtener_credenciales_seguras() -> Tuple[str, str, str]:
    """Obtiene credenciales priorizando Secrets y luego Variables de Entorno."""
    try:
        g_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        g_cx = st.secrets.get("GOOGLE_CX", os.getenv("GOOGLE_CX", ""))
        groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        return g_key, g_cx, groq_key
    except (AttributeError, FileNotFoundError):
        return os.getenv("GOOGLE_API_KEY", ""), os.getenv("GOOGLE_CX", ""), os.getenv("GROQ_API_KEY", "")

GOOGLE_API_KEY, GOOGLE_CX, GROQ_API_KEY = obtener_credenciales_seguras()
DUCKDUCKGO_ENABLED = (os.getenv("DUCKDUCKGO_ENABLED", "false").lower() == "true")
MAX_BACKGROUND_TASKS = 2
CACHE_EXPIRATION = timedelta(hours=12)

# --- CARGA SEGURA DEL MÓDULO DE IA EXTERNO ---
try:
    from ia_module import analizar_recurso_groq, chatgroq, GROQ_AVAILABLE
except ImportError:
    logger.warning("⚠️ Módulo ia_module.py no encontrado. IA desactivada.")
    GROQ_AVAILABLE = False
    def analizar_recurso_groq(titulo, descripcion, nivel, categoria, plataforma):
        return {
            "calidad_ia": 0.8,
            "relevancia_ia": 0.8,
            "recomendacion_personalizada": "IA no disponible (módulo ia_module.py no cargado).",
            "razones_calidad": [],
            "advertencias": ["Módulo ia_module no encontrado"]
        }
    def chatgroq(mensaje: str) -> str:
        return "🧠 IA no disponible. Usa el buscador superior para encontrar cursos ahora."

def validate_api_key(key: str, key_type: str) -> bool:
    if not key or len(key) < 10:
        return False
    if key_type == "google" and not key.startswith(("AIza", "AIz")):
        return False
    return True

# ============================================================
# 2. FEATURE FLAGS & CONFIGURACIONES AVANZADAS
# ============================================================
DEFAULT_FEATURES = {
    "enable_google_api": True,
    "enable_known_platforms": True,
    "enable_hidden_platforms": True,
    "enable_groq_analysis": True,
    "enable_chat_ia": True,
    "enable_favorites": True,
    "enable_feedback": True,
    "enable_export_import": True,
    "enable_offline_cache": True,
    "enable_ddg_fallback": False,
    "enable_debug_mode": False,
    "ui_theme": "auto",  # auto | dark | light
    "max_results": 15,
    "max_analysis": 5
}
def init_feature_flags():
    if "features" not in st.session_state:
        st.session_state.features = DEFAULT_FEATURES.copy()
    # Garantizar consistencia si ambientes cambian
    st.session_state.features["enable_google_api"] &= bool(validate_api_key(GOOGLE_API_KEY, "google") and GOOGLE_CX)
    st.session_state.features["enable_groq_analysis"] &= GROQ_AVAILABLE

# ============================================================
# 3. CACHÉ & CONCURRENCIA
# ============================================================
class ExpiringCache:
    """Caché con TTL y limpieza lazy."""
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
search_cache = ExpiringCache(ttl_seconds=int(CACHE_EXPIRATION.total_seconds()))
background_tasks: "queue.Queue[Dict[str, Any]]" = queue.Queue()
executor = ThreadPoolExecutor(max_workers=MAX_BACKGROUND_TASKS)

# ============================================================
# 4. MODELOS DE DATOS & UTILIDADES JSON
# ============================================================
# ... (el resto del archivo permanece exactamente igual hasta la sección de IA)

# [MANTENER TODO IGUAL desde "4. MODELOS DE DATOS" hasta "7. INTEGRACIÓN GROQ", 
# pero eliminamos las funciones internas de IA que ya están en ia_module.py]

# ============================================================
# 5. BASE DE DATOS (Context Manager, Migraciones, Auditoría)
# ============================================================
# ... (igual que en tu archivo original)
DB_PATH = "cursos_inteligentes_v3.db"
@contextlib.contextmanager
def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Error BD: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()

def migrate_database():
    # ... (igual que en tu archivo original)
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
            CREATE TABLE IF NOT EXISTS auditoria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evento TEXT NOT NULL,
                detalle TEXT,
                creado_en TEXT NOT NULL
            )
            ''')
            c.execute('''
            CREATE TABLE IF NOT EXISTS favoritos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_recurso TEXT NOT NULL,
                titulo TEXT NOT NULL,
                url TEXT NOT NULL,
                notas TEXT,
                creado_en TEXT NOT NULL
            )
            ''')
            c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_recurso TEXT NOT NULL,
                opinion TEXT,
                rating INTEGER,
                creado_en TEXT NOT NULL
            )
            ''')
            c.execute('''
            CREATE TABLE IF NOT EXISTS sesiones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                device TEXT,
                prefs_json TEXT
            )
            ''')
            c.execute('''
            CREATE TABLE IF NOT EXISTS configuracion (
                clave TEXT PRIMARY KEY,
                valor TEXT
            )
            ''')
            conn.commit()
            logger.info("✅ Migraciones aplicadas")
    except Exception as e:
        logger.error(f"❌ Error migrando DB: {e}")

def init_advanced_database() -> bool:
    # ... (igual que en tu archivo original)
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
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analiticas_busquedas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tema TEXT NOT NULL,
                idioma TEXT NOT NULL,
                nivel TEXT,
                timestamp TEXT NOT NULL,
                plataforma_origen TEXT,
                veces_mostrado INTEGER DEFAULT 0,
                veces_clickeado INTEGER DEFAULT 0,
                tiempo_promedio_uso REAL DEFAULT 0.0,
                satisfaccion_usuario REAL DEFAULT 0.0
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS certificaciones_verificadas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plataforma TEXT NOT NULL,
                curso_tema TEXT NOT NULL,
                tipo_certificacion TEXT NOT NULL,
                validez_internacional INTEGER DEFAULT 0,
                paises_validos TEXT DEFAULT '[]',
                costo_certificado REAL DEFAULT 0.0,
                reputacion_academica REAL DEFAULT 0.5,
                ultima_verificacion TEXT NOT NULL,
                veces_verificado INTEGER DEFAULT 1
            )
            ''')
            cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas")
            if cursor.fetchone()[0] == 0:
                plataformas_iniciales = [
                    {"nombre": "Aprende con Alf", "url_base": "https://aprendeconalf.es/?s={}", "descripcion": "Cursos gratuitos de programación, matemáticas y ciencia de datos con ejercicios prácticos", "idioma": "es", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.85, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["es"], "reputacion_academica": 0.90},
                    {"nombre": "Coursera", "url_base": "https://www.coursera.org/search?query={}&free=true", "descripcion": "Plataforma líder con cursos universitarios gratuitos (audit mode)", "idioma": "en", "categoria": "General", "nivel": "Avanzado", "confianza": 0.95, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.95},
                    {"nombre": "edX", "url_base": "https://www.edx.org/search?tab=course&availability=current&price=free&q={}", "descripcion": "Cursos de Harvard, MIT y otras universidades top (modo audit gratuito)", "idioma": "en", "categoria": "Académico", "nivel": "Avanzado", "confianza": 0.92, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.93},
                    {"nombre": "Kaggle Learn", "url_base": "https://www.kaggle.com/learn/search?q={}", "descripcion": "Microcursos prácticos de ciencia de datos con certificados gratuitos", "idioma": "en", "categoria": "Data Science", "nivel": "Intermedio", "confianza": 0.90, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.88},
                    {"nombre": "freeCodeCamp", "url_base": "https://www.freecodecamp.org/news/search/?query={}", "descripcion": "Certificados gratuitos completos en desarrollo web y ciencia de datos", "idioma": "en", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.93, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.91},
                    {"nombre": "PhET Simulations", "url_base": "https://phet.colorado.edu/en/search?q={}", "descripcion": "Simulaciones interactivas de ciencias y matemáticas", "idioma": "en", "categoria": "Ciencias", "nivel": "Todos", "confianza": 0.88, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.85},
                    {"nombre": "The Programming Historian", "url_base": "https://programminghistorian.org/en/lessons/?q={}", "descripcion": "Tutoriales académicos de programación y humanidades digitales", "idioma": "en", "categoria": "Programación", "nivel": "Avanzado", "confianza": 0.82, "tipo_certificacion": "gratuito", "validez_internacional": 0, "paises_validos": ["uk", "us", "ca"], "reputacion_academica": 0.80},
                    {"nombre": "Domestika (Gratuito)", "url_base": "https://www.domestika.org/es/search?query={}&free=1", "descripcion": "Cursos gratuitos de diseño creativo", "idioma": "es", "categoria": "Diseño", "nivel": "Intermedio", "confianza": 0.83, "tipo_certificacion": "pago", "validez_internacional": 1, "paises_validos": ["es", "mx", "ar", "cl"], "reputacion_academica": 0.82},
                    {"nombre": "Biblioteca Virtual Miguel de Cervantes", "url_base": "https://www.cervantesvirtual.com/buscar/?q={}", "descripcion": "Recursos académicos hispanos", "idioma": "es", "categoria": "Humanidades", "nivel": "Avanzado", "confianza": 0.87, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["es", "latam", "eu"], "reputacion_academica": 0.85},
                    {"nombre": "OER Commons", "url_base": "https://www.oercommons.org/search?q={}", "descripcion": "Recursos educativos abiertos", "idioma": "en", "categoria": "General", "nivel": "Todos", "confianza": 0.89, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.87}
                ]
                for p in plataformas_iniciales:
                    cursor.execute(
                        '''INSERT INTO plataformas_ocultas
                           (nombre, url_base, descripcion, idioma, categoria, nivel, confianza,
                            ultima_verificacion, activa, tipo_certificacion, validez_internacional,
                            paises_validos, reputacion_academica)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            p["nombre"], p["url_base"], p["descripcion"], p["idioma"], p["categoria"],
                            p["nivel"], p["confianza"], datetime.now().isoformat(), 1, p["tipo_certificacion"],
                            int(p["validez_internacional"]), safe_json_dumps(p["paises_validos"]), p["reputacion_academica"]
                        )
                    )
            conn.commit()
        logger.info("✅ Base de datos inicializada correctamente")
        migrate_database()
        return True
    except Exception as e:
        logger.error(f"❌ Error Init DB: {e}")
        return False

init_advanced_database()

# ============================================================
# 6. UTILIDADES GENERALES & CHAT PARCHEADO
# ============================================================
# ... (igual que en tu archivo original)

def get_codigo_idioma(nombre_idioma: str) -> str:
    return {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt", "es": "es", "en": "en", "pt": "pt"}.get(nombre_idioma, "es")

def generar_id_unico(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    if nivel_solicitado not in ("Cualquiera", "Todos"):
        return nivel_solicitado
    t = (texto or "").lower()
    if any(x in t for x in ['principiante', 'básico', 'beginner', 'desde cero', 'intro']):
        return "Principiante"
    if any(x in t for x in ['avanzado', 'advanced', 'experto', 'expert']):
        return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    tema = (tema or "").lower()
    if any(x in tema for x in ['python', 'java', 'javascript', 'web', 'code', 'programación', 'desarrollo']):
        return "Programación"
    if any(x in tema for x in ['data', 'datos', 'ia', 'ai', 'machine learning', 'ciencia de datos']):
        return "Data Science"
    if any(x in tema for x in ['design', 'diseño', 'ux', 'ui']):
        return "Diseño"
    if any(x in tema for x in ['marketing', 'negocios', 'business', 'finanzas', 'economía']):
        return "Negocios"
    return "General"

def extraer_plataforma(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        if 'youtube' in domain: return 'YouTube'
        if 'coursera' in domain: return 'Coursera'
        if 'udemy' in domain: return 'Udemy'
        if 'edx' in domain: return 'edX'
        if 'khanacademy' in domain: return 'Khan Academy'
        if 'freecodecamp' in domain: return 'freeCodeCamp'
        if not domain: return "Web"
        parts = domain.split('.')
        return parts[-2].title() if len(parts) >= 2 else domain.title()
    except:
        return "Web"

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    t = (url + (titulo or "") + (descripcion or "")).lower()
    invalidas = ['comprar', 'buy', 'precio', 'price', 'premium', 'paid', 'only', 'exclusive', 'suscripción', 'subscription', 'membership', 'register now', 'matrícula']
    validas = ['curso', 'tutorial', 'aprender', 'learn', 'gratis', 'free', 'class', 'education', 'educación', 'certificado', 'certificate']
    dominios = ['.edu', 'coursera', 'edx', 'khanacademy', 'udemy', 'youtube', 'freecodecamp', '.gov', '.gob', '.org']
    if any(i in t for i in invalidas): return False
    return any(v in t for v in validas) or any(d in url.lower() for d in dominios)

# --- PARCHE DE LIMPIEZA PARA CHAT ---
def limpiar_html_visible(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r'\{.*\}\s*$', '', texto, flags=re.DOTALL).strip()  # bloque JSON al final
    texto = re.sub(r'<[^>]+>', '', texto).strip()  # etiquetas HTML en toda la cadena
    return texto

def ui_chat_mostrar(mensaje: str, rol: str):
    texto_limpio = limpiar_html_visible(mensaje)
    if not texto_limpio:
        return
    if rol == "assistant":
        st.markdown(f"🤖 **IA:** {texto_limpio}")
    elif rol == "user":
        st.markdown(f"👤 **Tú:** {texto_limpio}")

# ============================================================
# 7. INTEGRACIÓN GROQ (Ahora usa ia_module.py)
# ============================================================

# ✅ LAS FUNCIONES `analizar_recurso_groq_sync` Y `chatgroq` YA FUERON DEFINIDAS EN LA SECCIÓN 1

def ejecutar_analisis_background(resultados: List[RecursoEducativo]):
    """Ejecuta análisis en segundo plano usando el módulo externo."""
    if not st.session_state.features.get("enable_groq_analysis", True):
        return
    pendientes = [r for r in resultados if r.analisis_pendiente]
    if not pendientes:
        return
    for r in pendientes:
        # Usamos la función del módulo externo
        analisis = analizar_recurso_groq(
            titulo=r.titulo,
            descripcion=r.descripcion,
            nivel=r.nivel,
            categoria=r.categoria,
            plataforma=r.plataforma
        )
        r.metadatos_analisis = analisis
        r.analisis_pendiente = False
        time.sleep(0.3)  # rate limiting ligero

# ============================================================
# 8. BÚSQUEDA MULTICAPA (Google, Conocidas, Ocultas, DDG opcional)
# ============================================================
# ... (igual que en tu archivo original)

@async_profile
async def buscar_en_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_google_api", True):
        return []
    if not validate_api_key(GOOGLE_API_KEY, "google") or not GOOGLE_CX:
        return []
    try:
        query_base = f"{tema} curso gratuito certificado"
        if nivel not in ("Cualquiera", "Todos"):
            query_base += f" nivel {nivel.lower()}"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': query_base, 'num': 5, 'lr': f'lang_{idioma}'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=8) as response:
                if response.status != 200:
                    return []
                data = await response.json()
                items = data.get('items', [])
                resultados: List[RecursoEducativo] = []
                for item in items:
                    url_item = item.get('link', '')
                    titulo = item.get('title', '')
                    descripcion = item.get('snippet', '')
                    if not es_recurso_educativo_valido(url_item, titulo, descripcion):
                        continue
                    nivel_calc = determinar_nivel(titulo + " " + descripcion, nivel)
                    confianza = 0.83
                    if any(d in url_item.lower() for d in ['.edu', 'coursera.org', 'edx.org', 'freecodecamp.org', '.gov']):
                        confianza = min(confianza + 0.1, 0.95)
                    resultados.append(RecursoEducativo(
                        id=generar_id_unico(url_item),
                        titulo=titulo or f"Recurso {generar_id_unico(url_item)}",
                        url=url_item,
                        descripcion=descripcion or "Sin descripción disponible.",
                        plataforma=extraer_plataforma(url_item),
                        idioma=idioma,
                        nivel=nivel_calc,
                        categoria=determinar_categoria(tema),
                        certificacion=None,
                        confianza=confianza,
                        tipo="verificada",
                        ultima_verificacion=datetime.now().isoformat(),
                        activo=True,
                        metadatos={'fuente': 'google_api'}
                    ))
                return resultados[:5]
    except Exception as e:
        logger.error(f"Error Google API: {e}")
        return []

def buscar_en_plataformas_conocidas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_known_platforms", True):
        return []
    recursos: List[RecursoEducativo] = []
    plataformas = {
        "es": [
            {"nombre": "YouTube Educativo", "url": f"https://www.youtube.com/results?search_query=curso+gratis+{quote_plus(tema)}"},
            {"nombre": "Coursera (ES)", "url": f"https://www.coursera.org/search?query={quote_plus(tema)}&languages=es&free=true"},
            {"nombre": "Udemy (Gratis)", "url": f"https://www.udemy.com/courses/search/?q={quote_plus(tema)}&price=price-free&lang=es"},
            {"nombre": "Khan Academy (ES)", "url": f"https://es.khanacademy.org/search?page_search_query={quote_plus(tema)}"}
        ],
        "en": [
            {"nombre": "YouTube Education", "url": f"https://www.youtube.com/results?search_query=free+course+{quote_plus(tema)}"},
            {"nombre": "Khan Academy", "url": f"https://www.khanacademy.org/search?page_search_query={quote_plus(tema)}"},
            {"nombre": "Coursera", "url": f"https://www.coursera.org/search?query={quote_plus(tema)}&free=true"},
            {"nombre": "Udemy (Free)", "url": f"https://www.udemy.com/courses/search/?q={quote_plus(tema)}&price=price-free&lang=en"},
            {"nombre": "edX", "url": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={quote_plus(tema)}"},
            {"nombre": "freeCodeCamp", "url": f"https://www.freecodecamp.org/news/search/?query={quote_plus(tema)}"}
        ],
        "pt": [
            {"nombre": "YouTube BR", "url": f"https://www.youtube.com/results?search_query=curso+gratuito+{quote_plus(tema)}"},
            {"nombre": "Coursera (PT)", "url": f"https://www.coursera.org/search?query={quote_plus(tema)}&languages=pt&free=true"},
            {"nombre": "Udemy (PT)", "url": f"https://www.udemy.com/courses/search/?q={quote_plus(tema)}&price=price-free&lang=pt"},
            {"nombre": "Khan Academy (PT)", "url": f"https://pt.khanacademy.org/search?page_search_query={quote_plus(tema)}"}
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
        if len(recursos) >= 6:
            break
    return recursos

def buscar_en_plataformas_ocultas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_hidden_platforms", True):
        return []
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            query = '''
            SELECT nombre, url_base, descripcion, nivel, confianza,
                   tipo_certificacion, validez_internacional, paises_validos, reputacion_academica
            FROM plataformas_ocultas
            WHERE activa = 1 AND idioma = ?
            '''
            params = [idioma]
            if nivel not in ("Cualquiera", "Todos"):
                query += " AND (nivel = ? OR nivel = 'Todos')"
                params.append(nivel)
            query += " ORDER BY confianza DESC LIMIT 6"
            cursor.execute(query, params)
            filas = cursor.fetchall()
            recursos: List[RecursoEducativo] = []
            for r in filas:
                nombre, url_base, descripcion, nivel_db, confianza, tipo_cert, validez_int, paises_json, reputacion = r
                url_completa = url_base.format(quote_plus(tema))
                nivel_calc = nivel_db if nivel in ("Cualquiera", "Todos") else nivel
                cert = None
                if tipo_cert and tipo_cert != "none":
                    paises_val = safe_json_loads(paises_json, default_value=[])
                    if isinstance(paises_val, dict):
                        paises_val = paises_val.get("paises", ["global"])
                    elif not isinstance(paises_val, list):
                        paises_val = ["global"]
                    cert = Certificacion(
                        plataforma=nombre,
                        curso=tema,
                        tipo=tipo_cert,
                        validez_internacional=bool(validez_int),
                        paises_validos=paises_val,
                        costo_certificado=0.0 if tipo_cert == "gratuito" else 49.99,
                        reputacion_academica=reputacion or 0.8,
                        ultima_verificacion=datetime.now().isoformat()
                    )
                recursos.append(RecursoEducativo(
                    id=generar_id_unico(url_completa),
                    titulo=f"💎 {nombre} — {tema}",
                    url=url_completa,
                    descripcion=descripcion or "Sin descripción.",
                    plataforma=nombre,
                    idioma=idioma,
                    nivel=nivel_calc,
                    categoria=determinar_categoria(tema),
                    certificacion=cert,
                    confianza=float(confianza or 0.8),
                    tipo="oculta",
                    ultima_verificacion=datetime.now().isoformat(),
                    activo=True,
                    metadatos={"fuente": "plataformas_ocultas", "confianza_db": confianza}
                ))
            return recursos
    except Exception as e:
        logger.error(f"Error al obtener plataformas ocultas: {e}")
        return []

def eliminar_duplicados(resultados: List[RecursoEducativo]) -> List[RecursoEducativo]:
    seen = set()
    unicos: List[RecursoEducativo] = []
    for r in resultados:
        if r.url not in seen:
            seen.add(r.url)
            unicos.append(r)
    return unicos

@async_profile
async def buscar_recursos_multicapa(tema: str, idioma_seleccion_ui: str, nivel: str) -> List[RecursoEducativo]:
    cache_key = f"{tema}|{idioma_seleccion_ui}|{nivel}"
    cached = search_cache.get(cache_key)
    if cached:
        return cached
    idioma = get_codigo_idioma(idioma_seleccion_ui)
    resultados: List[RecursoEducativo] = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Buscando en plataformas ocultas...")
    ocultas = buscar_en_plataformas_ocultas(tema, idioma, nivel)
    resultados.extend(ocultas)
    progress_bar.progress(0.3)
    status_text.text("Consultando Google API...")
    google_res = await buscar_en_google_api(tema, idioma, nivel)
    resultados.extend(google_res)
    progress_bar.progress(0.6)
    status_text.text("Buscando en plataformas conocidas...")
    conocidas = buscar_en_plataformas_conocidas(tema, idioma, nivel)
    resultados.extend(conocidas)
    progress_bar.progress(0.85)
    status_text.text("Procesando y deduplicando resultados...")
    resultados = eliminar_duplicados(resultados)
    resultados.sort(key=lambda x: x.confianza, reverse=True)
    if st.session_state.features.get("enable_groq_analysis", True) and GROQ_AVAILABLE:
        for r in resultados[:st.session_state.features.get("max_analysis", 5)]:
            r.analisis_pendiente = True
    final = resultados[:st.session_state.features.get("max_results", 15)]
    search_cache.set(cache_key, final)
    progress_bar.progress(1.0)
    time.sleep(0.1)
    progress_bar.empty()
    status_text.empty()
    return final

# ============================================================
# 9. PROCESAMIENTO EN SEGUNDO PLANO (Background Workers)
# ============================================================
# ... (igual que en tu archivo original, pero ahora llama a la función del módulo)

def analizar_resultados_en_segundo_plano(resultados: List[RecursoEducativo]):
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True)):
        return
    try:
        for recurso in resultados:
            if recurso.analisis_pendiente and not recurso.metadatos_analisis:
                # Usa la función del módulo externo
                analisis = analizar_recurso_groq(
                    titulo=recurso.titulo,
                    descripcion=recurso.descripcion,
                    nivel=recurso.nivel,
                    categoria=recurso.categoria,
                    plataforma=recurso.plataforma
                )
                recurso.metadatos_analisis = analisis
                recurso.analisis_pendiente = False
                time.sleep(0.3)
    except Exception as e:
        logger.error(f"Error análisis background: {e}")

def worker():
    while True:
        try:
            tarea = background_tasks.get(timeout=60)
            if tarea is None:
                break
            tipo = tarea.get('tipo')
            params = tarea.get('parametros', {})
            if tipo == 'analizar_resultados':
                analizar_resultados_en_segundo_plano(**params)
            background_tasks.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error en tarea background: {e}")
            background_tasks.task_done()

def iniciar_tareas_background():
    if 'background_started' not in st.session_state:
        num_workers = min(MAX_BACKGROUND_TASKS, os.cpu_count() or 1)
        for _ in range(num_workers):
            threading.Thread(target=worker, daemon=True).start()
        st.session_state.background_started = True
        logger.info(f"✅ Workers background iniciados: {num_workers}")

def planificar_analisis_ia(resultados: List[RecursoEducativo]):
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True)):
        return
    # Nota: ahora ya no hay "analizar_recurso_groq_sync", solo usamos el módulo
    tarea = {'tipo': 'analizar_resultados', 'parametros': {'resultados': [r for r in resultados if r.analisis_pendiente]}}
    background_tasks.put(tarea)
    logger.info(f"🧠 Tarea IA planificada: {len(tarea['parametros']['resultados'])} resultados")

# ============================================================
# 10. UI ESTILOS Y COMPONENTES
# ============================================================
# ... (igual que en tu archivo original, incluyendo `mostrar_recurso`)

st.set_page_config(page_title="🎓 Buscador Profesional de Cursos", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")
init_feature_flags()

def apply_theme(theme: str):
    if theme == "dark":
        st.markdown("""
        <style>
        body { background-color: #0f111a; color: #e6edf3; }
        .resultado-card { background: #14171f; color: #e6edf3; border-left-color: #4CAF50; }
        </style>
        """, unsafe_allow_html=True)
    elif theme == "light":
        pass  # default
    else:
        # auto -> deja por defecto (Streamlit maneja tema)
        pass

apply_theme(st.session_state.features.get("ui_theme", "auto"))

st.markdown("""
<style>
.main-header {
  background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
  color: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.main-header h1 { margin: 0; font-size: 2.3rem; }
.resultado-card {
  border-radius: 15px; padding: 20px; margin-bottom: 20px; background: white;
  box-shadow: 0 5px 20px rgba(0,0,0,0.08); border-left: 6px solid #4CAF50;
  transition: transform .2s;
}
.resultado-card:hover { transform: translateY(-3px); }
.nivel-principiante { border-left-color: #2196F3 !important; }
.nivel-intermedio { border-left-color: #4CAF50 !important; }
.nivel-avanzado { border-left-color: #FF9800 !important; }
.plataforma-oculta { border-left-color: #FF6B35 !important; background: #fff5f0; }
.certificado-badge { display:inline-block;padding:4px 10px;border-radius:12px;font-size:.8rem;font-weight:bold;background:#e8f5e9;color:#2e7d32;margin-right:5px; }
a { text-decoration: none !important; }
.status-badge { display:inline-block;padding:4px 10px;border-radius:15px;font-size:.8rem;font-weight:bold;background:rgba(255,255,255,0.2); }
.smalltext { font-size: 0.85rem; color: #607d8b; }
.badge-pendiente { display:inline-block; padding:3px 8px; background:#ede7f6; color:#6a1b9a; border-radius:12px; font-size:.75rem; }
.badge-ok { display:inline-block; padding:3px 8px; background:#e8f5e9; color:#2e7d32; border-radius:12px; font-size:.75rem; }
.tooltip { font-size:0.8rem; color:#78909c; }
</style>
""", unsafe_allow_html=True)

def link_button(url: str, label: str = "➡️ Acceder al recurso") -> str:
    if not url:
        return ""
    return f'''<a href="{url}" target="_blank" style="display:inline-block;background:linear-gradient(to right,#6a11cb,#2575fc);color:white;padding:10px 16px;border-radius:8px;font-weight:bold;">{label}</a>'''

def badge_certificacion(cert: Optional[Certificacion]) -> str:
    if not cert: return ""
    html = ""
    if cert.tipo == "gratuito":
        html += '<span class="certificado-badge">✅ Certificado Gratuito</span>'
    elif cert.tipo == "audit":
        html += '<span class="certificado-badge" style="background:#e3f2fd;color:#1565c0;">🎓 Modo Audit</span>'
    elif cert.tipo == "pago":
        html += '<span class="certificado-badge" style="background:#fff3e0;color:#ef6c00;">💰 Certificado de Pago</span>'
    if cert.validez_internacional:
        html += '<span class="certificado-badge" style="background:#e3f2fd;color:#1565c0;">🌐 Validez Internacional</span>'
    return html

def clase_nivel(nivel: str) -> str:
    return {"Principiante": "nivel-principiante", "Intermedio": "nivel-intermedio", "Avanzado": "nivel-avanzado"}.get(nivel, "")

def mostrar_recurso(r: RecursoEducativo, idx: int):
    extra_class = "plataforma-oculta" if r.tipo == "oculta" else ""
    nivel_class = clase_nivel(r.nivel)
    cert_html = badge_certificacion(r.certificacion)
    ia_block = ""
    estado = ""
    if r.metadatos_analisis:
        data = r.metadatos_analisis
        cal = int(data.get('calidad_ia', 0)*100)
        rel = int(data.get('relevancia_ia', 0)*100)
        ia_block = f"""
        <div style="background:#f3e5f5;padding:12px;border-radius:8px;margin:12px 0;border-left:4px solid #9c27b0;">
            <strong>🧠 Análisis IA:</strong> Calidad {cal}% • Relevancia {rel}%<br>
            {data.get('recomendacion_personalizada', '')}
        </div>"""
        estado = '<span class="badge-ok">IA listo</span>'
    elif r.analisis_pendiente:
        ia_block = "<div style='color:#9c27b0;font-size:0.9em;margin:5px 0;'>⏳ Analizando...</div>"
        estado = '<span class="badge-pendiente">IA pendiente</span>'
    else:
        estado = '<span class="smalltext tooltip">Sin IA</span>'
    desc = r.descripcion or "Sin descripción disponible."
    titulo = r.titulo or "Recurso Educativo"
    fav_btn = f"""<button onclick="window.parent.postMessage({{'action':'add_fav','id':'{r.id}'}}, '*')" style="margin-left:8px;padding:6px 10px;border-radius:8px;border:1px solid #e0e0e0;background:#fafafa;cursor:pointer;">⭐ Favorito</button>"""
    st.markdown(f"""
<div class="resultado-card {nivel_class} {extra_class}">
  <h3 style="margin-top:0;">{titulo} {estado}</h3>
  <p><strong>📚 {r.nivel}</strong> | 🌐 {r.plataforma} | 🏷️ {r.categoria}</p>
  <p style="color:#555;">{desc}</p>
  <div style="margin-bottom:10px;">{cert_html}</div>
  {ia_block}
  <div style="margin-top:15px;">{link_button(r.url, "➡️ Acceder al recurso")}{fav_btn}</div>
  <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.8rem; color: #888;">
    Confianza: {r.confianza*100:.0f}% | Verificado: {datetime.fromisoformat(r.ultima_verificacion).strftime('%d/%m/%Y')}
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 11. FAVORITOS, FEEDBACK, EXPORT/IMPORT
# ============================================================
# ... (igual que en tu archivo original)

# [Mantener todo igual desde "11. FAVORITOS" hasta el final]

# ============================================================
# 12–30. Resto del código (paneles, main, db, etc.)
# ============================================================
# (TODAS LAS DEMÁS SECCIONES PERMANECEN SIN CAMBIOS)

# ... (El resto del archivo es idéntico al original: paneles, favoritos, main, etc.)

# NOTA: Por brevedad, no se repite todo el código final, ya que no cambia.
# Solo se modificaron las secciones de IA y se añadió el import seguro arriba.

# ============================================================
# ACTIVACIÓN FINAL
# ============================================================
if __name__ == "__main__":
    # ... (igual que en tu archivo original)
    def main_extended():
        # ... (igual)
        pass
    main_extended()
