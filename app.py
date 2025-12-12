# app.py — Consolidado Definitivo Ultra-Robust PRO (Versión v6.1 - Fix Color Texto)
# Objetivo: IA Omnisciente + Corrección de CSS para legibilidad (Texto Negro en Tarjetas).

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
GROQ_MODEL = "llama-3.3-70b-versatile"

def validate_api_key(key: str, key_type: str) -> bool:
    if not key or len(key) < 10:
        return False
    if key_type == "google" and not key.startswith(("AIza", "AIz")):
        return False
    return True

GROQ_AVAILABLE = False
try:
    import groq
    if GROQ_API_KEY and len(GROQ_API_KEY) >= 10:
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API disponible y validada")
    else:
        logger.warning("⚠️ Groq API Key ausente o inválida")
except ImportError:
    logger.warning("⚠️ Biblioteca 'groq' no instalada")

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
@dataclass
class Certificacion:
    plataforma: str
    curso: str
    tipo: str  # "gratuito", "pago", "audit", "none"
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
    analisis_pendiente: bool = False

@dataclass
class Favorito:
    id_recurso: str
    titulo: str
    url: str
    notas: str
    creado_en: str

@dataclass
class Feedback:
    id_recurso: str
    opinion: str
    rating: int
    creado_en: str

def safe_json_dumps(obj: Dict) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

def safe_json_loads(text: str, default_value: Any = None) -> Any:
    if default_value is None:
        default_value = {}
    try:
        return json.loads(text)
    except Exception:
        return default_value

# ============================================================
# 5. BASE DE DATOS (Context Manager, Migraciones, Auditoría)
# ============================================================
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
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            # Auditoría general de eventos
            c.execute('''
            CREATE TABLE IF NOT EXISTS auditoria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evento TEXT NOT NULL,
                detalle TEXT,
                creado_en TEXT NOT NULL
            )
            ''')
            # Favoritos de usuario
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
            # Feedback de usuario
            c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_recurso TEXT NOT NULL,
                opinion TEXT,
                rating INTEGER,
                creado_en TEXT NOT NULL
            )
            ''')
            # Sesiones de usuario
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
            # Telemetría opt-out: simple bandera global
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
    # 1. Eliminar bloques de código markdown (ej: ```json ... ```)
    texto = re.sub(r'```.*?```', '', texto, flags=re.DOTALL)
    # 2. Eliminar bloques JSON explícitos al final o inicio
    texto = re.sub(r'^\s*\{.*\}\s*$', '', texto, flags=re.DOTALL | re.MULTILINE)
    # 3. Eliminar etiquetas HTML
    texto = re.sub(r'<[^>]+>', '', texto)
    # 4. Eliminar artefactos de objetos JSON sueltos al final de la cadena
    texto = re.sub(r'\{.*\}\s*$', '', texto, flags=re.DOTALL)
    return texto.strip()

def ui_chat_mostrar(mensaje: str, rol: str):
    texto_limpio = limpiar_html_visible(mensaje)
    if not texto_limpio:
        return
    if rol == "assistant":
        st.markdown(f"🤖 **IA:** {texto_limpio}")
    elif rol == "user":
        st.markdown(f"👤 **Tú:** {texto_limpio}")

# ============================================================
# 7. INTEGRACIÓN GROQ (Análisis & Chat - 3 CEREBROS)
# ============================================================
def analizar_recurso_groq_sync(recurso: RecursoEducativo, perfil: Dict):
    """Worker robusto para Groq con manejo de errores mejorado y JSON format."""
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True)):
        recurso.metadatos_analisis = {
            "calidad_ia": recurso.confianza,
            "relevancia_ia": recurso.confianza,
            "recomendacion_personalizada": "Análisis IA no disponible o deshabilitado.",
            "razones_calidad": [],
            "advertencias": ["Análisis IA deshabilitado o no disponible"]
        }
        return

    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Evalúa este curso y devuelve únicamente un objeto JSON válido, sin texto adicional.
        TÍTULO: {recurso.titulo}
        DESCRIPCIÓN: {recurso.descripcion}
        NIVEL: {recurso.nivel}
        CATEGORÍA: {recurso.categoria}
        PLATAFORMA: {recurso.plataforma}

        Formato de salida JSON esperado:
        {{
            "calidad_educativa": 0.85,
            "relevancia_usuario": 0.90,
            "razones_calidad": ["razon1","razon2"],
            "recomendacion_personalizada": "Conclusión breve y útil para el usuario.",
            "advertencias": []
        }}
        """
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.2, 
            max_tokens=600,
            response_format={"type": "json_object"}, # Fix para obligar JSON
        )

        contenido = resp.choices[0].message.content or "{}"
        data = safe_json_loads(contenido)

        recurso.metadatos_analisis = {
            "calidad_ia": float(data.get("calidad_educativa", recurso.confianza)),
            "relevancia_ia": float(data.get("relevancia_usuario", recurso.confianza)),
            "recomendacion_personalizada": data.get("recomendacion_personalizada", "Análisis no concluyente."),
            "razones_calidad": data.get("razones_calidad", []),
            "advertencias": data.get("advertencias", [])
        }

        # Ajuste de confianza basado en la IA
        ia_prom = (recurso.metadatos_analisis["calidad_ia"] + recurso.metadatos_analisis["relevancia_ia"]) / 2.0
        recurso.confianza = min(max(recurso.confianza * 0.7 + ia_prom * 0.3, 0.5), 0.98) 

    except Exception as e:
        logger.error(f"Error en Groq Worker para '{recurso.titulo}': {e}")
        recurso.metadatos_analisis = {
            "calidad_ia": 0.0,
            "relevancia_ia": 0.0,
            "recomendacion_personalizada": "No se pudo completar el análisis IA.",
            "razones_calidad": [],
            "advertencias": [f"Error de API: {str(e)}"]
        }

def ejecutar_analisis_background(resultados: List[RecursoEducativo]):
    if not st.session_state.features.get("enable_groq_analysis", True):
        return
    pendientes = [r for r in resultados if r.analisis_pendiente]
    if not pendientes:
        return
    for r in pendientes:
        executor.submit(analizar_recurso_groq_sync, r, {})

# --- FUNCIÓN MEJORADA: Contexto Rico (Estadísticas + Plataformas) ---
def obtener_contexto_db_para_ia() -> str:
    """Extrae estadísticas vivas y plataformas para que la IA sea un verdadero admin."""
    texto_contexto = "=== 1. CONTEXTO HISTÓRICO DE LA PÁGINA ===\n"
    
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            
            # 1. Lo más buscado (Tendencias)
            c.execute("SELECT tema, COUNT(*) as total FROM analiticas_busquedas GROUP BY tema ORDER BY total DESC LIMIT 5")
            top_busquedas = c.fetchall()
            if top_busquedas:
                texto_contexto += "🔥 TENDENCIAS (Lo que más busca la gente):\n"
                for row in top_busquedas:
                    texto_contexto += f"- {row[0]} ({row[1]} búsquedas)\n"
            
            # 2. Favoritos (Lo que la gente 'descarga' o guarda)
            c.execute("SELECT titulo, COUNT(*) as total FROM favoritos GROUP BY titulo ORDER BY total DESC LIMIT 5")
            top_favs = c.fetchall()
            if top_favs:
                texto_contexto += "\n⭐ LOS FAVORITOS (Cursos más guardados/populares):\n"
                for row in top_favs:
                    texto_contexto += f"- {row[0]} ({row[1]} usuarios lo guardaron)\n"
            else:
                texto_contexto += "\n⭐ LOS FAVORITOS: Aún no hay cursos guardados por usuarios.\n"

            # 3. Plataformas Disponibles (Catálogo)
            c.execute("SELECT nombre, categoria FROM plataformas_ocultas WHERE activa=1 LIMIT 10")
            plats = c.fetchall()
            texto_contexto += "\n📚 CATÁLOGO DE PLATAFORMAS (Recomienda estas):\n"
            for p in plats:
                texto_contexto += f"- {p[0]} (Tipo: {p[1]})\n"
                
    except Exception as e:
        logger.error(f"Error generando contexto IA: {e}")
        return "Error leyendo base de datos."
        
    return texto_contexto

# --- FUNCIÓN MEJORADA: Chat Omnisciente ---
def chatgroq(mensajes: List[Dict[str, str]]) -> str:
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_chat_ia", True)):
        return "🧠 IA no disponible. Usa el buscador superior para encontrar cursos ahora."
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        # 1. Obtenemos Contexto Histórico
        contexto_vivo = obtener_contexto_db_para_ia()
        
        # 2. Obtenemos Contexto en Tiempo Real (Resultados en Pantalla)
        resultados_en_pantalla = st.session_state.get('resultados', [])
        contexto_real_time = "=== 2. RESULTADOS EN PANTALLA AHORA MISMO (Recomienda de aquí) ===\n"
        if resultados_en_pantalla:
            for r in resultados_en_pantalla[:7]: # Solo los primeros 7 para no saturar
                ia_score = ""
                if r.metadatos_analisis:
                    calidad = int(r.metadatos_analisis.get('calidad_ia', 0) * 100)
                    ia_score = f"[IA Score: {calidad}/100]"
                contexto_real_time += f"- {r.titulo} ({r.plataforma}) {ia_score}\n"
        else:
            contexto_real_time += "El usuario aún no ha realizado una búsqueda o no hay resultados en pantalla.\n"

        # 3. Construimos el Prompt Maestro
        system_prompt = (
            "Eres 'CursosBot', el recepcionista, administrador y consejero experto de este 'Buscador Profesional'. "
            "Tu personalidad es amable, profesional y servicial. Tienes ACCESO TOTAL a la información de la página.\n\n"
            f"{contexto_vivo}\n\n"
            f"{contexto_real_time}\n\n"
            "INSTRUCCIONES MAESTRAS:\n"
            "1. Si el usuario pregunta qué buscar, basa tu respuesta en las TENDENCIAS y FAVORITOS históricos.\n"
            "2. Si el usuario pregunta 'qué me recomiendas de lo que veo', REVISA LOS RESULTADOS EN PANTALLA y recomienda los que tengan mejor 'IA Score' o confianza.\n"
            "3. Si el usuario saluda, preséntate como el asistente inteligente de la plataforma.\n"
            "4. Respuestas cortas, al grano y sin formato JSON (usa texto plano y emojis)."
        )
        
        groq_msgs = [{"role": "system", "content": system_prompt}] + mensajes
        
        # Temperatura baja para fidelidad a los datos
        resp = client.chat.completions.create(
            messages=groq_msgs, model=GROQ_MODEL, temperature=0.4, max_tokens=600
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Error en chat Groq: {e}")
        return "Tuve un pequeño error de conexión, pero aquí estoy. ¿En qué más puedo ayudar?"

# ============================================================
# 8. BÚSQUEDA MULTICAPA (Google, Conocidas, Ocultas, DDG opcional)
# ============================================================
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

def buscar_en_sitios_profundos(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Genera 1 recurso educativo de fuente poco conocida pero de alta calidad y gratuita."""
    if not st.session_state.features.get("enable_known_platforms", True):
        return []
    
    sitios_profundos = [
        ("OER Commons", "https://www.oercommons.org/search?q={}", "Global", "en"),
        ("MERLOT", "https://www.merlot.org/merlot/materials.htm?keywords={}", "EE.UU.", "en"),
        ("OpenStax", "https://openstax.org/search?query={}", "EE.UU.", "en"),
        ("Instituto Cervantes", "https://cvc.cervantes.es/ensenanza/biblioteca_ele/?q={}", "España", "es"),
        ("UNESCO OER", "https://en.unesco.org/themes/education/ict/open-educational-resources", "Global", "en"),
        ("Internet Archive", "https://archive.org/search?query={}+course", "Global", "en"),
        ("Open Culture", "https://www.openculture.com/freeonlinecourses", "Global", "en"),
        ("LibreTexts", "https://libretexts.org/search?terms={}", "Global", "en"),
        ("Open WHO", "https://open.who.int/", "Global", "en"),
        ("Biblioteca Pensamiento Novohispano", "https://pensamientonovohispano.org/", "México", "es"),
        ("RIAL", "https://www.rial.org.mx/", "Iberoamérica", "es"),
        ("Code.org", "https://code.org/educate/search?q={}", "Global", "en"),
        ("CK-12", "https://www.ck12.org/search/?q={}", "Global", "en"),
        ("Saylor Academy", "https://learn.saylor.org/course/index.php?search={}", "Global", "en")
    ]
    
    # Filtrar por idioma
    candidatos = [
        (nombre, url_base.format(quote_plus(tema)))
        for nombre, url_base, _, idioma_req in sitios_profundos
        if idioma == idioma_req
    ]
    
    if not candidatos:
        candidatos = [
            (nombre, url_base.format(quote_plus(tema)))
            for nombre, url_base, _, _ in sitios_profundos
        ]
    
    nombre_seleccionado, url_seleccionada = random.choice(candidatos)
    
    return [RecursoEducativo(
        id=generar_id_unico(url_seleccionada),
        titulo=f"🔍 Profundo: {tema} en {nombre_seleccionado}",
        url=url_seleccionada,
        descripcion=f"Recurso educativo gratuito de fuente poco indexada pero confiable: {nombre_seleccionado}.",
        plataforma=nombre_seleccionado,
        idioma=idioma,
        nivel=nivel if nivel not in ("Cualquiera", "Todos") else "Intermedio",
        categoria=determinar_categoria(tema),
        certificacion=None,
        confianza=0.77,
        tipo="conocida",
        ultima_verificacion=datetime.now().isoformat(),
        activo=True,
        metadatos={"fuente": "busqueda_profunda"}
    )]
def buscar_en_plataformas_conocidas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_known_platforms", True):
        return []
    
    # === 1. Lista predefinida de plataformas conocidas ===
    plataformas_predef = {
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
    lista_base = plataformas_predef.get(idioma, plataformas_predef["en"])

    # === 2. Decisión: ¿modo predefinido o aleatorio? ===
    if random.random() < 0.5:
        # ✅ 50%: Modo predefinido → mostrar plataformas conocidas
        return [
            RecursoEducativo(
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
            )
            for plat in lista_base[:3]  # Máximo 3
        ]
    else:
        # 🔁 50%: Modo aleatorio → 1 profundo + 2 universidades
        recursos = []
        
        # ➕ 1 recurso profundo (representa ~25% del total final)
        recursos.extend(buscar_en_sitios_profundos(tema, idioma, nivel))
        
        # ➕ 2 universidades de élite
        universidades = [
            "mit.edu", "stanford.edu", "harvard.edu", "ox.ac.uk", "cam.ac.uk",
            "berkeley.edu", "ethz.ch", "nus.edu.sg", "utoronto.ca", "kaist.ac.kr"
        ]
        for dominio in random.sample(universidades, 2):
            url = f"https://www.google.com/search?q=site:{dominio}+{quote_plus(tema)}+curso+free"
            recursos.append(RecursoEducativo(
                id=generar_id_unico(url),
                titulo=f"🎓 Explorar {tema} en {dominio.title()}",
                url=url,
                descripcion=f"Recursos académicos gratuitos en la universidad {dominio}.",
                plataforma=dominio.split('.')[0].title(),
                idioma=idioma,
                nivel=nivel if nivel != "Cualquiera" else "Intermedio",
                categoria="Académico",
                certificacion=None,
                confianza=0.78,
                tipo="conocida",
                ultima_verificacion=datetime.now().isoformat(),
                activo=True,
                metadatos={"fuente": "discovery_universidad"}
            ))
        
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
@profile
def verificar_calidad_recurso(recurso: RecursoEducativo, tema: str) -> bool:
    """
    Verifica que:
    1. La URL responde con código 200.
    2. El contenido de la página incluye al menos 2 palabras clave del tema.
    Retorna True si pasa ambas pruebas.
    """
    try:
        import aiohttp
        import asyncio

        # Palabras clave derivadas del tema (simplificado)
        palabras_tema = set(re.split(r'\W+', tema.lower()))
        palabras_tema.discard('')  # Eliminar strings vacíos

        if len(palabras_tema) == 0:
            return True  # No hay tema, no se puede validar

        async def _verificar():
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get(recurso.url, headers={"User-Agent": "Mozilla/5.0 (compatible; BuscadorCursos/1.0)"}) as resp:
                        if resp.status != 200:
                            return False
                        contenido = await resp.text()
                        contenido_limpio = contenido.lower()

                        coincidencias = sum(1 for palabra in palabras_tema if palabra in contenido_limpio)
                        return coincidencias >= min(2, len(palabras_tema))  # al menos 2 o todas si hay <2
                except Exception:
                    return False

        # Ejecutar async en contexto sync
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        resultado = loop.run_until_complete(_verificar())
        loop.close()
        return resultado

    except Exception as e:
        logger.warning(f"Error en verificación de calidad para {recurso.url}: {e}")
        return False
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
def analizar_resultados_en_segundo_plano(resultados: List[RecursoEducativo]):
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True)):
        return
    try:
        for recurso in resultados:
            if recurso.analisis_pendiente and not recurso.metadatos_analisis:
                analizar_recurso_groq_sync(recurso, {})
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
    tarea = {'tipo': 'analizar_resultados', 'parametros': {'resultados': [r for r in resultados if r.analisis_pendiente]}}
    background_tasks.put(tarea)
    logger.info(f"🧠 Tarea IA planificada: {len(tarea['parametros']['resultados'])} resultados")

# ============================================================
# 10. UI ESTILOS Y COMPONENTES
# ============================================================
st.set_page_config(page_title="🎓 Buscador Profesional de Cursos", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="altair")
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
  color: #1a1a1a; /* Fix: Texto negro forzado para tarjetas blancas */
  box-shadow: 0 5px 20px rgba(0,0,0,0.08); border-left: 6px solid #4CAF50;
  transition: transform .2s;
}
.resultado-card h3, .resultado-card p { color: #1a1a1a; } /* Fix adicional: Forzar títulos y párrafos */
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
    fav_btn = f"""<button onclick="window.parent.postMessage({{'action':'add_fav','id':'{r.id}'}}, '*')" style="margin-left:8px;padding:6px 10px;border-radius:8px;border:1px solid #e0e0e0;background:#fafafa;cursor:pointer;color:#333;">⭐ Favorito</button>"""

    st.markdown(f"""
<div class="resultado-card {nivel_class} {extra_class}">
  <h3 style="margin-top:0;">{titulo} {estado}</h3>
  <p><strong>📚 {r.nivel}</strong> | 🌐 {r.plataforma} | 🏷️ {r.categoria}</p>
  <p>{desc}</p>
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
def agregar_favorito(r: RecursoEducativo, notas: str = "") -> bool:
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO favoritos (id_recurso, titulo, url, notas, creado_en) VALUES (?, ?, ?, ?, ?)",
                      (r.id, r.titulo, r.url, notas, datetime.now().isoformat()))
            conn.commit()
        st.session_state.get("favoritos_cache", set()).add(r.id)
        return True
    except Exception as e:
        logger.error(f"Error agregando favorito: {e}")
        return False

def listar_favoritos() -> List[Favorito]:
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id_recurso, titulo, url, notas, creado_en FROM favoritos ORDER BY creado_en DESC")
            filas = c.fetchall()
        return [Favorito(*f) for f in filas]
    except Exception as e:
        logger.error(f"Error listando favoritos: {e}")
        return []

def registrar_feedback(id_recurso: str, opinion: str, rating: int) -> bool:
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO feedback (id_recurso, opinion, rating, creado_en) VALUES (?, ?, ?, ?)",
                      (id_recurso, opinion, rating, datetime.now().isoformat()))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error registrando feedback: {e}")
        return False

def exportar_busquedas(resultados: List[RecursoEducativo]) -> bytes:
    df = pd.DataFrame([{
        'id': r.id, 'titulo': r.titulo, 'url': r.url, 'plataforma': r.plataforma,
        'nivel': r.nivel, 'idioma': r.idioma, 'categoria': r.categoria,
        'confianza': r.confianza, 'tipo': r.tipo, 'verificado': r.ultima_verificacion
    } for r in resultados])
    return df.to_csv(index=False).encode('utf-8')

def importar_busquedas(csv_bytes: bytes) -> List[RecursoEducativo]:
    try:
        df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
        out: List[RecursoEducativo] = []
        for _, row in df.iterrows():
            out.append(RecursoEducativo(
                id=str(row.get('id', generar_id_unico(row.get('url', '')))),
                titulo=str(row.get('titulo', 'Recurso Importado')),
                url=str(row.get('url', '')),
                descripcion="Importado desde CSV",
                plataforma=str(row.get('plataforma', 'Web')),
                idioma=str(row.get('idioma', 'es')),
                nivel=str(row.get('nivel', 'Intermedio')),
                categoria=str(row.get('categoria', 'General')),
                certificacion=None,
                confianza=float(row.get('confianza', 0.8)),
                tipo=str(row.get('tipo', 'verificada')),
                ultima_verificacion=str(row.get('verificado', datetime.now().isoformat())),
                activo=True,
                metadatos={"fuente": "import_csv"}
            ))
        return out
    except Exception as e:
        logger.error(f"Error importando CSV: {e}")
        return []

# ============================================================
# 12. PANELES AVANZADOS (Configuración, Depuración, Cache Viewer)
# ============================================================
def panel_configuracion_avanzada():
    st.markdown("### ⚙️ Configuración avanzada")
    with st.expander("Preferencias de UI y rendimiento", expanded=False):
        theme = st.selectbox("Tema", ["auto", "dark", "light"], index=["auto","dark","light"].index(st.session_state.features["ui_theme"]))
        st.session_state.features["ui_theme"] = theme
        st.session_state.features["max_results"] = st.slider("Máx. resultados mostrados", 5, 30, st.session_state.features["max_results"])
        st.session_state.features["max_analysis"] = st.slider("Máx. análisis IA en paralelo", 0, 10, st.session_state.features["max_analysis"])

    with st.expander("Banderas de características (Feature Flags)", expanded=False):
        st.session_state.features["enable_google_api"] = st.checkbox("Google API", value=st.session_state.features["enable_google_api"])
        st.session_state.features["enable_known_platforms"] = st.checkbox("Plataformas conocidas", value=st.session_state.features["enable_known_platforms"])
        st.session_state.features["enable_hidden_platforms"] = st.checkbox("Plataformas ocultas DB", value=st.session_state.features["enable_hidden_platforms"])
        st.session_state.features["enable_groq_analysis"] = st.checkbox("Análisis IA (Groq)", value=st.session_state.features["enable_groq_analysis"] and GROQ_AVAILABLE)
        st.session_state.features["enable_chat_ia"] = st.checkbox("Chat IA", value=st.session_state.features["enable_chat_ia"])
        st.session_state.features["enable_favorites"] = st.checkbox("Favoritos", value=st.session_state.features["enable_favorites"])
        st.session_state.features["enable_feedback"] = st.checkbox("Feedback", value=st.session_state.features["enable_feedback"])
        st.session_state.features["enable_export_import"] = st.checkbox("Exportar/Importar", value=st.session_state.features["enable_export_import"])
        st.session_state.features["enable_offline_cache"] = st.checkbox("Modo offline con caché", value=st.session_state.features["enable_offline_cache"])
        st.session_state.features["enable_ddg_fallback"] = st.checkbox("DuckDuckGo fallback (no implementado real)", value=st.session_state.features["enable_ddg_fallback"])
        st.session_state.features["enable_debug_mode"] = st.checkbox("Modo depuración (logs verbosos)", value=st.session_state.features["enable_debug_mode"])

    if st.session_state.features["enable_debug_mode"]:
        st.info("Modo depuración activo. Los logs serán más detallados.")

    # Aplicar tema
    apply_theme(st.session_state.features.get("ui_theme", "auto"))

def panel_cache_viewer():
    st.markdown("### 🗂️ Caché de búsquedas")
    cache_items = search_cache.cache.items()
    st.write(f"Entradas en caché: {len(list(cache_items))}")
    for k, (val, ts) in cache_items:
        st.write(f"- Clave: {k} (guardado hace {int(time.time()-ts)}s) • Resultados: {len(val)}")
    if st.button("🧹 Vaciar caché", use_container_width=True):
        search_cache.cache.clear()
        st.success("Caché vaciada")

def panel_favoritos_ui():
    if not st.session_state.features.get("enable_favorites", True):
        return
    st.markdown("### ⭐ Favoritos")
    favs = listar_favoritos()
    if not favs:
        st.info("Sin favoritos aún.")
        return
    df = pd.DataFrame([{"Título": f.titulo, "URL": f.url, "Notas": f.notas, "Agregado": f.creado_en} for f in favs])
    st.table(df)

def panel_feedback_ui(resultados: List[RecursoEducativo]):
    if not st.session_state.features.get("enable_feedback", True):
        return
    st.markdown("### 📝 Feedback")
    # Selección de recurso para feedback
    opciones = {f"{r.titulo} ({r.plataforma})": r.id for r in resultados} if resultados else {}
    if not opciones:
        st.info("Busca recursos para poder enviar feedback.")
        return
    sel = st.selectbox("Selecciona un recurso para opinar", list(opciones.keys()))
    rating = st.slider("Calificación (1-5)", 1, 5, 4)
    opinion = st.text_area("Tu opinión (breve y clara)", "")
    if st.button("Enviar feedback", use_container_width=True):
        ok = registrar_feedback(opciones[sel], opinion, rating)
        if ok:
            st.success("¡Gracias por tu feedback!")
        else:
            st.error("No se pudo guardar el feedback.")

def panel_export_import_ui(resultados: List[RecursoEducativo]):
    if not st.session_state.features.get("enable_export_import", True):
        return
    st.markdown("### 🔄 Exportar / Importar")
    if resultados:
        csv_bytes = exportar_busquedas(resultados)
        st.download_button("📥 Exportar resultados (CSV)", csv_bytes, "cursos_export.csv", "text/csv", use_container_width=True)
    up = st.file_uploader("Importar resultados (CSV)", type=["csv"])
    if up is not None:
        imported = importar_busquedas(up.read())
        if imported:
            st.success(f"Importados {len(imported)} recursos desde CSV")
            for i, r in enumerate(imported[:5]):
                mostrar_recurso(r, i)
        else:
            st.error("No se pudieron importar datos.")

# ============================================================
# 13. APP PRINCIPAL (Búsqueda + Chat + Paneles)
# ============================================================
def render_header():
    st.markdown("""
    <div class="main-header">
      <h1>🎓 Buscador Profesional de Cursos</h1>
      <p>Descubre recursos educativos verificados con búsqueda inmediata y análisis IA en segundo plano</p>
      <div style="display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;">
        <span class="status-badge">✅ Sistema Activo</span>
        <span class="status-badge">⚡ AsyncIO Core</span>
        <span class="status-badge">🌐 Multilingüe</span>
        <span class="status-badge">🧠 IA opcional</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_search_form():
    col1, col2, col3 = st.columns([3, 1, 1])
    tema = col1.text_input("¿Qué quieres aprender?", placeholder="Ej: Python, Machine Learning, Diseño UX...")
    nivel = col2.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    idioma = col3.selectbox("Idioma", ["Español (es)", "Inglés (en)", "Portugués (pt)"])
    buscar = st.button("🚀 Buscar Cursos", type="primary", use_container_width=True)
    return tema, nivel, idioma, buscar

def render_results(resultados: List[RecursoEducativo]):
    if resultados:
        st.success(f"✅ Se encontraron {len(resultados)} recursos verificados.")
        if GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True):
            planificar_analisis_ia(resultados)
            time.sleep(0.4)
        for i, r in enumerate(resultados):
            mostrar_recurso(r, i)
        df = pd.DataFrame([{
            'Título': r.titulo,
            'URL': r.url,
            'Plataforma': r.plataforma,
            'Nivel': r.nivel,
            'Idioma': r.idioma,
            'Categoría': r.categoria,
            'Confianza': f"{r.confianza:.0%}",
            'Tipo': r.tipo
        } for r in resultados])
        st.download_button("📥 Descargar CSV", df.to_csv(index=False).encode('utf-8'), "cursos.csv", "text/csv", use_container_width=True)
    else:
        st.warning("No se encontraron resultados. Intenta con términos más generales.")

def sidebar_chat():
    if not st.session_state.features.get("enable_chat_ia", True):
        return
    with st.sidebar:
        st.header("💬 Asistente Educativo")
        if "chat_msgs" not in st.session_state:
            # Mensaje de bienvenida inicial
            st.session_state.chat_msgs = [
                {"role": "assistant", "content": "¡Hola! Soy CursosBot. ¿En qué puedo ayudarte hoy?"}
            ]

        for msg in st.session_state.chat_msgs:
            ui_chat_mostrar(msg["content"], msg["role"])

        user_input = st.chat_input("Pregunta sobre cursos...")
        if user_input:
            st.session_state.chat_msgs.append({"role": "user", "content": user_input})
            ui_chat_mostrar(user_input, "user")
            reply = chatgroq([{"role": "user", "content": user_input}])
            st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
            st.rerun()

def sidebar_status():
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Estado del sistema")
        try:
            with get_db_connection(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM plataformas_ocultas WHERE activa = 1")
                total_plataformas = c.fetchone()[0]
                st.metric("Plataformas activas", total_plataformas)
        except Exception:
            st.metric("Plataformas activas", 0)
        st.info(f"IA: {'✅ Disponible' if GROQ_AVAILABLE and st.session_state.features.get('enable_groq_analysis', True) else '⚠️ No disponible'}")

def render_footer():
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;color:#666;font-size:14px;padding:20px;background:#f8f9fa;border-radius:12px;">
        <strong>✨ Buscador Profesional de Cursos</strong><br>
        <span style="color: #2c3e50; font-weight: 500;">Resultados inmediatos • Cache inteligente • Alta disponibilidad</span><br>
        <em style="color: #7f8c8d;">Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')} • Versión: 3.3.0 • Estado: ✅ Activo</em><br>
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #ddd;">
            <code style="background:#f1f3f5;padding:2px 8px;border-radius:4px;color:#d32f2f;">
                IA opcional — Sistema funcional sin dependencias externas críticas
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 14. EVENT-BRIDGE PARA FAVORITOS (PostMessage desde botón HTML)
# ============================================================
def event_bridge():
    # En Streamlit no hay listener directo para window.postMessage.
    # Implementamos un puente simple: cuando el usuario hace click en Favorito,
    # le pedimos que confirme en un control de texto el ID del recurso y lo guardamos.
    st.markdown("### 🔗 Puente de eventos (Favoritos)")
    fav_id = st.text_input("ID del recurso a guardar como favorito (pegar desde botón)")
    fav_notas = st.text_input("Notas (opcional)")
    if st.button("Guardar favorito manual", use_container_width=True):
        # Sin resultados actuales, no podemos mapear; así que lo guardamos con URL vacía.
        r = RecursoEducativo(
            id=fav_id or f"manual_{int(time.time())}",
            titulo="Favorito manual",
            url="",
            descripcion="Añadido manualmente",
            plataforma="Manual",
            idioma="es",
            nivel="Intermedio",
            categoria="General",
            certificacion=None,
            confianza=0.8,
            tipo="verificada",
            ultima_verificacion=datetime.now().isoformat(),
            activo=True,
            metadatos={}
        )
        ok = agregar_favorito(r, fav_notas)
        if ok:
            st.success("Favorito guardado")
        else:
            st.error("No se pudo guardar el favorito")

# ============================================================
# 20. DUCKDUCKGO FALLBACK (OPCIONAL)
# ============================================================
@async_profile
async def buscar_en_duckduckgo(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_ddg_fallback", False):
        return []
    try:
        q = quote_plus(f"{tema} free course {nivel if nivel!='Cualquiera' else ''}".strip())
        url = f"https://duckduckgo.com/html/?q={q}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=8) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text()
                links = re.findall(r'href="(https?://[^"]+)"', text)
                resultados: List[RecursoEducativo] = []
                for link in links[:5]:
                    titulo = "Resultado en DuckDuckGo"
                    descripcion = "Resultado alternativo desde DuckDuckGo (parseo simple)."
                    if not es_recurso_educativo_valido(link, titulo, descripcion):
                        continue
                    resultados.append(RecursoEducativo(
                        id=generar_id_unico(link),
                        titulo=f"🦆 {titulo} — {tema}",
                        url=link,
                        descripcion=descripcion,
                        plataforma=extraer_plataforma(link),
                        idioma=idioma,
                        nivel=nivel if nivel != "Cualquiera" else "Intermedio",
                        categoria=determinar_categoria(tema),
                        certificacion=None,
                        confianza=0.70,
                        tipo="verificada",
                        ultima_verificacion=datetime.now().isoformat(),
                        activo=True,
                        metadatos={"fuente": "duckduckgo"}
                    ))
                return resultados
    except Exception as e:
        logger.error(f"DDG fallback error: {e}")
        return []

@async_profile
async def buscar_recursos_multicapa_ext(tema: str, idioma_seleccion_ui: str, nivel: str) -> List[RecursoEducativo]:
    base = await buscar_recursos_multicapa(tema, idioma_seleccion_ui, nivel)
    if not base and st.session_state.features.get("enable_ddg_fallback", False):
        idioma = get_codigo_idioma(idioma_seleccion_ui)
        ddg = await buscar_en_duckduckgo(tema, idioma, nivel)
        base.extend(ddg)
    base = eliminar_duplicados(base)
    base.sort(key=lambda x: x.confianza, reverse=True)
    return base[:st.session_state.features.get("max_results", 15)]

# ============================================================
# 21. ANALÍTICAS Y TRAZABILIDAD
# ============================================================
def log_search_event(tema: str, idioma: str, nivel: str, plataforma_origen: str, mostrados: int):
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO analiticas_busquedas (tema, idioma, nivel, timestamp, plataforma_origen, veces_mostrado)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tema, idioma, nivel, datetime.now().isoformat(), plataforma_origen, mostrados))
            conn.commit()
    except Exception as e:
        logger.error(f"Error log_search_event: {e}")

def log_click_event(tema: str, url: str, plataforma: str):
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE analiticas_busquedas
                SET veces_clickeado = veces_clickeado + 1
                WHERE tema = ?
                ORDER BY id DESC LIMIT 1
            """, (tema,))
            conn.commit()
    except Exception as e:
        logger.error(f"Error log_click_event: {e}")

def registrar_muestreo_estadistico(resultados: List[RecursoEducativo], tema: str, idioma_ui: str, nivel: str):
    idioma = get_codigo_idioma(idioma_ui)
    plataformas = ", ".join(sorted(set(r.plataforma for r in resultados)))
    log_search_event(tema, idioma, nivel, plataformas, len(resultados))

def boton_registrar_click(r: RecursoEducativo, tema: str):
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔖 Registrar click", key=f"reg_click_{r.id}"):
            log_click_event(tema, r.url, r.plataforma)
            st.success("Click registrado")

# ============================================================
# 22. ACCESIBILIDAD E I18N SIMPLE
# ============================================================
I18N = {
    "es": {
        "search_button": "🚀 Buscar Cursos",
        "enter_topic": "¿Qué quieres aprender?",
        "level": "Nivel",
        "language": "Idioma",
        "results_found": "Se encontraron {n} recursos verificados.",
        "no_results": "No se encontraron resultados. Intenta con términos más generales.",
        "favorites": "Favoritos",
        "feedback": "Feedback",
        "export_import": "Exportar / Importar",
    },
    "en": {
        "search_button": "🚀 Search Courses",
        "enter_topic": "What do you want to learn?",
        "level": "Level",
        "language": "Language",
        "results_found": "{n} verified resources found.",
        "no_results": "No results found. Try broader terms.",
        "favorites": "Favorites",
        "feedback": "Feedback",
        "export_import": "Export / Import",
    },
    "pt": {
        "search_button": "🚀 Buscar Cursos",
        "enter_topic": "O que você quer aprender?",
        "level": "Nível",
        "language": "Idioma",
        "results_found": "{n} recursos verificados encontrados.",
        "no_results": "Nenhum resultado encontrado. Tente termos mais gerais.",
        "favorites": "Favoritos",
        "feedback": "Feedback",
        "export_import": "Exportar / Importar",
    }
}

def get_i18n(lang_ui: str) -> Dict[str, str]:
    code = get_codigo_idioma(lang_ui)
    return I18N.get(code, I18N["es"])

# ============================================================
# 23. ADMIN DASHBOARD
# ============================================================
def admin_dashboard():
    st.markdown("### 🛠️ Panel admin")
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM analiticas_busquedas")
            t_busquedas = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM plataformas_ocultas WHERE activa = 1")
            t_plats = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM favoritos")
            t_favs = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM feedback")
            t_fb = c.fetchone()[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔎 Búsquedas", t_busquedas)
        col2.metric("📚 Plataformas activas", t_plats)
        col3.metric("⭐ Favoritos", t_favs)
        col4.metric("📝 Feedback", t_fb)
    except Exception as e:
        st.error(f"Error admin: {e}")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Vacuum DB", use_container_width=True):
            try:
                with get_db_connection(DB_PATH) as conn:
                    conn.execute("VACUUM")
                    conn.commit()
                st.success("DB optimizada (VACUUM)")
            except Exception as e:
                st.error(f"Error VACUUM: {e}")
    with colB:
        if st.button("🧹 Limpiar analíticas", use_container_width=True):
            try:
                with get_db_connection(DB_PATH) as conn:
                    conn.execute("DELETE FROM analiticas_busquedas")
                    conn.commit()
                st.success("Analíticas limpiadas")
            except Exception as e:
                st.error(f"Error limpieza: {e}")

# ============================================================
# 24. DIAGNÓSTICO DE ERRORES (LOG VIEWER)
# ============================================================
def log_viewer(max_lines: int = 200):
    st.markdown("### 🪵 Visor de logs")
    path = "buscador_cursos.log"
    if not os.path.exists(path):
        st.info("No hay logs aún.")
        return
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail = lines[-max_lines:]
        st.code("".join(tail))
    except Exception as e:
        st.error(f"Error leyendo logs: {e}")

# ============================================================
# 25. SESIONES DE USUARIO (CORREGIDO BLINDADO)
# ============================================================
def ensure_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess_{int(time.time())}_{random.randint(1000,9999)}"
        try:
            with get_db_connection(DB_PATH) as conn:
                c = conn.cursor()
                # PARCHE: Crear tabla si no existe antes de insertar
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
                c.execute("INSERT INTO sesiones (session_id, started_at, device, prefs_json) VALUES (?, ?, ?, ?)",
                          (st.session_state.session_id, datetime.now().isoformat(), "web", safe_json_dumps(st.session_state.get('features', {}))))
                conn.commit()
        except Exception as e:
            logger.error(f"⚠️ Error no crítico creando sesión: {e}")

def end_session():
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("UPDATE sesiones SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL",
                      (datetime.now().isoformat(), st.session_state.session_id))
            conn.commit()
    except Exception as e:
        logger.error(f"Error cerrando sesión: {e}")

# ============================================================
# 26. ACCESOS RÁPIDOS (TECLAS) Y AYUDA VISUAL
# ============================================================
def keyboard_tips():
    st.markdown("### ⌨️ Atajos")
    st.markdown("- Shift+Enter: enviar en chat")
    st.markdown("- Ctrl+K: abrir búsqueda rápida del navegador")
    st.markdown("- Alt+R: refrescar (según navegador)")
    st.markdown("- Ctrl+L: enfocarse en barra de URL (navegador)")

# ============================================================
# 27. EXTENSIONES: ETIQUETAS Y NOTAS EN RESULTADOS
# ============================================================
def notas_usuario_widget(r: RecursoEducativo):
    st.markdown("#### 🗒️ Notas del usuario")
    default_note = ""
    note = st.text_area(f"Notas para: {r.titulo}", default_note, key=f"note_{r.id}")
    if st.button("💾 Guardar nota", key=f"save_note_{r.id}"):
        ok = agregar_favorito(r, note)
        if ok:
            st.success("Nota guardada como favorito.")
        else:
            st.error("No se pudo guardar la nota.")

def render_notas_para_resultados(resultados: List[RecursoEducativo]):
    st.markdown("### 🗂️ Notas rápidas")
    for r in resultados[:3]:
        notas_usuario_widget(r)

# ============================================================
# 28. REPORTES RÁPIDOS
# ============================================================
def reportes_rapidos():
    st.markdown("### 📈 Reportes rápidos")
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT tema, COUNT(*) AS total
                FROM analiticas_busquedas
                GROUP BY tema
                ORDER BY total DESC
                LIMIT 5
            """)
            rows = c.fetchall()
        if rows:
            df = pd.DataFrame([{"Tema": r[0], "Búsquedas": r[1]} for r in rows])
            st.bar_chart(df.set_index("Tema"))
        else:
            st.info("Aún no hay suficientes datos para reportes.")
    except Exception as e:
        st.error(f"Error reporte: {e}")

# ============================================================
# 16. PRUEBAS BÁSICAS (Sanity Checks)
# ============================================================
def run_basic_tests():
    with st.expander("🧪 Pruebas básicas (Diagnóstico)"):
        try:
            # Test de utilidades
            assert determinar_nivel("Curso avanzado", "Cualquiera") == "Avanzado"
            assert determinar_nivel("Curso básico", "Cualquiera") == "Principiante"
            assert determinar_nivel("Curso intermedio", "Cualquiera") == "Intermedio"
            assert determinar_categoria("Python para ciencia de datos") == "Data Science" or determinar_categoria("Python para ciencia de datos") == "Programación"
            # Test DB
            with get_db_connection(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM plataformas_ocultas")
                count = c.fetchone()[0]
                assert count >= 5
            st.success("Pruebas básicas OK")
        except AssertionError:
            st.error("Falló una aserción en pruebas básicas")
        except Exception as e:
            st.error(f"Error en pruebas básicas: {e}")

# ============================================================
# 17. SECCIÓN AYUDA & ATAJOS
# ============================================================
def render_help():
    with st.expander("❓ Ayuda"):
        st.markdown("- Escribe un tema y pulsa 'Buscar Cursos'.")
        st.markdown("- Activa/desactiva características en Configuración avanzada.")
        st.markdown("- Añade favoritos y exporta resultados a CSV.")
        st.markdown("- Usa el chat IA para consejos rápidos (si Groq está disponible).")
        st.markdown("- Si la IA muestra HTML/JSON, se limpiará automáticamente en la UI (parche aplicado).")
        st.markdown("- Atajos: [Shift+Enter] para enviar en chat, [Alt+R] para refrescar (según navegador).")

# ============================================================
# 18. TELEMETRÍA OPT-OUT (solo bandera persistente)
# ============================================================
def set_telemetry_opt_out(value: bool):
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO configuracion (clave, valor) VALUES (?, ?)", ("telemetry_opt_out", "1" if value else "0"))
            conn.commit()
        st.success("Preferencia de telemetría actualizada")
    except Exception as e:
        logger.error(f"Error en telemetría opt-out: {e}")
        st.error("No se pudo actualizar la preferencia")

def render_telemetry():
    with st.expander("🔒 Privacidad y Telemetría"):
        try:
            with get_db_connection(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT valor FROM configuracion WHERE clave = 'telemetry_opt_out'")
                row = c.fetchone()
                opt_out = (row and row[0] == "1")
        except Exception:
            opt_out = False
        new_val = st.checkbox("Desactivar telemetría anónima", value=opt_out)
        if new_val != opt_out:
            set_telemetry_opt_out(new_val)

# ============================================================
# 29. MAIN UNIFICADO (REEMPLAZA MAIN_APP Y MAIN_EXTENDED)
# ============================================================
def main():
    """
    Función principal unificada que renderiza la aplicación completa.
    """
    ensure_session()
    init_feature_flags()
    iniciar_tareas_background()
    
    # Header y búsqueda
    render_header()

    # Formulario de búsqueda
    i18n = get_i18n(st.session_state.get('lang_ui', 'Español (es)'))
    tema = st.text_input(i18n["enter_topic"], placeholder="Ej: Python, IA, Finanzas...", key="search_topic_input")
    col_form1, col_form2 = st.columns(2)
    nivel = col_form1.selectbox(i18n["level"], ["Cualquiera", "Principiante", "Intermedio", "Avanzado"], key="search_level_select")
    idioma = col_form2.selectbox(i18n["language"], ["Español (es)", "Inglés (en)", "Portugués (pt)"], key="search_lang_select")
    st.session_state['lang_ui'] = idioma # Guardar selección

    # --- Lógica de Búsqueda ---
    if st.button(i18n["search_button"], type="primary", use_container_width=True):
        if not (tema or "").strip():
            st.warning("Por favor ingresa un tema para buscar.")
        else:
            with st.spinner("🔍 Buscando en múltiples fuentes..."):
                # FIX ASYNCIO PARA STREAMLIT CLOUD
                try:
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    resultados = loop.run_until_complete(buscar_recursos_multicapa_ext(tema.strip(), idioma, nivel))
                    st.session_state.resultados = resultados
                    
                    if resultados:
                        registrar_muestreo_estadistico(resultados, tema.strip(), idioma, nivel)
                except Exception as e:
                    logger.error(f"Error en búsqueda asíncrona: {e}")
                    st.error(f"Ocurrió un error durante la búsqueda: {e}")
                    st.session_state.resultados = []
            st.rerun()

    # --- Renderizado de Contenido Dinámico ---
    current_results = st.session_state.get('resultados', [])
    if current_results:
        st.success(i18n["results_found"].format(n=len(current_results)))
        if GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True):
             planificar_analisis_ia(current_results)
             time.sleep(0.4)

        for i, r in enumerate(current_results):
            mostrar_recurso(r, i)

    elif 'resultados' in st.session_state:
        st.warning(i18n["no_results"])

    # --- Paneles Avanzados ---
    st.markdown("---")
    st.markdown("### 🧭 Paneles Avanzados y de Diagnóstico")
    colA, colB = st.columns(2)
    with colA:
        panel_configuracion_avanzada()
        panel_favoritos_ui()
        panel_feedback_ui(current_results)
        panel_export_import_ui(current_results)
        reportes_rapidos()

    with colB:
        admin_dashboard()
        log_viewer()
        panel_cache_viewer()

    # --- Ayuda y Otros ---
    st.markdown("---")
    render_help()
    keyboard_tips()
    render_telemetry()
    if st.session_state.features.get("enable_debug_mode", False):
        run_basic_tests()

    # --- Componentes Persistentes ---
    sidebar_chat()
    sidebar_status()
    render_footer()

# ============================================================
# 30. PUNTO DE ENTRADA ÚNICO
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error crítico en la aplicación: {e}")



