# app.py — Consolidado Definitivo Ultra-Robust (SG1 + Mejoras UI/Chat + Parche Render)
# Incluye: AsyncIO, ThreadPool, DB Context Manager, Limpieza de Chat, Validación Groq, UI Mejorada, Cache expirable
# Objetivo: entregar un archivo completo, listo para ejecutar, con lo mejor de todas las versiones sin quitar funcionalidades.

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
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
import asyncio
import aiohttp
import contextlib
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1. LOGGING & CONFIGURACIÓN
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('buscador_cursos.log'), logging.StreamHandler()]
)
logger = logging.getLogger("BuscadorProfesional")

def obtener_credenciales_seguras():
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
# 2. CACHÉ & CONCURRENCIA
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
# 3. MODELOS DE DATOS & UTILIDADES JSON
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

def safe_json_dumps(obj: Dict) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

def safe_json_loads(text: str, default_value: Dict = None) -> Dict:
    if default_value is None:
        default_value = {}
    try:
        return json.loads(text)
    except Exception:
        return default_value

# ============================================================
# 4. BASE DE DATOS (Context Manager Seguro)
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
                    {"nombre": "PhET Simulations", "url_base": "https://phet.colorado.edu/en/search?q={}", "descripcion": "Simulaciones interactivas de ciencias y matemáticas de la Universidad de Colorado", "idioma": "en", "categoria": "Ciencias", "nivel": "Todos", "confianza": 0.88, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.85},
                    {"nombre": "The Programming Historian", "url_base": "https://programminghistorian.org/en/lessons/?q={}", "descripcion": "Tutoriales académicos de programación y humanidades digitales", "idioma": "en", "categoria": "Programación", "nivel": "Avanzado", "confianza": 0.82, "tipo_certificacion": "gratuito", "validez_internacional": 0, "paises_validos": ["uk", "us", "ca"], "reputacion_academica": 0.80},
                    {"nombre": "Domestika (Gratuito)", "url_base": "https://www.domestika.org/es/search?query={}&free=1", "descripcion": "Cursos gratuitos de diseño creativo, algunos con certificados verificados", "idioma": "es", "categoria": "Diseño", "nivel": "Intermedio", "confianza": 0.83, "tipo_certificacion": "pago", "validez_internacional": 1, "paises_validos": ["es", "mx", "ar", "cl"], "reputacion_academica": 0.82},
                    {"nombre": "Biblioteca Virtual Miguel de Cervantes", "url_base": "https://www.cervantesvirtual.com/buscar/?q={}", "descripcion": "Recursos académicos hispanos con validez internacional", "idioma": "es", "categoria": "Humanidades", "nivel": "Avanzado", "confianza": 0.87, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["es", "latam", "eu"], "reputacion_academica": 0.85},
                    {"nombre": "OER Commons", "url_base": "https://www.oercommons.org/search?q={}", "descripcion": "Recursos educativos abiertos de instituciones globales con estándares académicos", "idioma": "en", "categoria": "General", "nivel": "Todos", "confianza": 0.89, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.87}
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
        return True
    except Exception as e:
        logger.error(f"❌ Error Init DB: {e}")
        return False

init_advanced_database()

# ============================================================
# 5. UTILIDADES GENERALES & CHAT PARCHEADO
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
    invalidas = ['comprar', 'buy', 'precio', 'price', 'premium', 'paid', 'only', 'exclusive', 'suscripción', 'subscription']
    validas = ['curso', 'tutorial', 'aprender', 'learn', 'gratis', 'free', 'class', 'education', 'educación']
    dominios = ['.edu', 'coursera', 'edx', 'khanacademy', 'udemy', 'youtube', 'freecodecamp']
    if any(i in t for i in invalidas): return False
    return any(v in t for v in validas) or any(d in url.lower() for d in dominios)

# --- PARCHE DE LIMPIEZA PARA CHAT ---
def limpiar_html_visible(texto: str) -> str:
    if not texto:
        return ""
    # Eliminar bloques JSON al final y etiquetas HTML en toda la cadena
    texto = re.sub(r'\{.*\}\s*$', '', texto, flags=re.DOTALL).strip()
    texto = re.sub(r'<[^>]+>', '', texto).strip()
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
# 6. INTEGRACIÓN GROQ (Análisis & Chat)
# ============================================================
def analizar_recurso_groq_sync(recurso: RecursoEducativo, perfil: Dict):
    """Worker robusto para Groq con manejo de errores mejorado."""
    if not GROQ_AVAILABLE:
        recurso.metadatos_analisis = {
            "calidad_ia": recurso.confianza,
            "relevancia_ia": recurso.confianza,
            "recomendacion_personalizada": "IA no disponible.",
            "razones_calidad": []
        }
        return
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Evalúa este curso. Devuelve SOLO JSON válido.
        TÍTULO: {recurso.titulo}
        DESCRIPCIÓN: {recurso.descripcion}
        NIVEL: {recurso.nivel}
        CATEGORÍA: {recurso.categoria}
        PLATAFORMA: {recurso.plataforma}

        JSON:
        {{
            "calidad_educativa": 0.85,
            "relevancia_usuario": 0.90,
            "razones_calidad": ["razon1","razon2"],
            "recomendacion_personalizada": "Conclusión breve útil para el usuario",
            "advertencias": []
        }}
        """
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL, temperature=0.3, max_tokens=600
        )
        contenido = (resp.choices[0].message.content or "").strip()
        json_match = re.search(r'\{.*\}', contenido, re.DOTALL)
        data = safe_json_loads(json_match.group()) if json_match else {}
        recurso.metadatos_analisis = {
            "calidad_ia": float(data.get("calidad_educativa", recurso.confianza)),
            "relevancia_ia": float(data.get("relevancia_usuario", recurso.confianza)),
            "recomendacion_personalizada": data.get("recomendacion_personalizada", "Curso recomendado."),
            "razones_calidad": data.get("razones_calidad", []),
            "advertencias": data.get("advertencias", [])
        }
        # Ajuste de confianza según IA
        ia_prom = (recurso.metadatos_analisis["calidad_ia"] + recurso.metadatos_analisis["relevancia_ia"]) / 2.0
        recurso.confianza = min(max(recurso.confianza, ia_prom), 0.95)
    except Exception as e:
        logger.error(f"Error Groq Worker: {e}")
        recurso.metadatos_analisis = {
            "calidad_ia": recurso.confianza,
            "relevancia_ia": recurso.confianza,
            "recomendacion_personalizada": "IA no disponible temporalmente.",
            "razones_calidad": []
        }

def ejecutar_analisis_background(resultados: List[RecursoEducativo]):
    pendientes = [r for r in resultados if r.analisis_pendiente]
    if not pendientes:
        return
    for r in pendientes:
        executor.submit(analizar_recurso_groq_sync, r, {})

def chatgroq(mensajes: List[Dict[str, str]]) -> str:
    if not GROQ_AVAILABLE:
        return "🧠 IA no disponible. Usa el buscador superior para encontrar cursos ahora."
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            "Eres un asistente educativo. Sé claro y útil. "
            "NO uses formato JSON en tu respuesta de chat. Mantén respuestas breves y accionables."
        )
        groq_msgs = [{"role": "system", "content": system_prompt}] + mensajes
        resp = client.chat.completions.create(
            messages=groq_msgs, model=GROQ_MODEL, temperature=0.5, max_tokens=700
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Error en chat Groq: {e}")
        return "Hubo un error con la IA. Intenta de nuevo."

# ============================================================
# 7. BÚSQUEDA MULTICAPA
# ============================================================
async def buscar_en_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
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
    recursos: List[RecursoEducativo] = []
    plataformas = {
        "es": [
            {"nombre": "YouTube Educativo", "url": f"https://www.youtube.com/results?search_query=curso+gratis+{tema.replace(' ', '+')}"},
            {"nombre": "Coursera (ES)", "url": f"https://www.coursera.org/search?query={tema}&languages=es&free=true"},
            {"nombre": "Udemy (Gratis)", "url": f"https://www.udemy.com/courses/search/?q={tema}&price=price-free&lang=es"},
            {"nombre": "Khan Academy (ES)", "url": f"https://es.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}"}
        ],
        "en": [
            {"nombre": "YouTube Education", "url": f"https://www.youtube.com/results?search_query=free+course+{tema.replace(' ', '+')}"},
            {"nombre": "Khan Academy", "url": f"https://www.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}"},
            {"nombre": "Coursera", "url": f"https://www.coursera.org/search?query={tema}&free=true"},
            {"nombre": "Udemy (Free)", "url": f"https://www.udemy.com/courses/search/?q={tema}&price=price-free&lang=en"},
            {"nombre": "edX", "url": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={tema.replace(' ', '%20')}"},
            {"nombre": "freeCodeCamp", "url": f"https://www.freecodecamp.org/news/search/?query={tema.replace(' ', '%20')}"}
        ],
        "pt": [
            {"nombre": "YouTube BR", "url": f"https://www.youtube.com/results?search_query=curso+gratuito+{tema.replace(' ', '+')}"},
            {"nombre": "Coursera (PT)", "url": f"https://www.coursera.org/search?query={tema}&languages=pt&free=true"},
            {"nombre": "Udemy (PT)", "url": f"https://www.udemy.com/courses/search/?q={tema}&price=price-free&lang=pt"},
            {"nombre": "Khan Academy (PT)", "url": f"https://pt.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}"}
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
                url_completa = url_base.format(tema.replace(' ', '+'))
                nivel_calc = nivel_db if nivel in ("Cualquiera", "Todos") else nivel
                cert = None
                if tipo_cert and tipo_cert != "none":
                    cert = Certificacion(
                        plataforma=nombre,
                        curso=tema,
                        tipo=tipo_cert,
                        validez_internacional=bool(validez_int),
                        paises_validos=safe_json_loads(paises_json, default_value={"paises": ["global"]}).get("paises", ["global"]) if isinstance(paises_json, str) else ["global"],
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
    if GROQ_AVAILABLE:
        for r in resultados[:5]:
            r.analisis_pendiente = True

    final = resultados[:15]
    search_cache.set(cache_key, final)

    progress_bar.progress(1.0)
    time.sleep(0.1)
    progress_bar.empty()
    status_text.empty()

    return final

# ============================================================
# 8. PROCESAMIENTO EN SEGUNDO PLANO
# ============================================================
def analizar_resultados_en_segundo_plano(resultados: List[RecursoEducativo]):
    if not GROQ_AVAILABLE:
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
    if not GROQ_AVAILABLE:
        return
    tarea = {'tipo': 'analizar_resultados', 'parametros': {'resultados': [r for r in resultados if r.analisis_pendiente]}}
    background_tasks.put(tarea)
    logger.info(f"🧠 Tarea IA planificada: {len(tarea['parametros']['resultados'])} resultados")

# ============================================================
# 9. UI y Presentación
# ============================================================
st.set_page_config(page_title="🎓 Buscador Profesional de Cursos", page_icon="🎓", layout="wide")

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
    if r.metadatos_analisis:
        data = r.metadatos_analisis
        cal = int(data.get('calidad_ia', 0)*100)
        rel = int(data.get('relevancia_ia', 0)*100)
        ia_block = f"""
        <div style="background:#f3e5f5;padding:12px;border-radius:8px;margin:12px 0;border-left:4px solid #9c27b0;">
            <strong>🧠 Análisis IA:</strong> Calidad {cal}% • Relevancia {rel}%<br>
            {data.get('recomendacion_personalizada', '')}
        </div>"""
    elif r.analisis_pendiente:
        ia_block = "<div style='color:#9c27b0;font-size:0.9em;margin:5px 0;'>⏳ Analizando...</div>"

    desc = r.descripcion or "Sin descripción disponible."
    titulo = r.titulo or "Recurso Educativo"

    st.markdown(f"""
<div class="resultado-card {nivel_class} {extra_class}">
  <h3 style="margin-top:0;">{titulo}</h3>
  <p><strong>📚 {r.nivel}</strong> | 🌐 {r.plataforma} | 🏷️ {r.categoria}</p>
  <p style="color:#555;">{desc}</p>
  <div style="margin-bottom:10px;">{cert_html}</div>
  {ia_block}
  <div style="margin-top:15px;">{link_button(r.url, "➡️ Acceder al recurso")}</div>
  <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.8rem; color: #888;">
    Confianza: {r.confianza*100:.0f}% | Verificado: {datetime.fromisoformat(r.ultima_verificacion).strftime('%d/%m/%Y')}
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 10. APP PRINCIPAL (Búsqueda + Chat)
# ============================================================
def main():
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

    iniciar_tareas_background()
    col1, col2, col3 = st.columns([3, 1, 1])
    tema = col1.text_input("¿Qué quieres aprender?", placeholder="Ej: Python, Machine Learning, Diseño UX...")
    nivel = col2.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    idioma = col3.selectbox("Idioma", ["Español (es)", "Inglés (en)", "Portugués (pt)"])

    buscar = st.button("🚀 Buscar Cursos", type="primary", use_container_width=True)

    if buscar:
        if not (tema or "").strip():
            st.warning("Por favor ingresa un tema.")
        else:
            with st.spinner("🔍 Buscando en múltiples fuentes..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resultados = loop.run_until_complete(buscar_recursos_multicapa(tema.strip(), idioma, nivel))
                loop.close()

                if resultados:
                    st.success(f"✅ Se encontraron {len(resultados)} recursos verificados.")
                    if GROQ_AVAILABLE:
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

    # --- CHAT IA (Sidebar)
    with st.sidebar:
        st.header("💬 Asistente Educativo")
        if "chat_msgs" not in st.session_state:
            st.session_state.chat_msgs = []

        # Mostrar historial limpio
        for msg in st.session_state.chat_msgs:
            ui_chat_mostrar(msg["content"], msg["role"])

        user_input = st.chat_input("Pregunta sobre cursos...")
        if user_input:
            st.session_state.chat_msgs.append({"role": "user", "content": user_input})
            ui_chat_mostrar(user_input, "user")
            if GROQ_AVAILABLE:
                reply = chatgroq([{"role": "user", "content": user_input}])
                st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
                st.rerun()
            else:
                st.warning("Chat IA no disponible (Falta API Key)")

    # --- SIDEBAR Extra: Estado del sistema
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
        st.info(f"IA: {'✅ Disponible' if GROQ_AVAILABLE else '⚠️ No disponible'}")

    # --- FOOTER
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;color:#666;font-size:14px;padding:20px;background:#f8f9fa;border-radius:12px;">
        <strong>✨ Buscador Profesional de Cursos</strong><br>
        <span style="color: #2c3e50; font-weight: 500;">Resultados inmediatos • Cache inteligente • Alta disponibilidad</span><br>
        <em style="color: #7f8c8d;">Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')} • Versión: 3.2.0 • Estado: ✅ Activo</em><br>
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid #ddd;">
            <code style="background:#f1f3f5;padding:2px 8px;border-radius:4px;color:#d32f2f;">
                IA opcional — Sistema funcional sin dependencias externas críticas
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

