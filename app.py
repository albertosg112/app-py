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
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

# --- INTEGRACIÓN: MÓDULO EXTERNO ---
try:
    import ia_module
except ImportError:
    # Fallback silencioso para no romper la app si el archivo no está
    ia_module = None
    logging.warning("⚠️ ia_module.py no encontrado. La funcionalidad IA estará limitada." )

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
# 1. LOGGING & CONFIGURACIÓN (MEJORADO)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('buscador_cursos.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BuscadorProfesional")
ia_logger = logging.getLogger("IAModuleIntegration") # Logger específico para la IA

def obtener_credenciales_seguras() -> Tuple[str, str, str]:
    """Obtiene credenciales priorizando Secrets y luego Variables de Entorno."""
    try:
        # Usar st.secrets es la forma preferida en producción
        g_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        g_cx = st.secrets.get("GOOGLE_CX", os.getenv("GOOGLE_CX", ""))
        groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        
        if not g_key or not g_cx:
            logger.warning("Credenciales de Google incompletas.")
        if not groq_key:
            logger.warning("GROQ_API_KEY no encontrada.")
            
        return g_key, g_cx, groq_key
    except (AttributeError, FileNotFoundError):
        # Fallback para entornos donde st.secrets no existe
        logger.info("st.secrets no disponible, usando os.getenv como fallback.")
        return os.getenv("GOOGLE_API_KEY", ""), os.getenv("GOOGLE_CX", ""), os.getenv("GROQ_API_KEY", "")

GOOGLE_API_KEY, GOOGLE_CX, GROQ_API_KEY = obtener_credenciales_seguras()
DUCKDUCKGO_ENABLED = (os.getenv("DUCKDUCKGO_ENABLED", "false").lower() == "true")
MAX_BACKGROUND_TASKS = 4 # Aumentado para permitir más análisis en paralelo
CACHE_EXPIRATION = timedelta(hours=12)

# --- INYECCIÓN DE DEPENDENCIAS Y VALIDACIÓN DE IA (CLAVE) ---
GROQ_AVAILABLE = False
if ia_module:
    ia_logger.info("ia_module.py importado correctamente.")
    if GROQ_API_KEY and len(GROQ_API_KEY) > 10:
        # Inyectamos la clave API en el módulo importado.
        ia_module.GROQ_API_KEY = GROQ_API_KEY
        ia_module.GROQ_AVAILABLE = True # Forzamos la disponibilidad en el módulo
        GROQ_AVAILABLE = True
        ia_logger.info("✅ Groq API disponible y clave inyectada en ia_module.")
    else:
        ia_logger.warning("⚠️ ia_module cargado, pero la clave Groq API es inválida o no está configurada.")
        ia_module.GROQ_AVAILABLE = False
else:
    ia_logger.error("❌ ia_module.py no se pudo importar. Toda la funcionalidad de IA está deshabilitada.")

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
    "ui_theme": "auto",
    "max_results": 15,
    "max_analysis": 5
}

def init_feature_flags():
    if "features" not in st.session_state:
        st.session_state.features = DEFAULT_FEATURES.copy()
    st.session_state.features["enable_google_api"] &= bool(validate_api_key(GOOGLE_API_KEY, "google") and GOOGLE_CX)
    st.session_state.features["enable_groq_analysis"] &= GROQ_AVAILABLE

# ============================================================
# 3. CACHÉ & CONCURRENCIA
# ============================================================
class ExpiringCache:
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
    plataforma: str; curso: str; tipo: str; validez_internacional: bool
    paises_validos: List[str]; costo_certificado: float; reputacion_academica: float; ultima_verificacion: str

@dataclass
class RecursoEducativo:
    id: str; titulo: str; url: str; descripcion: str; plataforma: str; idioma: str
    nivel: str; categoria: str; certificacion: Optional[Certificacion]; confianza: float
    tipo: str; ultima_verificacion: str; activo: bool; metadatos: Dict[str, Any]
    metadatos_analisis: Optional[Dict[str, Any]] = None; analisis_pendiente: bool = False

@dataclass
class Favorito:
    id_recurso: str; titulo: str; url: str; notas: str; creado_en: str

@dataclass
class Feedback:
    id_recurso: str; opinion: str; rating: int; creado_en: str

def safe_json_dumps(obj: Dict) -> str:
    try: return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception: return "{}"

def safe_json_loads(text: str, default_value: Any = None) -> Any:
    if default_value is None: default_value = {}
    try: return json.loads(text)
    except Exception: return default_value

# ============================================================
# 5. BASE DE DATOS
# ============================================================
DB_PATH = "cursos_inteligentes_v3.db"

@contextlib.contextmanager
def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Error BD: {e}"); conn.rollback(); raise e
    finally:
        conn.close()

def init_advanced_database():
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            # Crear todas las tablas si no existen
            c.execute('CREATE TABLE IF NOT EXISTS auditoria (id INTEGER PRIMARY KEY, evento TEXT, detalle TEXT, creado_en TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS favoritos (id INTEGER PRIMARY KEY, id_recurso TEXT, titulo TEXT, url TEXT, notas TEXT, creado_en TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, id_recurso TEXT, opinion TEXT, rating INTEGER, creado_en TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS sesiones (id INTEGER PRIMARY KEY, session_id TEXT, started_at TEXT, ended_at TEXT, device TEXT, prefs_json TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS configuracion (clave TEXT PRIMARY KEY, valor TEXT)')
            c.execute('''CREATE TABLE IF NOT EXISTS plataformas_ocultas (id INTEGER PRIMARY KEY, nombre TEXT, url_base TEXT, descripcion TEXT, idioma TEXT, categoria TEXT, nivel TEXT, confianza REAL, ultima_verificacion TEXT, activa INTEGER, tipo_certificacion TEXT, validez_internacional INTEGER, paises_validos TEXT, reputacion_academica REAL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS analiticas_busquedas (id INTEGER PRIMARY KEY, tema TEXT, idioma TEXT, nivel TEXT, timestamp TEXT, plataforma_origen TEXT, veces_mostrado INTEGER, veces_clickeado INTEGER, tiempo_promedio_uso REAL, satisfaccion_usuario REAL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS certificaciones_verificadas (id INTEGER PRIMARY KEY, plataforma TEXT, curso_tema TEXT, tipo_certificacion TEXT, validez_internacional INTEGER, paises_validos TEXT, costo_certificado REAL, reputacion_academica REAL, ultima_verificacion TEXT, veces_verificado INTEGER)''')
            
            c.execute("SELECT COUNT(*) FROM plataformas_ocultas")
            if c.fetchone()[0] == 0:
                plataformas_iniciales = [
                    {"nombre": "Aprende con Alf", "url_base": "https://aprendeconalf.es/?s={}", "descripcion": "Cursos gratuitos de programación y ciencia de datos", "idioma": "es", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.85, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["es"], "reputacion_academica": 0.90},
                    {"nombre": "Coursera", "url_base": "https://www.coursera.org/search?query={}&free=true", "descripcion": "Cursos universitarios (audit mode )", "idioma": "en", "categoria": "General", "nivel": "Avanzado", "confianza": 0.95, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.95},
                    {"nombre": "edX", "url_base": "https://www.edx.org/search?tab=course&availability=current&price=free&q={}", "descripcion": "Cursos de Harvard, MIT y más (modo audit )", "idioma": "en", "categoria": "Académico", "nivel": "Avanzado", "confianza": 0.92, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.93},
                    {"nombre": "Kaggle Learn", "url_base": "https://www.kaggle.com/learn/search?q={}", "descripcion": "Microcursos de ciencia de datos con certificados gratuitos", "idioma": "en", "categoria": "Data Science", "nivel": "Intermedio", "confianza": 0.90, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.88},
                    {"nombre": "freeCodeCamp", "url_base": "https://www.freecodecamp.org/news/search/?query={}", "descripcion": "Certificados completos en desarrollo web", "idioma": "en", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.93, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.91},
                ]
                for p in plataformas_iniciales:
                    c.execute('''INSERT INTO plataformas_ocultas (nombre, url_base, descripcion, idioma, categoria, nivel, confianza, ultima_verificacion, activa, tipo_certificacion, validez_internacional, paises_validos, reputacion_academica ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (p["nombre"], p["url_base"], p["descripcion"], p["idioma"], p["categoria"], p["nivel"], p["confianza"], datetime.now().isoformat(), 1, p["tipo_certificacion"], int(p["validez_internacional"]), safe_json_dumps(p["paises_validos"]), p["reputacion_academica"]))
            conn.commit()
        logger.info("✅ Base de datos inicializada correctamente")
    except Exception as e:
        logger.error(f"❌ Error Init DB: {e}")

init_advanced_database()

# ============================================================
# 6. UTILIDADES GENERALES
# ============================================================
def get_codigo_idioma(nombre_idioma: str) -> str:
    return {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}.get(nombre_idioma, "es")

def generar_id_unico(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    if nivel_solicitado not in ("Cualquiera", "Todos"): return nivel_solicitado
    t = (texto or "").lower()
    if any(x in t for x in ['principiante', 'básico', 'beginner', 'desde cero']): return "Principiante"
    if any(x in t for x in ['avanzado', 'advanced', 'experto']): return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    tema = (tema or "").lower()
    if any(x in tema for x in ['python', 'java', 'javascript', 'web', 'programación']): return "Programación"
    if any(x in tema for x in ['data', 'datos', 'ia', 'ai', 'machine learning']): return "Data Science"
    if any(x in tema for x in ['design', 'diseño', 'ux', 'ui']): return "Diseño"
    return "General"

def extraer_plataforma(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower().replace('www.', '')
        parts = domain.split('.')
        return parts[0].title() if len(parts) > 1 else domain.title()
    except: return "Web"

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    t = (url + (titulo or "") + (descripcion or "")).lower()
    invalidas = ['comprar', 'precio', 'premium', 'suscripción', 'matrícula']
    validas = ['curso', 'tutorial', 'aprender', 'gratis', 'free', 'class', 'educación']
    dominios = ['.edu', 'coursera', 'edx', 'khanacademy', 'udemy', 'youtube', 'freecodecamp', '.gov']
    if any(i in t for i in invalidas): return False
    return any(v in t for v in validas) or any(d in url.lower() for d in dominios)

def limpiar_html_visible(texto: str) -> str:
    if not texto: return ""
    texto = re.sub(r'\{.*\}\s*$', '', texto, flags=re.DOTALL)
    texto = re.sub(r'<[^>]+>', '', texto).strip()
    return texto

def ui_chat_mostrar(mensaje: str, rol: str):
    texto_limpio = limpiar_html_visible(mensaje)
    if not texto_limpio: return
    st.markdown(f"**{'🤖 IA' if rol == 'assistant' else '👤 Tú'}:** {texto_limpio}")

# ============================================================
# 7. INTEGRACIÓN GROQ (MEJORADA PARA ROBUSTEZ Y NO BLOQUEO)
# ============================================================
def analizar_recurso_groq_sync(recurso: RecursoEducativo, perfil: Dict):
    """
    Worker robusto que utiliza el módulo externo ia_module en un hilo separado
    para no bloquear la aplicación principal, con timeout.
    """
    if not (st.session_state.features.get("enable_groq_analysis", True) and GROQ_AVAILABLE and ia_module and ia_module.GROQ_AVAILABLE):
        recurso.metadatos_analisis = {"calidad_ia": 0, "relevancia_ia": 0, "recomendacion_personalizada": "Análisis IA no disponible.", "advertencias": ["IA_DISABLED"]}
        ia_logger.warning(f"Análisis omitido para {recurso.id}: IA no disponible/deshabilitada.")
        return

    def task():
        return ia_module.analizar_recurso_groq(titulo=recurso.titulo, descripcion=recurso.descripcion, nivel=recurso.nivel, categoria=recurso.categoria, plataforma=recurso.plataforma)

    try:
        future = executor.submit(task)
        data = future.result(timeout=45)
        recurso.metadatos_analisis = data
        ia_prom = (data.get("calidad_ia", 0) + data.get("relevancia_ia", 0)) / 2.0
        recurso.confianza = max(recurso.confianza, (recurso.confianza + ia_prom) / 2)
        recurso.confianza = min(recurso.confianza, 0.98)
        ia_logger.info(f"Análisis IA completado para {recurso.id}")
    except TimeoutError:
        ia_logger.error(f"Timeout (45s) analizando recurso {recurso.id}.")
        recurso.metadatos_analisis = {"calidad_ia": 0, "relevancia_ia": 0, "recomendacion_personalizada": "El análisis IA tardó demasiado.", "advertencias": ["TIMEOUT_ERROR"]}
    except Exception as e:
        ia_logger.error(f"Error en worker Groq (ia_module) para {recurso.id}: {e}", exc_info=True)
        recurso.metadatos_analisis = {"calidad_ia": 0, "relevancia_ia": 0, "recomendacion_personalizada": "Error en análisis IA.", "advertencias": [str(e)]}
    finally:
        recurso.analisis_pendiente = False

def chatgroq(mensajes: List[Dict[str, str]]) -> str:
    if not (ia_module and GROQ_AVAILABLE and st.session_state.features.get("enable_chat_ia", True)):
        return "🧠 IA no disponible. Usa el buscador superior."
    try:
        ultimo_mensaje = next((m['content'] for m in reversed(mensajes) if m['role'] == 'user'), "")
        if not ultimo_mensaje: return "No entendí tu mensaje."
        return ia_module.chatgroq(ultimo_mensaje)
    except Exception as e:
        logger.error(f"Error en chat Groq (ia_module): {e}"); return "Hubo un error con la IA."

# ============================================================
# 8. BÚSQUEDA MULTICAPA
# ============================================================
@async_profile
async def buscar_en_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not (st.session_state.features.get("enable_google_api", True) and validate_api_key(GOOGLE_API_KEY, "google") and GOOGLE_CX): return []
    try:
        query = f'{tema} curso gratuito certificado' + (f' nivel {nivel.lower()}' if nivel not in ("Cualquiera", "Todos") else '')
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': query, 'num': 5, 'lr': f'lang_{idioma}'}
        async with aiohttp.ClientSession( ) as s, s.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=8 ) as resp:
            if resp.status != 200: return []
            items = (await resp.json()).get('items', [])
            return [RecursoEducativo(id=generar_id_unico(i['link']), titulo=i.get('title'), url=i['link'], descripcion=i.get('snippet'), plataforma=extraer_plataforma(i['link']), idioma=idioma, nivel=determinar_nivel(i.get('title', '') + i.get('snippet', ''), nivel), categoria=determinar_categoria(tema), certificacion=None, confianza=0.85, tipo="verificada", ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={'fuente': 'google_api'}) for i in items if es_recurso_educativo_valido(i.get('link', ''), i.get('title'), i.get('snippet'))]
    except Exception as e:
        logger.error(f"Error Google API: {e}"); return []

def buscar_en_plataformas_conocidas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_known_platforms", True): return []
    plataformas = {"es": [{"n": "YouTube", "u": f"https://www.youtube.com/results?search_query=curso+gratis+{quote_plus(tema )}"}, {"n": "Coursera (ES)", "u": f"https://www.coursera.org/search?query={quote_plus(tema )}&languages=es&free=true"}], "en": [{"n": "YouTube", "u": f"https://www.youtube.com/results?search_query=free+course+{quote_plus(tema )}"}, {"n": "edX", "u": f"https://www.edx.org/search?q={quote_plus(tema )}&price=free"}]}
    return [RecursoEducativo(id=generar_id_unico(p["u"]), titulo=f'🎯 {p["n"]} — {tema}', url=p["u"], descripcion=f'Búsqueda directa en {p["n"]}', plataforma=p["n"], idioma=idioma, nivel=nivel if nivel != "Cualquiera" else "Intermedio", categoria=determinar_categoria(tema), certificacion=None, confianza=0.85, tipo="conocida", ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={"fuente": "plataformas_conocidas"}) for p in plataformas.get(idioma, plataformas["en"])]

def buscar_en_plataformas_ocultas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_hidden_platforms", True): return []
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            q = 'SELECT nombre, url_base, descripcion, nivel, confianza, tipo_certificacion, validez_internacional, paises_validos, reputacion_academica FROM plataformas_ocultas WHERE activa = 1 AND idioma = ?'
            p = [idioma]
            if nivel not in ("Cualquiera", "Todos"): q += " AND (nivel = ? OR nivel = 'Todos')"; p.append(nivel)
            c.execute(q + " ORDER BY confianza DESC LIMIT 6", p)
            recursos = []
            for r in c.fetchall():
                nombre, url_base, desc, nivel_db, conf, tipo_cert, val_int, paises_json, reputacion = r
                url = url_base.format(quote_plus(tema))
                cert = Certificacion(plataforma=nombre, curso=tema, tipo=tipo_cert, validez_internacional=bool(val_int), paises_validos=safe_json_loads(paises_json, []), costo_certificado=0.0 if tipo_cert == "gratuito" else 49.99, reputacion_academica=reputacion or 0.8, ultima_verificacion=datetime.now().isoformat()) if tipo_cert and tipo_cert != "none" else None
                recursos.append(RecursoEducativo(id=generar_id_unico(url), titulo=f'💎 {nombre} — {tema}', url=url, descripcion=desc, plataforma=nombre, idioma=idioma, nivel=nivel_db if nivel in ("Cualquiera", "Todos") else nivel, categoria=determinar_categoria(tema), certificacion=cert, confianza=float(conf or 0.8), tipo="oculta", ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={"fuente": "plataformas_ocultas"}))
            return recursos
    except Exception as e:
        logger.error(f"Error plataformas ocultas: {e}"); return []

def eliminar_duplicados(resultados: List[RecursoEducativo]) -> List[RecursoEducativo]:
    seen = set(); unicos = []; [seen.add(r.url) or unicos.append(r) for r in resultados if r.url not in seen]; return unicos

@async_profile
async def buscar_recursos_multicapa(tema: str, idioma_ui: str, nivel: str) -> List[RecursoEducativo]:
    cache_key = f"{tema}|{idioma_ui}|{nivel}"
    if cached := search_cache.get(cache_key): return cached
    idioma = get_codigo_idioma(idioma_ui)
    progress_bar = st.progress(0, "Iniciando búsqueda...")
    
    ocultas = buscar_en_plataformas_ocultas(tema, idioma, nivel); progress_bar.progress(0.3, "Consultando Google API...")
    google_res = await buscar_en_google_api(tema, idioma, nivel); progress_bar.progress(0.6, "Buscando en plataformas conocidas...")
    conocidas = buscar_en_plataformas_conocidas(tema, idioma, nivel); progress_bar.progress(0.85, "Procesando resultados...")
    
    resultados = eliminar_duplicados(ocultas + google_res + conocidas)
    resultados.sort(key=lambda x: x.confianza, reverse=True)
    
    if GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True):
        for r in resultados[:st.session_state.features.get("max_analysis", 5)]: r.analisis_pendiente = True

    final = resultados[:st.session_state.features.get("max_results", 15)]
    search_cache.set(cache_key, final)
    progress_bar.empty(); return final

# ============================================================
# 9. PROCESAMIENTO EN SEGUNDO PLANO
# ============================================================
def worker():
    while True:
        try:
            tarea = background_tasks.get(timeout=60)
            if tarea is None: break
            if tarea.get('tipo') == 'analizar_recurso':
                analizar_recurso_groq_sync(**tarea.get('parametros', {}))
            background_tasks.task_done()
        except queue.Empty: continue
        except Exception as e: logger.error(f"Error en worker: {e}"); background_tasks.task_done()

def iniciar_tareas_background():
    if 'background_started' not in st.session_state:
        for _ in range(min(MAX_BACKGROUND_TASKS, os.cpu_count() or 1)):
            threading.Thread(target=worker, daemon=True).start()
        st.session_state.background_started = True

def planificar_analisis_ia(resultados: List[RecursoEducativo]):
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True)): return
    for r in resultados:
        if r.analisis_pendiente:
            background_tasks.put({'tipo': 'analizar_recurso', 'parametros': {'recurso': r, 'perfil': {}}})

# ============================================================
# 10. UI ESTILOS Y COMPONENTES
# ============================================================
st.set_page_config(page_title="🎓 Buscador Profesional de Cursos", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")
init_feature_flags()

st.markdown("""
<style>
.main-header { background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%); color: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.25); }
.main-header h1 { margin: 0; font-size: 2.3rem; }
.resultado-card { border-radius: 15px; padding: 20px; margin-bottom: 20px; background: white; box-shadow: 0 5px 20px rgba(0,0,0,0.08); border-left: 6px solid #4CAF50; transition: transform .2s; }
.resultado-card:hover { transform: translateY(-3px); }
.nivel-principiante { border-left-color: #2196F3 !important; } .nivel-intermedio { border-left-color: #4CAF50 !important; } .nivel-avanzado { border-left-color: #FF9800 !important; }
.plataforma-oculta { border-left-color: #FF6B35 !important; background: #fff5f0; }
.certificado-badge { display:inline-block;padding:4px 10px;border-radius:12px;font-size:.8rem;font-weight:
bold;background:#e8f5e9;color:#2e7d32;margin-right:5px; }
a { text-decoration: none !important; }
.status-badge { display:inline-block;padding:4px 10px;border-radius:15px;font-size:.8rem;font-weight:bold;background:rgba(255,255,255,0.2); }
.smalltext { font-size: 0.85rem; color: #607d8b; }
.badge-pendiente { display:inline-block; padding:3px 8px; background:#ede7f6; color:#6a1b9a; border-radius:12px; font-size:.75rem; }
.badge-ok { display:inline-block; padding:3px 8px; background:#e8f5e9; color:#2e7d32; border-radius:12px; font-size:.75rem; }
.badge-error { display:inline-block; padding:3px 8px; background:#ffebee; color:#c62828; border-radius:12px; font-size:.75rem; }
.tooltip { font-size:0.8rem; color:#78909c; }
</style>
""", unsafe_allow_html=True)

def link_button(url: str, label: str = "➡️ Acceder al recurso") -> str:
    if not url: return ""
    return f'''<a href="{url}" target="_blank" style="display:inline-block;background:linear-gradient(to right,#6a11cb,#2575fc);color:white;padding:10px 16px;border-radius:8px;font-weight:bold;">{label}</a>'''

def badge_certificacion(cert: Optional[Certificacion]) -> str:
    if not cert: return ""
    html = ""
    if cert.tipo == "gratuito": html += '<span class="certificado-badge">✅ Certificado Gratuito</span>'
    elif cert.tipo == "audit": html += '<span class="certificado-badge" style="background:#e3f2fd;color:#1565c0;">🎓 Modo Audit</span>'
    elif cert.tipo == "pago": html += '<span class="certificado-badge" style="background:#fff3e0;color:#ef6c00;">💰 Certificado de Pago</span>'
    if cert.validez_internacional: html += '<span class="certificado-badge" style="background:#e3f2fd;color:#1565c0;">🌐 Validez Internacional</span>'
    return html

def clase_nivel(nivel: str) -> str:
    return {"Principiante": "nivel-principiante", "Intermedio": "nivel-intermedio", "Avanzado": "nivel-avanzado"}.get(nivel, "")

def mostrar_recurso(r: RecursoEducativo, idx: int):
    extra_class = "plataforma-oculta" if r.tipo == "oculta" else ""
    nivel_class = clase_nivel(r.nivel)
    cert_html = badge_certificacion(r.certificacion)
    ia_block, estado_ia = "", ""

    if r.analisis_pendiente:
        estado_ia = '<span class="badge-pendiente" title="El análisis se está ejecutando">IA pendiente</span>'
        ia_block = "<div style='color:#9c27b0;font-size:0.9em;margin:5px 0;'>⏳ Analizando con IA...</div>"
    elif r.metadatos_analisis:
        data = r.metadatos_analisis
        advertencias = data.get("advertencias", [])
        if "TIMEOUT_ERROR" in advertencias or "IA_DISABLED" in advertencias or "Error" in str(advertencias):
             estado_ia = '<span class="badge-error" title="El análisis de IA falló o no está disponible">IA Error</span>'
             ia_block = f"<div style='background:#fff3e0;padding:12px;border-radius:8px;margin:12px 0;border-left:4px solid #ff9800;'><strong>🧠 Análisis IA:</strong> {data.get('recomendacion_personalizada', 'No disponible.')}</div>"
        else:
            cal, rel = int(data.get('calidad_ia', 0)*100), int(data.get('relevancia_ia', 0)*100)
            estado_ia = f'<span class="badge-ok" title="Calidad: {cal}% / Relevancia: {rel}%">IA listo</span>'
            ia_block = f"""<div style="background:#f3e5f5;padding:12px;border-radius:8px;margin:12px 0;border-left:4px solid #9c27b0;"><strong>🧠 Análisis IA:</strong> Calidad {cal}% • Relevancia {rel}%  
{data.get('recomendacion_personalizada', '')}</div>"""
    else:
        estado_ia = '<span class="smalltext tooltip" title="El análisis IA no se ejecutó">Sin IA</span>'

    st.markdown(f"""
<div class="resultado-card {nivel_class} {extra_class}">
  <h3 style="margin-top:0;">{r.titulo or "Recurso"} {estado_ia}</h3>
  <p><strong>📚 {r.nivel}</strong> | 🌐 {r.plataforma} | 🏷️ {r.categoria}</p>
  <p style="color:#555;">{r.descripcion or "Sin descripción."}</p>
  <div style="margin-bottom:10px;">{cert_html}</div>
  {ia_block}
  <div style="margin-top:15px;">{link_button(r.url)}</div>
  <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.8rem; color: #888;">
    Confianza: {r.confianza*100:.0f}% | Verificado: {datetime.fromisoformat(r.ultima_verificacion).strftime('%d/%m/%Y')}
  </div>
</div>""", unsafe_allow_html=True)

# ============================================================
# 11. FAVORITOS, FEEDBACK, EXPORT/IMPORT
# ============================================================
def agregar_favorito(r: RecursoEducativo, notas: str = ""):
    try:
        with get_db_connection(DB_PATH) as conn:
            conn.execute("INSERT INTO favoritos (id_recurso, titulo, url, notas, creado_en) VALUES (?, ?, ?, ?, ?)", (r.id, r.titulo, r.url, notas, datetime.now().isoformat())); conn.commit()
        st.toast("⭐ Agregado a favoritos", icon="✅")
    except Exception as e:
        logger.error(f"Error agregando favorito: {e}")

def listar_favoritos() -> List[Favorito]:
    try:
        with get_db_connection(DB_PATH) as conn:
            return [Favorito(*f) for f in conn.execute("SELECT id_recurso, titulo, url, notas, creado_en FROM favoritos ORDER BY creado_en DESC").fetchall()]
    except Exception as e:
        logger.error(f"Error listando favoritos: {e}"); return []

def registrar_feedback(id_recurso: str, opinion: str, rating: int):
    try:
        with get_db_connection(DB_PATH) as conn:
            conn.execute("INSERT INTO feedback (id_recurso, opinion, rating, creado_en) VALUES (?, ?, ?, ?)", (id_recurso, opinion, rating, datetime.now().isoformat())); conn.commit()
        st.toast("¡Gracias por tu feedback!", icon="👍")
    except Exception as e:
        logger.error(f"Error registrando feedback: {e}")

def exportar_busquedas(resultados: List[RecursoEducativo]) -> bytes:
    df = pd.DataFrame([{'id': r.id, 'titulo': r.titulo, 'url': r.url, 'plataforma': r.plataforma, 'nivel': r.nivel, 'confianza': r.confianza} for r in resultados])
    return df.to_csv(index=False).encode('utf-8')

# ============================================================
# 12. PANELES AVANZADOS
# ============================================================
def panel_configuracion_avanzada():
    with st.expander("⚙️ Configuración avanzada"):
        st.session_state.features["ui_theme"] = st.selectbox("Tema", ["auto", "dark", "light"], index=["auto","dark","light"].index(st.session_state.features["ui_theme"]))
        st.session_state.features["max_results"] = st.slider("Máx. resultados", 5, 30, st.session_state.features["max_results"])
        st.session_state.features["max_analysis"] = st.slider("Máx. análisis IA", 0, 10, st.session_state.features["max_analysis"])
        st.session_state.features["enable_groq_analysis"] = st.checkbox("Análisis IA (Groq)", value=st.session_state.features["enable_groq_analysis"] and GROQ_AVAILABLE)
        st.session_state.features["enable_debug_mode"] = st.checkbox("Modo depuración", value=st.session_state.features["enable_debug_mode"])

def panel_cache_viewer():
    with st.expander("🗂️ Caché de búsquedas"):
        st.write(f"Entradas en caché: {len(search_cache.cache)}")
        if st.button("🧹 Vaciar caché", use_container_width=True):
            search_cache.cache.clear(); st.toast("Caché vaciada")

def panel_favoritos_ui():
    with st.expander("⭐ Favoritos"):
        favs = listar_favoritos()
        if not favs: st.info("Sin favoritos aún.")
        else: st.dataframe([{"Título": f.titulo, "URL": f.url, "Notas": f.notas} for f in favs])

# ============================================================
# 13. APP PRINCIPAL
# ============================================================
def render_header():
    st.markdown("""<div class="main-header"><h1>🎓 Buscador Profesional de Cursos</h1><p>Descubre recursos educativos verificados con análisis IA</p></div>""", unsafe_allow_html=True)

def render_search_form():
    col1, col2, col3 = st.columns([3, 1, 1])
    tema = col1.text_input("¿Qué quieres aprender?", placeholder="Ej: Python, Machine Learning...", key="search_main")
    nivel = col2.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    idioma = col3.selectbox("Idioma", ["Español (es)", "Inglés (en)", "Portugués (pt)"])
    buscar = st.button("🚀 Buscar Cursos", type="primary", use_container_width=True)
    return tema, nivel, idioma, buscar

def render_results(resultados: List[RecursoEducativo]):
    if resultados:
        st.success(f"✅ Se encontraron {len(resultados)} recursos verificados.")
        planificar_analisis_ia(resultados)
        # Espera breve para que la UI se actualice con el estado "pendiente"
        time.sleep(0.5) 
        for i, r in enumerate(resultados):
            mostrar_recurso(r, i)
        csv = exportar_busquedas(resultados)
        st.download_button("📥 Descargar CSV", csv, "cursos.csv", "text/csv", use_container_width=True)
    else:
        st.warning("No se encontraron resultados. Intenta con términos más generales.")

def sidebar_chat():
    with st.sidebar:
        st.header("💬 Asistente Educativo")
        if not st.session_state.features.get("enable_chat_ia", True):
            st.info("El chat IA está deshabilitado en la configuración.")
            return
        
        if "chat_msgs" not in st.session_state: st.session_state.chat_msgs = []
        for msg in st.session_state.chat_msgs: ui_chat_mostrar(msg["content"], msg["role"])

        if user_input := st.chat_input("Pregunta sobre cursos..."):
            st.session_state.chat_msgs.append({"role": "user", "content": user_input})
            ui_chat_mostrar(user_input, "user")
            with st.spinner("Pensando..."):
                reply = chatgroq(st.session_state.chat_msgs)
            st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
            st.rerun()

def sidebar_status():
    with st.sidebar:
        st.markdown("---"); st.subheader("📊 Estado del sistema")
        try:
            with get_db_connection(DB_PATH) as conn:
                total_plataformas = conn.execute("SELECT COUNT(*) FROM plataformas_ocultas WHERE activa = 1").fetchone()[0]
                st.metric("Plataformas activas", total_plataformas)
        except Exception: st.metric("Plataformas activas", "Error")
        st.info(f"IA: {'✅ Disponible' if GROQ_AVAILABLE and st.session_state.features.get('enable_groq_analysis', True) else '⚠️ No disponible'}")

def render_footer():
    st.markdown("---")
    st.markdown(f"""<div style="text-align:center;color:#666;font-size:14px;">✨ Buscador Profesional de Cursos v3.5 • {datetime.now().strftime('%d/%m/%Y')}</div>""", unsafe_allow_html=True)

# ============================================================
# 14. MAIN APP (CON CORRECCIÓN DE `StreamlitDuplicateElementKey`)
# ============================================================
def main():
    render_header()
    iniciar_tareas_background()
    
    tema, nivel, idioma, buscar = render_search_form()

    if 'resultados' not in st.session_state:
        st.session_state.resultados = []

    if buscar:
        if not (tema or "").strip():
            st.warning("Por favor ingresa un tema.")
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            st.session_state.resultados = loop.run_until_complete(buscar_recursos_multicapa(tema.strip(), idioma, nivel))
            loop.close()
            # Forzar rerun para mostrar resultados y estados de IA actualizados
            st.rerun()

    if st.session_state.resultados:
        render_results(st.session_state.resultados)

    # Paneles avanzados
    st.markdown("### 🧭 Paneles avanzados")
    colA, colB, colC = st.columns(3)
    with colA: panel_configuracion_avanzada()
    with colB: panel_cache_viewer()
    with colC: panel_favoritos_ui()

    # Sidebar y Footer
    sidebar_chat()
    sidebar_status()
    render_footer()

# ============================================================
# 15. ARRANQUE
# ============================================================
if __name__ == "__main__":
    main()

