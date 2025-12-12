# app.py — Consolidado Definitivo Ultra-Robust PRO (Versión v6.2 - Active Validation)
# Objetivo: IA Omnisciente + Validación de URLs en tiempo real + CSS Legible.

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
    try:
        g_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        g_cx = st.secrets.get("GOOGLE_CX", os.getenv("GOOGLE_CX", ""))
        groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        return g_key, g_cx, groq_key
    except (AttributeError, FileNotFoundError):
        return os.getenv("GOOGLE_API_KEY", ""), os.getenv("GOOGLE_CX", ""), os.getenv("GROQ_API_KEY", "")

GOOGLE_API_KEY, GOOGLE_CX, GROQ_API_KEY = obtener_credenciales_seguras()
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
# 2. FEATURE FLAGS
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
# 4. MODELOS DE DATOS
# ============================================================
@dataclass
class Certificacion:
    plataforma: str
    curso: str
    tipo: str 
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
# 5. BASE DE DATOS
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
            c.execute('CREATE TABLE IF NOT EXISTS auditoria (id INTEGER PRIMARY KEY AUTOINCREMENT, evento TEXT NOT NULL, detalle TEXT, creado_en TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS favoritos (id INTEGER PRIMARY KEY AUTOINCREMENT, id_recurso TEXT NOT NULL, titulo TEXT NOT NULL, url TEXT NOT NULL, notas TEXT, creado_en TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, id_recurso TEXT NOT NULL, opinion TEXT, rating INTEGER, creado_en TEXT NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS sesiones (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, started_at TEXT NOT NULL, ended_at TEXT, device TEXT, prefs_json TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS configuracion (clave TEXT PRIMARY KEY, valor TEXT)')
            conn.commit()
    except Exception as e:
        logger.error(f"❌ Error migrando DB: {e}")

def init_advanced_database() -> bool:
    try:
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS plataformas_ocultas (id INTEGER PRIMARY KEY AUTOINCREMENT, nombre TEXT NOT NULL, url_base TEXT NOT NULL, descripcion TEXT, idioma TEXT NOT NULL, categoria TEXT, nivel TEXT, confianza REAL DEFAULT 0.7, ultima_verificacion TEXT, activa INTEGER DEFAULT 1, tipo_certificacion TEXT DEFAULT "audit", validez_internacional INTEGER DEFAULT 0, paises_validos TEXT DEFAULT "[]", reputacion_academica REAL DEFAULT 0.5)')
            cursor.execute('CREATE TABLE IF NOT EXISTS analiticas_busquedas (id INTEGER PRIMARY KEY AUTOINCREMENT, tema TEXT NOT NULL, idioma TEXT NOT NULL, nivel TEXT, timestamp TEXT NOT NULL, plataforma_origen TEXT, veces_mostrado INTEGER DEFAULT 0, veces_clickeado INTEGER DEFAULT 0, tiempo_promedio_uso REAL DEFAULT 0.0, satisfaccion_usuario REAL DEFAULT 0.0)')
            cursor.execute('CREATE TABLE IF NOT EXISTS certificaciones_verificadas (id INTEGER PRIMARY KEY AUTOINCREMENT, plataforma TEXT NOT NULL, curso_tema TEXT NOT NULL, tipo_certificacion TEXT NOT NULL, validez_internacional INTEGER DEFAULT 0, paises_validos TEXT DEFAULT "[]", costo_certificado REAL DEFAULT 0.0, reputacion_academica REAL DEFAULT 0.5, ultima_verificacion TEXT NOT NULL, veces_verificado INTEGER DEFAULT 1)')
            
            cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas")
            if cursor.fetchone()[0] == 0:
                plataformas_iniciales = [
                    {"nombre": "Aprende con Alf", "url_base": "https://aprendeconalf.es/?s={}", "descripcion": "Cursos gratuitos de programación Python", "idioma": "es", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.85, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["es"], "reputacion_academica": 0.90},
                    {"nombre": "Coursera Audit", "url_base": "https://www.coursera.org/search?query={}&free=true", "descripcion": "Modo auditoría gratuito", "idioma": "en", "categoria": "General", "nivel": "Avanzado", "confianza": 0.95, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.95},
                    {"nombre": "Kaggle Learn", "url_base": "https://www.kaggle.com/learn/search?q={}", "descripcion": "Data Science práctico", "idioma": "en", "categoria": "Data Science", "nivel": "Intermedio", "confianza": 0.90, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.88},
                    {"nombre": "freeCodeCamp", "url_base": "https://www.freecodecamp.org/news/search/?query={}", "descripcion": "Certificados Full Stack", "idioma": "en", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.93, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.91},
                    {"nombre": "OER Commons", "url_base": "https://www.oercommons.org/search?q={}", "descripcion": "Recursos educativos abiertos", "idioma": "en", "categoria": "General", "nivel": "Todos", "confianza": 0.89, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.87}
                ]
                for p in plataformas_iniciales:
                    cursor.execute('''INSERT INTO plataformas_ocultas (nombre, url_base, descripcion, idioma, categoria, nivel, confianza, ultima_verificacion, activa, tipo_certificacion, validez_internacional, paises_validos, reputacion_academica) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (p["nombre"], p["url_base"], p["descripcion"], p["idioma"], p["categoria"], p["nivel"], p["confianza"], datetime.now().isoformat(), 1, p["tipo_certificacion"], int(p["validez_internacional"]), safe_json_dumps(p["paises_validos"]), p["reputacion_academica"]))
            conn.commit()
        migrate_database()
        return True
    except Exception as e:
        logger.error(f"❌ Error Init DB: {e}")
        return False

init_advanced_database()

# ============================================================
# 6. UTILIDADES GENERALES
# ============================================================
def get_codigo_idioma(nombre_idioma: str) -> str:
    return {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt", "es": "es", "en": "en", "pt": "pt"}.get(nombre_idioma, "es")

def generar_id_unico(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    if nivel_solicitado not in ("Cualquiera", "Todos"): return nivel_solicitado
    t = (texto or "").lower()
    if any(x in t for x in ['principiante', 'básico', 'beginner', 'intro']): return "Principiante"
    if any(x in t for x in ['avanzado', 'advanced', 'experto']): return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    tema = (tema or "").lower()
    if any(x in tema for x in ['python', 'java', 'web', 'code']): return "Programación"
    if any(x in tema for x in ['data', 'datos', 'ia', 'ai']): return "Data Science"
    if any(x in tema for x in ['design', 'diseño']): return "Diseño"
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
    except:
        return "Web"

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    t = (url + (titulo or "") + (descripcion or "")).lower()
    invalidas = ['comprar', 'buy', 'price', 'premium', 'login', 'signup', 'register']
    if any(i in t for i in invalidas): return False
    return True

# --- CHAT UTILS ---
def limpiar_html_visible(texto: str) -> str:
    if not texto: return ""
    texto = re.sub(r'```.*?```', '', texto, flags=re.DOTALL)
    texto = re.sub(r'<[^>]+>', '', texto)
    return texto.strip()

def ui_chat_mostrar(mensaje: str, rol: str):
    texto_limpio = limpiar_html_visible(mensaje)
    if not texto_limpio: return
    if rol == "assistant":
        st.markdown(f"🤖 **IA:** {texto_limpio}")
    elif rol == "user":
        st.markdown(f"👤 **Tú:** {texto_limpio}")

# ============================================================
# 7. INTEGRACIÓN GROQ
# ============================================================
def analizar_recurso_groq_sync(recurso: RecursoEducativo, perfil: Dict):
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True)):
        recurso.metadatos_analisis = {"calidad_ia": 0.5, "relevancia_ia": 0.5, "recomendacion_personalizada": "IA desactivada.", "advertencias": []}
        return

    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Evalúa este curso y devuelve un JSON válido.
        TÍTULO: {recurso.titulo}
        DESCRIPCIÓN: {recurso.descripcion}
        
        JSON esperado: {{"calidad_educativa": 0.85, "relevancia_usuario": 0.90, "recomendacion_personalizada": "Texto breve", "advertencias": []}}
        """
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.2, 
            response_format={"type": "json_object"},
        )
        data = safe_json_loads(resp.choices[0].message.content or "{}")
        recurso.metadatos_analisis = {
            "calidad_ia": float(data.get("calidad_educativa", 0.7)),
            "relevancia_ia": float(data.get("relevancia_usuario", 0.7)),
            "recomendacion_personalizada": data.get("recomendacion_personalizada", "Sin recomendación."),
            "advertencias": data.get("advertencias", [])
        }
        recurso.confianza = (recurso.confianza + recurso.metadatos_analisis["calidad_ia"]) / 2
    except Exception as e:
        logger.error(f"Error Groq: {e}")
        recurso.metadatos_analisis = {"error": str(e)}

def obtener_contexto_db_para_ia() -> str:
    # Versión simplificada para no saturar tokens
    return "El usuario busca cursos en plataformas verificadas."

def chatgroq(mensajes: List[Dict[str, str]]) -> str:
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_chat_ia", True)):
        return "IA no disponible."
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        sys_msg = [{"role": "system", "content": "Eres un asistente educativo útil. Responde brevemente."}]
        resp = client.chat.completions.create(messages=sys_msg + mensajes, model=GROQ_MODEL, temperature=0.5)
        return resp.choices[0].message.content
    except Exception:
        return "Error de conexión con la IA."

def planificar_analisis_ia(resultados: List[RecursoEducativo]):
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True)):
        return
    # Analizar solo los top 3 para ahorrar quota
    pendientes = [r for r in resultados[:3] if r.analisis_pendiente]
    for r in pendientes:
        executor.submit(analizar_recurso_groq_sync, r, {})

# ============================================================
# 8. BÚSQUEDA MULTICAPA (CON VALIDACIÓN ACTIVA Y ASYNC)
# ============================================================

async def validar_url_activa(url: str, tema: str) -> bool:
    """Verifica si la URL responde y contiene el tema (status 200 y contenido)."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0"}
    try:
        timeout = aiohttp.ClientTimeout(total=2.5) # Muy rápido
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers, allow_redirects=True) as response:
                if response.status != 200: return False
                # Leer solo un poco para ser rápido
                content = await response.content.read(8192)
                text = content.decode('utf-8', errors='ignore').lower()
                tema_clean = tema.lower().split()[0]
                if tema_clean not in text: return False
                if "no results" in text or "page not found" in text: return False
                return True
    except Exception:
        return False # Fallo conservador

@async_profile
async def buscar_en_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_google_api", True) or not GOOGLE_API_KEY:
        return []
    try:
        q = f"{tema} curso gratuito"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': q, 'num': 4, 'lr': f'lang_{idioma}'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=5) as resp:
                if resp.status != 200: return []
                data = await resp.json()
                items = data.get('items', [])
                res = []
                for item in items:
                    link = item.get('link', '')
                    res.append(RecursoEducativo(
                        id=generar_id_unico(link), titulo=item.get('title', 'Curso'), url=link,
                        descripcion=item.get('snippet', ''), plataforma=extraer_plataforma(link),
                        idioma=idioma, nivel=nivel, categoria="Google Search", certificacion=None,
                        confianza=0.85, tipo="verificada", ultima_verificacion=datetime.now().isoformat(),
                        activo=True, metadatos={'fuente': 'google'}
                    ))
                return res
    except Exception:
        return []

def buscar_en_plataformas_ocultas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    # Síncrono (DB local es rápida)
    if not st.session_state.features.get("enable_hidden_platforms", True): return []
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT nombre, url_base, descripcion, nivel, confianza FROM plataformas_ocultas WHERE idioma=? LIMIT 5", (idioma,))
            rows = c.fetchall()
            res = []
            for r in rows:
                url = r[1].format(quote_plus(tema))
                res.append(RecursoEducativo(
                    id=generar_id_unico(url), titulo=f"{r[0]}: {tema}", url=url, descripcion=r[2],
                    plataforma=r[0], idioma=idioma, nivel=r[3], categoria="DB Oculta", certificacion=None,
                    confianza=r[4], tipo="oculta", ultima_verificacion=datetime.now().isoformat(),
                    activo=True, metadatos={'fuente': 'db_local'}
                ))
            return res
    except Exception:
        return []

@async_profile
async def buscar_en_sitios_elite(tema: str, idioma: str) -> List[RecursoEducativo]:
    # Sitios profundos con validación activa
    candidatos = [
        ("MIT OCW", "https://ocw.mit.edu/search/?q={}", "en"),
        ("EdX Search", "https://www.edx.org/search?q={}", "en"),
        ("OER Commons", "https://www.oercommons.org/search?q={}", "en"),
        ("UNAM", "https://cuaieed.unam.mx/descargas.php?q={}", "es"),
        ("Dialnet", "https://dialnet.unirioja.es/buscar/documentos?querysDismax.DOCUMENTAL_TODO={}", "es")
    ]
    # Filtrar por idioma
    targets = [c for c in candidatos if c[2] == idioma or c[2] == 'en']
    tasks = []
    
    for nombre, pat, lang in targets:
        url = pat.format(quote_plus(tema))
        tasks.append((nombre, url, validar_url_activa(url, tema)))
    
    # Ejecutar validaciones en paralelo
    resultados_raw = await asyncio.gather(*(t[2] for t in tasks))
    
    finales = []
    for i, valido in enumerate(resultados_raw):
        if valido:
            nombre, url, _ = tasks[i]
            finales.append(RecursoEducativo(
                id=generar_id_unico(url), titulo=f"🏛️ {nombre} - {tema}", url=url,
                descripcion=f"Recurso académico verificado en {nombre}", plataforma=nombre,
                idioma=idioma, nivel="Académico", categoria="Investigación", certificacion=None,
                confianza=0.92, tipo="conocida", ultima_verificacion=datetime.now().isoformat(),
                activo=True, metadatos={'fuente': 'elite_verified'}
            ))
    return finales

def eliminar_duplicados(resultados: List[RecursoEducativo]) -> List[RecursoEducativo]:
    seen = set()
    unique = []
    for r in resultados:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
    return unique

@async_profile
async def buscar_recursos_multicapa_ext(tema: str, idioma_ui: str, nivel: str) -> List[RecursoEducativo]:
    idioma = get_codigo_idioma(idioma_ui)
    
    # 1. DB Local (Inmediata)
    res_db = buscar_en_plataformas_ocultas(tema, idioma, nivel)
    
    # 2. Web (Async Paralelo)
    t1 = buscar_en_google_api(tema, idioma, nivel)
    t2 = buscar_en_sitios_elite(tema, idioma)
    
    # 3. DuckDuckGo Fallback (Opcional)
    t3 = buscar_en_duckduckgo(tema, idioma, nivel)
    
    # Ejecutar todo junto
    web_results = await asyncio.gather(t1, t2, t3)
    
    todos = res_db
    for lista in web_results:
        todos.extend(lista)
        
    todos = eliminar_duplicados(todos)
    todos.sort(key=lambda x: x.confianza, reverse=True)
    
    # Marcar para análisis IA
    if GROQ_AVAILABLE:
        for r in todos[:5]: r.analisis_pendiente = True
        
    return todos[:st.session_state.features["max_results"]]

@async_profile
async def buscar_en_duckduckgo(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_ddg_fallback", False): return []
    # (Implementación simplificada para el fallback)
    return []

# ============================================================
# 9. BACKGROUND TASKS
# ============================================================
def worker():
    while True:
        try:
            tarea = background_tasks.get(timeout=2)
            if tarea:
                pass # Aquí procesaríamos tareas pesadas si hubiera
            background_tasks.task_done()
        except queue.Empty:
            continue
        except Exception:
            pass

def iniciar_tareas_background():
    if 'bg_started' not in st.session_state:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        st.session_state.bg_started = True

# ============================================================
# 10. UI Y COMPONENTES
# ============================================================
def render_results(resultados: List[RecursoEducativo]):
    if resultados:
        st.success(f"✅ {len(resultados)} recursos encontrados.")
        planificar_analisis_ia(resultados)
        time.sleep(0.1) # Breve pausa para UI
        for i, r in enumerate(resultados):
            mostrar_recurso(r, i)
    else:
        st.warning("No se encontraron resultados. Intenta otros términos.")

def log_search_event(tema, idioma, nivel, count):
    try:
        with get_db_connection(DB_PATH) as conn:
            conn.execute("INSERT INTO analiticas_busquedas (tema, idioma, nivel, timestamp, veces_mostrado) VALUES (?, ?, ?, ?, ?)",
                         (tema, idioma, nivel, datetime.now().isoformat(), count))
            conn.commit()
    except: pass

# ============================================================
# MAIN
# ============================================================
def main():
    ensure_session()
    init_feature_flags()
    iniciar_tareas_background()
    
    render_header()
    
    i18n = get_i18n(st.session_state.get('lang_ui', 'Español (es)'))
    tema = st.text_input(i18n["enter_topic"], placeholder="Ej: Python, Marketing, Física...")
    c1, c2 = st.columns(2)
    nivel = c1.selectbox(i18n["level"], ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    idioma = c2.selectbox(i18n["language"], ["Español (es)", "Inglés (en)", "Portugués (pt)"])
    st.session_state['lang_ui'] = idioma

    if st.button(i18n["search_button"], type="primary"):
        if not tema:
            st.warning("Escribe un tema.")
        else:
            with st.spinner("🔍 Buscando y validando en tiempo real..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    res = loop.run_until_complete(buscar_recursos_multicapa_ext(tema, idioma, nivel))
                    st.session_state.resultados = res
                    log_search_event(tema, idioma, nivel, len(res))
                except Exception as e:
                    st.error(f"Error de búsqueda: {e}")
                    logger.error(e)
            st.rerun()

    current = st.session_state.get('resultados', [])
    render_results(current)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        panel_configuracion_avanzada()
        panel_favoritos_ui()
    with c2:
        admin_dashboard()
        log_viewer()

    sidebar_chat()
    sidebar_status()
    render_footer()

if __name__ == "__main__":
    main()
