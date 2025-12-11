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
# 0. INTEGRACIÓN SEGURA CON IA_MODULE
# ============================================================
try:
    import ia_module
except ImportError:
    ia_module = None
    # No mostramos error en UI para no ensuciar, solo en logs interna
    pass

# ============================================================
# 1. LOGGING & CONFIGURACIÓN BASE
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BuscadorProfesional")

# Decoradores de rendimiento
def profile(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt > 0.3:
            logger.info(f"⏱️ {func.__name__} tardó {dt:.3f}s")
        return out
    return wrapper

def async_profile(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = await func(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt > 0.3:
            logger.info(f"⏱️ {func.__name__} tardó {dt:.3f}s (async)")
        return out
    return wrapper

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

# Configuración y Sincronización con ia_module
GROQ_AVAILABLE = False
if ia_module and GROQ_API_KEY:
    ia_module.GROQ_API_KEY = GROQ_API_KEY
    if len(GROQ_API_KEY) >= 10:
        ia_module.GROQ_AVAILABLE = True
        GROQ_AVAILABLE = True
        logger.info("✅ IA Module sincronizado y activo")

def validate_api_key(key: str, key_type: str) -> bool:
    if not key or len(key) < 10: return False
    if key_type == "google" and not key.startswith(("AIza", "AIz")): return False
    return True

# ============================================================
# 2. FEATURE FLAGS & CACHÉ
# ============================================================
DEFAULT_FEATURES = {
    "enable_google_api": True, "enable_known_platforms": True, "enable_hidden_platforms": True,
    "enable_groq_analysis": True, "enable_chat_ia": True, "enable_favorites": True,
    "enable_feedback": True, "enable_export_import": True, "enable_offline_cache": True,
    "enable_ddg_fallback": False, "enable_debug_mode": False, "ui_theme": "auto",
    "max_results": 15, "max_analysis": 5
}

def init_feature_flags():
    if "features" not in st.session_state:
        st.session_state.features = DEFAULT_FEATURES.copy()
    st.session_state.features["enable_google_api"] &= bool(validate_api_key(GOOGLE_API_KEY, "google") and GOOGLE_CX)
    st.session_state.features["enable_groq_analysis"] &= GROQ_AVAILABLE

class ExpiringCache:
    def __init__(self, ttl_seconds=43200):
        self.cache = {}
        self.ttl = ttl_seconds
    def get(self, key):
        if key in self.cache:
            val, ts = self.cache[key]
            if time.time() - ts < self.ttl: return val
            del self.cache[key]
        return None
    def set(self, key, value):
        self.cache[key] = (value, time.time())

search_cache = ExpiringCache()
background_tasks: "queue.Queue[Dict[str, Any]]" = queue.Queue()
executor = ThreadPoolExecutor(max_workers=MAX_BACKGROUND_TASKS)

# ============================================================
# 3. MODELOS DE DATOS
# ============================================================
@dataclass
class Certificacion:
    plataforma: str; curso: str; tipo: str; validez_internacional: bool; paises_validos: List[str]; costo_certificado: float; reputacion_academica: float; ultima_verificacion: str

@dataclass
class RecursoEducativo:
    id: str; titulo: str; url: str; descripcion: str; plataforma: str; idioma: str; nivel: str; categoria: str; certificacion: Optional[Certificacion]; confianza: float; tipo: str; ultima_verificacion: str; activo: bool; metadatos: Dict[str, Any]; metadatos_analisis: Optional[Dict[str, Any]] = None; analisis_pendiente: bool = False

@dataclass
class Favorito:
    id_recurso: str; titulo: str; url: str; notas: str; creado_en: str

def safe_json_dumps(obj): return json.dumps(obj, default=str)
def safe_json_loads(t, d=None): 
    try: return json.loads(t) 
    except: return d if d else {}

# ============================================================
# 4. BASE DE DATOS ROBUSTA
# ============================================================
DB_PATH = "cursos_inteligentes_v3.db"

@contextlib.contextmanager
def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try: yield conn
    except sqlite3.Error as e:
        logger.error(f"Error BD: {e}")
        conn.rollback()
    finally: conn.close()

def init_advanced_database():
    with get_db_connection(DB_PATH) as conn:
        c = conn.cursor()
        # Tablas Core
        c.execute('CREATE TABLE IF NOT EXISTS favoritos (id INTEGER PRIMARY KEY, id_recurso TEXT, titulo TEXT, url TEXT, notas TEXT, creado_en TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, id_recurso TEXT, opinion TEXT, rating INTEGER, creado_en TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS sesiones (id INTEGER PRIMARY KEY, session_id TEXT, started_at TEXT, ended_at TEXT, device TEXT, prefs_json TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS configuracion (clave TEXT PRIMARY KEY, valor TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS analiticas_busquedas (id INTEGER PRIMARY KEY, tema TEXT, idioma TEXT, nivel TEXT, timestamp TEXT, plataforma_origen TEXT, veces_mostrado INTEGER DEFAULT 0, veces_clickeado INTEGER DEFAULT 0, tiempo_promedio_uso REAL DEFAULT 0, satisfaccion_usuario REAL DEFAULT 0)')
        
        # Tabla Plataformas Ocultas (La "base de datos secreta")
        c.execute('''CREATE TABLE IF NOT EXISTS plataformas_ocultas (id INTEGER PRIMARY KEY, nombre TEXT, url_base TEXT, descripcion TEXT, idioma TEXT, categoria TEXT, nivel TEXT, confianza REAL, ultima_verificacion TEXT, activa INTEGER, tipo_certificacion TEXT, validez_internacional INTEGER, paises_validos TEXT, reputacion_academica REAL)''')
        
        # Semilla si está vacía
        c.execute("SELECT COUNT(*) FROM plataformas_ocultas")
        if c.fetchone()[0] == 0:
            plataformas_iniciales = [
                {"nombre": "Aprende con Alf", "url_base": "https://aprendeconalf.es/?s={}", "idioma": "es", "confianza": 0.85},
                {"nombre": "Coursera", "url_base": "https://www.coursera.org/search?query={}&free=true", "idioma": "en", "confianza": 0.95},
                {"nombre": "edX", "url_base": "https://www.edx.org/search?tab=course&availability=current&price=free&q={}", "idioma": "en", "confianza": 0.92},
                {"nombre": "freeCodeCamp", "url_base": "https://www.freecodecamp.org/news/search/?query={}", "idioma": "en", "confianza": 0.93},
                {"nombre": "Domestika (Gratuito)", "url_base": "https://www.domestika.org/es/search?query={}&free=1", "idioma": "es", "confianza": 0.83}
            ]
            for p in plataformas_iniciales:
                c.execute("INSERT INTO plataformas_ocultas (nombre, url_base, idioma, activa, confianza, ultima_verificacion) VALUES (?, ?, ?, 1, ?, ?)", 
                          (p["nombre"], p["url_base"], p["idioma"], p["confianza"], datetime.now().isoformat()))
            conn.commit()

init_advanced_database()

# ============================================================
# 5. UTILIDADES Y LÓGICA DE NEGOCIO
# ============================================================
def generar_id_unico(url: str) -> str: return hashlib.md5(url.encode()).hexdigest()[:10]
def get_codigo_idioma(i: str) -> str: return {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}.get(i, "es")
def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    if nivel_solicitado not in ("Cualquiera", "Todos"): return nivel_solicitado
    t = (texto or "").lower()
    if any(x in t for x in ['principiante', 'básico', 'intro']): return "Principiante"
    if any(x in t for x in ['avanzado', 'experto']): return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    tema = (tema or "").lower()
    if any(x in tema for x in ['python', 'java', 'web', 'programación']): return "Programación"
    if any(x in tema for x in ['data', 'datos', 'ia']): return "Data Science"
    if any(x in tema for x in ['design', 'diseño']): return "Diseño"
    return "General"

def extraer_plataforma(url: str) -> str:
    try: return urlparse(url).netloc.replace('www.','').split('.')[0].title()
    except: return "Web"

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    t = (url + (titulo or "") + (descripcion or "")).lower()
    if any(i in t for i in ['precio', 'buy', 'premium', 'login', 'cart']): return False
    return True

# --- Wrapper IA con ia_module ---
def analizar_recurso_groq_sync(recurso: RecursoEducativo, perfil: Dict):
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True)):
        return
    try:
        # Llamada al módulo externo
        data = ia_module.analizar_recurso_groq(recurso.titulo, recurso.descripcion, recurso.nivel, recurso.categoria, recurso.plataforma)
        recurso.metadatos_analisis = data
        ia_prom = (data.get("calidad_ia", 0.5) + data.get("relevancia_ia", 0.5)) / 2.0
        recurso.confianza = min(max(recurso.confianza, ia_prom), 0.95)
    except Exception as e:
        logger.error(f"Error IA: {e}")

def chatgroq(mensajes: List[Dict[str, str]]) -> str:
    if not (GROQ_AVAILABLE and st.session_state.features.get("enable_chat_ia", True)):
        return "🧠 IA no disponible."
    try:
        # Extraemos último mensaje para el módulo
        last = next((m['content'] for m in reversed(mensajes) if m['role'] == 'user'), "")
        return ia_module.chatgroq(last)
    except Exception as e:
        return f"⚠️ Error IA: {str(e)}"

# ============================================================
# 6. MOTORES DE BÚSQUEDA
# ============================================================
@async_profile
async def buscar_en_google_api(tema, idioma, nivel) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_google_api", True) or not validate_api_key(GOOGLE_API_KEY, "google"): return []
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': f"{tema} curso gratis", 'num': 5, 'lr': f'lang_{idioma}'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200: return []
                data = await resp.json()
                res = []
                for item in data.get('items', []):
                    res.append(RecursoEducativo(
                        id=generar_id_unico(item['link']), titulo=item['title'], url=item['link'],
                        descripcion=item.get('snippet',''), plataforma=extraer_plataforma(item['link']), idioma=idioma,
                        nivel=nivel, categoria=determinar_categoria(tema), certificacion=None, confianza=0.85,
                        tipo="verificada", ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={}
                    ))
                return res
    except: return []

def buscar_en_plataformas_conocidas(tema, idioma, nivel) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_known_platforms", True): return []
    res = []
    # Generamos enlaces directos útiles
    platforms = [
        ("YouTube", f"https://www.youtube.com/results?search_query=curso+{quote_plus(tema)}"),
        ("Coursera", f"https://www.coursera.org/search?query={quote_plus(tema)}&free=true"),
        ("Udemy", f"https://www.udemy.com/courses/search/?q={quote_plus(tema)}&price=price-free")
    ]
    for nombre, url in platforms:
        res.append(RecursoEducativo(
            id=generar_id_unico(url), titulo=f"{nombre}: {tema}", url=url,
            descripcion=f"Búsqueda directa en {nombre}", plataforma=nombre, idioma=idioma, nivel=nivel,
            categoria="General", certificacion=None, confianza=0.9, tipo="conocida",
            ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={}
        ))
    return res

def buscar_en_plataformas_ocultas(tema, idioma, nivel) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_hidden_platforms", True): return []
    try:
        with get_db_connection(DB_PATH) as conn:
            rows = conn.execute("SELECT nombre, url_base, confianza FROM plataformas_ocultas WHERE activa=1 AND idioma=?", (idioma,)).fetchall()
        res = []
        for nombre, url_base, conf in rows:
            url = url_base.format(quote_plus(tema))
            res.append(RecursoEducativo(
                id=generar_id_unico(url), titulo=f"💎 {nombre} - {tema}", url=url,
                descripcion="Recurso de alta calidad verificado", plataforma=nombre, idioma=idioma, nivel=nivel,
                categoria="Premium", certificacion=None, confianza=conf, tipo="oculta",
                ultima_verificacion=datetime.now().isoformat(), activo=True, metadatos={}
            ))
        return res
    except: return []

@async_profile
async def buscar_recursos_multicapa(tema, idioma, nivel) -> List[RecursoEducativo]:
    # Cache Check
    cache_key = f"{tema}|{idioma}|{nivel}"
    if cached := search_cache.get(cache_key): return cached

    res = buscar_en_plataformas_conocidas(tema, idioma, nivel)
    res.extend(buscar_en_plataformas_ocultas(tema, idioma, nivel))
    
    try:
        g_res = await buscar_en_google_api(tema, idioma, nivel)
        res.extend(g_res)
    except: pass
    
    # Ordenar y limitar
    res.sort(key=lambda x: x.confianza, reverse=True)
    res = res[:st.session_state.features["max_results"]]

    # Planificar IA
    if GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True):
        for r in res[:st.session_state.features["max_analysis"]]: 
            r.analisis_pendiente = True
        planificar_analisis_ia(res)
    
    search_cache.set(cache_key, res)
    return res

# Background System
def worker():
    while True:
        try:
            task = background_tasks.get(timeout=2)
            if task and task['tipo'] == 'analizar':
                for r in task['data']: analizar_recurso_groq_sync(r, {})
            background_tasks.task_done()
        except queue.Empty: pass
        except Exception: pass

def iniciar_tareas_background():
    if 'bg_started' not in st.session_state:
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        st.session_state.bg_started = True

def planificar_analisis_ia(resultados):
    if GROQ_AVAILABLE:
        background_tasks.put({'tipo': 'analizar', 'data': [r for r in resultados if r.analisis_pendiente]})

# ============================================================
# 7. INTERFAZ DE USUARIO (UI)
# ============================================================
st.set_page_config(page_title="Buscador PRO", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")
init_feature_flags()

def apply_theme():
    theme = st.session_state.features["ui_theme"]
    css = """
    .resultado-card { padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #4CAF50; transition: transform .2s; }
    .resultado-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    """
    if theme == "dark":
        css += ".resultado-card { background: #1e1e1e; color: #ddd; }"
    else:
        css += ".resultado-card { background: white; color: #333; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def mostrar_recurso(r: RecursoEducativo):
    ia_html = ""
    if r.metadatos_analisis:
        ia_html = f"<div style='background:#e1bee7;color:#4a148c;padding:8px;border-radius:5px;margin-top:8px;font-size:0.9em'>🧠 <b>IA:</b> {r.metadatos_analisis.get('recomendacion_personalizada','')}</div>"
    elif r.analisis_pendiente:
        ia_html = "<div style='color:#9c27b0;font-size:0.8em;margin-top:5px'>⏳ Analizando con IA...</div>"

    # Botón de favoritos con KEY ÚNICA basada en el ID del recurso
    fav_key = f"fav_btn_{r.id}"
    
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"""
        <div class="resultado-card">
            <h4 style="margin:0"><a href="{r.url}" target="_blank" style="text-decoration:none;color:#1565C0">{r.titulo}</a></h4>
            <p style="margin:5px 0 0 0;font-size:0.95em">{r.descripcion}</p>
            <div style="margin-top:5px;font-size:0.85em;color:#666">
                🏷️ {r.plataforma} | 📊 Nivel: {r.nivel} | ⭐ Confianza: {int(r.confianza*100)}%
            </div>
            {ia_html}
        </div>
        """, unsafe_allow_html=True)
    with col2:
        # Usamos st.button nativo para evitar líos de JS
        if st.button("⭐", key=fav_key, help="Añadir a favoritos"):
            agregar_favorito_bd(r.titulo, r.url, "Guardado desde lista")

def render_search_form():
    c1, c2, c3 = st.columns([3, 1, 1])
    # KEYS ÚNICAS para evitar DuplicateElementId
    tema = c1.text_input("¿Qué quieres aprender?", placeholder="Ej: Python, Liderazgo...", key="search_input_main")
    nivel = c2.selectbox("Nivel", ["Cualquiera", "Principiante", "Avanzado"], key="search_level_main")
    idioma = c3.selectbox("Idioma", ["Español (es)", "Inglés (en)"], key="search_lang_main")
    buscar = st.button("🚀 Buscar Cursos", type="primary", use_container_width=True, key="search_btn_main")
    return tema, nivel, idioma, buscar

def sidebar_chat_ui():
    if not st.session_state.features["enable_chat_ia"]: return
    with st.sidebar:
        st.header("💬 Asistente IA")
        if "chat_msgs" not in st.session_state: st.session_state.chat_msgs = []
        
        for m in st.session_state.chat_msgs:
            with st.chat_message(m["role"]): st.write(m["content"])
            
        if p := st.chat_input("Pregunta sobre educación..."):
            st.session_state.chat_msgs.append({"role": "user", "content": p})
            with st.chat_message("user"): st.write(p)
            
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    resp = chatgroq(st.session_state.chat_msgs)
                    st.write(resp)
            st.session_state.chat_msgs.append({"role": "assistant", "content": resp})

# ============================================================
# 8. PANELES AVANZADOS (Recuperados)
# ============================================================
def agregar_favorito_bd(titulo, url, notas):
    with get_db_connection(DB_PATH) as conn:
        conn.execute("INSERT INTO favoritos (titulo, url, notas, creado_en) VALUES (?, ?, ?, ?)", (titulo, url, notas, datetime.now().isoformat()))
        conn.commit()
    st.toast(f"Guardado: {titulo}")

def panel_favoritos():
    st.markdown("### ⭐ Mis Favoritos")
    try:
        with get_db_connection(DB_PATH) as conn:
            favs = conn.execute("SELECT titulo, url, notas, creado_en FROM favoritos ORDER BY id DESC").fetchall()
        if favs:
            df = pd.DataFrame(favs, columns=["Título", "URL", "Notas", "Fecha"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No tienes favoritos guardados.")
    except Exception as e: st.error(f"Error cargando favoritos: {e}")

def panel_admin():
    st.markdown("### 🛠️ Admin Dashboard")
    c1, c2, c3 = st.columns(3)
    with get_db_connection(DB_PATH) as conn:
        n_favs = conn.execute("SELECT COUNT(*) FROM favoritos").fetchone()[0]
        n_logs = conn.execute("SELECT COUNT(*) FROM analiticas_busquedas").fetchone()[0]
        n_plats = conn.execute("SELECT COUNT(*) FROM plataformas_ocultas").fetchone()[0]
    
    c1.metric("Favoritos", n_favs)
    c2.metric("Búsquedas", n_logs)
    c3.metric("Plataformas", n_plats)

    if st.checkbox("Ver Logs del Sistema"):
        if os.path.exists("buscador_cursos.log"):
            with open("buscador_cursos.log", "r") as f:
                st.code(f.read()[-2000:])
        else: st.warning("No hay archivo de log.")

def panel_configuracion():
    st.markdown("### ⚙️ Configuración")
    with st.expander("Opciones Generales"):
        st.session_state.features["ui_theme"] = st.selectbox("Tema Visual", ["auto", "dark"], key="conf_theme")
        st.session_state.features["enable_groq_analysis"] = st.checkbox("Activar Análisis IA", value=True, key="conf_ia")
        st.session_state.features["max_results"] = st.slider("Resultados por búsqueda", 5, 50, 15, key="conf_res")

def panel_exportacion(resultados):
    st.markdown("### 📤 Exportar Datos")
    if resultados:
        df = pd.DataFrame([vars(r) for r in resultados])
        # Limpiamos columnas complejas para el CSV
        df = df.drop(columns=['certificacion', 'metadatos', 'metadatos_analisis'], errors='ignore')
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Resultados (CSV)", csv, "resultados.csv", "text/csv", key="btn_export_csv")
    else:
        st.info("Realiza una búsqueda para exportar.")

def event_bridge_manual():
    # Panel manual para agregar favoritos si no se usa el botón de la tarjeta
    st.markdown("---")
    st.caption("Añadir favorito manualmente")
    c1, c2, c3 = st.columns([2, 2, 1])
    tit = c1.text_input("Título", key="manual_fav_tit")
    url = c2.text_input("URL", key="manual_fav_url")
    if c3.button("Guardar", key="manual_fav_btn"):
        if tit and url: agregar_favorito_bd(tit, url, "Manual")

# ============================================================
# 9. MAIN APP (INTEGRACIÓN TOTAL)
# ============================================================
def main_extended():
    apply_theme()
    iniciar_tareas_background()
    
    st.title("🎓 Buscador Profesional de Cursos")
    st.caption(f"Sistema Verificado | IA: {'✅ Activa' if GROQ_AVAILABLE else '❌ Inactiva'}")

    tema, nivel, idioma, buscar = render_search_form()
    
    # Lógica de Búsqueda
    resultados = []
    if buscar and tema:
        with st.spinner(f"Buscando cursos sobre '{tema}'..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(buscar_recursos_multicapa(tema, idioma, nivel))
            loop.close()
            
            # Log analítica
            with get_db_connection(DB_PATH) as conn:
                conn.execute("INSERT INTO analiticas_busquedas (tema, idioma, nivel, timestamp) VALUES (?,?,?,?)", 
                             (tema, idioma, nivel, datetime.now().isoformat()))
                conn.commit()

        if resultados:
            st.success(f"Encontrados {len(resultados)} cursos verificados.")
            for r in resultados:
                mostrar_recurso(r)
        else:
            st.warning("No se encontraron resultados. Intenta términos más generales.")

    st.markdown("---")
    
    # Pestañas para funcionalidades avanzadas
    tab1, tab2, tab3, tab4 = st.tabs(["⭐ Favoritos", "⚙️ Configuración", "📤 Exportar", "🛠️ Admin"])
    
    with tab1: panel_favoritos()
    with tab2: panel_configuracion()
    with tab3: panel_exportacion(resultados)
    with tab4: panel_admin()
    
    event_bridge_manual()
    sidebar_chat_ui()

if __name__ == "__main__":
    main_extended()
    # ============================================================
# 10. FUNCIONALIDADES EXTRA (RECUPERADAS)
# ============================================================

def panel_feedback_ui(resultados: List[RecursoEducativo]):
    """Panel para enviar valoraciones sobre los recursos encontrados."""
    if not st.session_state.features.get("enable_feedback", True):
        return
    
    st.markdown("### 📝 Feedback del Usuario")
    
    # Crear diccionario de opciones
    opciones = {f"{r.titulo} ({r.plataforma})": r.id for r in resultados} if resultados else {}
    
    if not opciones:
        st.info("Realiza una búsqueda para poder dejar feedback sobre los resultados.")
        return

    # Formulario con KEYS ÚNICAS para evitar errores
    with st.form(key="feedback_form_main"):
        sel_label = st.selectbox("Selecciona el recurso:", list(opciones.keys()), key="fb_select_resource")
        rating = st.slider("Calificación", 1, 5, 4, key="fb_rating_slider")
        opinion = st.text_area("Tu opinión", placeholder="¿Fue útil este curso?", key="fb_comment_area")
        
        submit = st.form_submit_button("Enviar Feedback")
        
        if submit:
            rec_id = opciones[sel_label]
            try:
                with get_db_connection(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO feedback (id_recurso, opinion, rating, creado_en) VALUES (?, ?, ?, ?)",
                        (rec_id, opinion, rating, datetime.now().isoformat())
                    )
                    conn.commit()
                st.success("¡Gracias! Tu feedback ha sido registrado.")
            except Exception as e:
                st.error(f"Error guardando feedback: {e}")

@async_profile
async def buscar_en_duckduckgo(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Fallback opcional usando DuckDuckGo (HTML Parsing básico)."""
    if not st.session_state.features.get("enable_ddg_fallback", False):
        return []
    try:
        # Nota: DDG no tiene API pública oficial gratuita, esto es un scraper básico de respaldo
        q = quote_plus(f"{tema} free course {nivel}".strip())
        url = f"https://duckduckgo.com/html/?q={q}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=8) as resp:
                if resp.status != 200: return []
                text = await resp.text()
                
                # Extracción simple mediante Regex (para no depender de BeautifulSoup si no está)
                links = re.findall(r'href="(https?://[^"]+)"', text)
                resultados = []
                
                seen_urls = set()
                for link in links:
                    if len(resultados) >= 5: break
                    if link in seen_urls: continue
                    if "duckduckgo" in link or "google" in link: continue
                    
                    # Filtrado básico
                    if not es_recurso_educativo_valido(link, "", ""): continue
                    
                    seen_urls.add(link)
                    resultados.append(RecursoEducativo(
                        id=generar_id_unico(link),
                        titulo=f"🦆 Resultado Web: {extraer_plataforma(link)}",
                        url=link,
                        descripcion="Resultado alternativo vía DuckDuckGo.",
                        plataforma="DuckDuckGo",
                        idioma=idioma,
                        nivel=nivel,
                        categoria="General",
                        certificacion=None,
                        confianza=0.70,
                        tipo="verificada",
                        ultima_verificacion=datetime.now().isoformat(),
                        activo=True,
                        metadatos={"fuente": "duckduckgo"}
                    ))
                return resultados
    except Exception as e:
        logger.error(f"Error DDG Fallback: {e}")
        return []

@async_profile
async def buscar_recursos_multicapa_ext(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Extensión de la búsqueda que incluye el fallback si lo principal falla."""
    # 1. Búsqueda estándar (Google + DB + Conocidas)
    base = await buscar_recursos_multicapa(tema, idioma, nivel)
    
    # 2. Fallback si hay pocos resultados y está activado
    if len(base) < 3 and st.session_state.features.get("enable_ddg_fallback", False):
        ddg_res = await buscar_en_duckduckgo(tema, idioma, nivel)
        base.extend(ddg_res)
    
    # 3. Deduplicación final
    seen = set()
    unique = []
    for r in base:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
            
    # 4. Ordenar por confianza
    unique.sort(key=lambda x: x.confianza, reverse=True)
    return unique[:st.session_state.features["max_results"]]

def log_viewer_completo():
    """Visor de logs integrado en la UI principal."""
    st.markdown("### 🪵 Visor de Logs del Sistema")
    log_file = "buscador_cursos.log"
    
    if st.button("🔄 Refrescar Logs", key="refresh_logs_btn"):
        pass # Al hacer rerun se recarga
        
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            # Mostrar las últimas 50 líneas
            content = "".join(lines[-50:])
            st.code(content, language="log")
    else:
        st.info("El archivo de log aún no ha sido creado.")

def run_basic_tests():
    """Ejecuta pruebas de sanidad al inicio (solo si debug activo)."""
    if not st.session_state.features.get("enable_debug_mode", False):
        return
        
    st.sidebar.markdown("---")
    st.sidebar.caption("🧪 Tests Activos")
    try:
        assert determinar_nivel("curso avanzado", "Cualquiera") == "Avanzado"
        assert generar_id_unico("http://test.com")
        st.sidebar.success("Tests Unitarios: OK")
    except Exception as e:
        st.sidebar.error(f"Tests Fallidos: {e}")

def end_session():
    """Cierra la sesión lógica en la base de datos."""
    if "session_id" in st.session_state:
        try:
            with get_db_connection(DB_PATH) as conn:
                conn.execute(
                    "UPDATE sesiones SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL",
                    (datetime.now().isoformat(), st.session_state.session_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error cerrando sesión: {e}")

def ensure_session():
    """Inicia una sesión si no existe."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess_{int(time.time())}_{random.randint(1000,9999)}"
        try:
            with get_db_connection(DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO sesiones (session_id, started_at, device, prefs_json) VALUES (?, ?, ?, ?)",
                    (st.session_state.session_id, datetime.now().isoformat(), "web", safe_json_dumps(st.session_state.features))
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error iniciando sesión: {e}")

# ============================================================
# 11. MAIN APP ACTUALIZADO (REEMPLAZA AL ANTERIOR)
# ============================================================

def main_completo():
    """Función principal que orquesta toda la aplicación."""
    ensure_session()
    apply_theme()
    iniciar_tareas_background()
    run_basic_tests()
    
    st.title("🎓 Buscador Profesional de Cursos")
    st.caption(f"Versión Ultra-Robust | IA: {'✅ Activa' if GROQ_AVAILABLE else '❌ Inactiva'}")

    tema, nivel, idioma, buscar = render_search_form()
    
    # Variables de estado para resultados
    if "resultados_busqueda" not in st.session_state:
        st.session_state.resultados_busqueda = []

    # Lógica de Búsqueda
    if buscar and tema:
        with st.spinner(f"🔍 Analizando múltiples fuentes para '{tema}'..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Usamos la versión extendida con fallback
            resultados = loop.run_until_complete(buscar_recursos_multicapa_ext(tema, idioma, nivel))
            loop.close()
            
            st.session_state.resultados_busqueda = resultados
            
            # Registrar analítica
            try:
                with get_db_connection(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO analiticas_busquedas (tema, idioma, nivel, timestamp, veces_mostrado) VALUES (?,?,?,?,?)", 
                        (tema, idioma, nivel, datetime.now().isoformat(), len(resultados))
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Error analytics: {e}")

    # Renderizado de Resultados
    resultados = st.session_state.resultados_busqueda
    if resultados:
        st.success(f"✅ Se encontraron {len(resultados)} recursos verificados y analizados.")
        for r in resultados:
            mostrar_recurso(r)
    elif buscar:
        st.warning("⚠️ No se encontraron resultados. Intenta activar el 'Modo Offline' o 'DuckDuckGo Fallback' en configuración.")

    st.markdown("---")
    
    # SISTEMA DE PESTAÑAS COMPLETO
    tab_fav, tab_conf, tab_export, tab_feed, tab_admin = st.tabs([
        "⭐ Favoritos", 
        "⚙️ Configuración", 
        "📤 Exportar", 
        "📝 Feedback",
        "🛠️ Admin & Logs"
    ])
    
    with tab_fav:
        panel_favoritos()
        event_bridge_manual() # Agregado aquí para consistencia
        
    with tab_conf:
        panel_configuracion()
        # Añadir opción de DuckDuckGo aquí
        st.session_state.features["enable_ddg_fallback"] = st.checkbox(
            "Activar DuckDuckGo Fallback (Lento pero más resultados)", 
            value=st.session_state.features.get("enable_ddg_fallback", False),
            key="conf_ddg_check"
        )
        st.session_state.features["enable_debug_mode"] = st.checkbox(
            "Modo Debug", 
            value=st.session_state.features.get("enable_debug_mode", False),
            key="conf_debug_check"
        )
        
    with tab_export:
        panel_exportacion(resultados)
        
    with tab_feed:
        panel_feedback_ui(resultados)
        
    with tab_admin:
        panel_admin()
        log_viewer_completo()
    
    # Barra lateral siempre visible
    sidebar_chat_ui()

# Reemplaza la llamada final en el bloque __main__ por main_completo()
