# app.py — Versión Maestra Consolidada (SG1 + Mejoras Despliegue + Arquitectura Async)
# CARACTERÍSTICAS:
# 1. Base de datos completa y gestión de conexiones segura (Context Managers).
# 2. Inicialización retrasada de Groq para estabilidad en Streamlit Cloud.
# 3. Wrapper AsyncGroq para UI fluida y no bloqueante.
# 4. Limpieza visual de chat y resultados (Sin código basura en pantalla).
# 5. Manejo robusto de credenciales (Secrets + Env Vars).

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

# --- GESTIÓN DE CREDENCIALES ROBUSTA ---
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

MAX_BACKGROUND_TASKS = 2
CACHE_EXPIRATION = timedelta(hours=12)
GROQ_MODEL = "llama-3.1-70b-versatile"

def validate_api_key(key: str, key_type: str) -> bool:
    if not key or len(key) < 10: return False
    if key_type == "google" and not key.startswith(("AIza", "AIz")): return False
    return True

# Verificación de disponibilidad (Lazy Loading - No instancia cliente globalmente)
GROQ_AVAILABLE = False
try:
    import groq
    if validate_api_key(GROQ_API_KEY, "groq"):
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API configurada correctamente")
    else:
        logger.warning("⚠️ Groq API Key inválida o ausente")
except ImportError:
    logger.warning("⚠️ Biblioteca 'groq' no instalada")

# ----------------------------
# 2. CLASE ASYNC GROQ (Mejorada)
# ----------------------------
class AsyncGroqWrapper:
    """Wrapper para ejecutar llamadas síncronas de Groq de forma asíncrona y segura."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def chat_completion(self, messages, model, temperature=0.3, max_tokens=900, response_format=None):
        if not self.api_key: return None
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor()
        try:
            # Instanciamos el cliente aquí para evitar problemas de contexto global en Cloud
            client = groq.Groq(api_key=self.api_key)
            
            def _call():
                return client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
            
            response = await loop.run_in_executor(executor, _call)
            return response
        except Exception as e:
            logger.error(f"Error AsyncGroq: {e}")
            raise e
        finally:
            executor.shutdown(wait=False)

# ----------------------------
# 3. GESTIÓN DE CACHÉ
# ----------------------------
class ExpiringCache:
    """Caché con TTL y limpieza lazy."""
    def __init__(self, ttl_seconds=43200):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                return value
            del self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = (value, datetime.now())

search_cache = ExpiringCache(ttl_seconds=43200)
# Executor global para tareas background (no UI)
background_executor = ThreadPoolExecutor(max_workers=MAX_BACKGROUND_TASKS)
background_tasks: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# ----------------------------
# 4. MODELOS DE DATOS & UTILIDADES JSON
# ----------------------------
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

def safe_json_dumps(obj: Dict) -> str:
    try: return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception: return "{}"

def safe_json_loads(text: str, default_value: Dict = None) -> Dict:
    if default_value is None: default_value = {}
    try: return json.loads(text)
    except Exception: return default_value

# ----------------------------
# 5. BASE DE DATOS (Context Manager + Seed Completa)
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
            # Tablas adicionales para analítica y certificaciones
            cursor.execute('''CREATE TABLE IF NOT EXISTS analiticas_busquedas (id INTEGER PRIMARY KEY, tema TEXT, idioma TEXT, nivel TEXT, timestamp TEXT)''')
            
            cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas")
            if cursor.fetchone()[0] == 0:
                # LISTA COMPLETA RESTAURADA
                plataformas_iniciales = [
                    {"nombre": "Aprende con Alf", "url_base": "https://aprendeconalf.es/?s={}", "descripcion": "Tutoriales Python/Data", "idioma": "es", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.85, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["es"], "reputacion_academica": 0.90},
                    {"nombre": "Coursera", "url_base": "https://www.coursera.org/search?query={}&free=true", "descripcion": "Cursos universitarios audit", "idioma": "en", "categoria": "General", "nivel": "Avanzado", "confianza": 0.95, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.95},
                    {"nombre": "Kaggle Learn", "url_base": "https://www.kaggle.com/learn/search?q={}", "descripcion": "Microcursos Data Science", "idioma": "en", "categoria": "Data Science", "nivel": "Intermedio", "confianza": 0.90, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.88},
                    {"nombre": "freeCodeCamp", "url_base": "https://www.freecodecamp.org/news/search/?query={}", "descripcion": "Certificados Web/Dev", "idioma": "en", "categoria": "Programación", "nivel": "Intermedio", "confianza": 0.93, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.91},
                    {"nombre": "edX", "url_base": "https://www.edx.org/search?tab=course&availability=current&price=free&q={}", "descripcion": "Cursos Harvard/MIT", "idioma": "en", "categoria": "Académico", "nivel": "Avanzado", "confianza": 0.92, "tipo_certificacion": "audit", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.93},
                    {"nombre": "PhET Simulations", "url_base": "https://phet.colorado.edu/en/search?q={}", "descripcion": "Simulaciones Ciencias", "idioma": "en", "categoria": "Ciencias", "nivel": "Todos", "confianza": 0.88, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.85},
                    {"nombre": "Domestika (Gratuito)", "url_base": "https://www.domestika.org/es/search?query={}&free=1", "descripcion": "Cursos creativos", "idioma": "es", "categoria": "Diseño", "nivel": "Intermedio", "confianza": 0.83, "tipo_certificacion": "pago", "validez_internacional": 1, "paises_validos": ["es", "latam"], "reputacion_academica": 0.82},
                    {"nombre": "Biblioteca Cervantes", "url_base": "https://www.cervantesvirtual.com/buscar/?q={}", "descripcion": "Recursos académicos hispanos", "idioma": "es", "categoria": "Humanidades", "nivel": "Avanzado", "confianza": 0.87, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["es", "latam"], "reputacion_academica": 0.85},
                    {"nombre": "OER Commons", "url_base": "https://www.oercommons.org/search?q={}", "descripcion": "Recursos educativos abiertos", "idioma": "en", "categoria": "General", "nivel": "Todos", "confianza": 0.89, "tipo_certificacion": "gratuito", "validez_internacional": 1, "paises_validos": ["global"], "reputacion_academica": 0.87},
                    {"nombre": "Programming Historian", "url_base": "https://programminghistorian.org/en/lessons/?q={}", "descripcion": "Tutoriales humanidades digitales", "idioma": "en", "categoria": "Programación", "nivel": "Avanzado", "confianza": 0.82, "tipo_certificacion": "gratuito", "validez_internacional": 0, "paises_validos": ["uk", "us"], "reputacion_academica": 0.80}
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
# 6. UTILIDADES GENERALES & PARCHE DE UI
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

# --- PARCHE DE LIMPIEZA PARA CHAT ---
def limpiar_html_visible(texto: str) -> str:
    """Elimina bloques JSON y HTML del texto visible."""
    if not texto: return ""
    # Eliminar bloques JSON al final
    texto = re.sub(r'\{.*\}\s*$', '', texto, flags=re.DOTALL).strip()
    # Eliminar etiquetas HTML básicas
    texto = re.sub(r'<[^>]+>', '', texto).strip()
    return texto

def ui_chat_mostrar(mensaje: str, rol: str):
    """Muestra mensaje limpio en UI."""
    texto_limpio = limpiar_html_visible(mensaje)
    if not texto_limpio: return
    
    if rol == "assistant":
        st.markdown(f"🤖 **IA:** {texto_limpio}")
    elif rol == "user":
        st.markdown(f"👤 **Tú:** {texto_limpio}")

def obtener_perfil_usuario() -> Dict:
    return {
        "nivel_real": st.session_state.get("nivel_real", "intermedio"),
        "objetivos": st.session_state.get("objetivos", "mejorar habilidades"),
        "tiempo_disponible": "2-3 horas"
    }

# ----------------------------
# 7. LÓGICA DE BÚSQUEDA
# ----------------------------
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

def buscar_en_plataformas_conocidas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    recursos = []
    # DICCIONARIO DETALLADO
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
             {"nombre": "Udemy (PT)", "url": f"https://www.udemy.com/courses/search/?q={tema}&price=price-free&lang=pt"}
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
    res.extend(buscar_en_plataformas_ocultas(tema, get_codigo_idioma(idioma), nivel))
    res.extend(await buscar_en_google_api(tema, get_codigo_idioma(idioma), nivel))
    if len(res) < 5:
        res.extend(buscar_en_plataformas_conocidas(tema, get_codigo_idioma(idioma), nivel))
    
    seen, final = set(), []
    for r in res:
        if r.url not in seen:
            seen.add(r.url)
            final.append(r)
    
    final.sort(key=lambda x: x.confianza, reverse=True)
    final = final[:15]
    
    if GROQ_AVAILABLE:
        for r in final[:4]: r.analisis_pendiente = True
        
    search_cache.set(cache_key, final)
    return final

# ----------------------------
# 8. ANÁLISIS IA (Async con AsyncGroqWrapper)
# ----------------------------
async def analizar_calidad_curso(recurso: RecursoEducativo, perfil: Dict) -> Dict:
    if not GROQ_AVAILABLE: return {}
    try:
        wrapper = AsyncGroqWrapper(GROQ_API_KEY)
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
            "recomendacion_personalizada": "Tu conclusión breve",
            "advertencias": []
        }}
        """
        response = await wrapper.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=600,
            response_format={"type": "json_object"}
        )
        
        contenido = response.choices[0].message.content
        try:
            return json.loads(contenido)
        except:
            return {}
    except Exception as e:
        logger.error(f"Error Análisis: {e}")
        return {}

def analizar_resultados_en_segundo_plano(resultados: List[RecursoEducativo]):
    """Orquestador de análisis en background (Síncrono que llama a Async)."""
    if not GROQ_AVAILABLE: return
    
    # Filtrar pendientes
    pendientes = [r for r in resultados if r.analisis_pendiente and not r.metadatos_analisis]
    if not pendientes: return

    # Ejecutar en loop local para no bloquear
    async def _process():
        perfil = obtener_perfil_usuario()
        for r in pendientes:
            analisis = await analizar_calidad_curso(r, perfil)
            if analisis:
                calidad = float(analisis.get("calidad_educativa", 0.7))
                relevancia = float(analisis.get("relevancia_usuario", 0.7))
                r.metadatos_analisis = {
                    "calidad_ia": calidad,
                    "relevancia_ia": relevancia,
                    "recomendacion_personalizada": analisis.get("recomendacion_personalizada", ""),
                    "razones_calidad": analisis.get("razones_calidad", []),
                    "advertencias": analisis.get("advertencias", [])
                }
                # Actualizar confianza
                r.confianza = (r.confianza + calidad) / 2
            r.analisis_pendiente = False
            # Pequeña pausa para no saturar API
            await asyncio.sleep(0.5)

    # Hack para Streamlit: correr el loop async en un thread separado
    def _run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_process())
        loop.close()

    background_executor.submit(_run_async_loop)

def planificar_analisis_ia(resultados: List[RecursoEducativo]):
    # Añadir a la cola si usáramos worker dedicado, o lanzar directo al executor
    analizar_resultados_en_segundo_plano(resultados)

# ----------------------------
# 9. UI Y VISUALIZACIÓN
# ----------------------------
st.set_page_config(page_title="Buscador de Cursos PRO", layout="wide", page_icon="🎓")

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
    if not url: return "" 
    return f'''<a href="{url}" target="_blank" style="display: inline-block; background: linear-gradient(to right, #6a11cb, #2575fc); color: white; padding: 10px 20px; border-radius: 8px; font-weight: bold;">{label}</a>'''

def badge_certificacion(cert: Optional[Certificacion]) -> str:
    if not cert: return ""
    html = ""
    if cert.tipo == "gratuito": html += '<span class="certificado-badge">✅ Gratis</span>'
    elif cert.tipo == "audit": html += '<span class="certificado-badge" style="background:#e3f2fd;color:#1565c0;">🎓 Audit</span>'
    return html

def mostrar_recurso(r: RecursoEducativo, idx: int):
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

    desc = r.descripcion if r.descripcion else "Sin descripción disponible."
    titulo = r.titulo if r.titulo else "Recurso Educativo"

    st.markdown(f"""
<div class="resultado-card {nivel_class} {extra_class}" style="animation-delay: {idx * 0.08}s;">
    <h3 style="margin-top:0;">{titulo}</h3>
    <p><strong>📚 {r.nivel}</strong> | 🌐 {r.plataforma} | 🏷️ {r.categoria}</p>
    <p style="color:#555;">{desc}</p>
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
# 10. MAIN APP (CON CHAT IA INTEGRADO)
# ----------------------------
def main():
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(buscar_recursos_multicapa(tema.strip(), idioma, nivel))
            loop.close()
            
            if resultados:
                st.success(f"✅ Se encontraron {len(resultados)} recursos de alta calidad.")
                
                if GROQ_AVAILABLE:
                    planificar_analisis_ia(resultados)
                    # Forzar un rerun leve o usar placeholder para actualizar UI si fuera necesario
                    # en streamlit standard esto solo actualizará en la próxima interacción, 
                    # pero los datos se están procesando en background.

                for i, r in enumerate(resultados):
                    mostrar_recurso(r, i)
                    
                df = pd.DataFrame([{'Título': r.titulo, 'URL': r.url, 'Plataforma': r.plataforma, 'Confianza': r.confianza} for r in resultados])
                st.download_button("📥 Descargar CSV", df.to_csv(index=False).encode('utf-8'), "cursos.csv", "text/csv")
            else:
                st.warning("No se encontraron resultados. Intenta con términos más generales.")

    # --- CHAT IA (Sidebar) ---
    with st.sidebar:
        st.header("💬 Asistente Educativo")
        if "chat_msgs" not in st.session_state: st.session_state.chat_msgs = []
        
        # Mostrar historial limpio (usando ui_chat_mostrar)
        for msg in st.session_state.chat_msgs:
            ui_chat_mostrar(msg["content"], msg["role"])
            
        if user_input := st.chat_input("Pregunta sobre cursos..."):
            st.session_state.chat_msgs.append({"role": "user", "content": user_input})
            ui_chat_mostrar(user_input, "user")
            
            if GROQ_AVAILABLE:
                try:
                    # Usamos el Wrapper para no bloquear
                    async_wrapper = AsyncGroqWrapper(GROQ_API_KEY)
                    msgs = [{"role": "system", "content": "Eres un experto educativo. Sé breve y útil."}] + st.session_state.chat_msgs
                    
                    # Hack para ejecutar async en botón síncrono de streamlit
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    resp = loop.run_until_complete(async_wrapper.chat_completion(msgs, GROQ_MODEL, temperature=0.5))
                    loop.close()
                    
                    if resp:
                        reply = resp.choices[0].message.content
                        st.session_state.chat_msgs.append({"role": "assistant", "content": reply})
                        st.rerun()
                except Exception as e:
                    st.error(f"Error Chat: {e}")
            else:
                st.warning("Chat IA no disponible (Falta API Key)")

if __name__ == "__main__":
    main()
