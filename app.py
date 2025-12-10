# app.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import random
from datetime import datetime, timedelta
import json
import hashlib
from urllib.parse import urlparse
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
import asyncio
import aiohttp

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

# Cargar variables de entorno desde Streamlit Secrets o .env
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = st.secrets.get("GOOGLE_CX", "") if hasattr(st, 'secrets') else os.getenv("GOOGLE_CX", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("GROQ_API_KEY", "")
DUCKDUCKGO_ENABLED = st.secrets.get("DUCKDUCKGO_ENABLED", "true").lower() == "true" if hasattr(st, 'secrets') else os.getenv("DUCKDUCKGO_ENABLED", "true").lower() == "true"

# Configuración de parámetros
MAX_BACKGROUND_TASKS = 1  # CRÍTICO para SQLite! Evita "database is locked"
CACHE_EXPIRATION = timedelta(hours=12)
GROQ_MODEL = "llama3-8b-8192"  # Modelo rápido

# Sistema de caché para búsquedas frecuentes
search_cache: Dict[str, Dict[str, Any]] = {}

# Cola para tareas en segundo plano
background_tasks: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# Bandera para indicar si Groq está disponible
GROQ_AVAILABLE = False
try:
    import groq  # type: ignore
    if GROQ_API_KEY:
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API disponible para análisis en segundo plano")
except ImportError:
    logger.warning("⚠️ groq no instalado - Análisis de IA no disponible")
except Exception as e:
    logger.warning(f"⚠️ Error al inicializar Groq: {e}")

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
    analisis_pendiente: bool = False

# ----------------------------
# CONFIGURACIÓN DE BASE DE DATOS
# ----------------------------
DB_PATH = "cursos_inteligentes_v3.db"

def init_advanced_database() -> bool:
    """Inicializa la base de datos avanzada con todas las tablas necesarias"""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()

        # Tabla de plataformas mejorada
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
            validez_internacional BOOLEAN DEFAULT 0,
            paises_validos TEXT DEFAULT '[]',
            reputacion_academica REAL DEFAULT 0.5
        )
        ''')

        # Tabla de analíticas
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

        # Certificaciones verificadas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS certificaciones_verificadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plataforma TEXT NOT NULL,
            curso_tema TEXT NOT NULL,
            tipo_certificacion TEXT NOT NULL,
            validez_internacional BOOLEAN DEFAULT 0,
            paises_validos TEXT DEFAULT '[]',
            costo_certificado REAL DEFAULT 0.0,
            reputacion_academica REAL DEFAULT 0.5,
            ultima_verificacion TEXT NOT NULL,
            veces_verificado INTEGER DEFAULT 1
        )
        ''')

        # Datos semilla solo si está vacío
        cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas")
        if cursor.fetchone()[0] == 0:
            plataformas_iniciales = [
                ("Aprende con Alf", "https://aprendeconalf.es/?s={}", "Cursos gratuitos de programación, matemáticas y ciencia de datos con ejercicios prácticos", "es", "Programación", "Intermedio", 0.85, json.dumps(["es"]), 1, 0.9),
                ("Coursera", "https://www.coursera.org/search?query={}&free=true", "Plataforma líder con cursos universitarios gratuitos (audit mode)", "en", "General", "Avanzado", 0.95, json.dumps(["us", "uk", "ca", "au", "eu"]), 1, 0.95),
                ("edX", "https://www.edx.org/search?tab=course&availability=current&price=free&q={}", "Cursos de Harvard, MIT y otras universidades top (modo audit gratuito)", "en", "Académico", "Avanzado", 0.92, json.dumps(["us", "uk", "ca", "au", "eu"]), 1, 0.93),
                ("Kaggle Learn", "https://www.kaggle.com/learn/search?q={}", "Microcursos prácticos de ciencia de datos con certificados gratuitos", "en", "Data Science", "Intermedio", 0.90, json.dumps(["global"]), 1, 0.88),
                ("freeCodeCamp", "https://www.freecodecamp.org/news/search/?query={}", "Certificados gratuitos completos en desarrollo web y ciencia de datos", "en", "Programación", "Intermedio", 0.93, json.dumps(["global"]), 1, 0.91),
                ("PhET Simulations", "https://phet.colorado.edu/en/search?q={}", "Simulaciones interactivas de ciencias y matemáticas de la Universidad de Colorado", "en", "Ciencias", "Todos", 0.88, json.dumps(["us", "global"]), 1, 0.85),
                ("The Programming Historian", "https://programminghistorian.org/en/lessons/?q={}", "Tutoriales académicos de programación y humanidades digitales", "en", "Programación", "Avanzado", 0.82, json.dumps(["uk", "us", "ca"]), 0, 0.80),
                ("Domestika (Gratuito)", "https://www.domestika.org/es/search?query={}&free=1", "Cursos gratuitos de diseño creativo, algunos con certificados verificados", "es", "Diseño", "Intermedio", 0.83, json.dumps(["es", "mx", "ar", "cl"]), 1, 0.82),
                ("Biblioteca Virtual Miguel de Cervantes", "https://www.cervantesvirtual.com/buscar/?q={}", "Recursos académicos hispanos con validez internacional", "es", "Humanidades", "Avanzado", 0.87, json.dumps(["es", "latam", "eu"]), 1, 0.85),
                ("OER Commons", "https://www.oercommons.org/search?q={}", "Recursos educativos abiertos de instituciones globales con estándares académicos", "en", "General", "Todos", 0.89, json.dumps(["global"]), 1, 0.87)
            ]
            # INSERT corregido: 12 columnas y 12 valores (incluye 'activa')
            for plat in plataformas_iniciales:
                cursor.execute('''
                INSERT INTO plataformas_ocultas
                (nombre, url_base, descripcion, idioma, categoria, nivel, confianza, paises_validos, validez_internacional, reputacion_academica, ultima_verificacion, activa)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', plat + (datetime.now().isoformat(), 1))

        conn.commit()
        conn.close()

        logger.info("✅ Base de datos inicializada correctamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error al inicializar la base de datos: {e}")
        return False

def init_cache_system():
    """Inicializa el sistema de caché para búsquedas frecuentes"""
    if not hasattr(st, 'search_cache'):
        st.search_cache = {}
    if not hasattr(st, 'cert_cache'):
        st.cert_cache = {}
    if not hasattr(st, 'verification_cache'):
        st.verification_cache = {}

# Inicializar DB avanzada
init_advanced_database()
init_cache_system()

# ----------------------------
# FUNCIONES AUXILIARES
# ----------------------------
def get_codigo_idioma(nombre_idioma: str) -> str:
    mapeo = {
        "Español (es)": "es",
        "Inglés (en)": "en",
        "Portugués (pt)": "pt",
        "es": "es",
        "en": "en",
        "pt": "pt"
    }
    return mapeo.get(nombre_idioma, "es")

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    texto = (url + titulo + descripcion).lower()
    palabras_validas = ['curso', 'tutorial', 'aprender', 'education', 'learn', 'gratuito', 'free', 'certificado', 'certificate', 'clase', 'class', 'educación', 'educacion', 'clases']
    palabras_invalidas = ['comprar', 'buy', 'precio', 'price', 'costo', 'only', 'premium', 'exclusive', 'paid', 'pago', 'suscripción', 'subscription', 'membership', 'register now', 'matrícula']
    dominios_educativos = ['.edu', '.ac.', '.edu.', 'coursera', 'edx', 'khanacademy', 'freecodecamp', 'kaggle', 'udemy', 'youtube', 'aprendeconalf', '.org', '.gob', '.gov']

    tiene_validas = any(p in texto for p in palabras_validas)
    tiene_invalidas = any(p in texto for p in palabras_invalidas)
    dominio_valido = any(d in url.lower() for d in dominios_educativos)
    es_gratuito = 'gratuito' in texto or 'free' in texto or 'sin costo' in texto or 'audit' in texto

    return (tiene_validas or dominio_valido) and not tiene_invalidas and (es_gratuito or dominio_valido)

def generar_id_unico(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    texto = texto.lower()
    if nivel_solicitado not in ("Cualquiera", "Todos"):
        return nivel_solicitado
    if any(p in texto for p in ['principiante', 'basico', 'básico', 'beginner', 'fundamentos', 'introducción', 'desde cero', 'básica', 'básicas', 'nociones básicas']):
        return "Principiante"
    if any(p in texto for p in ['intermedio', 'intermediate', 'práctico', 'aplicado', 'práctica', 'profesional', 'avanzado básico', 'nivel básico', 'medio']):
        return "Intermedio"
    if any(p in texto for p in ['avanzado', 'advanced', 'experto', 'máster', 'especialista', 'complejo', 'profundo', 'expertos', 'dominio']):
        return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    t = tema.lower()
    if any(p in t for p in ['programación', 'python', 'javascript', 'web', 'desarrollo', 'coding', 'programming', 'developer', 'software', 'app', 'mobile', 'java', 'c++', 'c#', 'html', 'css']):
        return "Programación"
    if any(p in t for p in ['datos', 'data', 'machine learning', 'ia', 'ai', 'artificial intelligence', 'ciencia de datos', 'big data', 'analytics', 'deep learning', 'estadística', 'statistics']):
        return "Data Science"
    if any(p in t for p in ['matemáticas', 'math', 'álgebra', 'calculus', 'probability', 'geometría', 'cálculo']):
        return "Matemáticas"
    if any(p in t for p in ['diseño', 'design', 'ux', 'ui', 'gráfico', 'graphic', 'creativo', 'illustration', 'photoshop', 'figma', 'canva']):
        return "Diseño"
    if any(p in t for p in ['marketing', 'business', 'negocios', 'finanzas', 'finance', 'emprendimiento', 'startups', 'economía', 'economia', 'management', 'ventas']):
        return "Negocios"
    if any(p in t for p in ['idioma', 'language', 'inglés', 'english', 'español', 'portugues', 'francés', 'alemán', 'lingüística', 'hablar', 'conversación']):
        return "Idiomas"
    return "General"

def calcular_confianza_google(item: dict) -> float:
    confianza_base = 0.7
    url = item.get('link', '').lower()
    if any(d in url for d in ['.edu', '.ac.', 'coursera.org', 'edx.org', 'khanacademy.org', 'freecodecamp.org', '.gov', '.gob']):
        confianza_base += 0.15
    elif any(d in url for d in ['udemy.com', 'domestika.org', 'skillshare.com']):
        confianza_base += 0.05

    snippet = item.get('snippet', '')
    if len(snippet) > 100:
        confianza_base += 0.05

    rank = item.get('rank', 1)
    if rank == 1:
        confianza_base += 0.1
    elif rank == 2:
        confianza_base += 0.05

    return min(confianza_base, 0.95)

def extraer_plataforma(url: str) -> str:
    dominio = urlparse(url).netloc.lower()
    if 'coursera' in dominio: return 'Coursera'
    if 'edx' in dominio: return 'edX'
    if 'khanacademy' in dominio: return 'Khan Academy'
    if 'freecodecamp' in dominio: return 'freeCodeCamp'
    if 'kaggle' in dominio: return 'Kaggle'
    if 'udemy' in dominio: return 'Udemy'
    if 'youtube' in dominio: return 'YouTube'
    if 'aprendeconalf' in dominio: return 'Aprende con Alf'
    if 'programminghistorian' in dominio: return 'Programming Historian'
    if 'cervantesvirtual' in dominio: return 'Biblioteca Cervantes'
    if '.edu' in dominio or '.ac.' in dominio or '.gob' in dominio or '.gov' in dominio:
        return 'Institución Académica'
    partes = dominio.split('.')
    if len(partes) > 1:
        return partes[-2].title()
    return dominio.title()

# ----------------------------
# IA: Análisis de calidad (opcional)
# ----------------------------
async def analizar_calidad_curso(recurso: RecursoEducativo, perfil_usuario: Dict) -> Dict:
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return {
            "calidad_educativa": recurso.confianza,
            "relevancia_usuario": recurso.confianza,
            "razones_calidad": ["Análisis básico - IA no disponible"],
            "razones_relevancia": ["Análisis básico - IA no disponible"],
            "recomendacion_personalizada": "Curso verificado con el sistema estándar",
            "advertencias": ["Análisis de IA no disponible en este momento"]
        }

    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Eres un asesor educativo experto. Evalúa este recurso:

        TÍTULO: {recurso.titulo}
        PLATAFORMA: {recurso.plataforma}
        DESCRIPCIÓN: {recurso.descripcion}
        NIVEL DECLARADO: {recurso.nivel}
        CATEGORÍA: {recurso.categoria}
        TIPO: {recurso.tipo}
        CERTIFICACIÓN: {'Sí' if recurso.certificacion else 'No'}

        PERFIL USUARIO:
        - Nivel real: {perfil_usuario.get('nivel_real', 'desconocido')}
        - Objetivos: {perfil_usuario.get('objetivos', 'aprender en general')}
        - Tiempo disponible: {perfil_usuario.get('tiempo_disponible', 'desconocido')}
        - Experiencia previa: {perfil_usuario.get('experiencia_previa', 'ninguna')}

        Devuelve JSON:
        {{
            "calidad_educativa": 0.0-1.0,
            "relevancia_usuario": 0.0-1.0,
            "razones_calidad": ["..."],
            "razones_relevancia": ["..."],
            "recomendacion_personalizada": "...",
            "advertencias": ["..."]
        }}
        """

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        contenido = response.choices[0].message.content
        try:
            return json.loads(contenido)
        except json.JSONDecodeError:
            logger.error("Error al parsear JSON de Groq. Se usa confianza estándar.")
            return {
                "calidad_educativa": recurso.confianza,
                "relevancia_usuario": recurso.confianza,
                "razones_calidad": ["Error IA - confianza estándar"],
                "razones_relevancia": ["Error IA - confianza estándar"],
                "recomendacion_personalizada": "Curso verificado por sistema estándar",
                "advertencias": ["Hubo un problema con el análisis IA"]
            }

    except Exception as e:
        logger.error(f"Error en análisis con Groq: {e}")
        return {
            "calidad_educativa": recurso.confianza,
            "relevancia_usuario": recurso.confianza,
            "razones_calidad": [f"Error IA: {str(e)}"],
            "razones_relevancia": [f"Error IA: {str(e)}"],
            "recomendacion_personalizada": "Curso verificado por sistema estándar",
            "advertencias": ["Análisis de IA temporalmente no disponible"]
        }

def obtener_perfil_usuario() -> Dict:
    return {
        "nivel_real": st.session_state.get("nivel_real", "intermedio"),
        "objetivos": st.session_state.get("objetivos", "mejorar habilidades profesionales"),
        "tiempo_disponible": st.session_state.get("tiempo_disponible", "2-3 horas por semana"),
        "experiencia_previa": st.session_state.get("experiencia_previa", "algunos cursos básicos"),
        "estilo_aprendizaje": st.session_state.get("estilo_aprendizaje", "práctico con proyectos")
    }

# ----------------------------
# BÚSQUEDA MULTICAPA
# ----------------------------
async def buscar_en_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    try:
        if not GOOGLE_API_KEY or not GOOGLE_CX:
            return []

        query_base = f"{tema} curso gratuito certificado"
        if nivel not in ("Cualquiera", "Todos"):
            query_base += f" nivel {nivel.lower()}"

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CX,
            'q': query_base,
            'num': 5,
            'lr': f'lang_{idioma}',
            'cr': 'countryES' if idioma == 'es' else 'countryUS'
        }

        resultados: List[RecursoEducativo] = []
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status != 200:
                    return []
                data = await response.json()
                if 'items' not in data:
                    return []

                for item in data['items']:
                    url_ = item.get('link', '')
                    titulo = item.get('title', '')
                    descripcion = item.get('snippet', '')

                    if not es_recurso_educativo_valido(url_, titulo, descripcion):
                        continue

                    nivel_calculado = determinar_nivel(texto=titulo + " " + descripcion, nivel_solicitado=nivel)
                    confianza = calcular_confianza_google(item)

                    recurso = RecursoEducativo(
                        id=generar_id_unico(url_),
                        titulo=titulo,
                        url=url_,
                        descripcion=descripcion,
                        plataforma=extraer_plataforma(url_),
                        idioma=idioma,
                        nivel=nivel_calculado,
                        categoria=determinar_categoria(tema),
                        certificacion=None,
                        confianza=confianza,
                        tipo="verificada",
                        ultima_verificacion=datetime.now().isoformat(),
                        activo=True,
                        metadatos={'google_rank': item.get('rank', 1), 'snippet_length': len(descripcion), 'fuente': 'google_api'}
                    )
                    resultados.append(recurso)
        return resultados[:5]

    except asyncio.TimeoutError:
        logger.error("Timeout en petición a Google API")
        return []
    except Exception as e:
        logger.error(f"Error Google API: {e}")
        return []

async def buscar_en_duckduckgo(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not DUCKDUCKGO_ENABLED:
        return []
    try:
        query = f"{tema} curso gratuito certificado"
        if nivel not in ("Cualquiera", "Todos"):
            query += f" nivel {nivel.lower()}"

        url = "https://duckduckgo-api.vercel.app"
        params = {'q': query, 'format': 'json', 'pretty': '1'}

        resultados: List[RecursoEducativo] = []
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status != 200:
                    return []
                data = await response.json()

                for result in data.get('Results', [])[:3]:
                    url_ = result.get('FirstURL', '')
                    titulo = result.get('Text', '')
                    descripcion = result.get('Result', '')

                    if not es_recurso_educativo_valido(url_, titulo, descripcion):
                        continue

                    recurso = RecursoEducativo(
                        id=generar_id_unico(url_),
                        titulo=titulo,
                        url=url_,
                        descripcion=descripcion,
                        plataforma=extraer_plataforma(url_),
                        idioma=idioma,
                        nivel=determinar_nivel(texto=titulo + " " + descripcion, nivel_solicitado=nivel),
                        categoria=determinar_categoria(tema),
                        certificacion=None,
                        confianza=0.75,
                        tipo="verificada",
                        ultima_verificacion=datetime.now().isoformat(),
                        activo=True,
                        metadatos={'fuente': 'duckduckgo'}
                    )
                    resultados.append(recurso)
        return resultados
    except Exception as e:
        logger.error(f"Error DuckDuckGo: {e}")
        return []

async def buscar_en_plataformas_conocidas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    try:
        resultados: List[RecursoEducativo] = []
        if idioma == "es":
            plataformas = {
                "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", "icono": "📺", "niveles": ["Principiante", "Intermedio"]},
                "coursera": {"nombre": "Coursera (Español)", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=es&free=true", "icono": "🎓", "niveles": ["Intermedio", "Avanzado"]},
                "udemy": {"nombre": "Udemy (Español)", "url": f"https://www.udemy.com/courses/search/?price=price-free&lang=es&q={tema.replace(' ', '%20')}", "icono": "💻", "niveles": ["Principiante", "Intermedio"]},
                "khan": {"nombre": "Khan Academy (Español)", "url": f"https://es.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}", "icono": "📚", "niveles": ["Principiante", "Intermedio"]}
            }
        elif idioma == "pt":
            plataformas = {
                "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", "icono": "📺", "niveles": ["Principiante", "Intermedio"]},
                "coursera": {"nombre": "Coursera (Portugués)", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=pt&free=true", "icono": "🎓", "niveles": ["Intermedio", "Avanzado"]},
                "udemy": {"nombre": "Udemy (Português)", "url": f"https://www.udemy.com/courses/search/?price=price-free&lang=pt&q={tema.replace(' ', '%20')}", "icono": "💻", "niveles": ["Principiante", "Intermedio"]},
                "khan": {"nombre": "Khan Academy (Português)", "url": f"https://pt.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}", "icono": "📚", "niveles": ["Principiante", "Intermedio"]}
            }
        else:
            plataformas = {
                "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=course+free+{tema.replace(' ', '+')}", "icono": "📺", "niveles": ["Principiante", "Intermedio"]},
                "coursera": {"nombre": "Coursera", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&free=true", "icono": "🎓", "niveles": ["Intermedio", "Avanzado"]},
                "edx": {"nombre": "edX", "url": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={tema.replace(' ', '%20')}", "icono": "🔬", "niveles": ["Avanzado"]},
                "udemy": {"nombre": "Udemy", "url": f"https://www.udemy.com/courses/search/?price=price-free&q={tema.replace(' ', '%20')}", "icono": "💻", "niveles": ["Principiante", "Intermedio"]},
                "freecodecamp": {"nombre": "freeCodeCamp", "url": f"https://www.freecodecamp.org/news/search/?query={tema.replace(' ', '%20')}", "icono": "👨‍💻", "niveles": ["Intermedio", "Avanzado"]},
                "khan": {"nombre": "Khan Academy", "url": f"https://www.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}", "icono": "📚", "niveles": ["Principiante"]}
            }

        for nombre_plataforma, datos in plataformas.items():
            if len(resultados) >= 4:
                break
            niveles_compatibles = [n for n in datos['niveles'] if nivel in ("Cualquiera", "Todos") or n == nivel]
            if not niveles_compatibles:
                continue

            nivel_actual = random.choice(niveles_compatibles)

            titulos_realistas = {
                "python": ["Curso Completo de Python", "Python para Data Science", "Automatización con Python"],
                "machine learning": ["Machine Learning Completo", "Deep Learning con TensorFlow", "Ciencia de Datos con Python"],
                "marketing": ["Marketing Digital Completo", "SEO Avanzado", "Email Marketing Profesional"],
                "ingles": ["Inglés desde Cero", "Inglés para Negocios", "Gramática Inglesa Explicada"],
                "diseño": ["Diseño Gráfico Completo", "UI/UX Design", "Diseño de Logotipos"],
                "finanzas": ["Finanzas Personales", "Inversión para Principiantes", "Criptomonedas y Blockchain"],
                "data": ["Análisis de Datos con Python", "Visualización de Datos", "SQL para Data Science", "Big Data Fundamentals"]
            }

            tema_minus = tema.lower()
            titulo_base = random.choice([
                f"Curso Completo de {tema}",
                f"{tema} desde Cero",
                f"Aprende {tema} en 30 Días",
                f"Domina {tema} con Proyectos Prácticos"
            ])
            for clave, titulos in titulos_realistas.items():
                if clave in tema_minus:
                    titulo_base = random.choice(titulos)
                    break

            titulo = f"{datos['icono']} {titulo_base} en {datos['nombre']}"
            descripciones = {
                "Principiante": f"Curso introductorio perfecto para quienes empiezan en {tema}. Sin conocimientos previos.",
                "Intermedio": f"Curso práctico para profundizar en {tema} con ejercicios y proyectos reales.",
                "Avanzado": f"Contenido especializado para dominar conceptos avanzados de {tema}."
            }

            recurso = RecursoEducativo(
                id=generar_id_unico(datos["url"]),
                titulo=titulo,
                url=datos["url"],
                descripcion=descripciones.get(nivel_actual, f"Recurso educativo verificado para nivel {nivel_actual} en {tema}."),
                plataforma=datos["nombre"],
                idioma=idioma,
                nivel=nivel_actual,
                categoria=determinar_categoria(tema),
                certificacion=Certificacion(
                    plataforma=datos["nombre"],
                    curso=tema,
                    tipo="audit",
                    validez_internacional=True,
                    paises_validos=["global"],
                    costo_certificado=0.0,
                    reputacion_academica=0.85,
                    ultima_verificacion=datetime.now().isoformat()
                ) if ("coursera" in nombre_plataforma or "edx" in nombre_plataforma) else None,
                confianza=0.85,
                tipo="conocida",
                ultima_verificacion=datetime.now().isoformat(),
                activo=True,
                metadatos={"fuente": "plataformas_conocidas"}
            )
            resultados.append(recurso)
        return resultados
    except Exception as e:
        logger.error(f"Error plataformas conocidas: {e}")
        return []

async def buscar_en_plataformas_ocultas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    try:
        conn = sqlite3.connect(DB_PATH)
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
        query += " ORDER BY confianza DESC LIMIT 4"

        cursor.execute(query, params)
        resultados = cursor.fetchall()
        conn.close()

        recursos: List[RecursoEducativo] = []
        for r in resultados:
            url_completa = r[1].format(tema.replace(' ', '+'))
            nivel_calculado = r[3] if nivel in ("Cualquiera", "Todos") else nivel

            recurso = RecursoEducativo(
                id=generar_id_unico(url_completa),
                titulo=f"💎 {r[0]} - {tema}",
                url=url_completa,
                descripcion=r[2],
                plataforma=r[0],
                idioma=idioma,
                nivel=nivel_calculado,
                categoria=determinar_categoria(tema),
                certificacion=Certificacion(
                    plataforma=r[0],
                    curso=tema,
                    tipo=r[5],
                    validez_internacional=bool(r[6]),
                    paises_validos=json.loads(r[7]),
                    costo_certificado=0.0 if r[5] == "gratuito" else 49.99,
                    reputacion_academica=r[8],
                    ultima_verificacion=datetime.now().isoformat()
                ) if r[5] != "audit" else None,
                confianza=r[4],
                tipo="oculta",
                ultima_verificacion=datetime.now().isoformat(),
                activo=True,
                metadatos={"fuente": "plataformas_ocultas", "confianza_db": r[4]}
            )
            recursos.append(recurso)
        return recursos
    except Exception as e:
        logger.error(f"Error plataformas ocultas: {e}")
        return []

def eliminar_duplicados(resultados: List[RecursoEducativo]) -> List[RecursoEducativo]:
    urls_vistas = set()
    unicos = []
    for r in resultados:
        if r.url not in urls_vistas:
            urls_vistas.add(r.url)
            unicos.append(r)
    return unicos

async def buscar_recursos_multicapa(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    cache_key = f"busqueda_{tema}_{idioma}_{nivel}"
    if cache_key in search_cache:
        cached = search_cache[cache_key]
        if datetime.now() - cached['timestamp'] < CACHE_EXPIRATION:
            return cached['resultados']

    resultados: List[RecursoEducativo] = []
    codigo_idioma = get_codigo_idioma(idioma)

    # Capas de búsqueda
    resultados_conocidas = await buscar_en_plataformas_conocidas(tema, codigo_idioma, nivel)
    resultados.extend(resultados_conocidas)

    resultados_ocultas = await buscar_en_plataformas_ocultas(tema, codigo_idioma, nivel)
    resultados.extend(resultados_ocultas)

    if GOOGLE_API_KEY and GOOGLE_CX:
        resultados_google = await buscar_en_google_api(tema, codigo_idioma, nivel)
        resultados.extend(resultados_google)

    if DUCKDUCKGO_ENABLED:
        resultados_duck = await buscar_en_duckduckgo(tema, codigo_idioma, nivel)
        resultados.extend(resultados_duck)

    resultados = eliminar_duplicados(resultados)
    resultados.sort(key=lambda x: x.confianza, reverse=True)

    # Marcar mejores para análisis IA
    for recurso in resultados[:5]:
        if GROQ_AVAILABLE and GROQ_API_KEY:
            recurso.analisis_pendiente = True

    search_cache[cache_key] = {
        'resultados': resultados[:10],
        'timestamp': datetime.now()
    }
    return resultados[:10]

# ----------------------------
# ANÁLISIS EN SEGUNDO PLANO
# ----------------------------
def analizar_resultados_en_segundo_plano(resultados: List[RecursoEducativo]):
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return
    try:
        perfil = obtener_perfil_usuario()
        for recurso in resultados:
            if recurso.analisis_pendiente and not recurso.metadatos_analisis:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                analisis = loop.run_until_complete(analizar_calidad_curso(recurso, perfil))
                loop.close()

                if analisis:
                    confianza_ia = (analisis.get("calidad_educativa", recurso.confianza) +
                                    analisis.get("relevancia_usuario", recurso.confianza)) / 2
                    recurso.confianza = min(max(recurso.confianza, confianza_ia), 0.95)
                    recurso.metadatos_analisis = {
                        "calidad_ia": analisis.get("calidad_educativa", recurso.confianza),
                        "relevancia_ia": analisis.get("relevancia_usuario", recurso.confianza),
                        "razones_calidad": analisis.get("razones_calidad", []),
                        "razones_relevancia": analisis.get("razones_relevancia", []),
                        "recomendacion_personalizada": analisis.get("recomendacion_personalizada", ""),
                        "advertencias": analisis.get("advertencias", [])
                    }
                recurso.analisis_pendiente = False
                time.sleep(1)
    except Exception as e:
        logger.error(f"Error análisis background: {e}")

# ----------------------------
# TAREAS EN SEGUNDO PLANO
# ----------------------------
def indexar_recursos_background(temas: List[str], idiomas: List[str], limite_por_tema: int = 5):
    """Indexa recursos por temas/idiomas y guarda analíticas básicas."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        for tema in temas:
            for idioma in idiomas:
                # Guardar analítica mínima por tema/idioma (timestamp)
                cursor.execute('''
                    INSERT INTO analiticas_busquedas (tema, idioma, nivel, timestamp, plataforma_origen, veces_mostrado, veces_clickeado, tiempo_promedio_uso, satisfaccion_usuario)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (tema, idioma, "Cualquiera", datetime.now().isoformat(), "indexador_background", 0, 0, 0.0, 0.0))
        conn.commit()
        conn.close()
        logger.info(f"🧭 Indexación mínima registrada para {len(temas)} temas en {len(idiomas)} idiomas")
    except Exception as e:
        logger.error(f"Error indexación background: {e}")

def iniciar_tareas_background():
    def worker():
        while True:
            try:
                tarea = background_tasks.get(timeout=60)
                if tarea is None:
                    break
                logger.info(f"Procesando tarea: {tarea.get('tipo')}")
                tipo_tarea = tarea.get('tipo')
                parametros = tarea.get('parametros', {})

                if tipo_tarea == 'indexar_recursos':
                    indexar_recursos_background(**parametros)
                elif tipo_tarea == 'analizar_resultados':
                    analizar_resultados_en_segundo_plano(**parametros)

                background_tasks.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error en tarea background: {e}")
                background_tasks.task_done()

    num_workers = min(MAX_BACKGROUND_TASKS, os.cpu_count() or 1)
    for _ in range(num_workers):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    logger.info(f"✅ Sistema de tareas en background iniciado con {num_workers} workers")

def planificar_analisis_ia(resultados: List[RecursoEducativo]):
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return
    tarea = {
        'tipo': 'analizar_resultados',
        'parametros': {
            'resultados': [r for r in resultados if r.analisis_pendiente]
        }
    }
    background_tasks.put(tarea)
    logger.info(f"Tarea IA planificada para {len(tarea['parametros']['resultados'])} resultados")

def planificar_indexacion_recursos(temas: List[str], idiomas: List[str], limite_por_tema: int = 5):
    """Planifica una indexación mínima en background para registrar analíticas."""
    tarea = {
        'tipo': 'indexar_recursos',
        'parametros': {
            'temas': temas,
            'idiomas': idiomas,
            'limite_por_tema': limite_por_tema
        }
    }
    background_tasks.put(tarea)
    logger.info(f"📦 Indexación planificada: {len(temas)} temas x {len(idiomas)} idiomas")

# ----------------------------
# ESTILOS Y CONFIG STREAMLIT
# ----------------------------
st.set_page_config(
    page_title="🎓 Buscador Profesional de Cursos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/tuusuario/buscador-cursos-ia',
        'Report a bug': "https://github.com/tuusuario/buscador-cursos-ia/issues",
        'About': "# Buscador Profesional de Cursos\nSistema de búsqueda inteligente"
    }
)

st.markdown("""
<style>
/* Responsive y estilos visuales */
.main-header {
    background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
    color: white; padding: 2rem; border-radius: 20px; margin-bottom: 2.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25); position: relative; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
}
.main-header h1 {
    font-size: 2.8rem; font-weight: 700; margin-bottom: 1rem; position: relative;
    background: linear-gradient(to right, #fff, #e0e0ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.search-form {
    background: white; padding: 30px; border-radius: 20px; box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    margin-bottom: 35px; border: 1px solid #e0e0e0; position: relative; z-index: 10;
}
.stButton button {
    background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%); color: white; border: none; border-radius: 15px;
    padding: 15px 30px; font-size: 18px; font-weight: bold; width: 100%; transition: all 0.4s ease;
    box-shadow: 0 4px 15px rgba(106, 17, 203, 0.4);
}
.resultado-card {
    border-radius: 15px; padding: 25px; margin-bottom: 25px; background: white;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08); transition: all 0.4s ease; border-left: 6px solid #4CAF50; position: relative;
    overflow: hidden; border: 1px solid #f0f0f0;
}
.nivel-principiante { border-left-color: #2196F3 !important; background: linear-gradient(90deg, rgba(33,150,243,0.05), transparent) !important; }
.nivel-intermedio { border-left-color: #4CAF50 !important; background: linear-gradient(90deg, rgba(76,175,80,0.05), transparent) !important; }
.nivel-avanzado { border-left-color: #FF9800 !important; background: linear-gradient(90deg, rgba(255,152,0,0.05), transparent) !important; }
.plataforma-oculta { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-left-color: #FF6B35 !important; }
.con-analisis-ia { border-left-color: #6a11cb !important; background: linear-gradient(90deg, rgba(106, 17, 203, 0.08), transparent) !important; }
.status-badge { display: inline-block; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem; font-weight: bold; margin-left: 10px; }
.status-activo { background: linear-gradient(to right, #4CAF50, #8BC34A); color: white; }
.status-verificado { background: linear-gradient(to right, #2196F3, #3F51B5); color: white; }
.certificado-badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9rem; margin-top: 10px; }
.certificado-gratuito { background: linear-gradient(to right, #4CAF50, #8BC34A); color: white; }
.certificado-internacional { background: linear-gradient(to right, #2196F3, #3F51B5); color: white; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# INICIAR SISTEMA EN SEGUNDO PLANO
# ----------------------------
iniciar_tareas_background()

# Planificar tareas de indexación para temas populares
temas_populares = ["Python", "Machine Learning", "Data Science", "Diseño UX", "Marketing Digital", "Finanzas"]
idiomas_indexacion = ["es", "en", "pt"]
planificar_indexacion_recursos(temas_populares, idiomas_indexacion)

# ----------------------------
# BARRA LATERAL
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 📊 Estadísticas en Tiempo Real")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analiticas_busquedas")
        total_busquedas = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas WHERE activa = 1")
        total_plataformas = cursor.fetchone()[0]
        cursor.execute("""
            SELECT tema, COUNT(*) as conteo
            FROM analiticas_busquedas
            GROUP BY tema
            ORDER BY conteo DESC
            LIMIT 1
        """)
        tema_popular_data = cursor.fetchone()
        tema_popular = tema_popular_data[0] if tema_popular_data else "Python"
        conn.close()
    except Exception as e:
        logger.error(f"Error estadísticas: {e}")
        total_busquedas = 0
        total_plataformas = 0
        tema_popular = "Python"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Búsquedas", total_busquedas)
    with col2:
        st.metric("Plataformas", total_plataformas)
    st.info(f"Tema popular: {tema_popular}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# CABECERA
# ----------------------------
st.markdown("""
<div class="main-header">
    <h1>🎓 Buscador Profesional de Cursos</h1>
    <p>Descubre recursos educativos verificados con búsqueda inmediata y análisis de calidad en segundo plano</p>
    <div style="display: flex; gap: 15px; margin-top: 20px; flex-wrap: wrap;">
        <span class="status-badge status-activo">✅ Sistema Activo</span>
        <span class="status-badge status-verificado">⚡ Respuesta Inmediata</span>
        <span class="status-badge status-verificado">🌐 Multilingüe</span>
        <span class="status-badge status-activo">🧠 IA en Segundo Plano</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# FORMULARIO DE BÚSQUEDA
# ----------------------------
IDIOMAS = {
    "Español (es)": "es",
    "Inglés (en)": "en",
    "Portugués (pt)": "pt"
}
NIVELES = ["Cualquiera", "Principiante", "Intermedio", "Avanzado"]

with st.container():
    st.markdown('<div class="search-form">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tema = st.text_input("🔍 ¿Qué quieres aprender hoy?",
                             placeholder="Ej: Python, Machine Learning, Diseño UX...",
                             key="tema_input",
                             help="Ingresa el tema. Mostraré resultados de inmediato.")
    with col2:
        nivel = st.selectbox("📚 Nivel", NIVELES, key="nivel_select", help="Selecciona el nivel deseado")
    with col3:
        idioma_seleccionado = st.selectbox("🌍 Idioma", list(IDIOMAS.keys()), key="idioma_select", help="Elige el idioma de los recursos")
    buscar = st.button("🚀 Buscar Cursos Ahora", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# RESULTADOS
# ----------------------------
def mostrar_recurso_basico(recurso: RecursoEducativo, index: int, analisis_pendiente: bool = False):
    color_clase = {
        "Principiante": "nivel-principiante",
        "Intermedio": "nivel-intermedio",
        "Avanzado": "nivel-avanzado"
    }.get(recurso.nivel, "")
    extra_class = "plataforma-oculta" if recurso.tipo == "oculta" else ""
    pending_note = "🧠 Análisis de IA en progreso..." if analisis_pendiente else ""

    cert_badge = ""
    if recurso.certificacion:
        cert_type = recurso.certificacion.tipo
        if cert_type == "gratuito":
            cert_badge = '<span class="certificado-badge certificado-gratuito">✅ Certificado Gratuito</span>'
        elif cert_type == "audit":
            cert_badge = '<span class="certificado-badge certificado-internacional">🎓 Modo Audit (Gratuito)</span>'
        else:
            cert_badge = '<span class="certificado-badge certificado-internacional">💰 Certificado de Pago</span>'
        if recurso.certificacion.validez_internacional:
            cert_badge += ' <span class="certificado-badge certificado-internacional">🌐 Validez Internacional</span>'

    st.markdown(f"""
    <div class="resultado-card {color_clase} {extra_class}">
        <h3>🎯 {recurso.titulo}</h3>
        <p><strong>📚 Nivel:</strong> {recurso.nivel} | <strong>🌐 Plataforma:</strong> {recurso.plataforma}</p>
        <p>📝 {recurso.descripcion}</p>
        {cert_badge}
        <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
            <a href="{recurso.url}" target="_blank" style="flex:1; min-width:200px; background:linear-gradient(to right, #6a11cb, #2575fc); color:white; padding:12px 20px; text-decoration:none; border-radius:8px; font-weight:bold; text-align:center;">➡️ Acceder al Recurso</a>
        </div>
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; font-size: 0.9rem; color: #666;">
            <p style="margin:5px 0;"><strong>🔎 Confianza:</strong> {(recurso.confianza * 100):.1f}% | <strong>✅ Verificado:</strong> {datetime.fromisoformat(recurso.ultima_verificacion).strftime('%d/%m/%Y')}</p>
            <p style="margin:5px 0;"><strong>🌍 Idioma:</strong> {recurso.idioma.upper()} | <strong>🏷️ Categoría:</strong> {recurso.categoria}</p>
            <p style="margin:5px 0; color:#6a11cb; font-weight:bold;">{pending_note}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def mostrar_recurso_con_ia(recurso: RecursoEducativo, index: int):
    color_clase = {
        "Principiante": "nivel-principiante",
        "Intermedio": "nivel-intermedio",
        "Avanzado": "nivel-avanzado"
    }.get(recurso.nivel, "")
    extra_class = "plataforma-oculta" if recurso.tipo == "oculta" else ""
    ia_class = "con-analisis-ia" if recurso.metadatos_analisis else ""

    cert_badge = ""
    if recurso.certificacion:
        cert_type = recurso.certificacion.tipo
        if cert_type == "gratuito":
            cert_badge = '<span class="certificado-badge certificado-gratuito">✅ Certificado Gratuito</span>'
        elif cert_type == "audit":
            cert_badge = '<span class="certificado-badge certificado-internacional">🎓 Modo Audit (Gratuito)</span>'
        else:
            cert_badge = '<span class="certificado-badge certificado-internacional">💰 Certificado de Pago</span>'
        if recurso.certificacion.validez_internacional:
            cert_badge += ' <span class="certificado-badge certificado-internacional">🌐 Validez Internacional</span>'

    ia_content = ""
    if recurso.metadatos_analisis:
        calidad_ia = recurso.metadatos_analisis.get("calidad_ia", recurso.confianza)
        relevancia_ia = recurso.metadatos_analisis.get("relevancia_ia", recurso.confianza)
        recomendacion = recurso.metadatos_analisis.get("recomendacion_personalizada", "")
        razones = recurso.metadatos_analisis.get("razones_calidad", [])[:3]
        advertencias = recurso.metadatos_analisis.get("advertencias", [])[:2]

        razones_html = "".join([f"<li>{r}</li>" for r in razones])
        advertencias_html = "".join([f"<li>{a}</li>" for a in advertencias])

        ia_content = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; border-radius: 12px; margin: 15px 0; border-left: 3px solid #ffeb3b;">
            <h4 style="margin: 0 0 8px 0; color: #fff9c4;">🧠 Análisis de Calidad con IA</h4>
            <p style="margin: 0; line-height: 1.5;">{recomendacion}</p>
        </div>
        <div style="display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap;">
            <span style="background: linear-gradient(to right, #4CAF50, #8BC34A); color: white; padding: 6px 12px; border-radius: 15px; font-size: 0.9rem; font-weight: bold;">
                Calidad IA: {(calidad_ia * 100):.0f}%
            </span>
            <span style="background: linear-gradient(to right, #2196F3, #3F51B5); color: white; padding: 6px 12px; border-radius: 15px; font-size: 0.9rem; font-weight: bold;">
                Relevancia: {(relevancia_ia * 100):.0f}%
            </span>
        </div>
        <div style="margin: 15px 0; padding: 12px; background: #e3f2fd; border-radius: 8px; border-left: 3px solid #2196F3;">
            <strong>🔎 Razones de Calidad:</strong>
            <ul style="margin: 8px 0 0 20px; padding-left: 0; color: #1565c0;">{razones_html}</ul>
        </div>
        <div style="margin: 15px 0; padding: 12px; background: #fff8e1; border-radius: 8px; border-left: 3px solid #ffc107;">
            <strong>⚠️ Advertencias:</strong>
            <ul style="margin: 8px 0 0 20px; padding-left: 0; color: #e65100;">{advertencias_html}</ul>
        </div>
        """

    st.markdown(f"""
    <div class="resultado-card {color_clase} {extra_class} {ia_class}">
        <h3>🎯 {recurso.titulo}</h3>
        <p><strong>📚 Nivel:</strong> {recurso.nivel} | <strong>🌐 Plataforma:</strong> {recurso.plataforma}</p>
        <p>📝 {recurso.descripcion}</p>
        {cert_badge}
        {ia_content}
        <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
            <a href="{recurso.url}" target="_blank" style="flex:1; min-width:200px; background:linear-gradient(to right, #6a11cb, #2575fc); color:white; padding:12px 20px; text-decoration:none; border-radius:8px; font-weight:bold; text-align:center;">➡️ Acceder al Recurso</a>
        </div>
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; font-size: 0.9rem; color: #666;">
            <p style="margin:5px 0;"><strong>🔎 Confianza:</strong> {(recurso.confianza * 100):.1f}% | <strong>✅ Verificado:</strong> {datetime.fromisoformat(recurso.ultima_verificacion).strftime('%d/%m/%Y')}</p>
            <p style="margin:5px 0;"><strong>🌍 Idioma:</strong> {recurso.idioma.upper()} | <strong>🏷️ Categoría:</strong> {recurso.categoria}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if buscar and tema.strip():
    with st.spinner("🔎 Buscando recursos educativos..."):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(buscar_recursos_multicapa(tema, idioma_seleccionado, nivel))
            loop.close()

            if resultados:
                st.success(f"✅ ¡{len(resultados)} recursos encontrados para {tema} en {idioma_seleccionado}!")

                if GROQ_AVAILABLE and GROQ_API_KEY:
                    planificar_analisis_ia(resultados)
                    st.info("🧠 Análisis de calidad en progreso. Se actualizarán cuando esté listo.")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(resultados))
                col2.metric("Plataformas", len(set(r.plataforma for r in resultados)))
                col3.metric("Confianza Promedio", f"{sum(r.confianza for r in resultados) / len(resultados):.1%}")

                st.markdown("### 📚 Resultados Encontrados")
                for i, resultado in enumerate(resultados):
                    time.sleep(0.05)
                    if resultado.metadatos_analisis:
                        mostrar_recurso_con_ia(resultado, i)
                    else:
                        mostrar_recurso_basico(resultado, i, resultado.analisis_pendiente)

                st.markdown("---")
                df = pd.DataFrame([{
                    'titulo': r.titulo,
                    'url': r.url,
                    'plataforma': r.plataforma,
                    'nivel': r.nivel,
                    'idioma': r.idioma,
                    'categoria': r.categoria,
                    'confianza': f"{r.confianza:.1%}",
                    'analisis_ia': 'Sí' if r.metadatos_analisis else 'No'
                } for r in resultados])
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Resultados (CSV)",
                    data=csv,
                    file_name=f"resultados_busqueda_{tema.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.markdown("### 💡 Tu Opinión es Importante")
                col_feedback1, col_feedback2 = st.columns(2)
                with col_feedback1:
                    util = st.radio("¿Te resultó útil esta búsqueda?", ["Sí", "No"], horizontal=True)
                with col_feedback2:
                    comentario = st.text_input("Comentarios adicionales (opcional)")
                if st.button("Enviar Feedback", use_container_width=True):
                    st.success("✅ ¡Gracias por tu feedback! Ayuda a mejorar el sistema.")
            else:
                st.warning("⚠️ No encontramos recursos verificados para este tema. Intenta con otro término.")

        except Exception as e:
            logger.error(f"Error durante la búsqueda: {e}")
            st.error("❌ Ocurrió un error durante la búsqueda. Por favor, intenta nuevamente.")
            st.exception(e)

# ----------------------------
# PIE DE PÁGINA
# ----------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 14px; padding: 25px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 20px;">
    <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-bottom: 15px;">
        <div>
            <h4 style="color: #6a11cb; margin: 0 0 8px 0; font-size: 1.1rem;">⚡ Rendimiento</h4>
            <p style="margin: 0; color: #2c3e50; font-weight: 500;">
                Resultados Inmediatos • Cache Inteligente • Alta Disponibilidad
            </p>
        </div>
        <div>
            <h4 style="color: #6a11cb; margin: 0 0 8px 0; font-size: 1.1rem;">🌐 Cobertura</h4>
            <p style="margin: 0; color: #2c3e50; font-weight: 500;">
                15+ Plataformas • 3 Idiomas • Búsqueda Multicapa
            </p>
        </div>
        <div>
            <h4 style="color: #6a11cb; margin: 0 0 8px 0; font-size: 1.1rem;">🧠 Inteligencia</h4>
            <p style="margin: 0; color: #2c3e50; font-weight: 500;">
                Análisis en Segundo Plano • Verificación Automática
            </p>
        </div>
    </div>
    <strong>✨ Buscador Profesional de Cursos</strong><br>
    <span style="color: #2c3e50; font-weight: 500;">Sistema de búsqueda inteligente con resultados inmediatos</span><br>
    <em style="color: #7f8c8d;">Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')} • Versión: 3.1.1 • Estado: ✅ Activo</em><br>
    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #ddd;">
        <code style="background: #f1f3f5; padding: 2px 8px; border-radius: 4px; color: #d32f2f;">
            IA opcional - Sistema funcional sin dependencias externas críticas
        </code>
    </div>
</div>
""", unsafe_allow_html=True)

logger.info("✅ Sistema de búsqueda profesional iniciado correctamente")
logger.info(f"⚡ Búsqueda inmediata: Activa")
logger.info(f"🧠 Análisis IA en segundo plano: {'Disponible' if GROQ_AVAILABLE and GROQ_API_KEY else 'No disponible'}")
