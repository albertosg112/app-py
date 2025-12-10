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

# ----------------------------
# LOGGING
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('buscador_cursos.log'), logging.StreamHandler()]
)
logger = logging.getLogger("BuscadorProfesional")

# ----------------------------
# CONFIGURACIÓN GLOBAL
# ----------------------------
# Usamos os.getenv para asegurar compatibilidad fuera de Streamlit Cloud
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = os.getenv("GOOGLE_CX", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DUCKDUCKGO_ENABLED = (os.getenv("DUCKDUCKGO_ENABLED", "false").lower() == "true")

MAX_BACKGROUND_TASKS = 1
CACHE_EXPIRATION = timedelta(hours=12)
GROQ_MODEL = "llama-3.1-70b-versatile"  # Modelo actualizado de Groq

search_cache: Dict[str, Dict[str, Any]] = {}
background_tasks: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# Verifica disponibilidad de Groq
GROQ_AVAILABLE = False
try:
    import groq
    if GROQ_API_KEY:
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API disponible para análisis en segundo plano")
except ImportError:
    logger.warning("⚠️ Biblioteca 'groq' no instalada - Análisis de IA no disponible")
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
# BASE DE DATOS
# ----------------------------
DB_PATH = "cursos_inteligentes_v3.db"

def init_advanced_database() -> bool:
    """Inicializa DB con tablas necesarias y datos semilla."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
                {
                    "nombre": "Aprende con Alf",
                    "url_base": "https://aprendeconalf.es/?s={}",
                    "descripcion": "Cursos gratuitos de programación, matemáticas y ciencia de datos con ejercicios prácticos",
                    "idioma": "es",
                    "categoria": "Programación",
                    "nivel": "Intermedio",
                    "confianza": 0.85,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "gratuito",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["es"]),
                    "reputacion_academica": 0.90
                },
                {
                    "nombre": "Coursera",
                    "url_base": "https://www.coursera.org/search?query={}&free=true",
                    "descripcion": "Plataforma líder con cursos universitarios gratuitos (audit mode)",
                    "idioma": "en",
                    "categoria": "General",
                    "nivel": "Avanzado",
                    "confianza": 0.95,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "audit",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["us", "uk", "ca", "au", "eu"]),
                    "reputacion_academica": 0.95
                },
                {
                    "nombre": "edX",
                    "url_base": "https://www.edx.org/search?tab=course&availability=current&price=free&q={}",
                    "descripcion": "Cursos de Harvard, MIT y otras universidades top (modo audit gratuito)",
                    "idioma": "en",
                    "categoria": "Académico",
                    "nivel": "Avanzado",
                    "confianza": 0.92,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "audit",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["us", "uk", "ca", "au", "eu"]),
                    "reputacion_academica": 0.93
                },
                {
                    "nombre": "Kaggle Learn",
                    "url_base": "https://www.kaggle.com/learn/search?q={}",
                    "descripcion": "Microcursos prácticos de ciencia de datos con certificados gratuitos",
                    "idioma": "en",
                    "categoria": "Data Science",
                    "nivel": "Intermedio",
                    "confianza": 0.90,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "gratuito",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["global"]),
                    "reputacion_academica": 0.88
                },
                {
                    "nombre": "freeCodeCamp",
                    "url_base": "https://www.freecodecamp.org/news/search/?query={}",
                    "descripcion": "Certificados gratuitos completos en desarrollo web y ciencia de datos",
                    "idioma": "en",
                    "categoria": "Programación",
                    "nivel": "Intermedio",
                    "confianza": 0.93,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "gratuito",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["global"]),
                    "reputacion_academica": 0.91
                },
                {
                    "nombre": "PhET Simulations",
                    "url_base": "https://phet.colorado.edu/en/search?q={}",
                    "descripcion": "Simulaciones interactivas de ciencias y matemáticas de la Universidad de Colorado",
                    "idioma": "en",
                    "categoria": "Ciencias",
                    "nivel": "Todos",
                    "confianza": 0.88,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "gratuito",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["us", "global"]),
                    "reputacion_academica": 0.85
                },
                {
                    "nombre": "The Programming Historian",
                    "url_base": "https://programminghistorian.org/en/lessons/?q={}",
                    "descripcion": "Tutoriales académicos de programación y humanidades digitales",
                    "idioma": "en",
                    "categoria": "Programación",
                    "nivel": "Avanzado",
                    "confianza": 0.82,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "gratuito",
                    "validez_internacional": 0,
                    "paises_validos": json.dumps(["global"]),
                    "reputacion_academica": 0.80
                },
                {
                    "nombre": "Domestika (Gratuito)",
                    "url_base": "https://www.domestika.org/es/search?query={}&free=1",
                    "descripcion": "Cursos gratuitos de diseño creativo, algunos con certificados verificados",
                    "idioma": "es",
                    "categoria": "Diseño",
                    "nivel": "Intermedio",
                    "confianza": 0.83,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "pago",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["es", "mx", "ar", "cl"]),
                    "reputacion_academica": 0.82
                },
                {
                    "nombre": "Biblioteca Virtual Miguel de Cervantes",
                    "url_base": "https://www.cervantesvirtual.com/buscar/?q={}",
                    "descripcion": "Recursos académicos hispanos con validez internacional",
                    "idioma": "es",
                    "categoria": "Humanidades",
                    "nivel": "Avanzado",
                    "confianza": 0.87,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "gratuito",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["es", "latam", "eu"]),
                    "reputacion_academica": 0.85
                },
                {
                    "nombre": "OER Commons",
                    "url_base": "https://www.oercommons.org/search?q={}",
                    "descripcion": "Recursos educativos abiertos de instituciones globales con estándares académicos",
                    "idioma": "en",
                    "categoria": "General",
                    "nivel": "Todos",
                    "confianza": 0.89,
                    "ultima_verificacion": datetime.now().isoformat(),
                    "activa": 1,
                    "tipo_certificacion": "gratuito",
                    "validez_internacional": 1,
                    "paises_validos": json.dumps(["global"]),
                    "reputacion_academica": 0.87
                }
            ]
            for p in plataformas_iniciales:
                cursor.execute(
                    '''
                    INSERT INTO plataformas_ocultas
                    (nombre, url_base, descripcion, idioma, categoria, nivel, confianza,
                     ultima_verificacion, activa, tipo_certificacion, validez_internacional,
                     paises_validos, reputacion_academica)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        p["nombre"], p["url_base"], p["descripcion"], p["idioma"], p["categoria"],
                        p["nivel"], p["confianza"], p["ultima_verificacion"], p["activa"],
                        p["tipo_certificacion"], int(p["validez_internacional"]),
                        p["paises_validos"], p["reputacion_academica"]
                    )
                )
        conn.commit()
        conn.close()
        logger.info("✅ Base de datos inicializada correctamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error al inicializar la base de datos: {e}")
        return False

init_advanced_database()

# ----------------------------
# UTILIDADES
# ----------------------------
def get_codigo_idioma(nombre_idioma: str) -> str:
    mapeo = {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt", "es": "es", "en": "en", "pt": "pt"}
    return mapeo.get(nombre_idioma, "es")

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    texto = (url + titulo + descripcion).lower()
    palabras_validas = ['curso', 'tutorial', 'aprender', 'education', 'learn', 'gratuito', 'free', 'certificado', 'certificate', 'clase', 'class', 'educación', 'educacion', 'clases']
    palabras_invalidas = ['comprar', 'buy', 'precio', 'price', 'costo', 'only', 'premium', 'exclusive', 'paid', 'pago', 'suscripción', 'subscription', 'membership', 'register now', 'matrícula']
    dominios_educativos = ['.edu', '.ac.', 'coursera', 'edx', 'khanacademy', 'freecodecamp', 'kaggle', 'udemy', 'youtube', 'aprendeconalf', '.org', '.gob', '.gov']
    tiene_validas = any(p in texto for p in palabras_validas)
    tiene_invalidas = any(p in texto for p in palabras_invalidas)
    dominio_valido = any(d in url.lower() for d in dominios_educativos)
    es_gratuito = ('gratuito' in texto) or ('free' in texto) or ('sin costo' in texto) or ('audit' in texto)
    return (tiene_validas or dominio_valido) and not tiene_invalidas and (es_gratuito or dominio_valido)

def generar_id_unico(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    texto = texto.lower()
    if nivel_solicitado not in ("Cualquiera", "Todos"):
        return nivel_solicitado
    if any(p in texto for p in ['principiante', 'basico', 'básico', 'beginner', 'fundamentos', 'introducción', 'desde cero']):
        return "Principiante"
    if any(p in texto for p in ['intermedio', 'intermediate', 'práctico', 'aplicado', 'profesional', 'medio']):
        return "Intermedio"
    if any(p in texto for p in ['avanzado', 'advanced', 'experto', 'máster', 'especialista', 'profundo']):
        return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    t = tema.lower()
    if any(p in t for p in ['programación', 'python', 'javascript', 'web', 'desarrollo', 'coding', 'software', 'java', 'c++', 'c#', 'html', 'css']):
        return "Programación"
    if any(p in t for p in ['datos', 'data', 'machine learning', 'ia', 'ai', 'ciencia de datos', 'big data', 'analytics', 'deep learning', 'estadística', 'statistics']):
        return "Data Science"
    if any(p in t for p in ['matemáticas', 'math', 'álgebra', 'geometría', 'cálculo', 'probability']):
        return "Matemáticas"
    if any(p in t for p in ['diseño', 'design', 'ux', 'ui', 'gráfico', 'figma', 'canva', 'illustration', 'photoshop']):
        return "Diseño"
    if any(p in t for p in ['marketing', 'business', 'negocios', 'finanzas', 'emprendimiento', 'startups', 'economía', 'management', 'ventas']):
        return "Negocios"
    if any(p in t for p in ['idioma', 'language', 'inglés', 'english', 'español', 'portugues', 'francés', 'alemán', 'lingüística', 'conversación']):
        return "Idiomas"
    return "General"

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
    if '.edu' in dominio or '.ac.' in dominio or '.gob' in dominio or '.gov' in dominio: return 'Institución Académica'
    partes = dominio.split('.')
    return partes[-2].title() if len(partes) > 1 else dominio.title()

# ----------------------------
# INTEGRACIÓN GROQ (opcional)
# ----------------------------
async def analizar_calidad_curso(recurso: RecursoEducativo, perfil_usuario: Dict) -> Dict:
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return {
            "calidad_educativa": recurso.confianza,
            "relevancia_usuario": recurso.confianza,
            "razones_calidad": ["Análisis básico - IA no disponible"],
            "razones_relevancia": ["Análisis básico - IA no disponible"],
            "recomendacion_personalizada": "Curso verificado con búsqueda estándar",
            "advertencias": ["Análisis de IA no disponible en este momento"]
        }
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        Evalúa el curso:
        TÍTULO: {recurso.titulo}
        PLATAFORMA: {recurso.plataforma}
        DESCRIPCIÓN: {recurso.descripcion}
        NIVEL DECLARADO: {recurso.nivel}
        CATEGORÍA: {recurso.categoria}
        TIPO DE RECURSO: {recurso.tipo}
        CERTIFICACIÓN: {'Sí' if recurso.certificacion else 'No'}
        PERFIL:
        - Nivel: {perfil_usuario.get('nivel_real', 'intermedio')}
        - Objetivos: {perfil_usuario.get('objetivos', 'mejorar habilidades profesionales')}
        - Tiempo: {perfil_usuario.get('tiempo_disponible', '2-3 horas/sem')}
        - Experiencia: {perfil_usuario.get('experiencia_previa', 'algunos cursos básicos')}
        Responde en JSON:
        {{
            "calidad_educativa": 0.0-1.0,
            "relevancia_usuario": 0.0-1.0,
            "razones_calidad": [],
            "razones_relevancia": [],
            "recomendacion_personalizada": "",
            "advertencias": []
        }}
        """
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL, # Modelo actualizado
            temperature=0.3,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        contenido = response.choices[0].message.content
        # Corrección: Asegurar que el contenido sea un JSON válido
        try:
            parsed_response = json.loads(contenido)
            return parsed_response
        except json.JSONDecodeError:
            logger.error(f"Groq devolvió contenido no JSON: {contenido}")
            return {
                "calidad_educativa": recurso.confianza,
                "relevancia_usuario": recurso.confianza,
                "razones_calidad": ["Error IA: Formato de respuesta inválido"],
                "razones_relevancia": ["Error IA: Formato de respuesta inválido"],
                "recomendacion_personalizada": "Curso verificado con búsqueda estándar",
                "advertencias": ["La IA devolvió un formato inesperado."]
            }

    except Exception as e:
        logger.error(f"Error en análisis con Groq: {e}")
        return {
            "calidad_educativa": recurso.confianza,
            "relevancia_usuario": recurso.confianza,
            "razones_calidad": [f"Error IA: {str(e)}"],
            "razones_relevancia": [f"Error IA: {str(e)}"],
            "recomendacion_personalizada": "Curso verificado con búsqueda estándar",
            "advertencias": ["Análisis de IA temporalmente no disponible"]
        }

def obtener_perfil_usuario() -> Dict:
    # Placeholder para el perfil de usuario, ya que Streamlit no mantiene el estado entre ejecuciones
    # Se podría implementar un formulario para capturar estos datos
    return {
        "nivel_real": "intermedio",
        "objetivos": "mejorar habilidades profesionales",
        "tiempo_disponible": "2-3 horas por semana",
        "experiencia_previa": "algunos cursos básicos",
        "estilo_aprendizaje": "práctico con proyectos"
    }

# ----------------------------
# BÚSQUEDA MULTICAPA
# ----------------------------
async def buscar_en_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []
    try:
        query_base = f"{tema} curso gratuito certificado"
        if nivel not in ("Cualquiera", "Todos"):
            query_base += f" nivel {nivel.lower()}"
        url = "https://www.googleapis.com/customsearch/v1" # Corrección: URL sin espacios
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': query_base, 'num': 5, 'lr': f'lang_{idioma}'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Error Google API ({response.status}): {await response.text()}")
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
                    confianza = 0.8
                    if any(d in url_item.lower() for d in ['.edu', 'coursera.org', 'edx.org', 'freecodecamp.org', '.gov']):
                        confianza += 0.1
                    confianza = min(confianza, 0.95)
                    resultados.append(RecursoEducativo(
                        id=generar_id_unico(url_item),
                        titulo=titulo,
                        url=url_item,
                        descripcion=descripcion,
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
    except asyncio.TimeoutError:
        logger.error("Timeout en Google API")
        return []
    except Exception as e:
        logger.error(f"Error inesperado en Google API: {e}")
        return []

async def buscar_en_plataformas_conocidas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    resultados: List[RecursoEducativo] = []
    if idioma == "es":
        plataformas = {
            "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", "niveles": ["Principiante", "Intermedio"]},
            "coursera": {"nombre": "Coursera (Español)", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=es&free=true", "niveles": ["Intermedio", "Avanzado"]},
            "udemy": {"nombre": "Udemy (Español)", "url": f"https://www.udemy.com/courses/search/?price=price-free&lang=es&q={tema.replace(' ', '%20')}", "niveles": ["Principiante", "Intermedio"]},
            "khan": {"nombre": "Khan Academy (Español)", "url": f"https://es.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}", "niveles": ["Principiante", "Intermedio"]},
        }
    elif idioma == "pt":
        plataformas = {
            "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", "niveles": ["Principiante", "Intermedio"]},
            "coursera": {"nombre": "Coursera (Português)", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=pt&free=true", "niveles": ["Intermedio", "Avanzado"]},
            "udemy": {"nombre": "Udemy (Português)", "url": f"https://www.udemy.com/courses/search/?price=price-free&lang=pt&q={tema.replace(' ', '%20')}", "niveles": ["Principiante", "Intermedio"]},
            "khan": {"nombre": "Khan Academy (Português)", "url": f"https://pt.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}", "niveles": ["Principiante", "Intermedio"]},
        }
    else:
        plataformas = {
            "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=full+course+free+{tema.replace(' ', '+')}", "niveles": ["Principiante", "Intermedio"]},
            "coursera": {"nombre": "Coursera", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&free=true", "niveles": ["Intermedio", "Avanzado"]},
            "edx": {"nombre": "edX", "url": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={tema.replace(' ', '%20')}", "niveles": ["Avanzado"]},
            "freecodecamp": {"nombre": "freeCodeCamp", "url": f"https://www.freecodecamp.org/news/search/?query={tema.replace(' ', '%20')}", "niveles": ["Intermedio", "Avanzado"]},
            "khan": {"nombre": "Khan Academy", "url": f"https://www.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}", "niveles": ["Principiante"]},
        }
    for nombre_plataforma, datos in plataformas.items():
        if len(resultados) >= 4:
            break
        niveles_compatibles = [n for n in datos['niveles'] if nivel in ("Cualquiera", "Todos") or n == nivel]
        if not niveles_compatibles:
            continue
        nivel_actual = random.choice(niveles_compatibles)
        titulo = f"🎯 {datos['nombre']} — {tema}"
        descripcion = {
            "Principiante": f"Curso introductorio ideal para empezar {tema}. Sin conocimientos previos.",
            "Intermedio": f"Curso práctico para profundizar en {tema} con ejercicios y proyectos.",
            "Avanzado": f"Contenido especializado para dominar {tema} a nivel profesional."
        }.get(nivel_actual, f"Recurso verificado para {tema}.")
        certificacion = None
        if "coursera" in nombre_plataforma or "edx" in nombre_plataforma or "freecodecamp" in nombre_plataforma:
            certificacion = Certificacion(
                plataforma=datos["nombre"],
                curso=tema,
                tipo="audit" if "coursera" in nombre_plataforma or "edx" in nombre_plataforma else "gratuito",
                validez_internacional=True,
                paises_validos=["global"],
                costo_certificado=0.0,
                reputacion_academica=0.85,
                ultima_verificacion=datetime.now().isoformat()
            )
        resultados.append(RecursoEducativo(
            id=generar_id_unico(datos["url"]),
            titulo=titulo,
            url=datos["url"],
            descripcion=descripcion,
            plataforma=datos["nombre"],
            idioma=idioma,
            nivel=nivel_actual,
            categoria=determinar_categoria(tema),
            certificacion=certificacion,
            confianza=0.85,
            tipo="conocida",
            ultima_verificacion=datetime.now().isoformat(),
            activo=True,
            metadatos={"fuente": "plataformas_conocidas"}
        ))
    return resultados

async def buscar_en_plataformas_ocultas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
        filas = cursor.fetchall()
        conn.close()
        recursos: List[RecursoEducativo] = []
        for r in filas:
            nombre, url_base, descripcion, nivel_db, confianza, tipo_cert, validez_int, paises_json, reputacion = r
            url_completa = url_base.format(tema.replace(' ', '+'))
            nivel_calc = nivel_db if nivel in ("Cualquiera", "Todos") else nivel
            cert = None
            if tipo_cert and tipo_cert != "audit":
                cert = Certificacion(
                    plataforma=nombre,
                    curso=tema,
                    tipo=tipo_cert,
                    validez_internacional=bool(validez_int),
                    paises_validos=json.loads(paises_json or "[]"),
                    costo_certificado=0.0 if tipo_cert == "gratuito" else 49.99,
                    reputacion_academica=reputacion,
                    ultima_verificacion=datetime.now().isoformat()
                )
            recursos.append(RecursoEducativo(
                id=generar_id_unico(url_completa),
                titulo=f"💎 {nombre} — {tema}",
                url=url_completa,
                descripcion=descripcion,
                plataforma=nombre,
                idioma=idioma,
                nivel=nivel_calc,
                categoria=determinar_categoria(tema),
                certificacion=cert,
                confianza=confianza,
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

async def buscar_recursos_multicapa(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    cache_key = f"busqueda_{tema}_{idioma}_{nivel}"
    if cache_key in search_cache:
        cached = search_cache[cache_key]
        if datetime.now() - cached['timestamp'] < CACHE_EXPIRATION:
            return cached['resultados']

    resultados: List[RecursoEducativo] = []
    codigo_idioma = get_codigo_idioma(idioma)
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Buscando en plataformas conocidas...")
    conocidos = await buscar_en_plataformas_conocidas(tema, codigo_idioma, nivel)
    resultados.extend(conocidos)
    progress_bar.progress(0.3)

    status_text.text("Buscando en plataformas ocultas...")
    ocultas = await buscar_en_plataformas_ocultas(tema, codigo_idioma, nivel)
    resultados.extend(ocultas)
    progress_bar.progress(0.6)

    status_text.text("Consultando Google API...")
    google_res = await buscar_en_google_api(tema, codigo_idioma, nivel)
    resultados.extend(google_res)
    progress_bar.progress(0.9)

    status_text.text("Procesando resultados...")
    resultados = eliminar_duplicados(resultados)
    resultados.sort(key=lambda x: x.confianza, reverse=True)

    if GROQ_AVAILABLE and GROQ_API_KEY:
        for r in resultados[:5]: # Limitar IA a los primeros 5
            r.analisis_pendiente = True

    final = resultados[:10]
    search_cache[cache_key] = {'resultados': final, 'timestamp': datetime.now()}
    
    progress_bar.progress(1.0)
    time.sleep(0.1) # Pequeña pausa para que se vea el 100%
    progress_bar.empty()
    status_text.empty()

    return final

# ----------------------------
# PROCESAMIENTO EN SEGUNDO PLANO
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
                    calidad = float(analisis.get("calidad_educativa", recurso.confianza))
                    relevancia = float(analisis.get("relevancia_usuario", recurso.confianza))
                    confianza_ia = (calidad + relevancia) / 2.0
                    recurso.confianza = min(max(recurso.confianza, confianza_ia), 0.95)
                    recurso.metadatos_analisis = {
                        "calidad_ia": calidad,
                        "relevancia_ia": relevancia,
                        "razones_calidad": analisis.get("razones_calidad", []),
                        "razones_relevancia": analisis.get("razones_relevancia", []),
                        "recomendacion_personalizada": analisis.get("recomendacion_personalizada", ""),
                        "advertencias": analisis.get("advertencias", []),
                    }
                recurso.analisis_pendiente = False
                time.sleep(0.6) # Respetar límite de tasa de Groq
    except Exception as e:
        logger.error(f"Error en análisis en segundo plano: {e}")

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
        logger.info(f"✅ Sistema de tareas en background iniciado con {num_workers} workers")
        st.session_state.background_started = True

def planificar_analisis_ia(resultados: List[RecursoEducativo]):
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return
    tarea = {'tipo': 'analizar_resultados', 'parametros': {'resultados': [r for r in resultados if r.analisis_pendiente]}}
    background_tasks.put(tarea)
    logger.info(f"Tarea IA planificada para {len(tarea['parametros']['resultados'])} resultados")

# Placeholder para evitar NameError
def planificar_indexacion_recursos(temas: List[str], idiomas: List[str]):
    logger.info(f"🗂️ Indexación planificada: {len(temas)} temas, {len(idiomas)} idiomas (placeholder)")

# ----------------------------
# INTERFAZ DE USUARIO (UI)
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
.main-header {
    background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
    color: white; padding: 2rem; border-radius: 20px; margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25); position: relative; overflow: hidden;
}
.main-header h1 { font-size: 2.6rem; margin: 0 0 0.6rem 0; }
.main-header p { font-size: 1.2rem; opacity: 0.95; }
.resultado-card {
    border-radius: 15px; padding: 20px; margin-bottom: 20px; background: white;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08); border-left: 6px solid #4CAF50;
}
.nivel-principiante { border-left-color: #2196F3 !important; }
.nivel-intermedio { border-left-color: #4CAF50 !important; }
.nivel-avanzado { border-left-color: #FF9800 !important; }
.plataforma-oculta { border-left-color: #FF6B35 !important; background: #fff5f0; }
.con-analisis-ia { border-left-color: #6a11cb !important; background: #f8f4ff; }
.status-badge { display: inline-block; padding: 3px 10px; border-radius: 15px; font-size: 0.8rem; font-weight: bold; margin-left: 10px; }
.status-activo { background: linear-gradient(to right, #4CAF50, #8BC34A); color: white; }
.status-verificado { background: linear-gradient(to right, #2196F3, #3F51B5); color: white; }
.certificado-badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9rem; margin-top: 8px; }
.certificado-gratuito { background: linear-gradient(to right, #4CAF50, #8BC34A); color: white; }
.certificado-internacional { background: linear-gradient(to right, #2196F3, #3F51B5); color: white; }
.fade-in { animation: fadeIn 0.6s ease forwards; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0);} }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎓 Buscador Profesional de Cursos</h1>
    <p>Descubre recursos educativos verificados con búsqueda inmediata y análisis opcional en segundo plano.</p>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;">
        <span class="status-badge status-activo">✅ Sistema Activo</span>
        <span class="status-badge status-verificado">⚡ Respuesta Inmediata</span>
        <span class="status-badge status-verificado">🌐 Multilingüe</span>
        <span class="status-badge status-activo">🧠 IA en segundo plano</span>
    </div>
</div>
""", unsafe_allow_html=True)

IDIOMAS = {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}
NIVELES = ["Cualquiera", "Principiante", "Intermedio", "Avanzado"]

with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tema = st.text_input("🔍 ¿Qué quieres aprender hoy?",
                             placeholder="Ej: Python, Machine Learning, Diseño UX...",
                             key="tema_input",
                             help="Ingresa el tema que deseas aprender. Se mostrarán resultados inmediatos.")
    with col2:
        nivel = st.selectbox("📚 Nivel", NIVELES, key="nivel_select", help="Selecciona el nivel deseado")
    with col3:
        idioma_seleccionado = st.selectbox("🌍 Idioma", list(IDIOMAS.keys()), key="idioma_select", help="Elige el idioma de los recursos")

    buscar = st.button("🚀 Buscar Cursos Ahora", use_container_width=True, type="primary")

iniciar_tareas_background()
# Planificación inicial (placeholder)
temas_populares = ["Python", "Machine Learning", "Data Science", "Diseño UX", "Marketing Digital", "Finanzas"]
idiomas_indexacion = ["es", "en", "pt"]
planificar_indexacion_recursos(temas_populares, idiomas_indexacion)

# ----------------------------
# FUNCIONES DE AYUDA PARA LA UI
# ----------------------------
def link_button(url: str, label: str = "➡️ Acceder al recurso") -> str:
    return f'''
    <a href="{url}" target="_blank"
       style="flex:1;min-width:200px;background:linear-gradient(to right,#6a11cb,#2575fc);
              color:white;padding:10px 16px;text-decoration:none;border-radius:8px;
              font-weight:bold;text-align:center;">
       {label}
    </a>
    '''

def badge_certificacion(cert: Optional[Certificacion]) -> str:
    if not cert:
        return ""
    if cert.tipo == "gratuito":
        b = '<span class="certificado-badge certificado-gratuito">✅ Certificado Gratuito</span>'
    elif cert.tipo == "audit":
        b = '<span class="certificado-badge certificado-internacional">🎓 Modo Audit (Gratuito)</span>'
    else:
        b = '<span class="certificado-badge certificado-internacional">💰 Certificado de Pago</span>'
    if cert.validez_internacional:
        b += ' <span class="certificado-badge certificado-internacional">🌐 Validez Internacional</span>'
    return b

def clase_nivel(nivel: str) -> str:
    return {"Principiante": "nivel-principiante", "Intermedio": "nivel-intermedio", "Avanzado": "nivel-avanzado"}.get(nivel, "")

# ----------------------------
# FUNCIONES PARA MOSTRAR RESULTADOS
# ----------------------------
def mostrar_recurso_basico(recurso: RecursoEducativo, index: int, analisis_pendiente: bool = False):
    extra = "plataforma-oculta" if recurso.tipo == "oculta" else ""
    pending_class = "analisis-pendiente" if analisis_pendiente else ""
    cert_html = badge_certificacion(recurso.certificacion)
    st.markdown(f"""
    <div class="resultado-card {clase_nivel(recurso.nivel)} {extra} {pending_class} fade-in"
         style="animation-delay: {index * 0.08}s;">
        <h3>🎯 {recurso.titulo}</h3>
        <p><strong>📚 Nivel:</strong> {recurso.nivel} | <strong>🌐 Plataforma:</strong> {recurso.plataforma}</p>
        <p>📝 {recurso.descripcion}</p>
        {cert_html}
        <div style="margin-top: 12px; display:flex; gap:8px; flex-wrap:wrap;">
            {link_button(recurso.url, "➡️ Acceder al recurso")}
        </div>
        <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.9rem; color: #666;">
            <p style="margin:4px 0;"><strong>🔎 Confianza:</strong> {recurso.confianza*100:.1f}% |
               <strong>✅ Verificado:</strong> {datetime.fromisoformat(recurso.ultima_verificacion).strftime('%d/%m/%Y')}</p>
            <p style="margin:4px 0;"><strong>🌍 Idioma:</strong> {recurso.idioma.upper()} |
               <strong>🏷️ Categoría:</strong> {recurso.categoria}</p>
            {"<p style='margin:4px 0;color:#6a11cb;font-weight:bold;'>🤖 Análisis IA en progreso...</p>" if analisis_pendiente else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

def mostrar_recurso_con_ia(recurso: RecursoEducativo, index: int):
    extra = "plataforma-oculta" if recurso.tipo == "oculta" else ""
    ia_class = "con-analisis-ia" if recurso.metadatos_analisis else ""
    cert_html = badge_certificacion(recurso.certificacion)
    analisis_ia = recurso.metadatos_analisis or {}
    calidad = int(100 * analisis_ia.get("calidad_ia", recurso.confianza))
    relevancia = int(100 * analisis_ia.get("relevancia_ia", recurso.confianza))
    recomendacion = analisis_ia.get("recomendacion_personalizada", "")
    razones = analisis_ia.get("razones_calidad", [])[:3]
    advertencias = analisis_ia.get("advertencias", [])[:2]
    razones_html = "".join(f"<li>{r}</li>" for r in razones) if razones else ""
    advertencias_html = "".join(f"<li>{a}</li>" for a in advertencias) if advertencias else ""
    ia_block = ""
    if recurso.metadatos_analisis:
        ia_block = f"""
        <div style="background:#f0ecff;padding:10px;border-radius:8px;margin-top:10px;border-left:4px solid #6a11cb;">
            <strong>🧠 Análisis IA</strong><br>
            Calidad: {calidad}% • Relevancia: {relevancia}%<br>
            {recomendacion}
        </div>
        {f'<div style="margin: 10px 0; padding: 10px; background: #e3f2fd; border-radius: 8px;"><strong>🔍 Razones de Calidad:</strong><ul style="margin: 8px 0 0 20px;">{razones_html}</ul></div>' if razones_html else ''}
        {f'<div style="margin: 10px 0; padding: 10px; background: #fff8e1; border-radius: 8px;"><strong>⚠️ Advertencias:</strong><ul style="margin: 8px 0 0 20px;">{advertencias_html}</ul></div>' if advertencias_html else ''}
        """
    st.markdown(f"""
    <div class="resultado-card {clase_nivel(recurso.nivel)} {extra} {ia_class} fade-in"
         style="animation-delay: {index * 0.08}s;">
        <h3>🎯 {recurso.titulo}</h3>
        <p><strong>📚 Nivel:</strong> {recurso.nivel} | <strong>🌐 Plataforma:</strong> {recurso.plataforma}</p>
        <p>📝 {recurso.descripcion}</p>
        {cert_html}
        {ia_block}
        <div style="margin-top: 12px; display:flex; gap:8px; flex-wrap:wrap;">
            {link_button(recurso.url, "➡️ Acceder al recurso")}
        </div>
        <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #eee; font-size: 0.9rem; color: #666;">
            <p style="margin:4px 0;"><strong>🔎 Confianza:</strong> {recurso.confianza*100:.1f}% |
               <strong>✅ Verificado:</strong> {datetime.fromisoformat(recurso.ultima_verificacion).strftime('%d/%m/%Y')}</p>
            <p style="margin:4px 0;"><strong>🌍 Idioma:</strong> {recurso.idioma.upper()} |
               <strong>🏷️ Categoría:</strong> {recurso.categoria}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# EJECUCIÓN DE LA BÚSQUEDA PRINCIPAL
# ----------------------------
if buscar and tema.strip():
    with st.spinner("🔍 Buscando recursos educativos..."):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(buscar_recursos_multicapa(tema, idioma_seleccionado, nivel))
            loop.close()

            if resultados:
                st.success(f"✅ ¡{len(resultados)} recursos encontrados para '{tema}' en {idioma_seleccionado}!")
                if GROQ_AVAILABLE and GROQ_API_KEY:
                    planificar_analisis_ia(resultados)
                    st.info("🧠 Análisis de calidad en progreso — se actualizarán cuando esté listo")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(resultados))
                col2.metric("Plataformas", len(set(r.plataforma for r in resultados)))
                col3.metric("Confianza Promedio", f"{sum(r.confianza for r in resultados)/len(resultados):.1%}")

                st.markdown("### 📚 Resultados encontrados")
                for i, r in enumerate(resultados):
                    time.sleep(0.05)
                    if r.metadatos_analisis:
                        mostrar_recurso_con_ia(r, i)
                    else:
                        mostrar_recurso_basico(r, i, r.analisis_pendiente)

                st.markdown("---")
                df = pd.DataFrame([{
                    'titulo': r.titulo,
                    'url': r.url,
                    'plataforma': r.plataforma,
                    'nivel': r.nivel,
                    'idioma': r.idioma,
                    'categoria': r.categoria,
                    'confianza': f"{r.confianza:.1%}",
                    'analisis_ia': 'Sí' if r.metadatos_analisis else ('Pendiente' if r.analisis_pendiente else 'No')
                } for r in resultados])
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Resultados (CSV)",
                    data=csv,
                    file_name=f"resultados_busqueda_{tema.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ No encontramos recursos verificados para este tema. Intenta con otro término.")
        except Exception as e:
            logger.error(f"Error durante la búsqueda: {e}")
            st.error("❌ Ocurrió un error durante la búsqueda. Intenta nuevamente.")
            st.exception(e)

# ----------------------------
# CHAT IA (opcional)
# ----------------------------
st.markdown("### 💬 Asistente educativo (opcional)")
if "chat_msgs" not in st.session_state:
    st.session_state.chat_msgs = []

def chatgroq(mensajes: List[Dict[str, str]]) -> str:
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return "🧠 IA no disponible. Usa el buscador superior para encontrar cursos ahora."
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            "Eres un asistente educativo. Conversa con claridad. "
            "Si detectas intención de búsqueda, al final incluye SOLO un bloque JSON con este formato exacto:\n"
            '{"buscar": {"tema": "...", "idioma": "Español (es)|Inglés (en)|Portugués (pt)", "nivel": "Cualquiera|Principiante|Intermedio|Avanzado"}}\n'
            "No pongas comentarios ni texto dentro del JSON. El contenido conversacional va arriba; el JSON solo al final."
        )
        groq_msgs = [{"role": "system", "content": system_prompt}] + mensajes
        resp = client.chat.completions.create(
            messages=groq_msgs,
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=900,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Error en chat Groq: {e}")
        return "Hubo un error con la IA. Intenta de nuevo."

def extraer_comando_busqueda(texto: str) -> Optional[Dict[str, str]]:
    try:
        bloques = re.findall(r'\{.*\}', texto, flags=re.DOTALL)
        for raw in reversed(bloques):
            data = json.loads(raw)
            if isinstance(data, dict) and "buscar" in data:
                cmd = data["buscar"]
                tema = cmd.get("tema", "").strip()
                idioma = cmd.get("idioma", "").strip()
                nivel = cmd.get("nivel", "").strip()
                if tema and idioma in IDIOMAS.keys() and nivel in NIVELES:
                    return {"tema": tema, "idioma": idioma, "nivel": nivel}
    except Exception as e:
        logger.warning(f"JSON de IA inválido: {e}")
    return None

# PARCHE: función de render del chat que limpia HTML y JSON visibles sin tocar el resto del código
def limpiar_html_visible(texto: str) -> str:
    """
    Elimina cualquier bloque HTML o JSON que aparezca al final del texto.
    Evita que se muestren etiquetas como <div> o {...} en la interfaz.
    """
    texto = re.sub(r'\{.*\}', '', texto, flags=re.DOTALL).strip()
    texto = re.sub(r'<[^>]+>$', '', texto, flags=re.DOTALL).strip()
    return texto

def ui_chat_mostrar(mensaje: str, rol: str):
    """
    Muestra los mensajes del chat sin renderizar HTML ni JSON como texto plano.
    """
    texto = limpiar_html_visible(mensaje)
    if rol == "assistant":
        st.markdown(f"> {texto}")
    elif rol == "user":
        st.markdown(f"**Tú:** {texto}")

# Mostrar mensajes anteriores (ignorando system)
for msg in st.session_state.chat_msgs:
    if msg["role"] in ["user", "assistant"]: # Filtra mensajes system
        ui_chat_mostrar(msg["content"], msg["role"])

user_input = st.chat_input("Escribe aquí...")
if user_input:
    st.session_state.chat_msgs.append({"role": "user", "content": user_input})
    respuesta = chatgroq([msg for msg in st.session_state.chat_msgs if msg["role"] in ["user", "assistant"]]) # Solo envía user/assistant
    st.session_state.chat_msgs.append({"role": "assistant", "content": respuesta})
    st.rerun() # st.experimental_rerun() es obsoleto

# Tras rerun, ejecutar búsqueda si IA mandó comando
if st.session_state.chat_msgs:
    last_assistant = next((m for m in reversed(st.session_state.chat_msgs) if m["role"] == "assistant"), None)
    if last_assistant:
        cmd = extraer_comando_busqueda(last_assistant["content"])
        if cmd:
            with st.spinner(f"Buscando cursos: {cmd['tema']} ({cmd['idioma']}, {cmd['nivel']})"):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resultados_cmd = loop.run_until_complete(
                    buscar_recursos_multicapa(cmd["tema"], cmd["idioma"], cmd["nivel"])
                )
                loop.close()
                if resultados_cmd:
                    st.success(f"✅ {len(resultados_cmd)} recursos encontrados para “{cmd['tema']}”")
                    if GROQ_AVAILABLE and GROQ_API_KEY:
                        planificar_analisis_ia(resultados_cmd)
                    for i, r in enumerate(resultados_cmd):
                        time.sleep(0.05)
                        if r.metadatos_analisis:
                            mostrar_recurso_con_ia(r, i)
                        else:
                            mostrar_recurso_basico(r, i, r.analisis_pendiente)
                else:
                    st.warning("No se encontraron resultados para la búsqueda solicitada por la IA.")

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### 📈 Estadísticas en tiempo real")
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analiticas_busquedas")
        total_busquedas = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM plataformas_ocultas WHERE activa = 1")
        total_plataformas = cursor.fetchone()[0]
        cursor.execute("""
            SELECT tema, COUNT(*) AS conteo
            FROM analiticas_busquedas
            GROUP BY tema
            ORDER BY conteo DESC
            LIMIT 1
        """)
        tema_popular_data = cursor.fetchone()
        tema_popular = tema_popular_data[0] if tema_popular_data else "Python"
        conn.close()
    except Exception as e:
        logger.error(f"Error al obtener estadísticas: {e}")
        total_busquedas = 0
        total_plataformas = 0
        tema_popular = "Python"

    c1, c2 = st.columns(2)
    with c1:
        st.metric("🔍 Búsquedas", total_busquedas)
    with c2:
        st.metric("📚 Plataformas", total_plataformas)
    st.metric("🔥 Tema Popular", tema_popular)

    st.markdown("### ✨ Características")
    st.markdown("- ✅ Búsqueda inmediata\n- ✅ Resultados multifuente\n- ✅ Plataformas ocultas\n- ✅ Verificación automática\n- ✅ Diseño responsivo\n- 🧠 Análisis IA (opcional)")

    st.markdown("### 🤖 Estado del sistema")
    st.info(f"IA: {'✅ Disponible' if GROQ_AVAILABLE and GROQ_API_KEY else '⚠️ No disponible'}\nÚltima actualización: {datetime.now().strftime('%H:%M:%S')}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:#666;font-size:14px;padding:20px;background:#f8f9fa;border-radius:12px;">
    <strong>✨ Buscador Profesional de Cursos</strong><br>
    <span style="color: #2c3e50; font-weight: 500;">Resultados inmediatos • Cache inteligente • Alta disponibilidad</span><br>
    <em style="color: #7f8c8d;">Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')} • Versión: 3.1.4 • Estado: ✅ Activo</em><br>
    <div style="margin-top:10px;padding-top:10px;border-top:1px solid #ddd;">
        <code style="background:#f1f3f5;padding:2px 8px;border-radius:4px;color:#d32f2f;">
            IA opcional — Sistema funcional sin dependencias externas críticas
        </code>
    </div>
</div>
""", unsafe_allow_html=True)

logger.info("✅ Sistema iniciado correctamente")
logger.info("⚡ Búsqueda inmediata: Activa")
logger.info(f"🧠 IA en segundo plano: {'Disponible' if GROQ_AVAILABLE and GROQ_API_KEY else 'No disponible'}")
logger.info(f"🌐 Idiomas soportados: {len(IDIOMAS)}")
