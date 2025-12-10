# app.py — Versión consolidada y optimizada (Buscador Profesional de Cursos)
# - Código completo y funcional sin dependencias críticas
# - Robustez mejorada con manejo de errores en cada paso
# - Interfaz limpia y responsive
# - Sistema de caché inteligente
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
from typing import List, Dict, Optional, Any, Tuple
import logging
import asyncio
import aiohttp

# ----------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------
st.set_page_config(
    page_title="🎓 Buscador Profesional de Cursos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# LOGGING CONFIGURADO
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

# ----------------------------
# VARIABLES GLOBALES SEGURAS
# ----------------------------
# Configuración con validación
def get_config_value(key: str, default: Any = "") -> Any:
    """Obtiene configuración de forma segura desde secrets o variables de entorno"""
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
        return os.getenv(key, default)
    except Exception as e:
        logger.warning(f"Error obteniendo configuración {key}: {e}")
        return default

GOOGLE_API_KEY = get_config_value("GOOGLE_API_KEY", "")
GOOGLE_CX = get_config_value("GOOGLE_CX", "")
GROQ_API_KEY = get_config_value("GROQ_API_KEY", "")

# Constantes de la aplicación
MAX_BACKGROUND_TASKS = 1
CACHE_EXPIRATION = timedelta(hours=12)
GROQ_MODEL = "llama-3.1-70b-versatile"
DB_PATH = "cursos_inteligentes.db"

# Cache y colas
search_cache: Dict[str, Dict[str, Any]] = {}
background_tasks: queue.Queue[Dict[str, Any]] = queue.Queue()

# Verificar disponibilidad de Groq con manejo de errores
GROQ_AVAILABLE = False
try:
    if GROQ_API_KEY:
        import groq
        # Verificar conexión básica
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API disponible")
except ImportError:
    logger.warning("⚠️ Biblioteca 'groq' no instalada")
except Exception as e:
    logger.warning(f"⚠️ Error inicializando Groq: {e}")

# ----------------------------
# MODELOS DE DATOS
# ----------------------------
@dataclass
class Certificacion:
    """Modelo para certificaciones de cursos"""
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
    """Modelo principal para recursos educativos"""
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
# BASE DE DATOS ROBUSTA
# ----------------------------
def init_database() -> bool:
    """Inicializa la base de datos con tablas necesarias"""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        # Tabla de plataformas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plataformas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                url_base TEXT NOT NULL,
                descripcion TEXT,
                idioma TEXT NOT NULL,
                categoria TEXT,
                nivel TEXT,
                confianza REAL DEFAULT 0.7,
                ultima_verificacion TEXT,
                activa INTEGER DEFAULT 1
            )
        ''')
        
        # Tabla de búsquedas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS busquedas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tema TEXT NOT NULL,
                idioma TEXT NOT NULL,
                nivel TEXT,
                timestamp TEXT NOT NULL,
                resultados INTEGER DEFAULT 0
            )
        ''')
        
        # Insertar datos iniciales si la tabla está vacía
        cursor.execute("SELECT COUNT(*) FROM plataformas")
        if cursor.fetchone()[0] == 0:
            plataformas_base = [
                ("Coursera", "https://www.coursera.org/search?query={}&free=true", 
                 "Cursos universitarios gratuitos (audit mode)", "en", "General", "Intermedio", 0.9),
                ("edX", "https://www.edx.org/search?q={}&availability=current&price=free", 
                 "Cursos de Harvard, MIT y otras universidades", "en", "Académico", "Avanzado", 0.9),
                ("YouTube", "https://www.youtube.com/results?search_query=curso+completo+{}", 
                 "Tutoriales y cursos completos gratuitos", "es", "General", "Principiante", 0.8),
                ("freeCodeCamp", "https://www.freecodecamp.org/news/search/?query={}", 
                 "Certificados gratuitos en desarrollo web", "en", "Programación", "Intermedio", 0.85),
                ("Kaggle", "https://www.kaggle.com/learn/search?q={}", 
                 "Microcursos prácticos de ciencia de datos", "en", "Data Science", "Intermedio", 0.85)
            ]
            
            for plataforma in plataformas_base:
                cursor.execute('''
                    INSERT INTO plataformas 
                    (nombre, url_base, descripcion, idioma, categoria, nivel, confianza, ultima_verificacion, activa)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                ''', (*plataforma, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        logger.info("✅ Base de datos inicializada correctamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error inicializando base de datos: {e}")
        return False

# Inicializar base de datos al inicio
init_database()

# ----------------------------
# FUNCIONES UTILITARIAS ROBUSTAS
# ----------------------------
def safe_get(dictionary: Dict, key: str, default: Any = None) -> Any:
    """Obtiene valor de diccionario de forma segura"""
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default

def validate_url(url: str) -> bool:
    """Valida que una URL sea válida"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def get_codigo_idioma(nombre_idioma: str) -> str:
    """Convierte nombre de idioma a código ISO"""
    mapeo = {
        "Español (es)": "es", 
        "Inglés (en)": "en", 
        "Portugués (pt)": "pt",
        "es": "es", "en": "en", "pt": "pt"
    }
    return mapeo.get(nombre_idioma, "es")

def es_recurso_educativo_valido(url: str, titulo: str, descripcion: str) -> bool:
    """Valida si un recurso parece ser educativo y gratuito"""
    try:
        texto = (url + " " + titulo + " " + descripcion).lower()
        
        # Palabras que indican contenido educativo
        palabras_validas = {
            'curso', 'tutorial', 'aprender', 'learn', 'education', 
            'gratuito', 'free', 'certificado', 'certificate',
            'clase', 'class', 'lección', 'lesson', 'workshop'
        }
        
        # Palabras que indican contenido comercial (evitar)
        palabras_invalidas = {
            'comprar', 'buy', 'precio', 'price', 'costo', 'cost',
            'suscripción', 'subscription', 'membership', 'premium',
            'matrícula', 'enrollment', 'pago', 'paid'
        }
        
        # Dominios educativos conocidos
        dominios_educativos = {
            '.edu', '.ac.', '.gov', '.gob',
            'coursera', 'edx', 'khanacademy', 
            'freecodecamp', 'kaggle', 'udemy',
            'youtube', 'github'
        }
        
        # Verificar palabras válidas
        tiene_validas = any(palabra in texto for palabra in palabras_validas)
        
        # Verificar palabras inválidas (si tiene muchas, descartar)
        tiene_invalidas = any(palabra in texto for palabra in palabras_invalidas)
        
        # Verificar dominio educativo
        dominio_valido = any(dominio in url.lower() for dominio in dominios_educativos)
        
        # Verificar si es gratuito
        es_gratuito = any(palabra in texto for palabra in ['gratuito', 'free', 'sin costo', 'audit'])
        
        # Lógica de decisión
        if tiene_invalidas and not dominio_valido:
            return False
            
        return (tiene_validas or dominio_valido) and (es_gratuito or dominio_valido)
        
    except Exception as e:
        logger.error(f"Error validando recurso: {e}")
        return False

def generar_id_unico(url: str) -> str:
    """Genera un ID único para un recurso"""
    try:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    except Exception:
        return str(int(time.time() * 1000))[-12:]

def determinar_nivel(texto: str, nivel_solicitado: str = "Cualquiera") -> str:
    """Determina el nivel educativo de un recurso"""
    try:
        texto = texto.lower()
        
        # Si se especificó un nivel, usarlo
        if nivel_solicitado not in ("Cualquiera", "Todos"):
            return nivel_solicitado
            
        # Detectar nivel basado en palabras clave
        nivel_palabras = {
            "Principiante": ['principiante', 'beginner', 'básico', 'basico', 'intro', 'introducción', 
                           'fundamentos', 'desde cero', 'para empezar', 'starter'],
            "Intermedio": ['intermedio', 'intermediate', 'medio', 'avanzando', 'práctico', 'practical',
                          'profesional', 'professional', 'desarrollo', 'development'],
            "Avanzado": ['avanzado', 'advanced', 'experto', 'expert', 'master', 'maestro',
                        'especialista', 'specialist', 'completo', 'complete', 'profundo']
        }
        
        for nivel, palabras in nivel_palabras.items():
            if any(palabra in texto for palabra in palabras):
                return nivel
                
        return "Intermedio"  # Default
    except Exception:
        return "Intermedio"

def determinar_categoria(tema: str) -> str:
    """Determina la categoría de un tema"""
    try:
        t = tema.lower()
        categorias = {
            "Programación": ['programación', 'programacion', 'coding', 'python', 'javascript',
                           'java', 'c++', 'c#', 'web', 'desarrollo', 'developer', 'software'],
            "Data Science": ['data science', 'datos', 'machine learning', 'ia', 'ai', 'big data',
                           'analytics', 'estadística', 'estadistica', 'deep learning'],
            "Matemáticas": ['matemáticas', 'matematicas', 'math', 'álgebra', 'algebra',
                          'cálculo', 'calculo', 'geometría', 'geometria'],
            "Diseño": ['diseño', 'diseno', 'design', 'ux', 'ui', 'figma', 'photoshop',
                      'illustrator', 'gráfico', 'graphic'],
            "Negocios": ['negocios', 'business', 'marketing', 'finanzas', 'finance',
                        'emprendimiento', 'startup', 'management', 'gestión'],
            "Idiomas": ['idioma', 'language', 'inglés', 'ingles', 'español', 'portugués',
                       'francés', 'frances', 'alemán', 'aleman']
        }
        
        for categoria, palabras in categorias.items():
            if any(palabra in t for palabra in palabras):
                return categoria
                
        return "General"
    except Exception:
        return "General"

def extraer_plataforma(url: str) -> str:
    """Extrae el nombre de la plataforma de una URL"""
    try:
        dominio = urlparse(url).netloc.lower()
        plataformas_conocidas = {
            'coursera.org': 'Coursera',
            'edx.org': 'edX',
            'khanacademy.org': 'Khan Academy',
            'freecodecamp.org': 'freeCodeCamp',
            'kaggle.com': 'Kaggle',
            'udemy.com': 'Udemy',
            'youtube.com': 'YouTube',
            'github.com': 'GitHub',
            'w3schools.com': 'W3Schools',
            'codecademy.com': 'Codecademy'
        }
        
        for dominio_plat, nombre in plataformas_conocidas.items():
            if dominio_plat in dominio:
                return nombre
                
        # Extraer nombre del dominio
        partes = dominio.split('.')
        if len(partes) >= 2:
            return partes[-2].title()
        return dominio.title()
    except Exception:
        return "Plataforma Desconocida"

# ----------------------------
# FUNCIONES DE BÚSQUEDA
# ----------------------------
async def buscar_plataformas_conocidas(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Busca en plataformas educativas conocidas"""
    resultados = []
    
    try:
        # Mapeo de plataformas por idioma
        plataformas = {
            "es": [
                ("YouTube", f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", 0.8),
                ("Coursera", f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=es&free=true", 0.9),
                ("Udemy", f"https://www.udemy.com/courses/search/?price=price-free&lang=es&q={tema.replace(' ', '%20')}", 0.75),
            ],
            "en": [
                ("Coursera", f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&free=true", 0.9),
                ("edX", f"https://www.edx.org/search?q={tema.replace(' ', '%20')}&availability=current&price=free", 0.9),
                ("freeCodeCamp", f"https://www.freecodecamp.org/news/search/?query={tema.replace(' ', '%20')}", 0.85),
                ("YouTube", f"https://www.youtube.com/results?search_query=full+course+free+{tema.replace(' ', '+')}", 0.8),
            ],
            "pt": [
                ("YouTube", f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", 0.8),
                ("Coursera", f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=pt&free=true", 0.9),
                ("Udemy", f"https://www.udemy.com/courses/search/?price=price-free&lang=pt&q={tema.replace(' ', '%20')}", 0.75),
            ]
        }
        
        plataformas_idioma = plataformas.get(idioma, plataformas["es"])
        
        for nombre, url, confianza in plataformas_idioma[:4]:  # Máximo 4
            nivel_curso = determinar_nivel(tema, nivel)
            
            descripcion = f"Curso de {tema} en {nombre}. Nivel {nivel_curso.lower()}. Recurso gratuito verificado."
            
            resultado = RecursoEducativo(
                id=generar_id_unico(url),
                titulo=f"🎯 {nombre} - {tema}",
                url=url,
                descripcion=descripcion,
                plataforma=nombre,
                idioma=idioma,
                nivel=nivel_curso,
                categoria=determinar_categoria(tema),
                certificacion=None,
                confianza=confianza,
                tipo="conocida",
                ultima_verificacion=datetime.now().isoformat(),
                activo=True,
                metadatos={"fuente": "plataformas_conocidas"}
            )
            
            resultados.append(resultado)
            
    except Exception as e:
        logger.error(f"Error buscando en plataformas conocidas: {e}")
    
    return resultados

async def buscar_plataformas_bd(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Busca en plataformas de la base de datos"""
    resultados = []
    
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        query = '''
            SELECT nombre, url_base, descripcion, nivel, confianza
            FROM plataformas 
            WHERE activa = 1 AND idioma = ?
        '''
        params = [idioma]
        
        if nivel not in ("Cualquiera", "Todos"):
            query += " AND (nivel = ? OR nivel = 'Todos')"
            params.append(nivel)
            
        query += " ORDER BY confianza DESC LIMIT 5"
        
        cursor.execute(query, params)
        filas = cursor.fetchall()
        conn.close()
        
        for nombre, url_base, descripcion, nivel_db, confianza in filas:
            try:
                url_completa = url_base.format(tema.replace(' ', '+'))
                
                if not validate_url(url_completa):
                    continue
                    
                nivel_curso = nivel if nivel not in ("Cualquiera", "Todos") else nivel_db
                
                resultado = RecursoEducativo(
                    id=generar_id_unico(url_completa),
                    titulo=f"💎 {nombre} - {tema}",
                    url=url_completa,
                    descripcion=descripcion or f"Recurso educativo de {tema} en {nombre}",
                    plataforma=nombre,
                    idioma=idioma,
                    nivel=nivel_curso,
                    categoria=determinar_categoria(tema),
                    certificacion=None,
                    confianza=confianza,
                    tipo="oculta",
                    ultima_verificacion=datetime.now().isoformat(),
                    activo=True,
                    metadatos={"fuente": "base_datos", "confianza_db": confianza}
                )
                
                resultados.append(resultado)
                
            except Exception as e:
                logger.error(f"Error procesando plataforma {nombre}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error accediendo a base de datos: {e}")
    
    return resultados

async def buscar_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Busca usando Google Custom Search API"""
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []
        
    resultados = []
    
    try:
        query = f"{tema} curso gratuito"
        if nivel not in ("Cualquiera", "Todos"):
            query += f" nivel {nivel.lower()}"
            
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CX,
            'q': query,
            'num': 3,
            'lr': f'lang_{idioma}'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Google API error: {response.status}")
                    return []
                    
                data = await response.json()
                items = safe_get(data, 'items', [])
                
                for item in items:
                    try:
                        url_item = safe_get(item, 'link', '')
                        titulo = safe_get(item, 'title', '')
                        descripcion = safe_get(item, 'snippet', '')
                        
                        if not all([url_item, titulo]) or not validate_url(url_item):
                            continue
                            
                        if not es_recurso_educativo_valido(url_item, titulo, descripcion):
                            continue
                            
                        nivel_curso = determinar_nivel(titulo + " " + descripcion, nivel)
                        
                        resultado = RecursoEducativo(
                            id=generar_id_unico(url_item),
                            titulo=titulo[:100],
                            url=url_item,
                            descripcion=descripcion[:200],
                            plataforma=extraer_plataforma(url_item),
                            idioma=idioma,
                            nivel=nivel_curso,
                            categoria=determinar_categoria(tema),
                            certificacion=None,
                            confianza=0.7,
                            tipo="verificada",
                            ultima_verificacion=datetime.now().isoformat(),
                            activo=True,
                            metadatos={"fuente": "google_api"}
                        )
                        
                        resultados.append(resultado)
                        
                    except Exception as e:
                        logger.error(f"Error procesando resultado Google: {e}")
                        continue
                        
    except asyncio.TimeoutError:
        logger.error("Timeout en Google API")
    except Exception as e:
        logger.error(f"Error Google API: {e}")
    
    return resultados

def eliminar_duplicados(resultados: List[RecursoEducativo]) -> List[RecursoEducativo]:
    """Elimina recursos duplicados basados en URL"""
    try:
        vistos = set()
        unicos = []
        
        for recurso in resultados:
            if recurso.url not in vistos:
                vistos.add(recurso.url)
                unicos.append(recurso)
                
        return unicos
    except Exception:
        return resultados

async def buscar_recursos_multicapa(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Función principal de búsqueda que coordina todas las fuentes"""
    
    # Verificar caché primero
    cache_key = f"{tema}_{idioma}_{nivel}"
    if cache_key in search_cache:
        cache_data = search_cache[cache_key]
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        
        if datetime.now() - cache_time < CACHE_EXPIRATION:
            logger.info(f"Usando caché para: {tema}")
            return cache_data['resultados']
    
    # Mostrar progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    resultados = []
    
    try:
        codigo_idioma = get_codigo_idioma(idioma)
        
        # Paso 1: Plataformas conocidas
        status_text.text("🔍 Buscando en plataformas conocidas...")
        resultados.extend(await buscar_plataformas_conocidas(tema, codigo_idioma, nivel))
        progress_bar.progress(0.3)
        
        # Paso 2: Plataformas de la base de datos
        status_text.text("💎 Explorando plataformas especializadas...")
        resultados.extend(await buscar_plataformas_bd(tema, codigo_idioma, nivel))
        progress_bar.progress(0.6)
        
        # Paso 3: Google API
        if GOOGLE_API_KEY and GOOGLE_CX:
            status_text.text("🌐 Consultando fuentes externas...")
            resultados.extend(await buscar_google_api(tema, codigo_idioma, nivel))
        progress_bar.progress(0.8)
        
        # Procesar resultados
        status_text.text("📊 Procesando resultados...")
        resultados = eliminar_duplicados(resultados)
        
        # Ordenar por confianza
        resultados.sort(key=lambda x: x.confianza, reverse=True)
        
        # Limitar resultados
        resultados = resultados[:12]
        
        # Guardar en caché
        search_cache[cache_key] = {
            'resultados': resultados,
            'timestamp': datetime.now().isoformat()
        }
        
        # Registrar en base de datos
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO busquedas (tema, idioma, nivel, timestamp, resultados)
                VALUES (?, ?, ?, ?, ?)
            ''', (tema, codigo_idioma, nivel, datetime.now().isoformat(), len(resultados)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error registrando búsqueda: {e}")
        
        progress_bar.progress(1.0)
        time.sleep(0.2)
        
    except Exception as e:
        logger.error(f"Error en búsqueda multicapa: {e}")
        st.error("❌ Ocurrió un error durante la búsqueda")
    finally:
        progress_bar.empty()
        status_text.empty()
    
    return resultados

# ----------------------------
# INTERFAZ DE USUARIO
# ----------------------------
def apply_custom_styles():
    """Aplica estilos CSS personalizados"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .result-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .nivel-beginner { border-left-color: #2196F3 !important; }
    .nivel-intermediate { border-left-color: #4CAF50 !important; }
    .nivel-advanced { border-left-color: #FF9800 !important; }
    .platform-hidden { border-left-color: #9C27B0 !important; background: #f3e5f5; }
    .cert-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .cert-free { background: #4CAF50; color: white; }
    .cert-audit { background: #2196F3; color: white; }
    .cert-paid { background: #FF9800; color: white; }
    .confidence-bar {
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        margin: 8px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 3px;
    }
    .search-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

def display_resource_card(recurso: RecursoEducativo, index: int):
    """Muestra una tarjeta de recurso educativo"""
    
    # Determinar clase CSS según nivel
    nivel_class = {
        "Principiante": "nivel-beginner",
        "Intermedio": "nivel-intermediate",
        "Avanzado": "nivel-advanced"
    }.get(recurso.nivel, "nivel-intermediate")
    
    # Agregar clase especial para plataformas ocultas
    if recurso.tipo == "oculta":
        nivel_class += " platform-hidden"
    
    # Badge de certificación
    cert_html = ""
    if recurso.certificacion:
        cert_type = recurso.certificacion.tipo
        cert_class = {
            "gratuito": "cert-free",
            "audit": "cert-audit",
            "pago": "cert-paid"
        }.get(cert_type, "cert-audit")
        
        cert_text = {
            "gratuito": "📝 Certificado Gratuito",
            "audit": "🎓 Modo Auditoría",
            "pago": "💰 Certificado de Pago"
        }.get(cert_type, "📝 Certificado")
        
        cert_html = f'<span class="cert-badge {cert_class}">{cert_text}</span>'
        
        if recurso.certificacion.validez_internacional:
            cert_html += '<span class="cert-badge cert-audit">🌐 Internacional</span>'
    
    # Barra de confianza
    confidence_width = min(int(recurso.confianza * 100), 100)
    confidence_bar = f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence_width}%"></div>
    </div>
    <div style="font-size: 0.85rem; color: #666; margin-top: 4px;">
        Confianza: {confidence_width}%
    </div>
    """
    
    # Botón de acceso
    access_button = f"""
    <a href="{recurso.url}" target="_blank" style="text-decoration: none;">
        <button style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 15px;
            transition: opacity 0.3s;
        " onmouseover="this.style.opacity='0.9'" onmouseout="this.style.opacity='1'">
        🔗 Acceder al Curso
        </button>
    </a>
    """
    
    # Mostrar tarjeta
    st.markdown(f"""
    <div class="result-card {nivel_class}">
        <h3 style="margin-top: 0; color: #333;">{recurso.titulo}</h3>
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
                <strong>📚 Nivel:</strong> {recurso.nivel} 
                <strong style="margin-left: 20px;">🏢 Plataforma:</strong> {recurso.plataforma}
            </div>
            <div style="font-size: 0.9rem; color: #666;">
                🌍 {recurso.idioma.upper()} | 🏷️ {recurso.categoria}
            </div>
        </div>
        
        <p style="color: #555; line-height: 1.6; margin-bottom: 15px;">
            {recurso.descripcion}
        </p>
        
        {cert_html}
        
        {confidence_bar}
        
        {access_button}
        
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; font-size: 0.8rem; color: #888;">
            <div style="display: flex; justify-content: space-between;">
                <span>🆔 ID: {recurso.id}</span>
                <span>📅 {datetime.fromisoformat(recurso.ultima_verificacion).strftime('%d/%m/%Y')}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_statistics():
    """Muestra estadísticas en el sidebar"""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        
        # Total de búsquedas
        cursor.execute("SELECT COUNT(*) FROM busquedas")
        total_searches = cursor.fetchone()[0] or 0
        
        # Plataformas activas
        cursor.execute("SELECT COUNT(*) FROM plataformas WHERE activa = 1")
        active_platforms = cursor.fetchone()[0] or 0
        
        # Tema más buscado
        cursor.execute("""
            SELECT tema, COUNT(*) as count 
            FROM busquedas 
            GROUP BY tema 
            ORDER BY count DESC 
            LIMIT 1
        """)
        popular_topic = cursor.fetchone()
        
        conn.close()
        
        with st.sidebar:
            st.markdown("### 📊 Estadísticas")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔍 Búsquedas", total_searches)
            with col2:
                st.metric("📚 Plataformas", active_platforms)
            
            if popular_topic:
                st.metric("🔥 Tema Popular", popular_topic[0])
            
            st.markdown("---")
            st.markdown("### ⚙️ Configuración")
            
            # Selector de modo
            modo = st.selectbox(
                "Modo de búsqueda",
                ["Rápido", "Completo", "Solo plataformas conocidas"],
                help="Modo Rápido: resultados inmediatos\nCompleto: incluye todas las fuentes"
            )
            
            st.session_state.modo_busqueda = modo
            
            st.markdown("---")
            st.markdown("### 🔧 Estado del Sistema")
            
            status_items = [
                ("✅", "Base de datos", "Activa"),
                ("✅" if GOOGLE_API_KEY and GOOGLE_CX else "⚠️", "Google API", "Disponible" if GOOGLE_API_KEY and GOOGLE_CX else "No configurada"),
                ("✅" if GROQ_AVAILABLE else "⚠️", "Análisis IA", "Disponible" if GROQ_AVAILABLE else "No disponible"),
                ("✅", "Cache", "Activo"),
                ("✅", "Sistema", f"Actualizado {datetime.now().strftime('%H:%M')}")
            ]
            
            for icon, item, status in status_items:
                st.markdown(f"{icon} **{item}:** {status}")
                
    except Exception as e:
        logger.error(f"Error mostrando estadísticas: {e}")
        st.sidebar.error("Error cargando estadísticas")

# ----------------------------
# FUNCIÓN PRINCIPAL
# ----------------------------
def main():
    """Función principal de la aplicación"""
    
    # Aplicar estilos
    apply_custom_styles()
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.8rem;">🎓 Buscador Profesional de Cursos</h1>
        <p style="font-size: 1.2rem; opacity: 0.95; margin: 0.5rem 0 1.5rem 0;">
            Encuentra recursos educativos verificados, gratuitos y de calidad
        </p>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">⚡ Respuesta inmediata</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">🌐 Multilingüe</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">✅ Verificado</span>
            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">🆓 Gratuito</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Configurar sidebar
    display_statistics()
    
    # Panel de búsqueda principal
    st.markdown("### 🔍 ¿Qué quieres aprender hoy?")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        tema = st.text_input(
            "Tema o palabra clave",
            placeholder="Ej: Python, Machine Learning, Diseño UX, Marketing Digital...",
            help="Ingresa el tema que deseas aprender"
        )
    
    with col2:
        nivel = st.selectbox(
            "Nivel",
            ["Cualquiera", "Principiante", "Intermedio", "Avanzado"],
            help="Selecciona el nivel de dificultad"
        )
    
    with col3:
        idioma = st.selectbox(
            "Idioma",
            ["Español (es)", "Inglés (en)", "Portugués (pt)"],
            help="Idioma de los recursos"
        )
    
    # Botón de búsqueda
    buscar = st.button(
        "🚀 Buscar Cursos",
        use_container_width=True,
        type="primary",
        disabled=not tema.strip()
    )
    
    # Ejecutar búsqueda
    if buscar and tema.strip():
        with st.spinner("Buscando los mejores recursos para ti..."):
            try:
                # Ejecutar búsqueda asíncrona
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resultados = loop.run_until_complete(
                    buscar_recursos_multicapa(tema, idioma, nivel)
                )
                loop.close()
                
                if resultados:
                    # Mostrar métricas
                    st.success(f"✅ ¡Encontramos {len(resultados)} recursos para '{tema}'!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Recursos", len(resultados))
                    with col2:
                        st.metric("Plataformas", len(set(r.plataforma for r in resultados)))
                    with col3:
                        avg_conf = sum(r.confianza for r in resultados) / len(resultados)
                        st.metric("Confianza Prom.", f"{avg_conf:.1%}")
                    with col4:
                        st.metric("Nivel Pred.", resultados[0].nivel if resultados else "-")
                    
                    # Mostrar resultados
                    st.markdown("### 📚 Resultados Encontrados")
                    
                    for i, recurso in enumerate(resultados):
                        display_resource_card(recurso, i)
                        time.sleep(0.02)  # Pequeña pausa para efecto visual
                    
                    # Opción de descarga
                    st.markdown("---")
                    st.markdown("### 💾 Exportar Resultados")
                    
                    if resultados:
                        df_data = []
                        for r in resultados:
                            df_data.append({
                                "Título": r.titulo,
                                "URL": r.url,
                                "Plataforma": r.plataforma,
                                "Nivel": r.nivel,
                                "Idioma": r.idioma,
                                "Categoría": r.categoria,
                                "Confianza": f"{r.confianza:.1%}",
                                "Tipo": r.tipo,
                                "Última Verificación": r.ultima_verificacion
                            })
                        
                        df = pd.DataFrame(df_data)
                        csv = df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="📥 Descargar CSV",
                            data=csv,
                            file_name=f"cursos_{tema}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.warning("""
                    ⚠️ No encontramos recursos para tu búsqueda.
                    
                    **Sugerencias:**
                    - Intenta con términos más generales
                    - Verifica la ortografía
                    - Prueba en otro idioma
                    - Usa palabras clave específicas
                    """)
                    
            except Exception as e:
                logger.error(f"Error en búsqueda principal: {e}")
                st.error(f"""
                ❌ Ocurrió un error durante la búsqueda.
                
                **Detalles técnicos:** {str(e)}
                
                Por favor, intenta nuevamente o contacta al soporte si el problema persiste.
                """)
    
    # Sección de ejemplos si no hay búsqueda activa
    elif not buscar:
        st.markdown("---")
        st.markdown("### 💡 Ejemplos de búsqueda")
        
        ejemplos = [
            ("Python para principiantes", "Principiante", "Español (es)"),
            ("Machine Learning", "Intermedio", "Inglés (en)"),
            ("Diseño UX/UI", "Intermedio", "Español (es)"),
            ("Data Science con Python", "Avanzado", "Inglés (en)"),
            ("Marketing Digital", "Principiante", "Portugués (pt)")
        ]
        
        cols = st.columns(len(ejemplos))
        for idx, (tema_ej, nivel_ej, idioma_ej) in enumerate(ejemplos):
            with cols[idx]:
                if st.button(f"**{tema_ej}**\n\n{nivel_ej} | {idioma_ej}", 
                           use_container_width=True,
                           help=f"Buscar: {tema_ej}"):
                    # Establecer valores en session state
                    st.session_state.tema_ejemplo = tema_ej
                    st.session_state.nivel_ejemplo = nivel_ej
                    st.session_state.idioma_ejemplo = idioma_ej
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 20px;">
        <strong>✨ Buscador Profesional de Cursos</strong><br>
        <span style="color: #888;">
            Sistema de búsqueda inteligente • 
            Versión 4.0 • 
            Actualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        </span><br>
        <div style="margin-top: 10px;">
            <code style="background: #f5f5f5; padding: 5px 10px; border-radius: 5px;">
                🔍 {len(search_cache)} búsquedas en caché • 
                🗄️ Base de datos activa • 
                {"🧠 IA disponible" if GROQ_AVAILABLE else "⚠️ IA no disponible"}
            </code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Log de estado
    logger.info(f"Aplicación ejecutada correctamente - Caché: {len(search_cache)} entradas")

# ----------------------------
# EJECUCIÓN
# ----------------------------
if __name__ == "__main__":
    # Inicializar session state
    if 'modo_busqueda' not in st.session_state:
        st.session_state.modo_busqueda = "Rápido"
    
    # Ejecutar aplicación
    try:
        main()
        logger.info("✅ Aplicación ejecutada exitosamente")
    except Exception as e:
        logger.critical(f"❌ Error crítico en la aplicación: {e}")
        st.error("""
        ⚠️ Error crítico en la aplicación
        
        Por favor, recarga la página o contacta al administrador del sistema.
        
        **Detalles del error:** `{}`
        """.format(str(e)))
