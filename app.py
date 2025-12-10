import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import random
from datetime import datetime, timedelta
import json
import hashlib
import requests
import re
from urllib.parse import quote, urlparse
import threading
import queue
import concurrent.futures
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import asyncio
import aiohttp
from dotenv import load_dotenv
import sys

# ----------------------------
# CONFIGURACIÓN INICIAL - SEGURA PARA STREAMLIT CLOUD
# ----------------------------
# Configurar logging básico antes de cualquier operación
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BuscadorProfesional")

# Variables de entorno seguras para Streamlit Cloud
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = st.secrets.get("GOOGLE_CX", "") if hasattr(st, 'secrets') else os.getenv("GOOGLE_CX", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("GROQ_API_KEY", "")
DUCKDUCKGO_ENABLED = str(st.secrets.get("DUCKDUCKGO_ENABLED", "true")).lower() == "true" if hasattr(st, 'secrets') else os.getenv("DUCKDUCKGO_ENABLED", "true").lower() == "true"

# Configuración de parámetros
MAX_BACKGROUND_TASKS = 1
CACHE_EXPIRATION = timedelta(hours=12)
GROQ_MODEL = "llama3-8b-8192"  # Modelo rápido y gratuito

# Sistema de caché para búsquedas frecuentes
search_cache = {}
groq_cache = {}

# Cola para tareas en segundo plano
background_tasks = queue.Queue()

# ----------------------------
# MODELOS DE DATOS AVANZADOS
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
    tipo: str  # "conocida", "oculta", "verificada", "tor", "academico"
    ultima_verificacion: str
    activo: bool
    metadatos: Dict[str, Any]
    metadatos_analisis: Optional[Dict[str, Any]] = None  # Para análisis de IA

# ----------------------------
# FUNCIONES AUXILIARES BÁSICAS (siempre disponibles)
# ----------------------------
def get_codigo_idioma(nombre_idioma: str) -> str:
    """Convierte nombre de idioma a código ISO"""
    mapeo = {
        "Español (es)": "es",
        "Inglés (en)": "en",
        "Portugués (pt)": "pt",
        "es": "es",
        "en": "en",
        "pt": "pt"
    }
    return mapeo.get(nombre_idioma, "es")

def generar_id_unico(url: str) -> str:
    """Genera un ID único para un recurso basado en su URL"""
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    """Determina el nivel educativo basado en el texto"""
    texto = texto.lower()
    
    if nivel_solicitado != "Cualquiera" and nivel_solicitado != "Todos":
        return nivel_solicitado
    
    if any(palabra in texto for palabra in ['principiante', 'basico', 'básico', 'beginner', 'fundamentos', 'introducción', 'desde cero']):
        return "Principiante"
    elif any(palabra in texto for palabra in ['intermedio', 'intermediate', 'práctico', 'aplicado', 'práctica', 'profesional']):
        return "Intermedio"
    elif any(palabra in texto for palabra in ['avanzado', 'advanced', 'experto', 'máster', 'profesional', 'especialista']):
        return "Avanzado"
    else:
        return "Intermedio"

def determinar_categoria(tema: str) -> str:
    """Determina la categoría educativa basada en el tema"""
    tema = tema.lower()
    
    if any(palabra in tema for palabra in ['programación', 'python', 'javascript', 'web', 'desarrollo', 'coding', 'programming']):
        return "Programación"
    elif any(palabra in tema for palabra in ['datos', 'data', 'machine learning', 'ia', 'ai', 'artificial intelligence', 'ciencia de datos']):
        return "Data Science"
    elif any(palabra in tema for palabra in ['matemáticas', 'math', 'estadística', 'statistics', 'álgebra', 'calculus']):
        return "Matemáticas"
    elif any(palabra in tema for palabra in ['diseño', 'design', 'ux', 'ui', 'gráfico', 'graphic', 'creativo']):
        return "Diseño"
    elif any(palabra in tema for palabra in ['marketing', 'business', 'negocios', 'finanzas', 'finance', 'emprendimiento']):
        return "Negocios"
    elif any(palabra in tema for palabra in ['idioma', 'language', 'inglés', 'english', 'español', 'portugues']):
        return "Idiomas"
    else:
        return "General"

def extraer_plataforma(url: str) -> str:
    """Extrae el nombre de la plataforma de una URL"""
    dominio = urlparse(url).netloc.lower()
    
    if 'coursera' in dominio:
        return 'Coursera'
    elif 'edx' in dominio:
        return 'edX'
    elif 'khanacademy' in dominio:
        return 'Khan Academy'
    elif 'freecodecamp' in dominio:
        return 'freeCodeCamp'
    elif 'kaggle' in dominio:
        return 'Kaggle'
    elif 'udemy' in dominio:
        return 'Udemy'
    elif 'youtube' in dominio:
        return 'YouTube'
    elif 'aprendeconalf' in dominio:
        return 'Aprende con Alf'
    elif '.edu' in dominio or '.ac.' in dominio or '.gob' in dominio or '.gov' in dominio:
        return 'Institución Académica'
    else:
        partes = dominio.split('.')
        if len(partes) > 1:
            return partes[-2].title()
        return dominio.title()

# ----------------------------
# FUNCIONES DE BÚSQUEDA BÁSICAS
# ----------------------------
def buscar_recursos_basicos(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    """Búsqueda básica sin dependencias externas"""
    resultados = []
    codigo_idioma = get_codigo_idioma(idioma)
    
    # Definir plataformas según el idioma
    if codigo_idioma == "es":
        plataformas = [
            {
                "nombre": "YouTube",
                "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}",
                "descripcion": "Videotutoriales gratuitos con ejercicios prácticos",
                "niveles": ["Principiante", "Intermedio"]
            },
            {
                "nombre": "Coursera (Español)",
                "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=es&free=true",
                "descripcion": "Cursos universitarios gratuitos en modo auditor",
                "niveles": ["Intermedio", "Avanzado"]
            },
            {
                "nombre": "Udemy (Español)",
                "url": f"https://www.udemy.com/courses/search/?price=price-free&lang=es&q={tema.replace(' ', '%20')}",
                "descripcion": "Cursos gratuitos con certificados de finalización",
                "niveles": ["Principiante", "Intermedio"]
            }
        ]
    elif codigo_idioma == "pt":
        plataformas = [
            {
                "nombre": "YouTube",
                "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}",
                "descripcion": "Videotutoriales gratuitos en portugués",
                "niveles": ["Principiante", "Intermedio"]
            },
            {
                "nombre": "Coursera (Portugués)",
                "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&languages=pt&free=true",
                "descripcion": "Cursos gratuitos de universidades internacionales",
                "niveles": ["Intermedio", "Avanzado"]
            }
        ]
    else:  # Inglés
        plataformas = [
            {
                "nombre": "YouTube",
                "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}",
                "descripcion": "Videotutoriales gratuitos en inglés",
                "niveles": ["Principiante", "Intermedio"]
            },
            {
                "nombre": "Coursera",
                "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&free=true",
                "descripcion": "Cursos universitarios gratuitos (audit mode)",
                "niveles": ["Intermedio", "Avanzado"]
            },
            {
                "nombre": "edX",
                "url": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={tema.replace(' ', '%20')}",
                "descripcion": "Cursos de Harvard, MIT y otras universidades top",
                "niveles": ["Avanzado"]
            },
            {
                "nombre": "freeCodeCamp",
                "url": f"https://www.freecodecamp.org/news/search/?query={tema.replace(' ', '%20')}",
                "descripcion": "Certificados gratuitos en desarrollo web y ciencia de datos",
                "niveles": ["Intermedio", "Avanzado"]
            }
        ]
    
    # Filtrar por nivel
    niveles_permitidos = [nivel] if nivel != "Cualquiera" and nivel != "Todos" else ["Principiante", "Intermedio", "Avanzado"]
    
    for plat in plataformas:
        if any(nivel in plat["niveles"] for nivel in niveles_permitidos):
            nivel_seleccionado = next((nivel for nivel in niveles_permitidos if nivel in plat["niveles"]), "Intermedio")
            
            # Títulos realistas
            titulos = {
                "python": ["Curso Completo de Python", "Python para Principiantes", "Python Avanzado"],
                "machine learning": ["Introducción al Machine Learning", "ML para Data Science", "Deep Learning Completo"],
                "marketing": ["Marketing Digital Básico", "Estrategias Avanzadas de Marketing", "SEO y SEM"],
                "ingles": ["Inglés para Principiantes", "Conversación en Inglés", "Business English"],
                "diseño": ["Diseño Gráfico Básico", "UI/UX Design Profesional", "Adobe Creative Suite"],
                "finanzas": ["Finanzas Personales", "Inversión para Principiantes", "Análisis Financiero"]
            }
            
            tema_minus = tema.lower()
            titulo_base = f"Curso de {tema} para {nivel_seleccionado}"
            
            for clave, opciones in titulos.items():
                if clave in tema_minus:
                    titulo_base = random.choice(opciones)
                    break
            
            recurso = RecursoEducativo(
                id=generar_id_unico(plat["url"]),
                titulo=f"📚 {titulo_base} en {plat['nombre']}",
                url=plat["url"],
                descripcion=plat["descripcion"],
                plataforma=plat["nombre"],
                idioma=codigo_idioma,
                nivel=nivel_seleccionado,
                categoria=determinar_categoria(tema),
                certificacion=None,
                confianza=0.85,
                tipo="conocida",
                ultima_verificacion=datetime.now().isoformat(),
                activo=True,
                metadatos={"fuente": "busqueda_basica"}
            )
            resultados.append(recurso)
    
    return resultados[:8]  # Limitar a 8 resultados

# ----------------------------
# FUNCIONES DE MOSTRAR RESULTADOS (¡CORREGIDAS!)
# ----------------------------
def mostrar_recurso_basico(recurso: RecursoEducativo, index: int):
    """Muestra un recurso educativo en la interfaz"""
    
    # Clases CSS para estilos
    color_clase = {
        "Principiante": "nivel-principiante",
        "Intermedio": "nivel-intermedio", 
        "Avanzado": "nivel-avanzado"
    }.get(recurso.nivel, "")
    
    extra_class = "plataforma-oculta" if recurso.tipo == "oculta" else ""
    
    st.markdown(f"""
    <div class="resultado-card {color_clase} {extra_class} fade-in" style="animation-delay: {index * 0.1}s;">
        <h3>🎯 {recurso.titulo}</h3>
        <p><strong>📚 Nivel:</strong> {recurso.nivel} | <strong>🌐 Plataforma:</strong> {recurso.plataforma}</p>
        <p>📝 {recurso.descripcion}</p>
        
        <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
            <a href="{recurso.url}" target="_blank" style="flex: 1; min-width: 200px; 
                background: linear-gradient(to right, #6a11cb, #2575fc); color: white; 
                padding: 12px 20px; text-decoration: none; border-radius: 8px; 
                font-weight: bold; text-align: center; transition: all 0.3s ease;">
                ➡️ Acceder al Recurso
            </a>
        </div>
        
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee; font-size: 0.9rem; color: #666;">
            <p style="margin: 5px 0;">
                <strong>🔍 Confianza:</strong> {(recurso.confianza * 100):.1f}% | 
                <strong>✅ Verificado:</strong> {datetime.fromisoformat(recurso.ultima_verificacion).strftime('%d/%m/%Y')}
            </p>
            <p style="margin: 5px 0;">
                <strong>🌍 Idioma:</strong> {recurso.idioma.upper()} | 
                <strong>🏷️ Categoría:</strong> {recurso.categoria}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# CONFIGURACIÓN DE BASE DE DATOS SEGURA
# ----------------------------
DB_PATH = "cursos_inteligentes_v3.db"

def init_database():
    """Inicializa la base de datos de forma segura para Streamlit Cloud"""
    try:
        # Streamlit Cloud no permite escritura persistente, así que usamos una base de datos en memoria
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Crear tabla básica
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS plataformas_ocultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            url_base TEXT NOT NULL,
            descripcion TEXT,
            idioma TEXT NOT NULL,
            categoria TEXT,
            nivel TEXT,
            confianza REAL DEFAULT 0.7
        )
        ''')
        
        # Insertar datos de ejemplo
        plataformas_ejemplo = [
            ("Aprende con Alf", "https://aprendeconalf.es/?s={}", "Cursos gratuitos de programación y matemáticas", "es", "Programación", "Intermedio", 0.85),
            ("Coursera", "https://www.coursera.org/search?query={}&free=true", "Cursos universitarios gratuitos", "en", "General", "Avanzado", 0.95),
            ("edX", "https://www.edx.org/search?tab=course&availability=current&price=free&q={}", "Cursos de Harvard y MIT", "en", "Académico", "Avanzado", 0.92),
            ("Kaggle Learn", "https://www.kaggle.com/learn/search?q={}", "Microcursos de ciencia de datos", "en", "Data Science", "Intermedio", 0.90)
        ]
        
        cursor.executemany('''
        INSERT INTO plataformas_ocultas (nombre, url_base, descripcion, idioma, categoria, nivel, confianza)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', plataformas_ejemplo)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Base de datos en memoria inicializada correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al inicializar base de datos: {e}")
        return False

# ----------------------------
# ESTILOS RESPONSIVE (SIMPLIFICADOS)
# ----------------------------
st.set_page_config(
    page_title="🎓 Buscador de Cursos - Versión Lite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Estilos simplificados para mejor rendimiento */
    .main-header {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    
    .search-form {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        margin-bottom: 35px;
        border: 1px solid #e0e0e0;
    }
    
    .stButton button {
        background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(106, 17, 203, 0.4);
    }
    
    .resultado-card {
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        background: white;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        transition: all 0.4s ease;
        border-left: 6px solid #4CAF50;
    }
    
    .nivel-principiante { border-left-color: #2196F3 !important; }
    .nivel-intermedio { border-left-color: #4CAF50 !important; }
    .nivel-avanzado { border-left-color: #FF9800 !important; }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease forwards;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# INICIAR SISTEMA
# ----------------------------
init_database()

# ----------------------------
# INTERFAZ DE USUARIO PRINCIPAL
# ----------------------------
st.markdown("""
<div class="main-header fade-in">
    <h1>🎓 Buscador de Cursos - Versión Lite</h1>
    <p>Descubre recursos educativos gratuitos en múltiples idiomas con nuestra búsqueda optimizada</p>
</div>
""", unsafe_allow_html=True)

# Formulario de búsqueda
with st.container():
    st.markdown('<div class="search-form fade-in">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        tema = st.text_input("🔍 ¿Qué quieres aprender hoy?", 
                           placeholder="Ej: Python, Machine Learning, Diseño UX...",
                           key="tema_input")
    
    with col2:
        nivel = st.selectbox("📚 Nivel", 
                           ["Cualquiera", "Principiante", "Intermedio", "Avanzado"],
                           key="nivel_select")
    
    with col3:
        idioma_seleccionado = st.selectbox("🌍 Idioma", 
                                         ["Español (es)", "Inglés (en)", "Portugués (pt)"],
                                         key="idioma_select")
    
    buscar = st.button("🚀 Buscar Cursos", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# PROCESAR BÚSQUEDA
# ----------------------------
if buscar and tema.strip():
    with st.spinner("🔍 Buscando recursos educativos..."):
        try:
            # Usar búsqueda básica
            resultados = buscar_recursos_basicos(tema, idioma_seleccionado, nivel)
            
            if resultados:
                st.success(f"✅ ¡**{len(resultados)} recursos** encontrados para **{tema}** en **{idioma_seleccionado}**!")
                
                # Mostrar estadísticas
                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(resultados))
                col2.metric("Plataformas", len(set(r.plataforma for r in resultados)))
                col3.metric("Confianza Promedio", f"{sum(r.confianza for r in resultados) / len(resultados):.1%}")
                
                # Mostrar resultados
                st.markdown("### 📚 **Resultados Encontrados**")
                
                for i, resultado in enumerate(resultados):
                    time.sleep(0.1)
                    mostrar_recurso_basico(resultado, i)
                
                # Descarga de resultados
                st.markdown("---")
                df = pd.DataFrame([{
                    'titulo': r.titulo,
                    'url': r.url,
                    'plataforma': r.plataforma,
                    'nivel': r.nivel,
                    'idioma': r.idioma,
                    'categoria': r.categoria,
                    'confianza': f"{r.confianza:.1%}"
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
                st.warning("⚠️ No encontramos recursos para este tema. Intenta con otro término de búsqueda.")
        
        except Exception as e:
            logger.error(f"Error durante la búsqueda: {e}")
            st.error("❌ Ocurrió un error durante la búsqueda. Por favor, intenta nuevamente.")
            st.exception(e)

# ----------------------------
# SECCIÓN DE EJEMPLOS
# ----------------------------
else:
    st.info("💡 **Sistema listo para buscar**. Ingresa un tema, selecciona nivel e idioma para descubrir recursos educativos gratuitos.")
    
    # Ejemplos recomendados
    st.markdown("### 🚀 **Temas Populares**")
    
    ejemplos = {
        "es": [
            {"tema": "Python para Principiantes", "nivel": "Principiante"},
            {"tema": "Machine Learning", "nivel": "Intermedio"},
            {"tema": "Diseño UX/UI", "nivel": "Avanzado"}
        ],
        "en": [
            {"tema": "Python Programming", "nivel": "Principiante"},
            {"tema": "Data Science", "nivel": "Intermedio"},
            {"tema": "Web Development", "nivel": "Avanzado"}
        ],
        "pt": [
            {"tema": "Programação em Python", "nivel": "Principiante"},
            {"tema": "Ciência de Dados", "nivel": "Intermedio"},
            {"tema": "Marketing Digital", "nivel": "Intermedio"}
        ]
    }
    
    tabs = st.tabs(["🇪🇸 Español", "🇬🇧 Inglés", "🇵🇹 Portugués"])
    
    for tab_idx, (idioma_codigo, ejemplos_lista) in enumerate(ejemplos.items()):
        with tabs[tab_idx]:
            for ejemplo in ejemplos_lista:
                if st.button(f"🔍 Buscar: {ejemplo['tema']}", key=f"ejemplo_{tab_idx}_{ejemplo['tema']}", use_container_width=True):
                    st.session_state.tema_input = ejemplo['tema']
                    st.session_state.nivel_select = ejemplo['nivel']
                    st.session_state.idioma_select = [k for k, v in {
                        "es": "Español (es)",
                        "en": "Inglés (en)",
                        "pt": "Portugués (pt)"
                    }.items() if v == idioma_codigo][0]
                    st.experimental_rerun()

# ----------------------------
# PIE DE PÁGINA
# ----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; padding: 25px; background: #f8f9fa; border-radius: 15px; margin-top: 20px;">
    <strong>✨ Buscador de Cursos - Versión Lite</strong><br>
    <span style="color: #2c3e50;">Sistema de búsqueda optimizado para Streamlit Cloud</span><br>
    <em style="color: #7f8c8d;">Última actualización: {} • Versión: 1.0.0 • Estado: ✅ Activo</em>
</div>
""".format(datetime.now().strftime('%d/%m/%Y %H:%M')), unsafe_allow_html=True)

logger.info("✅ Sistema de búsqueda de cursos iniciado correctamente")
logger.info("⚡ Versión optimizada para Streamlit Cloud - Sin dependencias problemáticas")
