# app.py — Buscador de Cursos SG1 "Enterprise Edition"
# ==============================================================================
# AUTOR: Generative AI Assistant
# VERSIÓN: 4.0.0 (Release Candidate)
# LICENCIA: MIT
# ==============================================================================
# DESCRIPCIÓN:
# Sistema integral de búsqueda, análisis y gestión de recursos educativos.
# Incluye arquitectura asíncrona, gestión de base de datos relacional,
# análisis de IA mediante Groq, panel de administración, y analítica de datos.
# ==============================================================================

import streamlit as st
import pandas as pd
import sqlite3
import os
import time
import random
import json
import hashlib
import re
import logging
import asyncio
import aiohttp
import contextlib
import base64
import csv
import io
from datetime import datetime, timedelta
from urllib.parse import urlparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# ==============================================================================
# 1. CONFIGURACIÓN DEL SISTEMA Y LOGGING
# ==============================================================================

# Configuración de Logging Avanzada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s : %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('system_sg1.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SG1_Enterprise")

# Configuración de la Página de Streamlit
st.set_page_config(
    page_title="SG1 Enterprise Learning Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tuusuario/sg1-help',
        'Report a bug': "https://github.com/tuusuario/sg1-issues",
        'About': "SG1 Enterprise v4.0 - Sistema de Inteligencia Educativa"
    }
)

# Estilos CSS Profesionales (Dark/Light Mode Compatible)
st.markdown("""
<style>
    /* Variables Globales */
    :root {
        --primary-color: #4b6cb7;
        --secondary-color: #182848;
        --accent-color: #ff6b6b;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --bg-light: #f8f9fa;
        --text-dark: #2c3e50;
    }

    /* Header Principal */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        max-width: 600px;
    }

    /* Tarjetas de Resultados */
    .resource-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .resource-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .resource-card h3 {
        color: var(--text-dark);
        margin-top: 0;
        font-weight: 700;
    }
    
    /* Variantes de Nivel */
    .card-beginner { border-left-color: #2196F3; }
    .card-intermediate { border-left-color: #4CAF50; }
    .card-advanced { border-left-color: #9C27B0; }
    .card-special { border-left-color: #FF9800; background-color: #fffbf0; }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 5px;
    }
    .badge-free { background-color: #e8f5e9; color: #2e7d32; }
    .badge-paid { background-color: #ffebee; color: #c62828; }
    .badge-ai { background-color: #f3e5f5; color: #7b1fa2; border: 1px solid #e1bee7; }

    /* Botones Personalizados */
    .btn-access {
        display: inline-block;
        background: linear-gradient(90deg, #4b6cb7 0%, #2575fc 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        transition: opacity 0.3s;
        text-align: center;
        border: none;
        cursor: pointer;
    }
    .btn-access:hover {
        opacity: 0.9;
        text-decoration: none;
    }

    /* Métricas del Dashboard */
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: var(--primary-color); }
    .metric-label { font-size: 0.9rem; color: #666; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONSTANTES Y ENUMS
# ==============================================================================

class Language(Enum):
    ES = "es"
    EN = "en"
    PT = "pt"

class Level(Enum):
    ANY = "Cualquiera"
    BEGINNER = "Principiante"
    INTERMEDIATE = "Intermedio"
    ADVANCED = "Avanzado"

DB_PATH = "sg1_enterprise_v4.db"
CACHE_TTL = 43200  # 12 horas
GROQ_MODEL_ID = "llama-3.1-70b-versatile"
MAX_WORKERS = 4

# ==============================================================================
# 3. GESTOR DE SEGURIDAD Y CREDENCIALES
# ==============================================================================

class SecurityManager:
    """Gestiona claves API, validaciones y acceso seguro."""
    
    @staticmethod
    def get_credentials() -> Tuple[str, str, str]:
        """Recupera credenciales de Secrets o Variables de Entorno."""
        try:
            g_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
            g_cx = st.secrets.get("GOOGLE_CX", os.getenv("GOOGLE_CX", ""))
            groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
            return g_key, g_cx, groq_key
        except Exception as e:
            logger.error(f"Error recuperando credenciales: {e}")
            return "", "", ""

    @staticmethod
    def validate_api_key(key: str, service: str) -> bool:
        """Validación heurística de formato de API Keys."""
        if not key or len(key) < 8:
            return False
        if service == "google" and not key.startswith(("AIza", "AIz")):
            return False
        if service == "groq" and not key.startswith(("gsk_", "groq_")):
            # Algunas keys de groq empiezan con gsk_, otras son diferentes, validación laxa
            return len(key) > 20
        return True

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash simple para simulación de auth (usar bcrypt en prod real)."""
        return hashlib.sha256(password.encode()).hexdigest()

GOOGLE_API_KEY, GOOGLE_CX, GROQ_API_KEY = SecurityManager.get_credentials()

# ==============================================================================
# 4. GESTOR DE BASE DE DATOS (ORM LIGHT)
# ==============================================================================

@contextlib.contextmanager
def db_connection():
    """Context Manager para conexiones seguras a SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Permite acceder a columnas por nombre
        yield conn
    except sqlite3.Error as e:
        logger.critical(f"Error crítico de BD: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

class DatabaseManager:
    """Clase Singleton para manejo de todas las operaciones de BD."""
    
    @staticmethod
    def init_db():
        """Inicializa el esquema de la base de datos completo."""
        with db_connection() as conn:
            cursor = conn.cursor()
            
            # Tabla: Plataformas Ocultas (Recursos curados)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS plataformas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL UNIQUE,
                url_base TEXT NOT NULL,
                descripcion TEXT,
                idioma TEXT DEFAULT 'en',
                categoria TEXT DEFAULT 'General',
                nivel TEXT DEFAULT 'Todos',
                confianza REAL DEFAULT 0.8,
                activa INTEGER DEFAULT 1,
                tipo_certificacion TEXT DEFAULT 'none',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Tabla: Historial de Búsquedas (Analytics)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS historial_busquedas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id TEXT,
                tema TEXT,
                idioma TEXT,
                nivel TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                resultados_count INTEGER
            )
            ''')
            
            # Tabla: Favoritos de Usuarios
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS favoritos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id TEXT,
                recurso_id TEXT,
                titulo TEXT,
                url TEXT,
                plataforma TEXT,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(usuario_id, url)
            )
            ''')

            # Seed Data (Datos Semilla)
            DatabaseManager._seed_platforms(cursor)
            conn.commit()

    @staticmethod
    def _seed_platforms(cursor):
        """Poblar base de datos con datos iniciales si está vacía."""
        cursor.execute("SELECT COUNT(*) FROM plataformas")
        if cursor.fetchone()[0] == 0:
            platforms = [
                ("Aprende con Alf", "https://aprendeconalf.es/?s={}", "Tutoriales Python/Data", "es", "Programación", 0.9, "gratuito"),
                ("Coursera Free", "https://www.coursera.org/search?query={}&free=true", "Cursos universitarios", "en", "General", 0.95, "audit"),
                ("edX Free", "https://www.edx.org/search?q={}&tab=course", "Cursos Harvard/MIT", "en", "Académico", 0.92, "audit"),
                ("Kaggle Learn", "https://www.kaggle.com/learn", "Data Science práctico", "en", "Data Science", 0.94, "gratuito"),
                ("FreeCodeCamp", "https://www.freecodecamp.org/news/search/?query={}", "Desarrollo Web Fullstack", "en", "Programación", 0.96, "gratuito"),
                ("Khan Academy ES", "https://es.khanacademy.org/search?page_search_query={}", "Matemáticas y Ciencias", "es", "Ciencias", 0.93, "gratuito"),
                ("MIT OpenCourseWare", "https://ocw.mit.edu/search/?q={}", "Material académico MIT", "en", "Académico", 0.98, "gratuito"),
                ("Mozilla MDN", "https://developer.mozilla.org/es/search?q={}", "Documentación Web", "es", "Programación", 0.97, "gratuito"),
                ("Scikit-Learn Doc", "https://scikit-learn.org/stable/search.html?q={}", "ML Documentation", "en", "Data Science", 0.95, "gratuito"),
                ("Real Python", "https://realpython.com/search?q={}", "Tutoriales Python Pro", "en", "Programación", 0.88, "pago")
            ]
            cursor.executemany('''
                INSERT OR IGNORE INTO plataformas (nombre, url_base, descripcion, idioma, categoria, confianza, tipo_certificacion)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', platforms)
            logger.info("✅ Base de datos poblada con datos semilla.")

    @staticmethod
    def log_search(tema: str, idioma: str, nivel: str, count: int):
        """Registra una búsqueda para análisis posterior."""
        try:
            user_id = st.session_state.get('user_id', 'guest')
            with db_connection() as conn:
                conn.execute(
                    "INSERT INTO historial_busquedas (usuario_id, tema, idioma, nivel, resultados_count) VALUES (?, ?, ?, ?, ?)",
                    (user_id, tema, idioma, nivel, count)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"No se pudo loguear la búsqueda: {e}")

    @staticmethod
    def add_favorite(recurso: 'RecursoEducativo'):
        """Guarda un recurso en favoritos."""
        try:
            user_id = st.session_state.get('user_id', 'guest')
            with db_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO favoritos (usuario_id, recurso_id, titulo, url, plataforma) VALUES (?, ?, ?, ?, ?)",
                    (user_id, recurso.id, recurso.titulo, recurso.url, recurso.plataforma)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error guardando favorito: {e}")
            return False

    @staticmethod
    def get_favorites() -> List[Dict]:
        """Obtiene favoritos del usuario actual."""
        user_id = st.session_state.get('user_id', 'guest')
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM favoritos WHERE usuario_id = ? ORDER BY added_at DESC", (user_id,))
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_platform_stats() -> Dict[str, int]:
        """Obtiene estadísticas para el dashboard."""
        stats = {}
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM plataformas")
            stats['total_platforms'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM historial_busquedas")
            stats['total_searches'] = cursor.fetchone()[0]
            # Top tema
            cursor.execute("SELECT tema, COUNT(*) as c FROM historial_busquedas GROUP BY tema ORDER BY c DESC LIMIT 1")
            row = cursor.fetchone()
            stats['top_trend'] = row[0] if row else "N/A"
        return stats

# Inicialización de DB al importar
DatabaseManager.init_db()

# ==============================================================================
# 5. MODELOS DE DATOS (DATACLASSES)
# ==============================================================================

@dataclass
class Certificacion:
    tipo: str  # gratuito, audit, pago
    validez_global: bool = False
    costo: float = 0.0

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
    confianza: float
    certificacion: Optional[Certificacion] = None
    metadatos_ai: Optional[Dict] = None
    analisis_pendiente: bool = False
    
    def to_dict(self):
        return asdict(self)

# ==============================================================================
# 6. MOTOR DE INTELIGENCIA ARTIFICIAL (ASYNC WRAPPER)
# ==============================================================================

class AIWorker:
    """Maneja la comunicación asíncrona con LLMs (Groq)."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available = SecurityManager.validate_api_key(api_key, "groq")
        # No instanciamos el cliente aquí para evitar problemas de contexto en hilos

    async def analyze_course(self, recurso: RecursoEducativo, user_profile: Dict) -> Dict:
        """Analiza un curso específico usando el modelo LLM."""
        if not self.available:
            return {}

        import groq  # Importación lazy
        
        prompt = f"""
        Actúa como un consejero académico experto. Analiza el siguiente curso:
        
        TITULO: {recurso.titulo}
        DESCRIPCIÓN: {recurso.descripcion}
        PLATAFORMA: {recurso.plataforma}
        NIVEL DETECTADO: {recurso.nivel}
        
        PERFIL USUARIO: {user_profile}
        
        Devuelve estrictamente un objeto JSON con estas claves:
        {{
            "calidad_score": (float 0-1),
            "match_perfil": (float 0-1),
            "pros": [lista breve],
            "contras": [lista breve],
            "veredicto": "string corto",
            "tags_sugeridos": [lista]
        }}
        """
        
        # Ejecución en ThreadPool para no bloquear el Event Loop principal
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        try:
            def _sync_call():
                client = groq.Groq(api_key=self.api_key)
                return client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Eres un API que solo responde JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    model=GROQ_MODEL_ID,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )

            response = await loop.run_in_executor(executor, _sync_call)
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Fallo en análisis IA para {recurso.id}: {e}")
            return {"veredicto": "Análisis no disponible", "calidad_score": 0.5}
        finally:
            executor.shutdown(wait=False)

    async def chat_interaction(self, history: List[Dict]) -> str:
        """Maneja el chat educativo en el sidebar."""
        if not self.available: return "IA no configurada."
        
        import groq
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        try:
            def _call():
                client = groq.Groq(api_key=self.api_key)
                # Filtramos historial para evitar tokens excesivos
                msgs = [{"role": "system", "content": "Eres SG1-Bot, un experto en educación. Sé conciso y útil."}] + history[-5:]
                return client.chat.completions.create(
                    messages=msgs,
                    model=GROQ_MODEL_ID,
                    temperature=0.5,
                    max_tokens=500
                )
            resp = await loop.run_in_executor(executor, _call)
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error en chat: {str(e)}"

# Instancia global del Worker IA
ai_worker = AIWorker(GROQ_API_KEY)

# ==============================================================================
# 7. MOTOR DE BÚSQUEDA MULTICAPA
# ==============================================================================

class SearchEngine:
    """Motor de búsqueda que coordina múltiples fuentes de datos."""

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r'<[^>]+>', '', text).strip()

    @staticmethod
    def _generate_id(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]

    @staticmethod
    def _determine_level(text: str, target: str) -> str:
        text = text.lower()
        if target != Level.ANY.value: return target
        if any(x in text for x in ['intro', 'principiante', 'basic', '101', 'start']): return Level.BEGINNER.value
        if any(x in text for x in ['advance', 'expert', 'master', 'deep']): return Level.ADVANCED.value
        return Level.INTERMEDIATE.value

    @staticmethod
    def _is_valid_resource(url: str, text: str) -> bool:
        """Filtrado heurístico de spam o contenido pago agresivo."""
        blacklist = ['buy now', 'pricing', 'login', 'signup', 'cart', 'checkout']
        whitelist_domains = ['edu', 'org', 'gov', 'coursera', 'edx', 'udemy', 'youtube', 'kaggle']
        
        text_lower = text.lower()
        url_lower = url.lower()
        
        if any(b in text_lower for b in blacklist): return False
        if not any(d in url_lower for d in whitelist_domains) and 'course' not in text_lower: return False
        return True

    @staticmethod
    async def search_google(query: str, lang: str) -> List[RecursoEducativo]:
        """Búsqueda asíncrona en Google Custom Search API."""
        if not SecurityManager.validate_api_key(GOOGLE_API_KEY, "google") or not GOOGLE_CX:
            return []
        
        results = []
        api_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CX,
            'q': f"{query} course tutorial education",
            'lr': f'lang_{lang}',
            'num': 6,
            'safe': 'active'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get('items', []):
                            if SearchEngine._is_valid_resource(item['link'], item.get('snippet', '')):
                                results.append(RecursoEducativo(
                                    id=SearchEngine._generate_id(item['link']),
                                    titulo=item['title'],
                                    url=item['link'],
                                    descripcion=item.get('snippet', ''),
                                    plataforma=urlparse(item['link']).netloc.replace('www.', '').capitalize(),
                                    idioma=lang,
                                    nivel=Level.ANY.value, # Se refinará después
                                    categoria="General",
                                    confianza=0.85,
                                    certificacion=None,
                                    activo=True
                                ))
        except Exception as e:
            logger.warning(f"Google API Error: {e}")
        
        return results

    @staticmethod
    def search_internal_db(query: str, lang: str) -> List[RecursoEducativo]:
        """Búsqueda en la base de datos local curada."""
        results = []
        try:
            with db_connection() as conn:
                cursor = conn.cursor()
                # Búsqueda difusa simple
                sql = """
                    SELECT * FROM plataformas 
                    WHERE (nombre LIKE ? OR descripcion LIKE ?) 
                    AND idioma = ? AND activa = 1
                """
                like_q = f"%{query}%"
                cursor.execute(sql, (like_q, like_q, lang))
                rows = cursor.fetchall()
                
                for row in rows:
                    final_url = row['url_base'].format(query.replace(' ', '+'))
                    cert = Certificacion(tipo=row['tipo_certificacion'])
                    results.append(RecursoEducativo(
                        id=SearchEngine._generate_id(final_url),
                        titulo=f"📚 {row['nombre']} - {query.title()}",
                        url=final_url,
                        descripcion=row['descripcion'] or f"Recurso verificado en {row['nombre']}",
                        plataforma=row['nombre'],
                        idioma=row['idioma'],
                        nivel=row['nivel'],
                        categoria=row['categoria'],
                        confianza=row['confianza'],
                        certificacion=cert,
                        activo=True,
                        analisis_pendiente=True # Candidato a análisis IA
                    ))
        except Exception as e:
            logger.error(f"DB Search Error: {e}")
        return results

    @staticmethod
    def search_known_platforms(query: str, lang: str) -> List[RecursoEducativo]:
        """Generador de enlaces profundos a plataformas masivas."""
        # Lógica mejorada del código B
        platforms_map = {
            "es": [
                ("YouTube Edu", f"https://www.youtube.com/results?search_query=curso+{query}"),
                ("Udemy Free", f"https://www.udemy.com/courses/search/?q={query}&price=price-free&lang=es"),
                ("Coursera ES", f"https://www.coursera.org/search?query={query}&language=es")
            ],
            "en": [
                ("YouTube Edu", f"https://www.youtube.com/results?search_query=course+{query}"),
                ("MIT OCW", f"https://ocw.mit.edu/search/?q={query}"),
                ("Stanford Online", f"https://online.stanford.edu/search?search_api_fulltext={query}")
            ],
            "pt": [
                ("YouTube BR", f"https://www.youtube.com/results?search_query=curso+{query}"),
                ("Udemy PT", f"https://www.udemy.com/courses/search/?q={query}&lang=pt")
            ]
        }
        
        target_list = platforms_map.get(lang, platforms_map["en"])
        results = []
        for name, url in target_list:
            results.append(RecursoEducativo(
                id=SearchEngine._generate_id(url),
                titulo=f"🌐 {name}: {query}",
                url=url,
                descripcion=f"Búsqueda directa en catálogo de {name}",
                plataforma=name,
                idioma=lang,
                nivel=Level.ANY.value,
                categoria="General",
                confianza=0.80,
                activo=True
            ))
        return results

    @staticmethod
    async def execute_multilayer_search(query: str, lang: str, level: str) -> List[RecursoEducativo]:
        """Orquestador maestro de búsqueda."""
        
        # 1. Búsqueda en paralelo (DB Local + Google API)
        # La DB local es síncrona pero rápida, Google es Async
        internal_task = asyncio.to_thread(SearchEngine.search_internal_db, query, lang)
        google_task = SearchEngine.search_google(query, lang)
        
        results_internal, results_google = await asyncio.gather(internal_task, google_task)
        
        # 2. Generar enlaces conocidos (Fallback)
        results_known = SearchEngine.search_known_platforms(query, lang)
        
        # 3. Fusión y Deduplicación
        all_results = results_internal + results_google + results_known
        unique_results = {}
        for r in all_results:
            if r.url not in unique_results:
                # Refinamiento de nivel post-búsqueda
                r.nivel = SearchEngine._determine_level(f"{r.titulo} {r.descripcion}", level)
                if level != Level.ANY.value and r.nivel != level and r.confianza < 0.9:
                    continue # Filtrado suave
                unique_results[r.url] = r
                
        final_list = list(unique_results.values())
        
        # 4. Ordenamiento por confianza
        final_list.sort(key=lambda x: x.confianza, reverse=True)
        
        return final_list[:15] # Top 15

# ==============================================================================
# 8. COMPONENTES DE UI (RENDERIZADO)
# ==============================================================================

class UIRenderer:
    """Maneja la presentación visual de componentes."""

    @staticmethod
    def render_link_button(url: str, text: str = "Acceder al Recurso"):
        return f'''
        <a href="{url}" target="_blank" class="btn-access">
            {text} <span style="margin-left:5px;">🚀</span>
        </a>
        '''

    @staticmethod
    def render_badges(recurso: RecursoEducativo) -> str:
        html = ""
        if recurso.certificacion:
            if recurso.certificacion.tipo == "gratuito":
                html += '<span class="badge badge-free">Gratis</span>'
            elif recurso.certificacion.tipo == "pago":
                html += '<span class="badge badge-paid">Pago</span>'
            elif recurso.certificacion.tipo == "audit":
                html += '<span class="badge badge-ai">Audit Mode</span>'
        
        if recurso.metadatos_ai:
            score = int(recurso.metadatos_ai.get('calidad_score', 0) * 100)
            html += f'<span class="badge badge-ai">IA Score: {score}%</span>'
            
        return html

    @staticmethod
    def render_resource_card(r: RecursoEducativo, index: int):
        """Renderiza una tarjeta de recurso con HTML/CSS puro para evitar bugs de layout."""
        
        # Determinar clase de estilo según nivel
        level_class = "card-intermediate"
        if r.nivel == Level.BEGINNER.value: level_class = "card-beginner"
        elif r.nivel == Level.ADVANCED.value: level_class = "card-advanced"
        elif r.tipo == "oculta": level_class = "card-special"

        badges_html = UIRenderer.render_badges(r)
        button_html = UIRenderer.render_link_button(r.url)
        
        # Bloque de IA si existe
        ai_section = ""
        if r.metadatos_ai:
            veredicto = r.metadatos_ai.get('veredicto', 'Sin veredicto')
            ai_section = f"""
            <div style="margin-top:10px; padding:10px; background:#f0f2f6; border-radius:8px; font-size:0.9rem;">
                <strong>🤖 Análisis IA:</strong> {veredicto}
            </div>
            """
        elif r.analisis_pendiente:
            ai_section = f"""
            <div style="margin-top:10px; font-size:0.8rem; color:#666; font-style:italic;">
                ⏳ Analizando contenido...
            </div>
            """

        # HTML Minificado y limpio (sin indentación interna que rompa markdown)
        html_card = f"""
<div class="resource-card {level_class}" style="animation: fadeIn 0.5s ease forwards; animation-delay: {index * 0.1}s;">
    <div style="display:flex; justify-content:space-between; align-items:start;">
        <div>
            <h3>{r.titulo}</h3>
            <div style="margin-bottom:10px;">
                <span style="color:#666; font-size:0.9rem;">🏢 {r.plataforma}</span>
                <span style="color:#666; font-size:0.9rem; margin-left:10px;">📚 {r.nivel}</span>
            </div>
        </div>
        <div style="text-align:right;">
            {badges_html}
        </div>
    </div>
    <p style="color:#444; line-height:1.5;">{r.descripcion}</p>
    {ai_section}
    <div style="margin-top:15px; display:flex; justify-content:space-between; align-items:center;">
        {button_html}
        <span style="font-size:0.8rem; color:#888;">Confianza: {int(r.confianza*100)}%</span>
    </div>
</div>
"""
        st.markdown(html_card, unsafe_allow_html=True)
        
        # Botón de favorito (Streamlit native button outside HTML)
        col1, col2 = st.columns([0.85, 0.15])
        with col2:
            if st.button("❤️", key=f"fav_{r.id}", help="Guardar en favoritos"):
                if DatabaseManager.add_favorite(r):
                    st.toast(f"Guardado: {r.titulo}")

    @staticmethod
    def clean_chat_text(text: str) -> str:
        """Limpieza rigurosa de texto para el chat."""
        if not text: return ""
        text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL) # Quitar JSON
        text = re.sub(r'<[^>]*>', '', text) # Quitar HTML tags
        return text.strip()

# ==============================================================================
# 9. LÓGICA DE APLICACIÓN PRINCIPAL (MAIN LOOP)
# ==============================================================================

def main():
    # Inicialización de Sesión
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{random.randint(1000, 9999)}"
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []

    # --- SIDEBAR: NAVEGACIÓN Y CHAT ---
    with st.sidebar:
        st.title("🚀 Navegación")
        page = st.radio("Ir a:", ["Buscador", "Mis Favoritos", "Analytics Dashboard", "Admin Panel"])
        
        st.divider()
        
        st.subheader("💬 Asistente IA")
        chat_container = st.container()
        
        # Renderizado de chat
        with chat_container:
            for msg in st.session_state.chat_history:
                role_icon = "👤" if msg['role'] == "user" else "🤖"
                st.markdown(f"**{role_icon}**: {UIRenderer.clean_chat_text(msg['content'])}")
        
        user_input = st.chat_input("Pregúntame algo...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Llamada Asíncrona simulada en botón síncrono (Trick)
            if GROQ_AVAILABLE:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(ai_worker.chat_interaction(st.session_state.chat_history))
                loop.close()
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            else:
                st.warning("IA no disponible. Configura GROQ_API_KEY.")

    # --- PÁGINA: BUSCADOR (HOME) ---
    if page == "Buscador":
        st.markdown('<div class="main-header"><h1>🎓 SG1 Enterprise Search</h1><p>El buscador de cursos más avanzado del mercado.</p></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            query = st.text_input("Tema de búsqueda", placeholder="Ej: Machine Learning, React, Python...")
        with col2:
            lang_opt = st.selectbox("Idioma", [l.value for l in Language], index=0)
        with col3:
            level_opt = st.selectbox("Nivel", [l.value for l in Level], index=0)
        with col4:
            st.write("") 
            st.write("")
            search_btn = st.button("Buscar 🔍", type="primary", use_container_width=True)

        if search_btn and query:
            with st.spinner("🚀 Iniciando motores de búsqueda multicapa..."):
                # Ejecución Asíncrona
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(
                    SearchEngine.execute_multilayer_search(query, lang_opt, level_opt)
                )
                loop.close()
                
                st.session_state.search_results = results
                
                # Loguear búsqueda
                DatabaseManager.log_search(query, lang_opt, level_opt, len(results))
                
                # Disparar análisis en background (Fake thread trigger for Streamlit Cloud compatibility)
                if GROQ_AVAILABLE:
                    # En una app real usaríamos Celery/Redis. Aquí usamos un ThreadPool simple.
                    def background_analysis(res_list):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        for r in res_list[:3]: # Solo analizar top 3 para ahorrar quota
                            if r.analisis_pendiente:
                                analysis = loop.run_until_complete(ai_worker.analyze_course(r, {"nivel": level_opt}))
                                r.metadatos_ai = analysis
                                r.analisis_pendiente = False
                        loop.close()
                    
                    executor = ThreadPoolExecutor(max_workers=1)
                    executor.submit(background_analysis, st.session_state.search_results)
                    st.toast("Analizando resultados con IA en segundo plano...")

        # Mostrar Resultados
        if st.session_state.search_results:
            st.success(f"Encontrados {len(st.session_state.search_results)} recursos relevantes.")
            
            # Exportar
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Titulo", "URL", "Plataforma", "Confianza"])
            for r in st.session_state.search_results:
                writer.writerow([r.titulo, r.url, r.plataforma, r.confianza])
            
            st.download_button(
                label="📥 Descargar Reporte CSV",
                data=csv_buffer.getvalue(),
                file_name="sg1_report.csv",
                mime="text/csv"
            )

            for i, r in enumerate(st.session_state.search_results):
                UIRenderer.render_resource_card(r, i)
        
        elif search_btn:
            st.warning("No se encontraron resultados. Intenta ampliar tus términos de búsqueda.")

    # --- PÁGINA: FAVORITOS ---
    elif page == "Mis Favoritos":
        st.title("❤️ Mis Cursos Guardados")
        favs = DatabaseManager.get_favorites()
        if favs:
            for f in favs:
                st.info(f"**[{f['plataforma']}]** {f['titulo']} - [Abrir]({f['url']})")
        else:
            st.write("Aún no tienes favoritos.")

    # --- PÁGINA: ANALYTICS DASHBOARD ---
    elif page == "Analytics Dashboard":
        st.title("📊 Panel de Control")
        stats = DatabaseManager.get_platform_stats()
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-container'><div class='metric-value'>{stats['total_platforms']}</div><div class='metric-label'>Plataformas Indexadas</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-container'><div class='metric-value'>{stats['total_searches']}</div><div class='metric-label'>Búsquedas Totales</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-container'><div class='metric-value' style='font-size:1.5rem; padding-top:10px;'>{stats['top_trend']}</div><div class='metric-label'>Tendencia #1</div></div>", unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("Actividad Reciente")
        with db_connection() as conn:
            df = pd.read_sql("SELECT tema, idioma, timestamp FROM historial_busquedas ORDER BY timestamp DESC LIMIT 50", conn)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df['tema'].value_counts())

    # --- PÁGINA: ADMIN PANEL ---
    elif page == "Admin Panel":
        st.title("🛠️ Administración del Sistema")
        pwd = st.text_input("Contraseña de Administrador", type="password")
        if pwd == "admin123": # Mock password
            st.success("Acceso Concedido")
            
            with st.expander("➕ Agregar Nueva Plataforma Manualmente"):
                with st.form("add_plat"):
                    name = st.text_input("Nombre")
                    url = st.text_input("URL Base (con {})")
                    cat = st.selectbox("Categoría", ["Programación", "Data Science", "General", "Idiomas"])
                    submitted = st.form_submit_button("Guardar en BD")
                    if submitted:
                        with db_connection() as conn:
                            conn.execute("INSERT INTO plataformas (nombre, url_base, categoria) VALUES (?, ?, ?)", (name, url, cat))
                            conn.commit()
                        st.success("Plataforma agregada.")
            
            st.subheader("Base de Datos Actual")
            with db_connection() as conn:
                df = pd.read_sql("SELECT id, nombre, categoria, confianza FROM plataformas", conn)
                st.dataframe(df)
        elif pwd:
            st.error("Contraseña incorrecta")

# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================
if __name__ == "__main__":
    main()
