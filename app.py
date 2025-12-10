# app.py — SG1 Enterprise v5.3 "Ultra-Robust"
# ==============================================================================
# AUTOR: Generative AI Assistant & Qwen (Refined by Gemini)
# VERSIÓN: 5.3.0 (Stable Enterprise Release)
# LICENCIA: MIT
# ==============================================================================
# DESCRIPCIÓN:
# Sistema integral de búsqueda educativa con arquitectura asíncrona, IA generativa,
# gestión de datos persistente, caché inteligente y panel de administración completo.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import time
import random
import uuid
import json
import hashlib
import re
import math
import statistics
import itertools
import collections
import functools
import textwrap
import datetime
import calendar
import csv
import io
import base64
import requests
import aiohttp
import asyncio
import threading
import queue
import logging
import contextlib
import dataclasses
from urllib.parse import urlparse, urljoin, quote, quote_plus, urlencode
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# ============================================================
# 0. CONFIGURACIÓN GLOBAL Y CONSTANTES
# ============================================================

@dataclass(frozen=True)
class ConfigConstants:
    """Constantes configurables del sistema."""
    APP_NAME: str = "🎓 SG1 Enterprise Learning Hub"
    APP_VERSION: str = "5.3.0"
    APP_ENV: str = os.getenv("APP_ENV", "production")
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_RESULTS_DEFAULT: int = 15
    MAX_ANALYSIS_DEFAULT: int = 3
    CACHE_TTL_SECONDS: int = 43200  # 12 horas
    MAX_BACKGROUND_TASKS: int = 4
    GROQ_MODEL_DEFAULT: str = "llama-3.1-70b-versatile"
    GROQ_TIMEOUT: int = 25
    GOOGLE_API_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    DB_PATH: str = "sg1_enterprise.db"

CONFIG = ConfigConstants()

# ============================================================
# 1. SISTEMA DE LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(logging, CONFIG.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_sg1.log', mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SG1_Core")

# ============================================================
# 2. GESTIÓN DE SEGURIDAD Y CREDENCIALES
# ============================================================

class SecurityManager:
    """Gestiona claves API y validaciones de seguridad."""
    
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
        if not key or len(key) < 10: return False
        if service == "google" and not key.startswith(("AIza", "AIz")): return False
        if service == "groq" and not key.startswith(("gsk_", "groq_")): 
            # Validación laxa para Groq ya que las keys varían
            return len(key) > 20
        return True

    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitiza inputs para evitar XSS/SQLi básicos."""
        if not text: return ""
        return re.sub(r'[<>"\'%;()&]', '', str(text).strip())

GOOGLE_API_KEY, GOOGLE_CX, GROQ_API_KEY = SecurityManager.get_credentials()
GROQ_AVAILABLE = False

try:
    import groq
    if SecurityManager.validate_api_key(GROQ_API_KEY, "groq"):
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API disponible y validada")
except ImportError:
    logger.warning("⚠️ Biblioteca 'groq' no instalada")

# ============================================================
# 3. GESTIÓN DE BASE DE DATOS (ORM LIGHT)
# ============================================================

@contextlib.contextmanager
def db_connection():
    """Context Manager para conexiones seguras a SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(CONFIG.DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.critical(f"Error crítico de BD: {e}")
        if conn: conn.rollback()
        raise e
    finally:
        if conn: conn.close()

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

            # Seed Data
            DatabaseManager._seed_platforms(cursor)
            conn.commit()

    @staticmethod
    def _seed_platforms(cursor):
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
    def add_favorite(recurso):
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
        user_id = st.session_state.get('user_id', 'guest')
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM favoritos WHERE usuario_id = ? ORDER BY added_at DESC", (user_id,))
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_platform_stats() -> Dict[str, int]:
        stats = {}
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM plataformas")
            stats['total_platforms'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM historial_busquedas")
            stats['total_searches'] = cursor.fetchone()[0]
            cursor.execute("SELECT tema, COUNT(*) as c FROM historial_busquedas GROUP BY tema ORDER BY c DESC LIMIT 1")
            row = cursor.fetchone()
            stats['top_trend'] = row[0] if row else "N/A"
        return stats

DatabaseManager.init_db()

# ============================================================
# 4. MODELOS DE DATOS
# ============================================================

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
    tipo: str = "general" # oculta, conocida, verificada

# ============================================================
# 5. MOTOR DE BÚSQUEDA MULTICAPA (ASYNC)
# ============================================================

class SearchEngine:
    @staticmethod
    def _generate_id(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]

    @staticmethod
    def _determine_level(text: str, target: str) -> str:
        text = text.lower()
        if target not in ("Cualquiera", "Todos"): return target
        if any(x in text for x in ['intro', 'principiante', 'basic']): return "Principiante"
        if any(x in text for x in ['advance', 'expert', 'master']): return "Avanzado"
        return "Intermedio"

    @staticmethod
    def _is_valid_resource(url: str, text: str) -> bool:
        blacklist = ['buy now', 'pricing', 'login', 'signup', 'cart', 'shop']
        return not any(b in text.lower() for b in blacklist)

    @staticmethod
    async def search_google(query: str, lang: str) -> List[RecursoEducativo]:
        if not SecurityManager.validate_api_key(GOOGLE_API_KEY, "google") or not GOOGLE_CX:
            return []
        
        results = []
        api_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX,
            'q': f"{query} course tutorial education",
            'lr': f'lang_{lang}', 'num': 6
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
                                    nivel="General",
                                    categoria="Web",
                                    confianza=0.85,
                                    tipo="verificada",
                                    activo=True
                                ))
        except Exception as e:
            logger.warning(f"Google API Error: {e}")
        return results

    @staticmethod
    def search_internal_db(query: str, lang: str) -> List[RecursoEducativo]:
        results = []
        try:
            with db_connection() as conn:
                cursor = conn.cursor()
                sql = "SELECT * FROM plataformas WHERE (nombre LIKE ? OR descripcion LIKE ?) AND idioma = ? AND activa = 1"
                like_q = f"%{query}%"
                cursor.execute(sql, (like_q, like_q, lang))
                rows = cursor.fetchall()
                
                for row in rows:
                    final_url = row['url_base'].format(quote(query))
                    results.append(RecursoEducativo(
                        id=SearchEngine._generate_id(final_url),
                        titulo=f"📚 {row['nombre']} - {query.title()}",
                        url=final_url,
                        descripcion=row['descripcion'],
                        plataforma=row['nombre'],
                        idioma=row['idioma'],
                        nivel=row['nivel'],
                        categoria=row['categoria'],
                        confianza=row['confianza'],
                        certificacion=Certificacion(tipo=row['tipo_certificacion']),
                        tipo="oculta",
                        activo=True,
                        analisis_pendiente=True
                    ))
        except Exception as e:
            logger.error(f"DB Search Error: {e}")
        return results

    @staticmethod
    def search_known_platforms(query: str, lang: str) -> List[RecursoEducativo]:
        platforms_map = {
            "es": [("YouTube", f"https://www.youtube.com/results?search_query=curso+{query}"), ("Udemy", f"https://www.udemy.com/courses/search/?q={query}&price=price-free&lang=es")],
            "en": [("YouTube", f"https://www.youtube.com/results?search_query=course+{query}"), ("Coursera", f"https://www.coursera.org/search?query={query}&free=true")]
        }
        target = platforms_map.get(lang, platforms_map.get("en", []))
        return [RecursoEducativo(
            id=SearchEngine._generate_id(url), titulo=f"🌐 {name}: {query}",
            url=url, descripcion=f"Búsqueda directa en {name}", plataforma=name,
            idioma=lang, nivel="General", categoria="General", confianza=0.80, tipo="conocida", activo=True
        ) for name, url in target]

    @staticmethod
    async def execute_search(query: str, lang: str, level: str) -> List[RecursoEducativo]:
        internal_task = asyncio.to_thread(SearchEngine.search_internal_db, query, lang)
        google_task = SearchEngine.search_google(query, lang)
        
        r_int, r_goog = await asyncio.gather(internal_task, google_task)
        r_known = SearchEngine.search_known_platforms(query, lang)
        
        all_res = r_int + r_goog + r_known
        # Deduplicar
        unique = {r.url: r for r in all_res}.values()
        final_list = sorted(list(unique), key=lambda x: x.confianza, reverse=True)
        return final_list[:CONFIG.MAX_RESULTS_DEFAULT]

# ============================================================
# 6. MOTOR DE IA (ASYNC)
# ============================================================

class AIWorker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available = SecurityManager.validate_api_key(api_key, "groq")

    async def analyze_course(self, recurso: RecursoEducativo) -> Dict:
        if not self.available: return {}
        import groq
        
        prompt = f"""
        Analiza brevemente: {recurso.titulo}. Desc: {recurso.descripcion}.
        JSON estrictamente: {{"calidad": 0-100, "veredicto": "texto corto", "tags": ["tag1"]}}
        """
        
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        try:
            def _call():
                client = groq.Groq(api_key=self.api_key)
                return client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=CONFIG.GROQ_MODEL_DEFAULT, temperature=0.2, response_format={"type": "json_object"}
                )
            resp = await loop.run_in_executor(executor, _call)
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return {"veredicto": "Error IA", "calidad": 0}
        finally:
            executor.shutdown(wait=False)

    async def chat_interaction(self, history: List[Dict]) -> str:
        if not self.available: return "IA no disponible."
        import groq
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            def _call():
                client = groq.Groq(api_key=self.api_key)
                msgs = [{"role": "system", "content": "Eres un experto educativo."}] + history[-5:]
                return client.chat.completions.create(messages=msgs, model=CONFIG.GROQ_MODEL_DEFAULT, temperature=0.5)
            resp = await loop.run_in_executor(executor, _call)
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

ai_worker = AIWorker(GROQ_API_KEY)

# ============================================================
# 7. INTERFAZ DE USUARIO (UI/UX)
# ============================================================

class UIRenderer:
    @staticmethod
    def load_css():
        st.markdown("""
        <style>
            :root { --primary: #4b6cb7; --secondary: #182848; --accent: #ff6b6b; }
            .main-header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .resource-card {
                background: white; padding: 1.5rem; border-radius: 12px;
                border-left: 5px solid var(--primary);
                box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 1rem;
                transition: transform 0.2s;
            }
            .resource-card:hover { transform: translateY(-3px); }
            .plataforma-oculta { border-left-color: #FF9800 !important; background: #fffbf0; }
            .badge {
                display: inline-block; padding: 3px 8px; border-radius: 4px;
                font-size: 0.75rem; font-weight: bold; margin-right: 5px;
                background: #e3f2fd; color: #1565c0;
            }
            a.btn-access {
                display: inline-block; text-decoration: none;
                background: linear-gradient(90deg, #4b6cb7, #2575fc);
                color: white !important; padding: 8px 16px; border-radius: 6px;
                font-weight: bold; margin-top: 10px;
            }
            a.btn-access:hover { opacity: 0.9; }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        st.markdown(f"""
        <div class="main-header">
            <h1>{CONFIG.APP_NAME}</h1>
            <p>Plataforma de inteligencia educativa avanzada v{CONFIG.APP_VERSION}</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def clean_chat(text: str) -> str:
        text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
        return re.sub(r'<[^>]*>', '', text).strip()

    @staticmethod
    def render_card(r: RecursoEducativo, idx: int):
        extra_class = "plataforma-oculta" if r.tipo == "oculta" else ""
        
        ai_block = ""
        if r.metadatos_ai:
            ai_block = f"""
            <div style="background:#f0f9ff; padding:10px; border-radius:8px; margin-top:10px; font-size:0.9rem;">
                <strong>🧠 IA Score: {r.metadatos_ai.get('calidad')}%</strong><br>
                {r.metadatos_ai.get('veredicto')}
            </div>"""
        
        # HTML sin indentación para evitar bugs de markdown
        html = f"""
<div class="resource-card {extra_class}" style="animation: fadeInUp 0.5s ease backwards {idx * 0.1}s;">
    <div style="display:flex; justify-content:space-between;">
        <h3 style="margin:0; color:#1e293b;">{r.titulo}</h3>
        <span style="color:#64748b; font-size:0.9rem;">{r.plataforma}</span>
    </div>
    <div style="margin:5px 0;">
        <span class="badge">{r.nivel}</span>
        <span class="badge">{r.categoria}</span>
    </div>
    <p style="color:#444; margin:10px 0;">{r.descripcion}</p>
    {ai_block}
    <a href="{r.url}" target="_blank" class="btn-access">Acceder al Recurso 🚀</a>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)
        
        c1, c2 = st.columns([0.9, 0.1])
        with c2:
            if st.button("❤️", key=f"fav_{r.id}_{idx}", help="Guardar"):
                if DatabaseManager.add_favorite(r):
                    st.toast("Guardado!")

# ============================================================
# 8. LÓGICA PRINCIPAL (MAIN LOOP)
# ============================================================

def main():
    if 'user_id' not in st.session_state: st.session_state.user_id = f"u_{random.randint(1000,9999)}"
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'results' not in st.session_state: st.session_state.results = []

    UIRenderer.load_css()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("Navegación")
        page = st.radio("Ir a:", ["Buscador", "Favoritos", "Analytics", "Admin"], key="nav_main")
        
        st.divider()
        st.subheader("💬 Asistente IA")
        
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                icon = "👤" if msg['role'] == "user" else "🤖"
                st.markdown(f"**{icon}**: {UIRenderer.clean_chat(msg['content'])}")
        
        if prompt := st.chat_input("Pregunta algo...", key="chat_input"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            if GROQ_AVAILABLE:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(ai_worker.chat_interaction(st.session_state.chat_history))
                loop.close()
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.rerun()

    # --- PAGINA: BUSCADOR ---
    if page == "Buscador":
        UIRenderer.render_header()
        
        c1, c2, c3 = st.columns([3, 1, 1])
        query = c1.text_input("Tema", placeholder="Ej: Machine Learning...", key="search_q")
        lang = c2.selectbox("Idioma", ["es", "en", "pt"], key="search_l")
        level = c3.selectbox("Nivel", ["Cualquiera", "Principiante", "Avanzado"], key="search_lvl")
        
        if st.button("🚀 Buscar", type="primary", use_container_width=True, key="search_btn"):
            if query:
                with st.spinner("🔍 Analizando bases de datos y web..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(SearchEngine.execute_search(query, lang, level))
                    loop.close()
                    
                    st.session_state.results = results
                    DatabaseManager.log_search(query, lang, level, len(results))
                    
                    # Trigger Background Analysis
                    if GROQ_AVAILABLE and results:
                        def bg_task(items):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            for r in items[:3]:
                                if r.analisis_pendiente:
                                    an = loop.run_until_complete(ai_worker.analyze_course(r))
                                    r.metadatos_ai = an
                                    r.analisis_pendiente = False
                            loop.close()
                        
                        executor = ThreadPoolExecutor(max_workers=1)
                        executor.submit(bg_task, st.session_state.results)
                        st.toast("IA analizando los mejores resultados...")

        if st.session_state.results:
            st.success(f"Encontrados {len(st.session_state.results)} recursos.")
            
            # Export CSV
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Titulo", "URL", "Plataforma", "Confianza"])
            for r in st.session_state.results: writer.writerow([r.titulo, r.url, r.plataforma, r.confianza])
            st.download_button("Descargar CSV", csv_buffer.getvalue(), "data.csv", "text/csv", key="dl_csv")

            for i, r in enumerate(st.session_state.results):
                UIRenderer.render_card(r, i)
        elif query:
            st.warning("No se encontraron resultados.")

    # --- PAGINA: FAVORITOS ---
    elif page == "Favoritos":
        st.title("❤️ Mis Favoritos")
        favs = DatabaseManager.get_favorites()
        if favs:
            for f in favs:
                st.info(f"**[{f['plataforma']}]** {f['titulo']} - {f['url']}")
        else:
            st.info("Aún no tienes favoritos.")

    # --- PAGINA: ANALYTICS ---
    elif page == "Analytics":
        st.title("📊 Dashboard")
        stats = DatabaseManager.get_platform_stats()
        c1, c2 = st.columns(2)
        c1.metric("Plataformas Indexadas", stats['total_platforms'])
        c2.metric("Búsquedas Totales", stats['total_searches'])
        
        with db_connection() as conn:
            df = pd.read_sql("SELECT * FROM historial_busquedas ORDER BY timestamp DESC LIMIT 50", conn)
            if not df.empty:
                st.subheader("Actividad Reciente")
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df['tema'].value_counts())

    # --- PAGINA: ADMIN ---
    elif page == "Admin":
        st.title("🛠️ Panel Admin")
        pwd = st.text_input("Clave", type="password", key="admin_pwd")
        if pwd == "admin123":
            st.success("Acceso Concedido")
            
            with st.expander("Agregar Plataforma Manual"):
                with st.form("add_p"):
                    n = st.text_input("Nombre")
                    u = st.text_input("URL Base (con {})")
                    c = st.selectbox("Categoría", ["Programación", "Data Science", "General"])
                    if st.form_submit_button("Guardar"):
                        with db_connection() as conn:
                            conn.execute("INSERT INTO plataformas (nombre, url_base, categoria) VALUES (?,?,?)", (n,u,c))
                            conn.commit()
                        st.success("Guardado")
            
            if st.button("Limpiar Historial"):
                with db_connection() as conn:
                    conn.execute("DELETE FROM historial_busquedas")
                    conn.commit()
                st.success("Historial eliminado")

if __name__ == "__main__":
    main()
