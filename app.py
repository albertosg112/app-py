# app.py — Buscador de Cursos SG1 "Enterprise Edition" (Fixed v5.2)
# ==============================================================================
# CORRECCIONES APLICADAS:
# 1. FIX: Se agregaron 'keys' únicos a todos los widgets para evitar StreamlitDuplicateElementId.
# 2. FIX: Se eliminó la indentación en los bloques HTML para evitar que se muestren como código.
# 3. ROBUST: Manejo de errores mejorado en Groq y DB.
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
        'About': "SG1 Enterprise v5.2 Fixed"
    }
)

# Estilos CSS Profesionales
st.markdown("""
<style>
    :root {
        --primary-color: #4b6cb7;
        --secondary-color: #182848;
        --accent-color: #ff6b6b;
        --success-color: #4CAF50;
        --bg-light: #f8f9fa;
        --text-dark: #2c3e50;
    }

    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .main-header h1 { font-family: sans-serif; font-weight: 800; font-size: 2.5rem; margin: 0; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin-top: 5px; }

    .resource-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    .resource-card:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }
    
    .card-beginner { border-left-color: #2196F3; }
    .card-intermediate { border-left-color: #4CAF50; }
    .card-advanced { border-left-color: #9C27B0; }
    .card-special { border-left-color: #FF9800; background-color: #fffbf0; }

    .badge {
        display: inline-block; padding: 4px 8px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600; margin-right: 5px;
    }
    .badge-free { background-color: #e8f5e9; color: #2e7d32; }
    .badge-paid { background-color: #ffebee; color: #c62828; }
    .badge-ai { background-color: #f3e5f5; color: #7b1fa2; border: 1px solid #e1bee7; }

    /* Fix para botones de enlace */
    a.btn-access {
        display: inline-block !important;
        background: linear-gradient(90deg, #4b6cb7 0%, #2575fc 100%);
        color: white !important;
        padding: 8px 16px;
        border-radius: 8px;
        text-decoration: none !important;
        font-weight: bold;
        text-align: center;
        border: none;
    }
    a.btn-access:hover { opacity: 0.9; }

    .metric-container {
        background: white; padding: 15px; border-radius: 10px;
        text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: var(--primary-color); }
    .metric-label { font-size: 0.85rem; color: #666; text-transform: uppercase; }
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
CACHE_TTL = 43200
GROQ_MODEL_ID = "llama-3.1-70b-versatile"
MAX_WORKERS = 4

# ==============================================================================
# 3. GESTOR DE SEGURIDAD Y CREDENCIALES
# ==============================================================================

class SecurityManager:
    @staticmethod
    def get_credentials() -> Tuple[str, str, str]:
        try:
            g_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
            g_cx = st.secrets.get("GOOGLE_CX", os.getenv("GOOGLE_CX", ""))
            groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
            return g_key, g_cx, groq_key
        except Exception:
            return "", "", ""

    @staticmethod
    def validate_api_key(key: str, service: str) -> bool:
        if not key or len(key) < 8: return False
        if service == "google" and not key.startswith(("AIza", "AIz")): return False
        return True

GOOGLE_API_KEY, GOOGLE_CX, GROQ_API_KEY = SecurityManager.get_credentials()

# ==============================================================================
# 4. GESTOR DE BASE DE DATOS
# ==============================================================================

@contextlib.contextmanager
def db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as e:
        logger.critical(f"Error BD: {e}")
        if conn: conn.rollback()
        raise e
    finally:
        if conn: conn.close()

class DatabaseManager:
    @staticmethod
    def init_db():
        with db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS plataformas (
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
            )''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS historial_busquedas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id TEXT, tema TEXT, idioma TEXT, nivel TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, resultados_count INTEGER
            )''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS favoritos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id TEXT, recurso_id TEXT, titulo TEXT, url TEXT, plataforma TEXT,
                added_at DATETIME DEFAULT CURRENT_TIMESTAMP, UNIQUE(usuario_id, url)
            )''')

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
                ("MIT OpenCourseWare", "https://ocw.mit.edu/search/?q={}", "Material académico MIT", "en", "Académico", 0.98, "gratuito")
            ]
            cursor.executemany('''INSERT OR IGNORE INTO plataformas 
                (nombre, url_base, descripcion, idioma, categoria, confianza, tipo_certificacion)
                VALUES (?, ?, ?, ?, ?, ?, ?)''', platforms)

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
        except Exception: pass

    @staticmethod
    def add_favorite(recurso: 'RecursoEducativo'):
        try:
            user_id = st.session_state.get('user_id', 'guest')
            with db_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO favoritos (usuario_id, recurso_id, titulo, url, plataforma) VALUES (?, ?, ?, ?, ?)",
                    (user_id, recurso.id, recurso.titulo, recurso.url, recurso.plataforma)
                )
                conn.commit()
            return True
        except Exception: return False

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

# ==============================================================================
# 5. MODELOS DE DATOS
# ==============================================================================

@dataclass
class Certificacion:
    tipo: str
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
    tipo: str = "general"

# ==============================================================================
# 6. MOTOR DE INTELIGENCIA ARTIFICIAL (ASYNC WRAPPER)
# ==============================================================================

class AIWorker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available = SecurityManager.validate_api_key(api_key, "groq")

    async def analyze_course(self, recurso: RecursoEducativo, user_profile: Dict) -> Dict:
        if not self.available: return {}
        import groq
        
        prompt = f"""
        Analiza este curso. TITULO: {recurso.titulo}. DESC: {recurso.descripcion}.
        Responde SOLO JSON: {{"calidad_score": 0.8, "veredicto": "breve opinion", "tags_sugeridos": ["tag1"]}}
        """
        
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        try:
            def _sync_call():
                client = groq.Groq(api_key=self.api_key)
                return client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=GROQ_MODEL_ID, temperature=0.2, response_format={"type": "json_object"}
                )
            response = await loop.run_in_executor(executor, _sync_call)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error IA: {e}")
            return {"veredicto": "Error de análisis", "calidad_score": 0.5}
        finally:
            executor.shutdown(wait=False)

    async def chat_interaction(self, history: List[Dict]) -> str:
        if not self.available: return "IA no configurada."
        import groq
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            def _call():
                client = groq.Groq(api_key=self.api_key)
                msgs = [{"role": "system", "content": "Eres SG1-Bot. Sé breve."}] + history[-5:]
                return client.chat.completions.create(messages=msgs, model=GROQ_MODEL_ID, temperature=0.5, max_tokens=300)
            resp = await loop.run_in_executor(executor, _call)
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

ai_worker = AIWorker(GROQ_API_KEY)

# ==============================================================================
# 7. MOTOR DE BÚSQUEDA
# ==============================================================================

class SearchEngine:
    @staticmethod
    def _generate_id(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]

    @staticmethod
    def _is_valid_resource(url: str, text: str) -> bool:
        blacklist = ['buy', 'login', 'signup']
        return not any(b in text.lower() for b in blacklist)

    @staticmethod
    async def search_google(query: str, lang: str) -> List[RecursoEducativo]:
        if not SecurityManager.validate_api_key(GOOGLE_API_KEY, "google") or not GOOGLE_CX: return []
        
        results = []
        api_url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': f"{query} course free", 'lr': f'lang_{lang}', 'num': 5}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data.get('items', []):
                            if SearchEngine._is_valid_resource(item['link'], item.get('snippet', '')):
                                results.append(RecursoEducativo(
                                    id=SearchEngine._generate_id(item['link']),
                                    titulo=item['title'], url=item['link'],
                                    descripcion=item.get('snippet', ''),
                                    plataforma=urlparse(item['link']).netloc,
                                    idioma=lang, nivel="General", categoria="Web",
                                    confianza=0.85, activo=True
                                ))
        except Exception: pass
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
                for row in cursor.fetchall():
                    final_url = row['url_base'].format(query.replace(' ', '+'))
                    results.append(RecursoEducativo(
                        id=SearchEngine._generate_id(final_url),
                        titulo=f"📚 {row['nombre']} - {query.title()}",
                        url=final_url,
                        descripcion=row['descripcion'],
                        plataforma=row['nombre'], idioma=row['idioma'],
                        nivel=row['nivel'], categoria=row['categoria'],
                        confianza=row['confianza'],
                        certificacion=Certificacion(tipo=row['tipo_certificacion']),
                        activo=True, analisis_pendiente=True, tipo="oculta"
                    ))
        except Exception: pass
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
            url=url, descripcion=f"Búsqueda en {name}", plataforma=name,
            idioma=lang, nivel="General", categoria="General", confianza=0.80, activo=True
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
        return sorted(list(unique), key=lambda x: x.confianza, reverse=True)[:15]

# ==============================================================================
# 8. COMPONENTES DE UI (RENDERIZADO) - CORREGIDO
# ==============================================================================

class UIRenderer:
    @staticmethod
    def render_badges(recurso: RecursoEducativo) -> str:
        html = ""
        if recurso.certificacion:
            c = recurso.certificacion.tipo
            cls = "badge-free" if c == "gratuito" else "badge-paid"
            html += f'<span class="badge {cls}">{c.title()}</span>'
        
        if recurso.metadatos_ai:
            score = int(recurso.metadatos_ai.get('calidad_score', 0) * 100)
            html += f'<span class="badge badge-ai">IA: {score}%</span>'
        return html

    @staticmethod
    def render_resource_card(r: RecursoEducativo, index: int):
        level_class = "card-intermediate"
        if r.tipo == "oculta": level_class = "card-special"

        badges = UIRenderer.render_badges(r)
        
        ai_block = ""
        if r.metadatos_ai:
            veredicto = r.metadatos_ai.get('veredicto', '')
            # HTML SIN INDENTACIÓN PARA EVITAR BUG
            ai_block = f"""<div style="margin-top:10px;padding:8px;background:#f0f2f6;border-radius:6px;font-size:0.85rem;"><strong>🤖 IA:</strong> {veredicto}</div>"""
        elif r.analisis_pendiente:
            ai_block = """<div style="margin-top:10px;font-size:0.8rem;color:#666;">⏳ Analizando...</div>"""

        # HTML FLAT (Sin indentación al inicio de las líneas)
        html = f"""
<div class="resource-card {level_class}" style="animation: fadeIn 0.5s ease forwards {index * 0.1}s;">
<div style="display:flex;justify-content:space-between;">
    <h3 style="margin:0;">{r.titulo}</h3>
    <span style="font-size:0.8rem;color:#666;">{r.plataforma}</span>
</div>
<div style="margin:8px 0;">{badges}</div>
<p style="color:#444;font-size:0.95rem;">{r.descripcion}</p>
{ai_block}
<div style="margin-top:12px;">
    <a href="{r.url}" target="_blank" class="btn-access">Acceder 🚀</a>
</div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)
        
        # Botón nativo de Streamlit fuera del HTML para funcionalidad
        c1, c2 = st.columns([0.9, 0.1])
        with c2:
            # KEY ÚNICA GENERADA CON ID + INDEX PARA EVITAR DUPLICADOS
            if st.button("❤️", key=f"fav_{r.id}_{index}", help="Guardar"):
                if DatabaseManager.add_favorite(r):
                    st.toast("Guardado!")

    @staticmethod
    def clean_chat(text: str) -> str:
        text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
        return re.sub(r'<[^>]*>', '', text).strip()

# ==============================================================================
# 9. MAIN APP (FIXED)
# ==============================================================================

def main():
    if 'user_id' not in st.session_state: st.session_state.user_id = f"u_{random.randint(1000,9999)}"
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'results' not in st.session_state: st.session_state.results = []

    with st.sidebar:
        st.title("Navegación")
        # KEY ÚNICA
        page = st.radio("Ir a:", ["Buscador", "Favoritos", "Analytics", "Admin"], key="nav_radio")
        st.divider()
        st.subheader("💬 Chat IA")
        
        chat_cont = st.container()
        with chat_cont:
            for msg in st.session_state.chat_history:
                icon = "👤" if msg['role'] == "user" else "🤖"
                st.markdown(f"**{icon}**: {UIRenderer.clean_chat(msg['content'])}")
        
        # KEY ÚNICA
        user_input = st.chat_input("Pregunta...", key="sidebar_chat_input")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            if GROQ_AVAILABLE:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resp = loop.run_until_complete(ai_worker.chat_interaction(st.session_state.chat_history))
                loop.close()
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.rerun()

    if page == "Buscador":
        st.markdown('<div class="main-header"><h1>🎓 SG1 Enterprise</h1><p>Búsqueda Inteligente</p></div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([3, 1, 1])
        # KEYS ÚNICAS
        query = c1.text_input("Tema", placeholder="Python, Marketing...", key="search_query")
        lang = c2.selectbox("Idioma", ["es", "en", "pt"], key="search_lang")
        level = c3.selectbox("Nivel", ["Cualquiera", "Principiante"], key="search_level")
        
        # KEY ÚNICA
        if st.button("Buscar", type="primary", use_container_width=True, key="search_btn"):
            if query:
                with st.spinner("Buscando..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    res = loop.run_until_complete(SearchEngine.execute_search(query, lang, level))
                    loop.close()
                    st.session_state.results = res
                    DatabaseManager.log_search(query, lang, level, len(res))
                    
                    if GROQ_AVAILABLE and res:
                        def bg_analysis(items):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            for r in items[:3]:
                                if r.analisis_pendiente:
                                    an = loop.run_until_complete(ai_worker.analyze_course(r, {}))
                                    r.metadatos_ai = an
                                    r.analisis_pendiente = False
                            loop.close()
                        executor = ThreadPoolExecutor(max_workers=1)
                        executor.submit(bg_analysis, st.session_state.results)
                        st.toast("IA Analizando...")

        if st.session_state.results:
            st.success(f"{len(st.session_state.results)} resultados.")
            
            # Export CSV
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Titulo", "URL", "Plataforma"])
            for r in st.session_state.results: writer.writerow([r.titulo, r.url, r.plataforma])
            
            # KEY ÚNICA
            st.download_button("Descargar CSV", csv_buffer.getvalue(), "data.csv", "text/csv", key="dl_csv")

            for i, r in enumerate(st.session_state.results):
                UIRenderer.render_resource_card(r, i)

    elif page == "Favoritos":
        st.title("❤️ Favoritos")
        favs = DatabaseManager.get_favorites()
        if favs:
            for f in favs:
                st.info(f"[{f['plataforma']}] {f['titulo']} - {f['url']}")
        else:
            st.warning("No hay favoritos guardados.")

    elif page == "Analytics":
        st.title("📊 Datos")
        stats = DatabaseManager.get_platform_stats()
        c1, c2 = st.columns(2)
        c1.metric("Plataformas", stats['total_platforms'])
        c2.metric("Búsquedas", stats['total_searches'])
        
        with db_connection() as conn:
            df = pd.read_sql("SELECT * FROM historial_busquedas ORDER BY timestamp DESC LIMIT 20", conn)
            st.dataframe(df, use_container_width=True)

    elif page == "Admin":
        st.title("Panel Admin")
        # KEY ÚNICA
        pwd = st.text_input("Clave", type="password", key="admin_pwd")
        if pwd == "admin123":
            with st.form("add_p_form"): # FORM CON KEY IMPLÍCITA
                n = st.text_input("Nombre", key="add_n")
                u = st.text_input("URL", key="add_u")
                sub = st.form_submit_button("Guardar")
                if sub:
                    with db_connection() as conn:
                        conn.execute("INSERT INTO plataformas (nombre, url_base) VALUES (?,?)", (n, u))
                        conn.commit()
                    st.success("Guardado")

if __name__ == "__main__":
    main()
