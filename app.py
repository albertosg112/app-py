# app.py — Parte 1

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
# CONFIG
# ----------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX = st.secrets.get("GOOGLE_CX", "") if hasattr(st, 'secrets') else os.getenv("GOOGLE_CX", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else os.getenv("GROQ_API_KEY", "")

MAX_BACKGROUND_TASKS = 1
CACHE_EXPIRATION = timedelta(hours=12)
GROQ_MODEL = "llama3-8b-8192"

search_cache: Dict[str, Dict[str, Any]] = {}
background_tasks: "queue.Queue[Dict[str, Any]]" = queue.Queue()

# ----------------------------
# MODELOS DE DATOS
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

# ----------------------------
# DB
# ----------------------------
DB_PATH = "cursos_inteligentes_v3.db"

def init_advanced_database() -> bool:
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
    palabras_validas = ['curso','tutorial','aprender','education','learn','gratuito','free','certificado','clase']
    palabras_invalidas = ['comprar','buy','precio','pago','suscripción','subscription']
    dominios_educativos = ['.edu','coursera','edx','khanacademy','freecodecamp','kaggle','udemy','youtube']
    tiene_validas = any(p in texto for p in palabras_validas)
    tiene_invalidas = any(p in texto for p in palabras_invalidas)
    dominio_valido = any(d in url.lower() for d in dominios_educativos)
    es_gratuito = ('gratuito' in texto) or ('free' in texto) or ('audit' in texto)
    return (tiene_validas or dominio_valido) and not tiene_invalidas and (es_gratuito or dominio_valido)

def generar_id_unico(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]

def determinar_nivel(texto: str, nivel_solicitado: str) -> str:
    texto = texto.lower()
    if nivel_solicitado not in ("Cualquiera", "Todos"):
        return nivel_solicitado
    if "principiante" in texto or "beginner" in texto: return "Principiante"
    if "intermedio" in texto or "intermediate" in texto: return "Intermedio"
    if "avanzado" in texto or "advanced" in texto: return "Avanzado"
    return "Intermedio"

def determinar_categoria(tema: str) -> str:
    t = tema.lower()
    if "python" in t or "programación" in t: return "Programación"
    if "data" in t or "machine learning" in t: return "Data Science"
    if "matemáticas" in t: return "Matemáticas"
    if "diseño" in t: return "Diseño"
    if "marketing" in t or "negocios" in t: return "Negocios"
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
    return dominio.title()

# ----------------------------
# BÚSQUEDA REAL
# ----------------------------
async def buscar_en_google_api(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return []
    try:
        query_base = f"{tema} curso gratuito certificado"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': GOOGLE_API_KEY, 'cx': GOOGLE_CX, 'q': query_base, 'num': 5, 'lr': f'lang_{idioma}'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status != 200:
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
                    resultados
                    # app.py — Parte 2 (continuación)

# ----------------------------
# Guardado dinámico en base de datos
# ----------------------------
def guardar_recurso_en_db(recurso: RecursoEducativo):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        # Tabla para recursos indexados (si no existe, la creamos)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS recursos_indexados (
            id TEXT PRIMARY KEY,
            titulo TEXT,
            url TEXT UNIQUE,
            descripcion TEXT,
            plataforma TEXT,
            idioma TEXT,
            nivel TEXT,
            categoria TEXT,
            confianza REAL,
            ultima_verificacion TEXT,
            veces_aparecido INTEGER DEFAULT 1
        )
        ''')
        # Insertar o actualizar frecuencia
        cursor.execute("SELECT id, veces_aparecido FROM recursos_indexados WHERE url = ?", (recurso.url,))
        row = cursor.fetchone()
        if row:
            veces = int(row[1]) + 1
            cursor.execute("UPDATE recursos_indexados SET veces_aparecido = ?, ultima_verificacion = ? WHERE url = ?",
                           (veces, datetime.now().isoformat(), recurso.url))
        else:
            cursor.execute('''
            INSERT INTO recursos_indexados
            (id, titulo, url, descripcion, plataforma, idioma, nivel, categoria, confianza, ultima_verificacion, veces_aparecido)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                recurso.id, recurso.titulo, recurso.url, recurso.descripcion, recurso.plataforma,
                recurso.idioma, recurso.nivel, recurso.categoria, recurso.confianza,
                recurso.ultima_verificacion, 1
            ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error guardando recurso en DB: {e}")

def registrar_busqueda(tema: str, idioma: str, nivel: str, total_mostrado: int):
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO analiticas_busquedas
        (tema, idioma, nivel, timestamp, plataforma_origen, veces_mostrado, veces_clickeado, tiempo_promedio_uso, satisfaccion_usuario)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tema, idioma, nivel, datetime.now().isoformat(), "busqueda_web",
            total_mostrado, 0, 0.0, 0.0
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error registrando analítica de búsqueda: {e}")

# ----------------------------
# Búsqueda multicapa (completa con guardado)
# ----------------------------
async def buscar_recursos_multicapa(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    cache_key = f"busqueda_{tema}_{idioma}_{nivel}"
    if cache_key in search_cache:
        cached = search_cache[cache_key]
        if datetime.now() - cached['timestamp'] < CACHE_EXPIRATION:
            return cached['resultados']

    resultados: List[RecursoEducativo] = []
    codigo_idioma = get_codigo_idioma(idioma)

    # 1) Plataformas conocidas (accesos rápidos verificados)
    resultados_conocidos = await buscar_en_plataformas_conocidas(tema, codigo_idioma, nivel)
    resultados.extend(resultados_conocidos)

    # 2) Ocultas desde DB
    # (Se mantiene como en Parte 1: buscar_en_plataformas_ocultas)
    # Para robustez, volvemos a llamarla aquí si no fue incluida antes:
    # Nota: si ya la tenés en Parte 1, esta llamada funciona igual.
    try:
        resultados_ocultas = await buscar_en_plataformas_ocultas(tema, codigo_idioma, nivel)
        resultados.extend(resultados_ocultas)
    except Exception as e:
        logger.error(f"Error en plataformas ocultas: {e}")

    # 3) Google API (real web search si tienes credenciales)
    try:
        resultados_google = await buscar_en_google_api(tema, codigo_idioma, nivel)
        resultados.extend(resultados_google)
    except Exception as e:
        logger.error(f"Error en Google API: {e}")

    # Filtrar, ordenar, marcar IA
    resultados = [r for r in resultados if es_recurso_educativo_valido(r.url, r.titulo, r.descripcion)]
    resultados = sorted(eliminar_duplicados(resultados), key=lambda x: (x.confianza), reverse=True)
    for r in resultados[:10]:
        guardar_recurso_en_db(r)

    # Marcar top 5 para IA si está disponible
    if 'GROQ_API_KEY' in globals() and GROQ_API_KEY:
        for r in resultados[:5]:
            r.analisis_pendiente = True

    final = resultados[:10]
    search_cache[cache_key] = {'resultados': final, 'timestamp': datetime.now()}
    registrar_busqueda(tema, codigo_idioma, nivel, len(final))
    return final

# ----------------------------
# Análisis IA en segundo plano
# ----------------------------
GROQ_AVAILABLE = False
try:
    import groq
    if GROQ_API_KEY:
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API disponible para análisis en segundo plano")
except Exception:
    GROQ_AVAILABLE = False

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
        NIVEL: {recurso.nivel}
        CATEGORÍA: {recurso.categoria}
        CERTIFICACIÓN: {'Sí' if recurso.certificacion else 'No'}

        PERFIL:
        - Nivel: intermedio
        - Objetivos: mejorar habilidades profesionales
        - Tiempo: 2-3 horas/semana

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
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        contenido = response.choices[0].message.content
        return json.loads(contenido)
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
    return {
        "nivel_real": "intermedio",
        "objetivos": "mejorar habilidades profesionales",
        "tiempo_disponible": "2-3 horas por semana",
        "experiencia_previa": "algunos cursos básicos",
        "estilo_aprendizaje": "práctico con proyectos"
    }

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
                    recurso.confianza = min(max(recurso.confianza, confianza_ia), 0.98)
                    recurso.metadatos_analisis = {
                        "calidad_ia": calidad,
                        "relevancia_ia": relevancia,
                        "razones_calidad": analisis.get("razones_calidad", []),
                        "razones_relevancia": analisis.get("razones_relevancia", []),
                        "recomendacion_personalizada": analisis.get("recomendacion_personalizada", ""),
                        "advertencias": analisis.get("advertencias", []),
                    }
                recurso.analisis_pendiente = False
                time.sleep(0.4)
    except Exception as e:
        logger.error(f"Error en análisis en segundo plano: {e}")

# ----------------------------
# Tareas en background
# ----------------------------
def iniciar_tareas_background():
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

    num_workers = min(MAX_BACKGROUND_TASKS, os.cpu_count() or 1)
    for _ in range(num_workers):
        threading.Thread(target=worker, daemon=True).start()
    logger.info(f"✅ Background iniciado con {num_workers} worker(s)")

def planificar_analisis_ia(resultados: List[RecursoEducativo]):
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return
    tarea = {'tipo': 'analizar_resultados', 'parametros': {'resultados': [r for r in resultados if r.analisis_pendiente]}}
    background_tasks.put(tarea)
    logger.info(f"Tarea IA planificada: {len(tarea['parametros']['resultados'])} resultados")

def planificar_indexacion_recursos(temas: List[str], idiomas: List[str]):
    logger.info(f"🗂️ Indexación planificada: {len(temas)} temas, {len(idiomas)} idiomas (placeholder)")

# ----------------------------
# UI
# ----------------------------
st.set_page_config(
    page_title="🎓 Buscador Profesional de Cursos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.resultado-card {
    border-radius: 15px; padding: 20px; margin-bottom: 20px; background: white;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08); border-left: 6px solid #4CAF50;
}
.nivel-principiante { border-left-color: #2196F3 !important; }
.nivel-intermedio { border-left-color: #4CAF50 !important; }
.nivel-avanzado { border-left-color: #FF9800 !important; }
.plataforma-oculta { border-left-color: #FF6B35 !important; background: #fff5f0; }
.con-analisis-ia { border-left-color: #6a11cb !important; background: #f8f4ff; }
.certificado-badge { display:inline-block; padding:5px 12px; border-radius:20px; font-weight:bold; font-size:0.9rem; margin-top:8px; }
.certificado-gratuito { background: linear-gradient(to right, #4CAF50, #8BC34A); color: white; }
.certificado-internacional { background: linear-gradient(to right, #2196F3, #3F51B5); color: white; }
.fade-in { animation: fadeIn 0.6s ease forwards; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0);} }
</style>
""", unsafe_allow_html=True)

# Formulario principal
IDIOMAS = {"Español (es)": "es", "Inglés (en)": "en", "Portugués (pt)": "pt"}
NIVELES = ["Cualquiera", "Principiante", "Intermedio", "Avanzado"]

col1, col2, col3 = st.columns([2,1,1])
with col1:
    tema = st.text_input("🔍 ¿Qué quieres aprender hoy?", placeholder="Ej: Python, Machine Learning, Diseño UX...")
with col2:
    nivel = st.selectbox("📚 Nivel", NIVELES, index=2)
with col3:
    idioma_seleccionado = st.selectbox("🌍 Idioma", list(IDIOMAS.keys()), index=0)

buscar = st.button("🚀 Buscar Cursos Ahora", type="primary", use_container_width=True)

iniciar_tareas_background()
planificar_indexacion_recursos(["Python","Machine Learning","Excel","Cocina Francesa"], ["es","en","pt"])

# ----------------------------
# Utilidades UI
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
# Render de resultados
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
            {"<p style='margin:4px 0;color:#6a11cb;font-weight:bold;'>🧠 Análisis IA en progreso...</p>" if analisis_pendiente else ""}
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
# Ejecución de búsqueda
# ----------------------------
if buscar and tema.strip():
    with st.spinner("🔍 Buscando recursos educativos..."):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            resultados = loop.run_until_complete(buscar_recursos_multicapa(tema, idioma_seleccionado, nivel))
            loop.close()

            if resultados:
                st.success(f"✅ ¡{len(resultados)} recursos encontrados para {tema} en {idioma_seleccionado}!")
                if GROQ_AVAILABLE and GROQ_API_KEY:
                    planificar_analisis_ia(resultados)
                    st.info("🧠 Análisis de calidad en progreso — se actualizarán cuando esté listo")

                colA, colB, colC = st.columns(3)
                colA.metric("Total", len(resultados))
                colB.metric("Plataformas", len(set(r.plataforma for r in resultados)))
                colC.metric("Confianza Promedio", f"{sum(r.confianza for r in resultados)/len(resultados):.1%}")

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
                st.download_button("📥 Descargar Resultados (CSV)", csv,
                                   file_name=f"resultados_{tema.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv", use_container_width=True)
            else:
                st.warning("⚠️ No encontramos recursos verificados para este tema. Intenta con otro término.")
        except Exception as e:
            logger.error(f"Error durante la búsqueda: {e}")
            st.error("❌ Ocurrió un error durante la búsqueda. Intenta nuevamente.")
            st.exception(e)

# ----------------------------
# Chat IA (opcional) — oculta JSON y ejecuta búsquedas
# ----------------------------
st.markdown("### 💬 Asistente educativo (opcional)")
if "chat_msgs" not in st.session_state:
    st.session_state.chat_msgs = [{"role": "system", "content": "Asistente educativo claro y útil."}]

def chatgroq(mensajes: List[Dict[str, str]]) -> str:
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        return "IA no disponible. Usa el buscador superior para encontrar cursos ahora."
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            "Eres un asistente educativo. Conversa con claridad. "
            "Si detectas intención de búsqueda, al final incluye SOLO un bloque JSON con este formato exacto:\n"
            "{\"buscar\": {\"tema\": \"...\", \"idioma\": \"Español (es)|Inglés (en)|Portugués (pt)\", \"nivel\": \"Cualquiera|Principiante|Intermedio|Avanzado\"}}\n"
            "No pongas comentarios ni texto dentro del JSON. El contenido conversacional va arriba; el JSON solo al final."
        )
        groq_msgs = [{"role": "system", "content": system_prompt}] + mensajes
        resp = client.chat.completions.create(
            messages=groq_msgs, model=GROQ_MODEL, temperature=0.3, max_tokens=900
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

def ui_chat_mostrar(mensaje: str, rol: str):
    texto = re.sub(r'\{.*\}\s*$', '', mensaje, flags=re.DOTALL).strip()
    if rol == "assistant":
        st.markdown(f"> {texto}")
    elif rol == "user":
        st.markdown(f"**Tú:** {texto}")

for m in st.session_state.chat_msgs:
    ui_chat_mostrar(m["content"], m["role"])

user_input = st.chat_input("Escribe aquí...")
if user_input:
    st.session_state.chat_msgs.append({"role": "user", "content": user_input})
    try:
        respuesta = chatgroq(st.session_state.chat_msgs)
        if isinstance(respuesta, str):
            st.session_state.chat_msgs.append({"role": "assistant", "content": respuesta})
        else:
            st.session_state.chat_msgs.append({"role": "assistant", "content": "⚠️ La IA respondió en un formato inesperado."})
    except Exception as e:
        logger.error(f"Error al procesar respuesta Groq: {e}")
        st.session_state.chat_msgs.append({"role": "assistant", "content": "❌ Error al procesar la respuesta de la IA."})
    st.experimental_rerun()

# Tras rerun, ejecutar búsqueda si IA mandó comando
if st.session_state.chat_msgs:
    last_assistant = next((m for m in reversed(st.session_state.chat_msgs) if m["role"] == "assistant"), None)
    if last_assistant:
        cmd = extraer_comando_busqueda(last_assistant["content"])
        if cmd:
            with st.spinner(f"Buscando cursos: {cmd['tema']} ({cmd['idioma']}, {cmd['nivel']})"):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                resultados_cmd = loop.run_until_complete(buscar_recursos_multicapa(cmd["tema"], cmd["idioma"], cmd["nivel"]))
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
# Sidebar
# ----------------------------
with st.sidebar:
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
        logger.error(f"Error estadísticas: {e}")
        total_busquedas, total_plataformas, tema_popular = 0, 0, "Python"

    c1, c2 = st.columns(2)
    with c1: st.metric("🔍 Búsquedas", total_busquedas)
    with c2: st.metric("📚 Plataformas", total_plataformas)
    st.metric("🔥 Tema Popular", tema_popular)

    st.markdown("### ✨ Características")
    st.markdown("- ✅ Búsqueda inmediata\n- ✅ Resultados multifuente\n- ✅ Plataformas ocultas\n- ✅ Verificación automática\n- ✅ Diseño responsivo\n- 🧠 Análisis IA (opcional)")

    st.markdown("### 🤖 Estado del sistema")
    st.info(f"IA: {'✅ Disponible' if GROQ_AVAILABLE and GROQ_API_KEY else '⚠️ No disponible'}\n\nÚltima actualización: {datetime.now().strftime('%H:%M:%S')}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:#666;font-size:14px;padding:20px;background:#f8f9fa;border-radius:12px;">
    <strong>✨ Buscador Profesional de Cursos</strong><br>
    <span style="color: #2c3e50; font-weight: 500;">Resultados inmediatos • Cache inteligente • Alta disponibilidad</span><br>
    <em style="color: #7f8c8d;">Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')} • Versión: 3.2.0 • Estado: ✅ Activo</em><br>
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
logger.info("🌐 Idiomas soportados: %s", list(IDIOMAS.keys()))
