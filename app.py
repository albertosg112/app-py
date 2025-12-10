# app.py — Versión CORREGIDA y FUNCIONAL
import streamlit as st
import os
import json
import time
import asyncio
from urllib.parse import quote_plus, parse_qs
import aiohttp
import requests
from dataclasses import dataclass, asdict
from typing import Optional, List, Any
from groq import Groq

# --------------------------------------------------
# 1. CONFIGURACIÓN BÁSICA
# --------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GOOGLE_API_AVAILABLE = bool(GOOGLE_API_KEY and GOOGLE_CX)
GROQ_AVAILABLE = bool(GROQ_API_KEY)

if GROQ_AVAILABLE:
    groq_client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------
# 2. MODELOS DE DATOS
# --------------------------------------------------
@dataclass
class Certificacion:
    tipo: str
    validez_internacional: bool
    paises_validos: List[str]

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
    puntuacion_final: float = 0.0
    analisis: Optional[dict] = None

# --------------------------------------------------
# 3. FUNCIONES AUXILIARES
# --------------------------------------------------
def get_codigo_idioma(idioma_str: str) -> str:
    mapping = {
        "Español (es)": "es",
        "Inglés (en)": "en",
        "Portugués (pt)": "pt"
    }
    return mapping.get(idioma_str, "es")

def extract_platform(url: str) -> str:
    domain = url.lower()
    platforms = {
        'youtube': 'YouTube',
        'coursera': 'Coursera',
        'edx': 'edX',
        'udemy': 'Udemy',
        'khanacademy': 'Khan Academy',
    }
    for k, v in platforms.items():
        if k in domain:
            return v
    return "Web"

def is_valid_educational_resource(url: str, title: str, desc: str) -> bool:
    text = (url + " " + title + " " + desc).lower()
    edu_words = ['curso', 'tutorial', 'learn', 'gratis', 'free', 'certificado', 'class']
    return any(w in text for w in edu_words)

# --------------------------------------------------
# 4. BÚSQUEDA CON GOOGLE API
# --------------------------------------------------
async def buscar_google(query: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not GOOGLE_API_AVAILABLE:
        return []
    base_query = f"{query} curso gratuito"
    if nivel not in ("Cualquiera", "Todos"):
        base_query += f" nivel {nivel.lower()}"
    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CX,
        'q': base_query,
        'num': 5,
        'lr': f'lang_{idioma}'
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                items = data.get('items', [])
                recursos = []
                for item in items:
                    url = item.get('link', '')
                    title = item.get('title', '')
                    snippet = item.get('snippet', '')
                    if not is_valid_educational_resource(url, title, snippet):
                        continue
                    cert = Certificacion("gratuito", True, ["global"]) if "certificado" in (title + snippet).lower() else None
                    recurso = RecursoEducativo(
                        id=url[:12].replace("/", "_"),
                        titulo=title,
                        url=url,
                        descripcion=snippet,
                        plataforma=extract_platform(url),
                        idioma=idioma,
                        nivel=nivel if nivel != "Cualquiera" else "Intermedio",
                        categoria="General",
                        certificacion=cert,
                        confianza=0.85,
                        tipo="verificada",
                        ultima_verificacion=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        activo=True
                    )
                    recursos.append(recurso)
                return recursos
    except Exception as e:
        st.warning(f"⚠️ Error en Google API: {e}")
        return []

# --------------------------------------------------
# 5. ANÁLISIS CON IA (GROQ)
# --------------------------------------------------
async def analizar_con_ia(recurso: RecursoEducativo) -> dict:
    if not GROQ_AVAILABLE:
        return {"recomendacion_personalizada": "Análisis no disponible (IA desactivada)."}
    prompt = f"""
Analiza este recurso educativo y responde SOLO con un JSON con esta estructura:
{{
  "calidad_educativa": 0.8,
  "relevancia_usuario": 0.85,
  "recomendacion_personalizada": "Excelente curso para aprender {recurso.categoria.lower()}."
}}
TÍTULO: {recurso.titulo}
DESCRIPCIÓN: {recurso.descripcion[:400]}
PLATAFORMA: {recurso.plataforma}
NIVEL: {recurso.nivel}
"""
    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=300
        )
        content = completion.choices[0].message.content.strip()
        # Extraer JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = content[start:end]
            return json.loads(json_str)
    except Exception as e:
        st.warning(f"⚠️ Error en análisis IA: {e}")
    return {"recomendacion_personalizada": "Análisis fallido. Recurso válido pero sin evaluación detallada."}

# --------------------------------------------------
# 6. INTERFAZ DE USUARIO (STREAMLIT)
# --------------------------------------------------
def main():
    st.set_page_config(page_title="🎓 Buscador Profesional de Cursos", layout="wide")
    st.title("🎓 Buscador Profesional de Cursos")
    st.caption("Búsqueda inteligente + Análisis con IA (Groq)")

    # Formulario
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        tema = st.text_input("¿Qué quieres aprender?", placeholder="Ej: Python, Machine Learning...")
    with col2:
        nivel = st.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    with col3:
        idioma_ui = st.selectbox("Idioma", ["Español (es)", "Inglés (en)", "Portugués (pt)"])

    buscar = st.button("🚀 Buscar Cursos", type="primary", use_container_width=True)

    if buscar and tema.strip():
        idioma = get_codigo_idioma(idioma_ui)
        with st.spinner("🔍 Buscando recursos..."):
            loop = asyncio.new_event_loop()
            recursos = loop.run_until_complete(buscar_google(tema.strip(), idioma, nivel))

        if recursos:
            st.success(f"✅ Encontrados {len(recursos)} recursos")
            for recurso in recursos:
                with st.spinner(f"🧠 Analizando: {recurso.titulo[:50]}..."):
                    analisis = loop.run_until_complete(analizar_con_ia(recurso))
                recurso.analisis = analisis

                # Mostrar tarjeta
                with st.container(border=True):
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.subheader(recurso.titulo)
                        st.write(f"**Plataforma**: {recurso.plataforma} | **Nivel**: {recurso.nivel}")
                        st.write(recurso.descripcion[:300] + ("..." if len(recurso.descripcion) > 300 else ""))
                        if recurso.analisis:
                            st.info("🧠 **IA dice**: " + recurso.analisis.get("recomendacion_personalizada", "Sin análisis."))
                    with cols[1]:
                        st.link_button("➡️ Acceder", recurso.url, use_container_width=True)
        else:
            st.warning("No se encontraron resultados. Intenta con otras palabras.")

    # Pie de página
    st.markdown("---")
    st.caption("Sistema funcional • Groq IA opcional • Google API requerida")

if __name__ == "__main__":
    main()
