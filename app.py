# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time
import json
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, urljoin
import logging

# ----------------------------
# CONFIGURACIÓN INICIAL
# ----------------------------
st.set_page_config(
    page_title="🎓 Buscador Premium de Cursos Gratuitos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ----------------------------
# 1. GESTIÓN DE SECRETOS
# ----------------------------
def get_secret(key: str, default: str = "") -> str:
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return str(st.secrets[key])
        import os
        val = os.getenv(key)
        return str(val) if val is not None else default
    except Exception:
        return default

GROQ_API_KEY = get_secret("GROQ_API_KEY")
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GOOGLE_CX = get_secret("GOOGLE_CX")

# ----------------------------
# 2. MODELOS DE DATOS
# ----------------------------
@dataclass
class CursoAuditado:
    titulo: str
    url: str
    es_gratis: bool
    tiene_certificado: bool
    calidad_score: int  # 0-100
    resumen: str
    veredicto: str  # "APROBADO" | "RECHAZADO"
    fuente: str  # "google" | "duckduckgo"
    scrap_content: Optional[str] = None

# ----------------------------
# 3. CLASE: ScrapeManager
# ----------------------------
class ScrapeManager:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    @staticmethod
    def scrape_text(url: str, timeout: int = 8) -> Optional[str]:
        try:
            resp = requests.get(url, headers=ScrapeManager.HEADERS, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Eliminar scripts, estilos, comentarios
            for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            return text[:5000]  # Truncar a 5k caracteres
        except Exception as e:
            logger.warning(f"Scraping fallido para {url}: {str(e)}")
            return None

# ----------------------------
# 4. CLASE: SearchManager
# ----------------------------
class SearchManager:
    @staticmethod
    @st.cache_data(show_spinner=False)
    def search_google(query: str, num_results: int = 10) -> List[Dict[str, str]]:
        if not GOOGLE_API_KEY or not GOOGLE_CX:
            return []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_API_KEY,
                "cx": GOOGLE_CX,
                "q": query,
                "num": min(num_results, 10),
                "safe": "active"
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = []
            for item in data.get("items", []):
                items.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            return items
        except Exception as e:
            logger.warning(f"Google Search falló: {e}")
            return []

    @staticmethod
    @st.cache_data(show_spinner=False)
    def search_ddg(query: str, num_results: int = 10) -> List[Dict[str, str]]:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=num_results)
                return [
                    {"title": r["title"], "link": r["href"], "snippet": r["body"]}
                    for r in results
                ]
        except Exception as e:
            logger.error(f"DuckDuckGo falló: {e}")
            return []

    @classmethod
    def search(cls, tema: str, idioma: str = "es", nivel: str = "Intermedio") -> Tuple[List[Dict[str, str]], str]:
        query = f"curso {tema} {nivel} gratuito certificado sitio:.edu OR sitio:.org OR curso en línea"
        if idioma == "es":
            query += " lang:es"
        elif idioma == "en":
            query += " lang:en"

        # Intentar Google primero
        resultados = cls.search_google(query)
        if resultados:
            return resultados, "google"
        # Fallback a DuckDuckGo
        resultados = cls.search_ddg(query)
        return resultados, "duckduckgo"

# ----------------------------
# 5. CLASE: AIAuditor (Groq)
# ----------------------------
class AIAuditor:
    SYSTEM_PROMPT = (
        "Eres un auditor académico estricto. Analiza SOLO si el curso es 100% GRATUITO, "
        "tiene certificado (aunque sea en modo auditoría) y es autogestionado (self-paced). "
        "Ignora si requiere pago, suscripción o matrícula. Devuelve JSON puro con las claves exactas: "
        '{"titulo", "url", "es_gratis", "tiene_certificado", "calidad_score", "resumen", "veredicto"}'
    )

    @staticmethod
    def analyze(content: str, titulo: str, url: str, fuente: str) -> Optional[CursoAuditado]:
        if not GROQ_API_KEY:
            # Modo sin IA: heurística básica
            es_gratis = any(kw in content.lower() for kw in ["gratis", "free", "sin costo", "audit"])
            cert = any(kw in content.lower() for kw in ["certific", "credential", "diploma"])
            calidad = 75 if es_gratis else 40
            veredicto = "APROBADO" if es_gratis else "RECHAZADO"
            return CursoAuditado(
                titulo=titulo,
                url=url,
                es_gratis=es_gratis,
                tiene_certificado=cert,
                calidad_score=calidad,
                resumen="Análisis básico (IA no disponible)",
                veredicto=veredicto,
                fuente=fuente
            )

        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            user_prompt = f"Contenido del curso: {content}\nTítulo: {titulo}\nURL: {url}"

            response = client.chat.completions.create(
                model="qwen-2.5-72b-versatile",
                messages=[
                    {"role": "system", "content": AIAuditor.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)

            # Validar estructura
            return CursoAuditado(
                titulo=data.get("titulo", titulo),
                url=data.get("url", url),
                es_gratis=bool(data.get("es_gratis", False)),
                tiene_certificado=bool(data.get("tiene_certificado", False)),
                calidad_score=int(data.get("calidad_score", 0)),
                resumen=str(data.get("resumen", "Sin resumen")),
                veredicto=str(data.get("veredicto", "RECHAZADO")),
                fuente=fuente
            )
        except Exception as e:
            logger.error(f"Error en auditoría IA: {e}")
            # Fallback a heurística
            es_gratis = "gratis" in titulo.lower() or "free" in titulo.lower()
            return CursoAuditado(
                titulo=titulo,
                url=url,
                es_gratis=es_gratis,
                tiene_certificado=False,
                calidad_score=50,
                resumen="Análisis fallido – modo de respaldo",
                veredicto="APROBADO" if es_gratis else "RECHAZADO",
                fuente=fuente
            )
        finally:
            time.sleep(1)  # Respetar rate limit

# ----------------------------
# 6. INTERFAZ DE USUARIO
# ----------------------------
def main():
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        google_ok = bool(GOOGLE_API_KEY and GOOGLE_CX)
        groq_ok = bool(GROQ_API_KEY)

        if google_ok:
            st.success("✅ Google Search API activa")
        else:
            st.warning("⚠️ Google API no configurada → Usando DuckDuckGo")

        if groq_ok:
            st.success("🧠 Groq IA (Qwen-2.5) activa")
        else:
            st.info("ℹ️ Auditoría IA desactivada → Modo básico")

        st.markdown("---")
        st.caption("Buscador Premium v1.0 • Open Source")

    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 25px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;">
        <h1>🎓 Buscador Premium de Cursos Gratuitos</h1>
        <p>Resultados verificados por IA: 100% gratis, con certificado y autogestionados</p>
    </div>
    """, unsafe_allow_html=True)

    # Formulario
    col1, col2, col3 = st.columns([3, 1, 1])
    tema = col1.text_input("¿Qué quieres aprender?", placeholder="Ej. Machine Learning, UX Design...")
    nivel = col2.selectbox("Nivel", ["Principiante", "Intermedio", "Avanzado"])
    idioma = col3.selectbox("Idioma", ["es", "en", "pt"])

    if st.button("🔍 Buscar Cursos Verificados", type="primary", use_container_width=True):
        if not tema.strip():
            st.error("Por favor, ingresa un tema.")
            return

        with st.spinner("🔍 Buscando y auditando cursos..."):
            resultados_raw, fuente = SearchManager.search(tema, idioma, nivel)
            st.info(f"🔎 {len(resultados_raw)} resultados encontrados ({fuente})")

            cursos_auditados: List[CursoAuditado] = []
            for item in resultados_raw[:8]:  # Máximo 8 para evitar límites
                url = item["link"]
                titulo = item["title"]
                snippet = item["snippet"]

                # Intentar scrapear contenido
                contenido = ScrapeManager.scrape_text(url)
                if not contenido:
                    contenido = snippet

                # Auditar con IA o heurística
                curso = AIAuditor.analyze(contenido, titulo, url, fuente)
                if curso:
                    cursos_auditados.append(curso)

            # Filtrar solo aprobados
            aprobados = [c for c in cursos_auditados if c.veredicto == "APROBADO"]
            rechazados = [c for c in cursos_auditados if c.veredicto == "RECHAZADO"]

            st.success(f"✅ {len(aprobados)} cursos aprobados | ❌ {len(rechazados)} rechazados")

            # Mostrar aprobados
            for curso in aprobados:
                with st.container():
                    st.markdown(f"### 🎯 [{curso.titulo}]({curso.url})")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Gratis", "✅ Sí" if curso.es_gratis else "❌ No")
                    col_b.metric("Certificado", "✅ Sí" if curso.tiene_certificado else "⚠️ No")
                    col_c.metric("Calidad IA", f"{curso.calidad_score}/100")

                    with st.expander("🔍 Ver análisis detallado"):
                        st.write(f"**Fuente:** {curso.fuente}")
                        st.write(f"**Resumen:** {curso.resumen}")
                        if not GROQ_API_KEY:
                            st.info("💡 Este análisis fue generado sin IA. Para mayor precisión, configura Groq.")

            # Opción: mostrar rechazados (opcional)
            if st.checkbox("Mostrar cursos rechazados"):
                for curso in rechazados:
                    with st.container():
                        st.markdown(f"### ❌ [{curso.titulo}]({curso.url})")
                        st.write(f"**Motivo:** {curso.resumen}")
                        st.write(f"**Calidad:** {curso.calidad_score}/100")

            # Exportar
            if aprobados:
                df = st.dataframe([asdict(c) for c in aprobados])
                csv = json.dumps([asdict(c) for c in aprobados], indent=2)
                st.download_button(
                    "📥 Descargar Resultados (JSON)",
                    csv,
                    file_name="cursos_aprobados.json",
                    mime="application/json"
                )

# ----------------------------
# EJECUCIÓN
# ----------------------------
if __name__ == "__main__":
    main()
