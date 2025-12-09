import streamlit as st
import pandas as pd
import time
import random
import os
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime, timedelta
import json
import hashlib

# Configuración de la página
st.set_page_config(
    page_title="🎓 Buscador Inteligente de Cursos Gratis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== BASE DE DATOS - CONFIGURACIÓN INICIAL ==========
DB_PATH = "/mount/src/app-py/educacion_oculta.db"  # Ruta en Streamlit Cloud
HIDDEN_SITES_FILE = "/mount/src/app-py/sites_poco_conocidos.json"

# Crear base de datos si no existe
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabla para sitios poco conocidos
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sitios_ocultos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        nombre TEXT NOT NULL,
        descripcion TEXT,
        categoria TEXT,
        nivel TEXT,
        confianza REAL DEFAULT 0.5,
        ultima_verificacion TEXT,
        veces_usado INTEGER DEFAULT 0,
        activo INTEGER DEFAULT 1
    )
    ''')
    
    # Tabla para búsquedas de usuarios (para aprendizaje)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS busquedas_usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tema TEXT NOT NULL,
        nivel TEXT,
        timestamp TEXT NOT NULL,
        resultados_mostrados INTEGER
    )
    ''')
    
    # Tabla para feedback de usuarios
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        util BOOLEAN,
        comentario TEXT,
        timestamp TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

# Inicializar la base de datos
init_database()

# ========== CARGAR SITIOS POCO CONOCIDOS ==========
def cargar_sites_poco_conocidos():
    """Cargar sitios educativos poco conocidos desde archivo JSON o usar valores por defecto"""
    sites_default = [
        {
            "nombre": "The Programming Historian",
            "url": "https://programminghistorian.org/es/lecciones/",
            "descripcion": "Tutoriales de programación y humanidades digitales",
            "categoria": "Programación",
            "nivel": "Intermedio"
        },
        {
            "nombre": "Aprende con Alf",
            "url": "https://aprendeconalf.es/",
            "descripcion": "Cursos gratuitos de programación, matemáticas y ciencia de datos",
            "categoria": "Programación",
            "nivel": "Principiante"
        },
        {
            "nombre": "Internet Archive - Cursos",
            "url": "https://archive.org/details/opensource_movies?and[]=education",
            "descripcion": "Archivo digital con miles de cursos y documentales educativos",
            "categoria": "General",
            "nivel": "Todos"
        },
        {
            "nombre": "Bartleby",
            "url": "https://bartleby.com/",
            "descripcion": "Literatura clásica, manuales de referencia y recursos educativos",
            "categoria": "Humanidades",
            "nivel": "Intermedio"
        },
        {
            "nombre": "Project Gutenberg",
            "url": "https://www.gutenberg.org/",
            "descripcion": "60,000+ libros electrónicos gratuitos, muchos educativos",
            "categoria": "Libros",
            "nivel": "Todos"
        },
        {
            "nombre": "OER Commons",
            "url": "https://www.oercommons.org/",
            "descripcion": "Recursos educativos abiertos de instituciones globales",
            "categoria": "General",
            "nivel": "Todos"
        },
        {
            "nombre": "CVC - Centro Virtual de Noticias de Educación Matemática",
            "url": "http://cvc.instituto.camoes.pt/hemeroteca/index.html",
            "descripcion": "Recursos especializados en educación matemática",
            "categoria": "Matemáticas",
            "nivel": "Avanzado"
        },
        {
            "nombre": "Replit Learn",
            "url": "https://replit.com/site/learn",
            "descripcion": "Tutoriales interactivos de programación en un entorno live",
            "categoria": "Programación",
            "nivel": "Principiante"
        },
        {
            "nombre": "MIT Open Learning Library",
            "url": "https://openlearninglibrary.mit.edu/",
            "descripcion": "Cursos del MIT sin certificación pero con todo el contenido",
            "categoria": "Universitario",
            "nivel": "Avanzado"
        },
        {
            "nombre": "PhET Simulations",
            "url": "https://phet.colorado.edu/es/",
            "descripcion": "Simulaciones interactivas de ciencias y matemáticas",
            "categoria": "Ciencias",
            "nivel": "Todos"
        },
        {
            "nombre": "Kaggle Learn",
            "url": "https://www.kaggle.com/learn",
            "descripcion": "Microcursos prácticos de ciencia de datos y machine learning",
            "categoria": "Data Science",
            "nivel": "Intermedio"
        },
        {
            "nombre": "Rosetta Project",
            "url": "https://rosettaproject.org/",
            "descripcion": "Recursos para aprender idiomas poco comunes y preservar el conocimiento lingüístico",
            "categoria": "Idiomas",
            "nivel": "Avanzado"
        },
        {
            "nombre": "The Open University - OpenLearn",
            "url": "https://www.open.edu/openlearn/",
            "descripcion": "Cursos gratuitos de la universidad británica Open University",
            "categoria": "Universitario",
            "nivel": "Intermedio"
        },
        {
            "nombre": "Code.org",
            "url": "https://code.org/learn",
            "descripcion": "Plataforma para aprender programación desde cero, ideal para principiantes",
            "categoria": "Programación",
            "nivel": "Principiante"
        },
        {
            "nombre": "Brilliant",
            "url": "https://brilliant.org/courses/",
            "descripcion": "Cursos interactivos de matemáticas, ciencia y programación (parte gratuita)",
            "categoria": "STEM",
            "nivel": "Intermedio"
        }
    ]
    
    try:
        if os.path.exists(HIDDEN_SITES_FILE):
            with open(HIDDEN_SITES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Guardar los valores por defecto
            with open(HIDDEN_SITES_FILE, 'w', encoding='utf-8') as f:
                json.dump(sites_default, f, ensure_ascii=False, indent=2)
            return sites_default
    except Exception as e:
        st.warning(f"⚠️ Error al cargar sitios poco conocidos: {e}")
        return sites_default

SITIOS_POCO_CONOCIDOS = cargar_sites_poco_conocidos()

# ========== FUNCIONES DE BASE DE DATOS ==========
def guardar_sitio_oculto(url, nombre, descripcion="", categoria="General", nivel="Todos", confianza=0.5):
    """Guardar un nuevo sitio en la base de datos"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verificar si ya existe
        cursor.execute("SELECT id FROM sitios_ocultos WHERE url = ?", (url,))
        if cursor.fetchone():
            conn.close()
            return False
        
        # Insertar nuevo sitio
        cursor.execute('''
        INSERT INTO sitios_ocultos (url, nombre, descripcion, categoria, nivel, confianza, ultima_verificacion, activo)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        ''', (url, nombre, descripcion, categoria, nivel, confianza, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al guardar sitio: {e}")
        return False

def obtener_sitios_recomendados(tema, nivel="Cualquiera", limite=10):
    """Obtener sitios recomendados de la base de datos basados en tema y nivel"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = '''
    SELECT url, nombre, descripcion, nivel, confianza 
    FROM sitios_ocultos 
    WHERE activo = 1
    '''
    params = []
    
    if nivel != "Cualquiera" and nivel != "Todos":
        query += " AND nivel IN (?, 'Todos')"
        params.append(nivel)
    
    # Buscar por coincidencia con el tema (en nombre o descripción)
    query += " AND (LOWER(nombre) LIKE ? OR LOWER(descripcion) LIKE ?)"
    params.append(f"%{tema.lower()}%")
    params.append(f"%{tema.lower()}%")
    
    query += " ORDER BY confianza DESC, veces_usado DESC LIMIT ?"
    params.append(limite)
    
    cursor.execute(query, params)
    resultados = cursor.fetchall()
    conn.close()
    
    return [{
        "url": r[0],
        "nombre": r[1],
        "descripcion": r[2],
        "nivel": r[3],
        "confianza": r[4]
    } for r in resultados]

def actualizar_uso_sitio(url):
    """Actualizar contador de uso de un sitio"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE sitios_ocultos SET veces_usado = veces_usado + 1 WHERE url = ?", (url,))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def registrar_busqueda(tema, nivel, resultados):
    """Registrar una búsqueda de usuario para aprendizaje"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO busquedas_usuarios (tema, nivel, timestamp, resultados_mostrados)
        VALUES (?, ?, ?, ?)
        ''', (tema, nivel, datetime.now().isoformat(), resultados))
        conn.commit()
        conn.close()
        return True
    except:
        return False

# ========== FUNCIÓN DE BÚSQUEDA AVANZADA ==========
def buscar_cursos_avanzado(tema, nivel_seleccionado):
    """Buscar cursos incluyendo sitios poco conocidos de nuestra base de datos"""
    resultados = []
    
    # 1. Buscar en base de datos de sitios ocultos
    sitios_recomendados = obtener_sitios_recomendados(tema, nivel_seleccionado, limite=8)
    
    # Añadir sitios recomendados
    for sitio in sitios_recomendados:
        resultados.append({
            "nivel": sitio["nivel"],
            "titulo": f"{sitio['nombre']} - {tema}",
            "plataforma": sitio["nombre"],
            "url": sitio["url"],
            "descripcion": sitio["descripcion"] or f"Cursos gratuitos de {tema} en una plataforma educativa especializada",
            "tipo": "sitio_oculto",
            "confianza": sitio["confianza"]
        })
        actualizar_uso_sitio(sitio["url"])
    
    # 2. Plataformas conocidas (como fallback)
    plataformas_conocidas = {
        "youtube": {
            "nombre": "YouTube (Curso Completo)",
            "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}"
        },
        "coursera": {
            "nombre": "Coursera (Auditoría Gratuita)",
            "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&free=true"
        },
        "edx": {
            "nombre": "EdX (Cursos Universitarios)",
            "url": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={tema.replace(' ', '%20')}"
        }
    }
    
    # 3. Añadir algunas plataformas conocidas si hay pocos resultados
    if len(resultados) < 5:
        niveles_reales = ["Principiante", "Intermedio", "Avanzado"]
        if nivel_seleccionado != "Cualquiera" and nivel_seleccionado != "Todos":
            niveles_reales = [nivel_seleccionado]
        
        for nombre_plataforma, datos in plataformas_conocidas.items():
            if len(resultados) >= 10:
                break
            
            nivel_actual = random.choice(niveles_reales)
            resultados.append({
                "nivel": nivel_actual,
                "titulo": f"Curso de {tema} en {datos['nombre']}",
                "plataforma": datos["nombre"],
                "url": datos["url"],
                "descripcion": f"Recursos educativos gratuitos para nivel {nivel_actual}",
                "tipo": "plataforma_conocida",
                "confianza": 0.9
            })
    
    # 4. Mezclar y ordenar por confianza
    resultados.sort(key=lambda x: x["confianza"], reverse=True)
    
    # Registrar la búsqueda
    registrar_busqueda(tema, nivel_seleccionado, len(resultados))
    
    return resultados[:10]  # Límite de 10 resultados

# ========== DESCUBRIDOR DE NUEVOS SITIOS (para PythonAnywhere) ==========
def descubrir_nuevos_sitios(temas_populares=["programación", "matemáticas", "idiomas", "ciencia de datos"]):
    """Función para descubrir nuevos sitios educativos (se ejecutará desde PythonAnywhere)"""
    nuevos_sitios = []
    
    for tema in temas_populares:
        try:
            # Buscar en Google con términos específicos para encontrar sitios poco conocidos
            query = f"site:.edu OR site:.org {tema} curso gratuito recursos educativos abiertos"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Esta es una búsqueda simulada - en producción usarías una API o scraping controlado
            resultados_simulados = [
                {
                    "url": f"https://repositorio.ejemplo-{tema.replace(' ', '-')}.edu/educacion",
                    "titulo": f"Repositorio de {tema.title()} - Universidad Ejemplo",
                    "descripcion": f"Recursos educativos abiertos sobre {tema}"
                },
                {
                    "url": f"https://educacionabierta.ejemplo-{tema.replace(' ', '-')}.org",
                    "titulo": f"Educación Abierta en {tema.title()}",
                    "descripcion": f"Plataforma de cursos gratuitos sobre {tema}"
                }
            ]
            
            for res in resultados_simulados:
                if guardar_sitio_oculto(
                    url=res["url"],
                    nombre=res["titulo"],
                    descripcion=res["descripcion"],
                    categoria=tema.title(),
                    nivel="Todos",
                    confianza=0.3  # Baja confianza inicial para nuevos sitios
                ):
                    nuevos_sitios.append(res["titulo"])
            
            # Pausa para evitar bloqueos
            time.sleep(2)
            
        except Exception as e:
            st.warning(f"Error al buscar nuevos sitios para {tema}: {e}")
    
    return nuevos_sitios

# ========== INTERFAZ DE USUARIO ==========
# Estilos personalizados
st.markdown("""
<style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
    }
    .resultado-card {
        border: 1px solid #e1e1e1;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .nivel-principiante { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .nivel-intermedio { background-color: #e8f5e9; border-left: 4px solid #4caf50; }
    .nivel-avanzado { background-color: #fff8e1; border-left: 4px solid #ff9800; }
    .sitio-oculto { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("🎓 Buscador Inteligente de Cursos Gratuitos")
st.markdown("### Descubre recursos educativos **conocidos y ocultos** para aprender cualquier habilidad")

# Formulario de búsqueda
with st.form("busqueda_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        tema = st.text_input("¿Qué quieres aprender hoy?", placeholder="Ej: Python, Fotografía, Finanzas personales...")
    with col2:
        nivel = st.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    
    formato = st.selectbox("Formato de resultados", ["Vista web (recomendado)", "CSV (Excel)"])
    buscar = st.form_submit_button("🔍 Buscar Cursos Gratuitos", use_container_width=True)

# Procesar búsqueda
if buscar and tema.strip():
    with st.spinner("🧠 Buscando recursos conocidos y sitios ocultos..."):
        resultados = buscar_cursos_avanzado(tema, nivel)
    
    if resultados:
        st.success(f"✅ ¡Encontramos {len(resultados)} recursos para **{tema}**!")
        
        # Contador de sitios ocultos encontrados
        sitios_ocultos = sum(1 for r in resultados if r.get("tipo") == "sitio_oculto")
        
        if sitios_ocultos > 0:
            st.info(f"✨ Hemos incluido {sitios_ocultos} sitios educativos poco conocidos que podrían interesarte")
        
        # Mostrar resultados
        for i, resultado in enumerate(resultados):
            clase_nivel = {
                "Principiante": "nivel-principiante",
                "Intermedio": "nivel-intermedio", 
                "Avanzado": "nivel-avanzado"
            }.get(resultado["nivel"], "")
            
            # Estilos personalizados para cada nivel
            color_borde = {
                "Principiante": "#2196f3",  # Azul
                "Intermedio": "#4caf50",    # Verde
                "Avanzado": "#ff9800"       # Naranja
            }.get(resultado["nivel"], "#9e9e9e")
            
            # Estilo especial para sitios ocultos
            if resultado.get("tipo") == "sitio_oculto":
                st.markdown(f"""
                <div class="sitio-oculto">
                    <h3>💎 {resultado['titulo']}</h3>
                    <p><b>Plataforma:</b> {resultado['plataforma']} | <b>Nivel:</b> {resultado['nivel']}</p>
                    <p>{resultado['descripcion']}</p>
                    <a href="{resultado['url']}" target="_blank" style="display: inline-block; background-color: white; color: #667eea; padding: 8px 16px; text-decoration: none; border-radius: 4px; margin-top: 8px; font-weight: bold;">
                        ➡️ Explorar este recurso oculto
                    </a>
                    <p style="font-size: 0.8em; margin-top: 8px; opacity: 0.8;">
                        Este es un recurso poco conocido que nuestro sistema ha descubierto especialmente para ti
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.container():
                    st.markdown(f"""
                    <div style="border: 2px solid {color_borde}; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h3>🎯 {resultado['titulo']}</h3>
                        <p>📚 <b>Nivel:</b> {resultado['nivel']} | 🌐 <b>Plataforma:</b> {resultado['plataforma']}</p>
                        <p>📝 {resultado['descripcion']}</p>
                        <a href="{resultado['url']}" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; margin-top: 8px; font-weight: bold;">
                            ➡️ Acceder al curso
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Botones de descarga
        st.markdown("---")
        df = pd.DataFrame(resultados)
        
        if formato == "CSV (Excel)":
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Descargar en formato Excel (CSV)",
                data=csv,
                file_name=f"rutas_aprendizaje_{tema.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.warning("⚠️ No encontramos recursos para este tema. Por favor, intenta con otro término o nivel.")
        
        # Sugerir temas populares basados en búsquedas anteriores
        st.markdown("### 📌 Temas sugeridos basados en búsquedas populares:")
        temas_sugeridos = ["Python", "Inglés", "Marketing Digital", "Diseño Gráfico", "Finanzas Personales"]
        cols = st.columns(len(temas_sugeridos))
        for i, tema_sugerido in enumerate(temas_sugeridos):
            with cols[i]:
                if st.button(f"{tema_sugerido}", key=f"sugerido_{i}"):
                    tema = tema_sugerido
                    st.experimental_rerun()

# Mensaje inicial si no hay búsqueda
else:
    st.info("💡 Ingresa el tema que deseas aprender y selecciona el nivel para comenzar")
    
    # Mostrar estadísticas de nuestra base de conocimiento
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Estadísticas
    cursor.execute("SELECT COUNT(*) FROM sitios_ocultos WHERE activo = 1")
    total_sitios = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM busquedas_usuarios")
    total_busquedas = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(confianza) FROM sitios_ocultos WHERE activo = 1")
    confianza_promedio = cursor.fetchone()[0] or 0.5
    
    conn.close()
    
    # Estadísticas visualizadas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sitios Ocultos", total_sitios)
    with col2:
        st.metric("Búsquedas Realizadas", total_busquedas)
    with col3:
        st.metric("Confianza Promedio", f"{confianza_promedio:.2f}")
    
    # Ejemplos de sitios poco conocidos
    st.markdown("### 💎 Algunos tesoros educativos ocultos que hemos descubierto:")
    
    cols = st.columns(3)
    for i, sitio in enumerate(SITIOS_POCO_CONOCIDOS[:6]):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="border: 1px solid #e1e1e1; border-radius: 8px; padding: 12px; margin-bottom: 10px; background-color: #f8f9fa;">
                <h4 style="color: #667eea; margin-top: 0;">{sitio['nombre']}</h4>
                <p style="font-size: 0.9em; color: #666;">{sitio['descripcion'][:60]}...</p>
                <span style="background-color: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">
                    {sitio['nivel']}
                </span>
            </div>
            """, unsafe_allow_html=True)

# Barra lateral con información
with st.sidebar:
    st.header("🤖 Sobre nuestro Buscador Inteligente")
    st.markdown("""
    Esta herramienta no solo busca en plataformas conocidas, sino que:
    - 🕵️ **Descubre sitios educativos ocultos** que pocos conocen
    - 📈 **Aprende de tus búsquedas** para ofrecerte mejores resultados
    - ♻️ **Se actualiza constantemente** con nuevos recursos
    - 🌐 **Conecta con PythonAnywhere** para descubrir nuevos sitios
    """)
    
    st.markdown("---")
    st.subheader("📊 Estadísticas de Nuestra Base de Conocimiento")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sitios_ocultos")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT categoria, COUNT(*) FROM sitios_ocultos GROUP BY categoria")
        categorias = cursor.fetchall()
        
        conn.close()
        
        st.metric("Total de Sitios", total)
        
        if categorias:
            st.markdown("### 📂 Categorías")
            for cat, count in categorias:
                st.markdown(f"- **{cat}**: {count} sitios")
    except:
        st.warning("Base de datos no disponible")
    
    st.markdown("---")
    st.subheader("✨ ¿Cómo contribuir?")
    st.markdown("""
    ¿Conoces un sitio educativo poco conocido?
    1. Búscalo en nuestra herramienta
    2. Si no aparece, envíanos el enlace
    3. Lo analizamos y lo añadimos a nuestra base
    """)
    
    # Feedback del usuario
    with st.expander("📤 Enviar un sitio educativo"):
        with st.form("feedback_form"):
            url = st.text_input("URL del sitio educativo")
            nombre = st.text_input("Nombre del sitio")
            descripcion = st.text_area("Breve descripción")
            categoria = st.selectbox("Categoría", ["Programación", "Idiomas", "Ciencias", "Humanidades", "Arte", "Otros"])
            submit = st.form_submit_button("Enviar para revisión")
            
            if submit and url and nombre:
                if guardar_sitio_oculto(url, nombre, descripcion, categoria, "Todos", 0.2):
                    st.success("¡Gracias! Hemos recibido tu sugerencia y la revisaremos pronto.")
                else:
                    st.warning("Este sitio ya está en nuestra base de datos o hubo un error.")

# Pie de página
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 14px;">
    ✨ Base de conocimiento en crecimiento constante - Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}<br>
    🤖 Sistema de descubrimiento automático conectado a PythonAnywhere<br>
    💚 Proyecto de código abierto para democratizar el acceso al conocimiento
</div>
""", unsafe_allow_html=True)

# ========== CÓDIGO PARA PYTHONANYWHERE (ejecutado por programación) ==========
# Este código se ejecutaría en PythonAnywhere como tarea programada
if "ejecutar_descubrimiento" in st.query_params:
    st.title("🔍 Ejecutando descubrimiento automático de nuevos sitios")
    st.write("Este proceso se ejecuta automáticamente desde PythonAnywhere")
    
    with st.spinner("Descubriendo nuevos sitios educativos..."):
        nuevos = descubrir_nuevos_sitios()
    
    if nuevos:
        st.success(f"¡Descubiertos {len(nuevos)} nuevos sitios educativos!")
        for sitio in nuevos:
            st.write(f"✅ {sitio}")
    else:
        st.info("No se descubrieron nuevos sitios en esta ejecución")
    
    st.write("Próxima ejecución automática en 24 horas")
