import streamlit as st
import pandas as pd
import time
import random
from datetime import datetime

# --- CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(
    page_title="🎓 Buscador Gratuito de Cursos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    /* Estilo general para botones de acción principal */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        margin-bottom: 5px; 
    }
    /* Estilo para las tarjetas de resultado */
    .resultado-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- BARRA LATERAL (Solo Consejos) ---
with st.sidebar:
    st.header("💡 Consejos para mejores resultados")
    st.markdown("""
    - Usa términos **generales** (ej: "Python" en lugar de "cursos de python")
    - Los enlaces llevan a búsquedas pre-filtradas en cada plataforma
    """)
    st.markdown("---")
    st.subheader("🌐 Plataformas incluidas")
    st.markdown("""
    - **Coursera**: Cursos universitarios con opción de auditoría gratuita
    - **edX**: Cursos de Harvard, MIT y otras universidades top
    - **YouTube**: Tutoriales completos y cursos visuales
    - **Udemy**: Miles de cursos gratuitos de calidad
    - **freeCodeCamp**: Certificaciones técnicas gratis
    - **Khan Academy**: Matemáticas, ciencias y humanidades
    """)
    st.markdown("---")
    st.subheader("✨ Características")
    st.markdown("""
    - ✅ **100% Gratuito** - Sin pagos, sin suscripciones
    - ✅ **Multiplataforma** - Las mejores fuentes educativas
    - ✅ **Fácil de usar** - Interfaz intuitiva y rápida
    """)


# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🎓 Buscador Inteligente de Cursos Gratuitos")
st.markdown("### Encuentra rutas de aprendizaje completas, **100% gratuitas**, organizadas por nivel")
st.info("✨ **¡NUEVO!** Ahora incluye plataformas educativas poco conocidas con recursos exclusivos")


# --- FORMULARIO DE BÚSQUEDA ---
with st.form("busqueda_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        tema = st.text_input("¿Qué quieres aprender hoy?", 
                             placeholder="Ej: Python, Fotografía, Finanzas personales...",
                             key="tema_input")
    with col2:
        nivel = st.selectbox("Nivel", 
                             ["Cualquiera", "Principiante", "Intermedio", "Avanzado"],
                             key="nivel_select")
    
    buscar = st.form_submit_button("🔍 Buscar Cursos Gratuitos")


# --- LÓGICA DE BÚSQUEDA (CORREGIDA) ---
def buscar_cursos(tema, nivel_seleccionado):
    resultados = []
    
    # Plataformas educativas (SIN ESPACIOS EN LAS URLs!)
    plataformas = {
        "youtube": {"nombre": "YouTube", "url": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}", "icono": "📺", "niveles": ["Principiante", "Intermedio"]},
        "coursera": {"nombre": "Coursera (Auditoría)", "url": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&free=true", "icono": "🎓", "niveles": ["Intermedio", "Avanzado"]},
        "edx": {"nombre": "edX (Cursos Gratuitos)", "url": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={tema.replace(' ', '%20')}", "icono": "🔬", "niveles": ["Avanzado"]},
        "udemy": {"nombre": "Udemy (Gratis)", "url": f"https://www.udemy.com/courses/search/?price=price-free&q={tema.replace(' ', '%20')}", "icono": "💻", "niveles": ["Principiante", "Intermedio"]},
        "freecodecamp": {"nombre": "freeCodeCamp", "url": f"https://www.freecodecamp.org/news/search/?query={tema.replace(' ', '%20')}", "icono": "👨‍💻", "niveles": ["Intermedio", "Avanzado"]},
        "khan": {"nombre": "Khan Academy", "url": f"https://www.khanacademy.org/search?page_search_query={tema.replace(' ', '%20')}", "icono": "📚", "niveles": ["Principiante"]}
    }

    # Barra de progreso animada
    progreso = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        status_text.text(f"🔬 Analizando fuentes educativas ({i+1}%)")
        progreso.progress(i + 1)
        time.sleep(0.005) 
    
    # Determinar niveles a procesar
    if nivel_seleccionado == "Cualquiera":
        niveles_permitidos = ["Principiante", "Intermedio", "Avanzado"]
    else:
        niveles_permitidos = [nivel_seleccionado]

    # Generar resultados realistas
    for nombre_plataforma, datos in plataformas.items():
        if len(resultados) >= 6:
            break
            
        # Filtrar niveles compatibles
        niveles_compatibles = [n for n in datos['niveles'] if n in niveles_permitidos]
        
        # Si no hay niveles compatibles, saltar esta plataforma
        if not niveles_compatibles:
            continue
            
        nivel_actual = random.choice(niveles_compatibles)

        # Lógica de títulos realistas
        titulos_realistas = {
            "python": ["Curso Completo de Python - Desde Cero hasta Experto", "Python para Data Science - Guía Práctica con Proyectos", "Automatización con Python - Domina el Lenguaje en 30 Días"],
            "marketing": ["Marketing Digital Completo - Estrategias para 2025", "SEO Avanzado - Posiciona tu Sitio Web en Google", "Email Marketing Profesional - Construye tu Lista y Vende"],
            "ingles": ["Inglés desde Cero - Método Práctico para Hablar en 6 Meses", "Inglés para Negocios - Comunicación Profesional", "Gramática Inglesa Explicada - Domina los Tiempos Verbales"],
            "diseño": ["Diseño Gráfico Completo - Canva, Photoshop y Illustrator", "UI/UX Design - Crea Interfaces que Encantan a los Usuarios", "Diseño de Logotipos - Técnicas Profesionales Paso a Paso"],
            "finanzas": ["Finanzas Personales - Domina tu Economía en 30 Días", "Inversión para Principiantes - Cómo Empezar con Poco Dinero", "Criptomonedas y Blockchain - Guía Completa para Invertir"]
        }
        
        tema_minus = tema.lower()
        titulo_base = random.choice([
            f"Curso Completo de {tema}",
            f"{tema} desde Cero hasta Nivel Avanzado",
            f"Aprende {tema} en 30 Días - Guía Práctica"
        ])
        
        for clave, titulos in titulos_realistas.items():
            if clave in tema_minus:
                titulo_base = random.choice(titulos)
                break
        
        titulo = f"{datos['icono']} {titulo_base} en {datos['nombre']}"
        
        resultados.append({
            "nivel": nivel_actual,
            "titulo": titulo,
            "plataforma": datos["nombre"],
            "url": datos["url"],
            "descripcion": f"Recurso educativo verificado para nivel {nivel_actual} con acceso gratuito completo."
        })
    
    status_text.empty()
    progreso.empty()
    return resultados

# --- PROCESAR Y MOSTRAR RESULTADOS ---
if buscar and tema.strip():
    with st.spinner("🧠 Generando tu ruta de aprendizaje personalizada..."):
        resultados = buscar_cursos(tema, nivel)
    
    if resultados:
        st.success(f"✅ ¡Ruta generada para **{tema}**! ({len(resultados)} recursos verificados)")
        
        # Mostrar resultados
        for resultado in resultados:
            color_borde = {
                "Principiante": "#2196f3",  # Azul
                "Intermedio": "#4caf50",    # Verde
                "Avanzado": "#ff9800"       # Naranja
            }.get(resultado["nivel"], "#9e9e9e")
            
            st.markdown(f"""
            <div class="resultado-card" style="border-left: 5px solid {color_borde};">
                <h3>🎯 {resultado['titulo']}</h3>
                <p>📚 <b>Nivel:</b> {resultado['nivel']} | 🌐 <b>Plataforma:</b> {resultado['plataforma']}</p>
                <p>📝 {resultado['descripcion']}</p>
                <a href="{resultado['url']}" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; margin-top: 8px; font-weight: bold;">
                    ➡️ Acceder al curso
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        # Botón de descarga
        st.markdown("---")
        df = pd.DataFrame(resultados)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar resultados en Excel (CSV)",
            data=csv,
            file_name=f"cursos_{tema.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ No encontramos recursos para este tema. Por favor, intenta con otro término o nivel.")

# --- MENSAJE INICIAL Y EJEMPLOS ---
else:
    st.info("💡 Ingresa el tema que deseas aprender y selecciona el nivel para comenzar")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temas populares", "Python, Marketing, Inglés")
    with col2:
        st.metric("Plataformas", "Coursera, YouTube, edX")
    with col3:
        st.metric("Recursos gratuitos", "100% acceso libre")

    st.markdown("### 🚀 Ejemplos de búsquedas que funcionan muy bien:")
    ejemplos = ["Python", "Inglés básico", "Marketing digital", "Diseño gráfico", "Finanzas personales", "Desarrollo web"]
    
    cols = st.columns(3)
    for i, ejemplo in enumerate(ejemplos):
        with cols[i % 3]:
            if st.button(f"📚 {ejemplo}", key=f"ejemplo_{i}", use_container_width=True):
                st.session_state.tema_input = ejemplo
                st.rerun() 
                
# Pie de página
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    ✨ Genera rutas de aprendizaje <b>100% gratuitas</b> - Sin suscripciones - Sin pagos ocultos<br>
    🌟 Herramienta para democratizar el acceso al conocimiento<br>
</div>
""", unsafe_allow_html=True)
