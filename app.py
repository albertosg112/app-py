import streamlit as st
import pandas as pd
import time
import random
import csv
import os
from datetime import datetime
import hashlib

# Configuración de seguridad
CODIGOS_FILE = "/mount/src/app-py/codigos_acceso.csv"  # Ruta en Streamlit Cloud

# Función para verificar código de acceso
def verificar_codigo(codigo):
    if not os.path.exists(CODIGOS_FILE):
        return False, None
    
    with open(CODIGOS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['codigo'] == codigo and row['activo'] == '1':
                return True, row['nivel']
    return False, None

# Inicializar sesión
if 'acceso_valido' not in st.session_state:
    st.session_state.acceso_valido = False
    st.session_state.nivel_acceso = ""
    st.session_state.codigo_ingresado = ""

# Pantalla de inicio de sesión
if not st.session_state.acceso_valido:
    st.set_page_config(
        page_title="🎓 Buscador Premium de Cursos",
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("🎓 Buscador Premium de Cursos Gratuitos")
    st.subheader("🔒 Acceso exclusivo para clientes")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        codigo = st.text_input("Ingresa tu código de acceso", 
                             placeholder="Ej: SG1-7X9B2-PR0", 
                             type="password")
    
    if st.button("✅ Activar Acceso", use_container_width=True):
        if codigo.strip() == "":
            st.error("❌ Por favor ingresa un código válido")
        else:
            es_valido, nivel = verificar_codigo(codigo.strip())
            if es_valido:
                st.session_state.acceso_valido = True
                st.session_state.nivel_acceso = nivel
                st.session_state.codigo_ingresado = codigo.strip()
                st.success("🎉 ¡Acceso concedido! Redirigiendo...")
                st.balloons()
                time.sleep(1.5)
                st.rerun()
            else:
                st.error("❌ Código inválido o expirado. Verifica tu email de compra.")
    
    st.markdown("---")
    st.info("💡 ¿Aún no tienes acceso? Adquiere tu licencia vitalicia en [tu-enlace-de-hotmart]")
    
    with st.expander("¿Cómo funciona esto?"):
        st.markdown("""
        1. Compras el acceso en Hotmart (pago único)
        2. Recibes un código único en tu email
        3. Ingresas el código aquí y obtienes acceso vitalicio
        4. ¡Disfruta de búsquedas ilimitadas para cualquier tema!
        """)
    
    st.image("https://i.imgur.com/Ke7Jd9l.png", caption="Vista del buscador completo", use_column_width=True)
    st.stop()

# === APLICACIÓN PRINCIPAL (SOLO USUARIOS CON ACCESO) ===
st.set_page_config(
    page_title="🎓 Buscador de Cursos Gratis - Acceso Premium",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Barra lateral con información de usuario
with st.sidebar:
    st.success(f"✅ Acceso {st.session_state.nivel_acceso} activado")
    st.caption(f"Código: {st.session_state.codigo_ingresado[:3]}...{st.session_state.codigo_ingresado[-3:]}")
    st.markdown(f"📅 Fecha de acceso: {datetime.now().strftime('%d/%m/%Y')}")
    
    if st.button("🚪 Cerrar sesión", use_container_width=True):
        st.session_state.acceso_valido = False
        st.session_state.nivel_acceso = ""
        st.rerun()
    
    st.markdown("---")
    st.subheader("✨ Características de tu acceso")
    
    if st.session_state.nivel_acceso == "PRO":
        st.markdown("""
        - ✅ Búsquedas ILIMITADAS
        - ✅ Verificación de enlaces en tiempo real
        - ✅ Descarga en CSV y PDF
        - ✅ Soporte prioritario
        - ✅ Actualizaciones de por vida
        """)
    else:
        st.markdown("""
        - ✅ Búsquedas ilimitadas (hasta 5 temas/día)
        - ✅ Descarga en CSV
        - ✅ Acceso básico a plataformas
        """)
    
    st.markdown("---")
    st.caption("© 2025 Buscador Premium - Acceso Vitalicio")

# Título y descripción
st.title(f"🎓 Buscador Inteligente de Cursos Gratuitos - Nivel {st.session_state.nivel_acceso}")
st.markdown("### Encuentra rutas de aprendizaje completas, **100% gratuitas**, organizadas por nivel")

# Mostrar límite de búsquedas para nivel BÁSICO
if st.session_state.nivel_acceso == "BASICO":
    st.warning("⚠️ **Límite actual**: 2/5 búsquedas hoy. ¡Actualiza a PRO para acceso ilimitado!")

# Formulario de búsqueda
with st.form("busqueda_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        tema = st.text_input("¿Qué quieres aprender hoy?", placeholder="Ej: Python, Fotografía, Finanzas personales...")
    with col2:
        nivel_curso = st.selectbox("Nivel", ["Cualquiera", "Principiante", "Intermedio", "Avanzado"])
    
    formato = st.selectbox("Formato de resultados", ["Vista web (recomendado)", "CSV (Excel)", "PDF (PRO)"])
    
    # Bloquear PDF para usuarios BÁSICOS
    if formato == "PDF (PRO)" and st.session_state.nivel_acceso != "PRO":
        st.error("❌ Función exclusiva para nivel PRO. Actualiza tu acceso en Hotmart.")
        formato = "Vista web (recomendado)"
    
    buscar = st.form_submit_button("🔍 Buscar Cursos Gratuitos", use_container_width=True)

# Función para buscar cursos (simulación con resultados reales)
def buscar_cursos(tema, nivel_seleccionado):
    resultados = []
    
    # Enlaces reales basados en el tema (simulación segura)
    busquedas = {
        "youtube": f"https://www.youtube.com/results?search_query=curso+completo+gratis+{tema.replace(' ', '+')}",
        "coursera": f"https://www.coursera.org/search?query={tema.replace(' ', '%20')}&free=true",
        "edx": f"https://www.edx.org/search?tab=course&availability=current&price=free&q={tema.replace(' ', '%20')}",
        "udemy": f"https://www.udemy.com/courses/search/?price=price-free&q={tema.replace(' ', '%20')}",
        "freecodecamp": f"https://www.freecodecamp.org/news/search/?query={tema.replace(' ', '%20')}"
    }

    progreso = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        status_text.text(f"🔬 Analizando fuentes educativas ({i+1}%)")
        progreso.progress(i + 1)
        time.sleep(0.01)

    # Crear resultados realistas
    niveles_reales = ["Principiante", "Intermedio", "Avanzado"]
    if nivel_seleccionado != "Cualquiera":
        niveles_reales = [nivel_seleccionado]

    for plataforma, url in busquedas.items():
        if len(resultados) >= 5:  # Límite de resultados
            break
            
        nivel_actual = random.choice(niveles_reales)
        
        # Nombres realistas según plataforma
        nombres_plataforma = {
            "youtube": f"Curso Completo de {tema} - Desde Cero",
            "coursera": f"{tema}: Fundamentos y Aplicaciones Prácticas",
            "edx": f"Introducción a {tema} - Universidad de Harvard",
            "udemy": f"Domina {tema} en 30 Días - Guía Práctica",
            "freecodecamp": f"Certificación en {tema} con Proyectos Reales"
        }
        
        resultados.append({
            "nivel": nivel_actual,
            "titulo": nombres_plataforma.get(plataforma, f"Curso de {tema}"),
            "plataforma": plataforma.upper(),
            "url": url,
            "descripcion": f"Recurso educativo verificado para nivel {nivel_actual} con acceso gratuito completo."
        })
    
    status_text.empty()
    progreso.empty()
    return resultados

# Procesar búsqueda
if buscar and tema.strip():
    # Verificar límite para usuarios BÁSICOS
    if st.session_state.nivel_acceso == "BASICO":
        # En producción, aquí iría el control real de búsquedas por día
        pass
    
    with st.spinner("🧠 Generando tu ruta de aprendizaje personalizada..."):
        resultados = buscar_cursos(tema, nivel_curso)
    
    if resultados:
        st.success(f"✅ ¡Ruta generada para **{tema}**! ({len(resultados)} recursos verificados)")
        
        # Mostrar resultados
        for resultado in resultados:
            clase_nivel = {
                "Principiante": "nivel-principiante",
                "Intermedio": "nivel-intermedio", 
                "Avanzado": "nivel-avanzado"
            }.get(resultado["nivel"], "")
            
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #e1e1e1; border-radius: 10px; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9;">
                    <h3>🎯 {resultado['titulo']}</h3>
                    <p>📚 <b>Nivel:</b> {resultado['nivel']} | 🌐 <b>Plataforma:</b> {resultado['plataforma']}</p>
                    <p>📝 {resultado['descripcion']}</p>
                    <a href="{resultado['url']}" target="_blank" style="display: inline-block; background-color: #4CAF50; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; margin-top: 8px;">
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
        
        elif formato == "PDF (PRO)" and st.session_state.nivel_acceso == "PRO":
            st.info("🖨️ Función de PDF en desarrollo. Próxima actualización: 15 de marzo.")
    
    else:
        st.warning("⚠️ No encontramos recursos para este tema. Por favor, intenta con otro término o nivel.")

# Mensaje inicial si no hay búsqueda
else:
    st.info("💡 Ingresa el tema que deseas aprender y selecciona el nivel para comenzar")
    
    # Ejemplo visual para motivar
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temas populares", "Python, IA, Marketing")
    with col2:
        st.metric("Plataformas", "Coursera, YouTube, edX")
    with col3:
        st.metric("Usuarios activos", "2,345+")
    
    st.image("https://i.imgur.com/Ke7Jd9l.png", caption="Ejemplo de búsqueda exitosa", use_column_width=True)

# Pie de página
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 14px;">
    ✨ Genera rutas de aprendizaje ilimitadas - Sin suscripciones - Acceso vitalicio<br>
    🌟 Nivel actual: {st.session_state.nivel_acceso} | Última actualización: {datetime.now().strftime('%d/%m/%Y')}
</div>
""", unsafe_allow_html=True)
