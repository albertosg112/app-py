def main():
    """
    Función principal que renderiza la aplicación completa de Streamlit.
    """
    # --- Inicializaciones y renderizado de UI estática ---
    ensure_session()
    init_feature_flags()
    iniciar_tareas_background() # Debe llamarse solo una vez

    render_header()

    # --- Formulario de búsqueda ---
    # Usando I18N para internacionalización
    i18n = get_i18n(st.session_state.get('lang_ui', 'Español (es)'))

    tema = st.text_input(i18n["enter_topic"], placeholder="Ej: Python, IA, Finanzas...", key="search_topic_input")
    col_form1, col_form2 = st.columns(2)
    nivel = col_form1.selectbox(i18n["level"], ["Cualquiera", "Principiante", "Intermedio", "Avanzado"], key="search_level_select")
    idioma = col_form2.selectbox(i18n["language"], ["Español (es)", "Inglés (en)", "Portugués (pt)"], key="search_lang_select")
    st.session_state['lang_ui'] = idioma # Guardar selección para recargas

    # --- Lógica de Búsqueda y Estado ---
    if st.button(i18n["search_button"], type="primary", use_container_width=True):
        if not (tema or "").strip():
            st.warning("Por favor ingresa un tema para buscar.")
        else:
            with st.spinner("🔍 Buscando en múltiples fuentes..."):
                # ### CORRECCIÓN ASYNCIO ###
                # Manera más segura de correr un bucle en Streamlit
                try:
                    resultados = asyncio.run(buscar_recursos_multicapa_ext(tema.strip(), idioma, nivel))
                    st.session_state.resultados = resultados
                    if resultados:
                        registrar_muestreo_estadistico(resultados, tema.strip(), idioma, nivel)
                except Exception as e:
                    logger.error(f"Error en la ejecución de búsqueda asíncrona: {e}")
                    st.error(f"Ocurrió un error durante la búsqueda: {e}")
                    st.session_state.resultados = []
            st.rerun()

    # --- Renderizado de Contenido Dinámico ---
    current_results = st.session_state.get('resultados', [])
    if current_results:
        st.success(i18n["results_found"].format(n=len(current_results)))
        if GROQ_AVAILABLE and st.session_state.features.get("enable_groq_analysis", True):
             planificar_analisis_ia(current_results)
             # Pequeña pausa para que la UI se actualice mientras los workers empiezan
             time.sleep(0.4)

        for i, r in enumerate(current_results):
            mostrar_recurso(r, i) # Esta función ya contiene el botón de registrar click
            # boton_registrar_click(r, tema) # Eliminar esta llamada duplicada

    elif 'resultados' in st.session_state: # Mostrar solo si se ha hecho una búsqueda
        st.warning(i18n["no_results"])


    # --- Paneles Avanzados y de Diagnóstico ---
    st.markdown("---")
    st.markdown("### 🧭 Paneles Avanzados y de Diagnóstico")
    colA, colB = st.columns(2)
    with colA:
        panel_configuracion_avanzada()
        panel_favoritos_ui()
        panel_feedback_ui(current_results)
        panel_export_import_ui(current_results) # Movido aquí para agrupar
        reportes_rapidos() # Movido aquí

    with colB:
        admin_dashboard()
        log_viewer()
        panel_cache_viewer()

    # --- Secciones de Ayuda y Otros ---
    st.markdown("---")
    render_help()
    keyboard_tips()
    render_telemetry()
    if st.session_state.features.get("enable_debug_mode", False):
        run_basic_tests() # Ejecutar solo en modo debug

    # --- Componentes Persistentes (Sidebar y Footer) ---
    sidebar_chat()
    sidebar_status()
    render_footer()

    # La gestión de fin de sesión (`end_session`) es compleja en Streamlit
    # porque el script no "termina". Se deja fuera del flujo principal por ahora.
# ============================================================
# 20. DUCKDUCKGO FALLBACK (OPCIONAL)
# ============================================================
@async_profile
async def buscar_en_duckduckgo(tema: str, idioma: str, nivel: str) -> List[RecursoEducativo]:
    if not st.session_state.features.get("enable_ddg_fallback", False):
        return []
    try:
        q = quote_plus(f"{tema} free course {nivel if nivel!='Cualquiera' else ''}".strip())
        url = f"https://duckduckgo.com/html/?q={q}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=8) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text()
                links = re.findall(r'href="(https?://[^"]+)"', text)
                resultados: List[RecursoEducativo] = []
                for link in links[:5]:
                    titulo = "Resultado en DuckDuckGo"
                    descripcion = "Resultado alternativo desde DuckDuckGo (parseo simple)."
                    if not es_recurso_educativo_valido(link, titulo, descripcion):
                        continue
                    resultados.append(RecursoEducativo(
                        id=generar_id_unico(link),
                        titulo=f"🦆 {titulo} — {tema}",
                        url=link,
                        descripcion=descripcion,
                        plataforma=extraer_plataforma(link),
                        idioma=idioma,
                        nivel=nivel if nivel != "Cualquiera" else "Intermedio",
                        categoria=determinar_categoria(tema),
                        certificacion=None,
                        confianza=0.70,
                        tipo="verificada",
                        ultima_verificacion=datetime.now().isoformat(),
                        activo=True,
                        metadatos={"fuente": "duckduckgo"}
                    ))
                return resultados
    except Exception as e:
        logger.error(f"DDG fallback error: {e}")
        return []

@async_profile
async def buscar_recursos_multicapa_ext(tema: str, idioma_seleccion_ui: str, nivel: str) -> List[RecursoEducativo]:
    base = await buscar_recursos_multicapa(tema, idioma_seleccion_ui, nivel)
    if not base and st.session_state.features.get("enable_ddg_fallback", False):
        idioma = get_codigo_idioma(idioma_seleccion_ui)
        ddg = await buscar_en_duckduckgo(tema, idioma, nivel)
        base.extend(ddg)
    base = eliminar_duplicados(base)
    base.sort(key=lambda x: x.confianza, reverse=True)
    return base[:st.session_state.features.get("max_results", 15)]

# ============================================================
# 21. ANALÍTICAS Y TRAZABILIDAD
# ============================================================
def log_search_event(tema: str, idioma: str, nivel: str, plataforma_origen: str, mostrados: int):
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO analiticas_busquedas (tema, idioma, nivel, timestamp, plataforma_origen, veces_mostrado)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tema, idioma, nivel, datetime.now().isoformat(), plataforma_origen, mostrados))
            conn.commit()
    except Exception as e:
        logger.error(f"Error log_search_event: {e}")

def log_click_event(tema: str, url: str, plataforma: str):
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE analiticas_busquedas
                SET veces_clickeado = veces_clickeado + 1
                WHERE tema = ?
                ORDER BY id DESC LIMIT 1
            """, (tema,))
            conn.commit()
    except Exception as e:
        logger.error(f"Error log_click_event: {e}")

def registrar_muestreo_estadistico(resultados: List[RecursoEducativo], tema: str, idioma_ui: str, nivel: str):
    idioma = get_codigo_idioma(idioma_ui)
    plataformas = ", ".join(sorted(set(r.plataforma for r in resultados)))
    log_search_event(tema, idioma, nivel, plataformas, len(resultados))

def boton_registrar_click(r: RecursoEducativo, tema: str):
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔖 Registrar click", key=f"reg_click_{r.id}"):
            log_click_event(tema, r.url, r.plataforma)
            st.success("Click registrado")

# ============================================================
# 23. ADMIN DASHBOARD
# ============================================================
def admin_dashboard():
    st.markdown("### 🛠️ Panel admin")
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM analiticas_busquedas")
            t_busquedas = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM plataformas_ocultas WHERE activa = 1")
            t_plats = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM favoritos")
            t_favs = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM feedback")
            t_fb = c.fetchone()[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔎 Búsquedas", t_busquedas)
        col2.metric("📚 Plataformas activas", t_plats)
        col3.metric("⭐ Favoritos", t_favs)
        col4.metric("📝 Feedback", t_fb)
    except Exception as e:
        st.error(f"Error admin: {e}")

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Vacuum DB", use_container_width=True):
            try:
                with get_db_connection(DB_PATH) as conn:
                    conn.execute("VACUUM")
                    conn.commit()
                st.success("DB optimizada (VACUUM)")
            except Exception as e:
                st.error(f"Error VACUUM: {e}")
    with colB:
        if st.button("🧹 Limpiar analíticas", use_container_width=True):
            try:
                with get_db_connection(DB_PATH) as conn:
                    conn.execute("DELETE FROM analiticas_busquedas")
                    conn.commit()
                st.success("Analíticas limpiadas")
            except Exception as e:
                st.error(f"Error limpieza: {e}")

# ============================================================
# 24. DIAGNÓSTICO DE ERRORES (LOG VIEWER)
# ============================================================
def log_viewer(max_lines: int = 200):
    st.markdown("### 🪵 Visor de logs")
    path = "buscador_cursos.log"
    if not os.path.exists(path):
        st.info("No hay logs aún.")
        return
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail = lines[-max_lines:]
        st.code("".join(tail))
    except Exception as e:
        st.error(f"Error leyendo logs: {e}")

# ============================================================
# 25. SESIONES DE USUARIO (CORREGIDO Y BLINDADO)
# ============================================================
def ensure_session():
    # 1. Crear el ID en memoria de Streamlit primero
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess_{int(time.time())}_{random.randint(1000,9999)}"
        
        # 2. Intentar guardar en Base de Datos de forma segura
        try:
            with get_db_connection(DB_PATH) as conn:
                c = conn.cursor()
                
                # --- PARCHE IMPORTANTE ---
                # Creamos la tabla AQUÍ mismo por si acaso no existe todavía.
                # Esto evita el error "no such table: sesiones" al iniciar.
                c.execute('''
                CREATE TABLE IF NOT EXISTS sesiones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    device TEXT,
                    prefs_json TEXT
                )
                ''')
                # -------------------------

                # Ahora sí insertamos sin miedo
                c.execute("INSERT INTO sesiones (session_id, started_at, device, prefs_json) VALUES (?, ?, ?, ?)",
                          (st.session_state.session_id, datetime.now().isoformat(), "web", safe_json_dumps(st.session_state.get('features', {}))))
                conn.commit()
                
        except Exception as e:
            # Si la base de datos falla (por permisos o bloqueo), 
            # solo lo registramos en los logs PERO NO ROMPEMOS la app.
            logger.error(f"⚠️ Error no crítico iniciando sesión DB: {e}")
# ============================================================
# 26. ACCESOS RÁPIDOS (TECLAS) Y AYUDA VISUAL
# ============================================================
def keyboard_tips():
    st.markdown("### ⌨️ Atajos")
    st.markdown("- Shift+Enter: enviar en chat")
    st.markdown("- Ctrl+K: abrir búsqueda rápida del navegador")
    st.markdown("- Alt+R: refrescar (según navegador)")
    st.markdown("- Ctrl+L: enfocarse en barra de URL (navegador)")

# ============================================================
# 27. EXTENSIONES: ETIQUETAS Y NOTAS EN RESULTADOS
# ============================================================
def notas_usuario_widget(r: RecursoEducativo):
    st.markdown("#### 🗒️ Notas del usuario")
    default_note = ""
    note = st.text_area(f"Notas para: {r.titulo}", default_note, key=f"note_{r.id}")
    if st.button("💾 Guardar nota", key=f"save_note_{r.id}"):
        ok = agregar_favorito(r, note)
        if ok:
            st.success("Nota guardada como favorito.")
        else:
            st.error("No se pudo guardar la nota.")

def render_notas_para_resultados(resultados: List[RecursoEducativo]):
    st.markdown("### 🗂️ Notas rápidas")
    for r in resultados[:3]:
        notas_usuario_widget(r)

# ============================================================
# 28. REPORTES RÁPIDOS
# ============================================================
def reportes_rapidos():
    st.markdown("### 📈 Reportes rápidos")
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT tema, COUNT(*) AS total
                FROM analiticas_busquedas
                GROUP BY tema
                ORDER BY total DESC
                LIMIT 5
            """)
            rows = c.fetchall()
        if rows:
            df = pd.DataFrame([{"Tema": r[0], "Búsquedas": r[1]} for r in rows])
            st.bar_chart(df.set_index("Tema"))
        else:
            st.info("Aún no hay suficientes datos para reportes.")
    except Exception as e:
        st.error(f"Error reporte: {e}")

# ============================================================
# 16. PRUEBAS BÁSICAS (Sanity Checks)
# ============================================================
def run_basic_tests():
    with st.expander("🧪 Pruebas básicas (Diagnóstico)"):
        try:
            # Test de utilidades
            assert determinar_nivel("Curso avanzado", "Cualquiera") == "Avanzado"
            assert determinar_nivel("Curso básico", "Cualquiera") == "Principiante"
            assert determinar_nivel("Curso intermedio", "Cualquiera") == "Intermedio"
            assert determinar_categoria("Python para ciencia de datos") == "Data Science" or determinar_categoria("Python para ciencia de datos") == "Programación"
            # Test DB
            with get_db_connection(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM plataformas_ocultas")
                count = c.fetchone()[0]
                assert count >= 5
            st.success("Pruebas básicas OK")
        except AssertionError:
            st.error("Falló una aserción en pruebas básicas")
        except Exception as e:
            st.error(f"Error en pruebas básicas: {e}")

# ============================================================
# 17. SECCIÓN AYUDA & ATAJOS
# ============================================================
def render_help():
    with st.expander("❓ Ayuda"):
        st.markdown("- Escribe un tema y pulsa 'Buscar Cursos'.")
        st.markdown("- Activa/desactiva características en Configuración avanzada.")
        st.markdown("- Añade favoritos y exporta resultados a CSV.")
        st.markdown("- Usa el chat IA para consejos rápidos (si Groq está disponible).")
        st.markdown("- Si la IA muestra HTML/JSON, se limpiará automáticamente en la UI (parche aplicado).")
        st.markdown("- Atajos: [Shift+Enter] para enviar en chat, [Alt+R] para refrescar (según navegador).")

# ============================================================
# 18. TELEMETRÍA OPT-OUT (solo bandera persistente)
# ============================================================
def set_telemetry_opt_out(value: bool):
    try:
        with get_db_connection(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO configuracion (clave, valor) VALUES (?, ?)", ("telemetry_opt_out", "1" if value else "0"))
            conn.commit()
        st.success("Preferencia de telemetría actualizada")
    except Exception as e:
        logger.error(f"Error en telemetría opt-out: {e}")
        st.error("No se pudo actualizar la preferencia")

def render_telemetry():
    with st.expander("🔒 Privacidad y Telemetría"):
        try:
            with get_db_connection(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT valor FROM configuracion WHERE clave = 'telemetry_opt_out'")
                row = c.fetchone()
                opt_out = (row and row[0] == "1")
        except Exception:
            opt_out = False
        new_val = st.checkbox("Desactivar telemetría anónima", value=opt_out)
        if new_val != opt_out:
            set_telemetry_opt_out(new_val)

# ============================================================
# 29. EXTENDER MAIN CON NUEVAS SECCIONES (CORE LOGIC)
# ============================================================
def main_extended():
    ensure_session()
    # Header y búsqueda
    render_header()
    iniciar_tareas_background()
    tema, nivel, idioma, buscar = render_search_form()

    resultados: List[RecursoEducativo] = []
    
    # Manejo de estado de resultados para persistencia durante interacciones
    if 'last_results' not in st.session_state:
        st.session_state.last_results = []

    if buscar:
        if not (tema or "").strip():
            st.warning("Por favor ingresa un tema.")
        else:
            with st.spinner("🔍 Buscando en múltiples fuentes..."):
                # PARCHE ASYNCIO PARA STREAMLIT CLOUD
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                resultados = loop.run_until_complete(buscar_recursos_multicapa_ext(tema.strip(), idioma, nivel))
                st.session_state.last_results = resultados
            
            registrar_muestreo_estadistico(resultados, tema.strip(), idioma, nivel)
    else:
        resultados = st.session_state.last_results

    if resultados:
        render_results(resultados)
        render_notas_para_resultados(resultados)

    # Paneles avanzados
    st.markdown("### 🧭 Paneles avanzados")
    colA, colB, colC = st.columns(3)
    with colA:
        panel_configuracion_avanzada()
    with colB:
        panel_cache_viewer()
    with colC:
        panel_favoritos_ui()

    # Feedback y export/import
    st.markdown("---")
    panel_feedback_ui(resultados)
    panel_export_import_ui(resultados)

    # Admin y diagnósticos
    st.markdown("---")
    admin_dashboard()
    reportes_rapidos()
    log_viewer()

    # Ayuda y atajos
    render_help()
    keyboard_tips()
    render_telemetry()
    run_basic_tests()

    # Sidebar y Footer
    sidebar_chat()
    sidebar_status()
    render_footer()

# ============================================================
# 30. ARRANQUE (PUNTO DE ENTRADA ÚNICO)
# ============================================================
if __name__ == "__main__":
    try:
        main_extended()
    except Exception as e:
        st.error(f"Error crítico en la aplicación: {e}")


