# ia_module.py
# Módulo autónomo de IA para análisis educativo y chat
# Compatible con Groq (API) y fallback si no está disponible
# Reutilizable en cualquier app (Streamlit, Flask, script, etc.)

import os
import logging
import json
import re
from typing import Dict, Any, Optional, List

# Configuración básica de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IA_Module")

# --- 1. Carga la clave de Groq desde variable de entorno ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- 2. Detecta si Groq está disponible ---
GROQ_AVAILABLE = False
try:
    import groq
    if GROQ_API_KEY and len(GROQ_API_KEY) >= 10:
        GROQ_AVAILABLE = True
        logger.info("✅ Groq API lista para usar")
    else:
        logger.warning("⚠️ GROQ_API_KEY no configurada o inválida")
except ImportError:
    logger.warning("❌ Librería 'groq' no instalada. Ejecuta: pip install groq")

# --- 3. Funciones de utilidad segura ---
def safe_json_loads(text: str, default_value: Any = None) -> Any:
    if default_value is None:
        default_value = {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default_value

# --- 4. Función principal: análisis de curso educativo ---
def analizar_recurso_groq(
    titulo: str,
    descripcion: str,
    nivel: str,
    categoria: str,
    plataforma: str
) -> Dict[str, Any]:
    """
    Evalúa un curso con IA y devuelve métricas estructuradas.
    Si Groq no está disponible, devuelve valores por defecto.
    """
    if not GROQ_AVAILABLE:
        return {
            "calidad_ia": 0.8,
            "relevancia_ia": 0.8,
            "recomendacion_personalizada": "IA no disponible. Sistema en modo básico.",
            "razones_calidad": [],
            "advertencias": ["Groq desactivado o sin clave API"]
        }

    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        prompt = f"""
Evalúa este curso educativo. Devuelve SOLO un objeto JSON válido con estas claves:
- "calidad_educativa": número entre 0.0 y 1.0
- "relevancia_usuario": número entre 0.0 y 1.0
- "razones_calidad": lista de 2-3 razones breves (strings)
- "recomendacion_personalizada": string útil de 1-2 oraciones
- "advertencias": lista (puede estar vacía)

Título: {titulo}
Descripción: {descripcion}
Nivel: {nivel}
Categoría: {categoria}
Plataforma: {plataforma}

JSON:
"""
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=600
        )
        contenido = (resp.choices[0].message.content or "").strip()

        # Extraer bloque JSON incluso si hay texto adicional
        json_match = re.search(r'\{.*\}', contenido, re.DOTALL)
        if json_match:
            data = safe_json_loads(json_match.group())
            return {
                "calidad_ia": float(data.get("calidad_educativa", 0.8)),
                "relevancia_ia": float(data.get("relevancia_usuario", 0.8)),
                "recomendacion_personalizada": str(data.get("recomendacion_personalizada", "Curso recomendado.")),
                "razones_calidad": list(data.get("razones_calidad", [])),
                "advertencias": list(data.get("advertencias", []))
            }
        else:
            raise ValueError("No se encontró JSON válido en la respuesta de Groq")

    except Exception as e:
        logger.error(f"Error en análisis IA: {e}")
        return {
            "calidad_ia": 0.8,
            "relevancia_ia": 0.8,
            "recomendacion_personalizada": "Error temporal en IA.",
            "razones_calidad": [],
            "advertencias": [str(e)]
        }

# --- 5. Función de chat simple ---
def chatgroq(mensaje: str) -> str:
    """Envía un mensaje y recibe una respuesta de la IA (útil para asistentes)."""
    if not GROQ_AVAILABLE:
        return "🧠 IA no disponible. El sistema sigue funcionando sin análisis avanzado."

    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un asistente educativo útil. Responde de forma clara, breve y sin formato JSON ni HTML."},
                {"role": "user", "content": mensaje}
            ],
            model=GROQ_MODEL,
            temperature=0.5,
            max_tokens=500
        )
        return resp.choices[0].message.content or "Sin respuesta de IA."
    except Exception as e:
        logger.error(f"Error en chat IA: {e}")
        return "Lo siento, hubo un error con el asistente IA."

# --- 6. Prueba automática (solo si se ejecuta directamente) ---
def test_ia():
    print("🧪 Iniciando prueba del módulo de IA...\n")

    if not GROQ_AVAILABLE:
        print("⚠️ Groq no está disponible. Modo básico activo.\n")
        return

    # Prueba de análisis
    print("🔍 Analizando curso de ejemplo...")
    resultado = analizar_recurso_groq(
        titulo="Curso de Python para principiantes",
        descripcion="Aprende Python desde cero con ejercicios prácticos y proyectos reales.",
        nivel="Principiante",
        categoria="Programación",
        plataforma="freeCodeCamp"
    )
    print("✅ Resultado análisis:", resultado, "\n")

    # Prueba de chat
    print("💬 Probando chat IA...")
    respuesta = chatgroq("¿Qué curso me recomiendas para aprender IA generativa gratis?")
    print("🤖 IA dice:", respuesta, "\n")

    print("🎉 Prueba completada.")

if __name__ == "__main__":
    test_ia()