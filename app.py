# @title 🎓 BUSCADOR DE CURSOS GRATUITOS INTELIGENTE
# @markdown ### 1. Escribe qué quieres aprender y dale al botón de "Play" a la izquierda.
tema_a_aprender = "Marketing Digital" # @param {type:"string"}

# Instalamos la librería necesaria de forma silenciosa
!pip install googlesearch-python -q
import pandas as pd
from googlesearch import search
from time import sleep

def buscar_cursos(tema):
    print(f"🔎 Buscando los mejores recursos gratuitos para: {tema}...")
    print("------------------------------------------------------")

    niveles = ["Nivel Principiante / Desde Cero", "Nivel Intermedio", "Nivel Avanzado / Experto"]
    # Sitios recomendados para filtrar basura y encontrar cursos reales
    sitios_top = "site:coursera.org OR site:edx.org OR site:youtube.com OR site:udemy.com OR site:freecodecamp.org"

    todos_los_recursos = []

    for nivel in niveles:
        print(f"   ...Buscando recursos de {nivel}...")
        # Creamos una búsqueda inteligente
        query = f"curso completo gratis {tema} {nivel} {sitios_top}"

        try:
            # Buscamos 5 enlaces de alta calidad por nivel
            resultados = search(query, num_results=5, advanced=True)
            
            for resultado in resultados:
                todos_los_recursos.append({
                    "Nivel": nivel,
                    "Tema": tema,
                    "Título": resultado.title,
                    "Descripción": result.description if hasattr(resultado, 'description') else "Enlace directo al curso",
                    "Enlace": resultado.url
                })
            sleep(2) # Pausa para no bloquear Google
        except Exception as e:
            print(f"⚠️ Pequeña pausa de seguridad... continuando.")

    # Crear el archivo para descargar
    if todos_los_recursos:
        df = pd.DataFrame(todos_los_recursos)
        nombre_archivo = f"Ruta_Aprendizaje_{tema.replace(' ', '_')}.csv"
        df.to_csv(nombre_archivo, index=False)
        print("\n✅ ¡LISTO! Tu ruta de aprendizaje se ha generado.")
        print(f"📂 Busca el archivo '{nombre_archivo}' en la carpeta de la izquierda para descargarlo.")
    else:
        print("❌ No se encontraron resultados. Intenta ser más específico.")

# Ejecutar la función
buscar_cursos(tema_a_aprender)