#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Prueba - Verificar que Apache Arrow está listo

Ejecuta este script para verificar que tu entorno está configurado correctamente.
"""

import sys
from pathlib import Path


def verificar_instalacion():
    """Verifica que todas las librerías necesarias estén instaladas"""
    print("=" * 70)
    print("🔍 VERIFICACIÓN DE INSTALACIÓN")
    print("=" * 70)
    
    librerias = {
        "polars": "Polars",
        "pyarrow": "PyArrow",
        "openpyxl": "OpenPyXL (para Excel)",
        "fastapi": "FastAPI (para n8n)",
        "uvicorn": "Uvicorn (servidor)"
    }
    
    instaladas = []
    faltantes = []
    
    for modulo, nombre in librerias.items():
        try:
            __import__(modulo)
            print(f"✅ {nombre:30} → Instalado")
            instaladas.append(nombre)
        except ImportError:
            print(f"❌ {nombre:30} → No instalado")
            faltantes.append(modulo)
    
    print("\n" + "=" * 70)
    
    if faltantes:
        print(f"⚠️  Faltan {len(faltantes)} librerías")
        print("\n💡 Instálalas con:")
        print(f"   pip install {' '.join(faltantes)}")
        return False
    else:
        print(f"✅ Todas las librerías están instaladas ({len(instaladas)}/{len(librerias)})")
        return True


def prueba_basica_polars():
    """Prueba básica de creación de dataset con Polars"""
    print("\n" + "=" * 70)
    print("🧪 PRUEBA BÁSICA - Crear Dataset de Ejemplo")
    print("=" * 70)
    
    try:
        import polars as pl
        import pyarrow.parquet as pq
        
        # Crear datos de ejemplo
        datos_ejemplo = {
            "id": [1, 2, 3, 4, 5],
            "nombre": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "edad": [25, 30, 35, 40, 45],
            "ciudad": ["Buenos Aires", "Neuquén", "Centenario", "Plottier", "Cutral-Có"],
            "activo": [True, True, False, True, True]
        }
        
        # Crear DataFrame
        df = pl.DataFrame(datos_ejemplo)
        
        print("\n📊 Dataset de ejemplo creado:")
        print(df)
        
        # Guardar como Parquet
        archivo_prueba = Path("test_dataset.parquet")
        df.write_parquet(archivo_prueba, compression="zstd")
        
        tamaño_kb = archivo_prueba.stat().st_size / 1024
        print(f"\n💾 Archivo guardado: {archivo_prueba}")
        print(f"📦 Tamaño: {tamaño_kb:.2f} KB")
        
        # Leer de nuevo para verificar
        df_leido = pl.read_parquet(archivo_prueba)
        print(f"\n✅ Archivo leído correctamente: {len(df_leido)} filas")
        
        # Mostrar esquema
        print(f"\n📋 Esquema del dataset:")
        for nombre, tipo in df_leido.schema.items():
            print(f"   • {nombre:15} → {tipo}")
        
        # Limpiar archivo de prueba
        archivo_prueba.unlink()
        print(f"\n🗑️  Archivo de prueba eliminado")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        return False


def prueba_procesamiento_lazy():
    """Prueba el procesamiento lazy (sin cargar en memoria)"""
    print("\n" + "=" * 70)
    print("🧪 PRUEBA AVANZADA - Procesamiento Lazy")
    print("=" * 70)
    
    try:
        import polars as pl
        
        # Crear un dataset más grande
        n_filas = 100000
        print(f"\n📊 Creando dataset de {n_filas:,} filas...")
        
        df = pl.DataFrame({
            "id": range(n_filas),
            "valor": [i * 2.5 for i in range(n_filas)],
            "categoria": [f"Cat_{i % 10}" for i in range(n_filas)]
        })
        
        # Guardar
        archivo = Path("test_large.parquet")
        df.write_parquet(archivo)
        
        tamaño_mb = archivo.stat().st_size / (1024 * 1024)
        print(f"✅ Dataset guardado: {tamaño_mb:.2f} MB")
        
        # Procesamiento Lazy (no carga en memoria)
        print("\n🚀 Procesamiento lazy (sin cargar en memoria):")
        
        resultado = pl.scan_parquet(archivo).filter(
            pl.col("valor") > 50000
        ).group_by("categoria").agg([
            pl.count().alias("total"),
            pl.col("valor").mean().alias("promedio")
        ]).collect()
        
        print(resultado)
        print(f"\n✅ Procesamiento completado sin cargar todo el archivo")
        
        # Limpiar
        archivo.unlink()
        print(f"🗑️  Archivo de prueba eliminado")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error durante la prueba: {e}")
        return False


def verificar_scripts():
    """Verifica que los scripts principales existan"""
    print("\n" + "=" * 70)
    print("📁 VERIFICACIÓN DE SCRIPTS")
    print("=" * 70)
    
    scripts = [
        "unificar_seguridad.py",
        "convertir_a_parquet.py",
        "n8n_arrow_bridge.py"
    ]
    
    disponibles = []
    faltantes = []
    
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script:30} → Disponible")
            disponibles.append(script)
        else:
            print(f"❌ {script:30} → No encontrado")
            faltantes.append(script)
    
    if faltantes:
        print(f"\n⚠️  Algunos scripts no se encuentran")
        return False
    else:
        print(f"\n✅ Todos los scripts están disponibles")
        return True


def mostrar_version_python():
    """Muestra la versión de Python"""
    print("\n" + "=" * 70)
    print("🐍 VERSIÓN DE PYTHON")
    print("=" * 70)
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Versión compatible (Python 3.8+)")
        return True
    else:
        print("⚠️  Se recomienda Python 3.8 o superior")
        return False


def main():
    """Función principal"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         SCRIPT DE VERIFICACIÓN - APACHE ARROW                   ║
    ║         Verifica que todo esté listo para usar                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    resultados = []
    
    # Ejecutar todas las verificaciones
    resultados.append(("Versión Python", mostrar_version_python()))
    resultados.append(("Librerías", verificar_instalacion()))
    resultados.append(("Scripts", verificar_scripts()))
    resultados.append(("Prueba Básica", prueba_basica_polars()))
    resultados.append(("Prueba Lazy", prueba_procesamiento_lazy()))
    
    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 70)
    
    exitosas = sum(1 for _, exito in resultados if exito)
    total = len(resultados)
    
    for nombre, exito in resultados:
        estado = "✅" if exito else "❌"
        print(f"{estado} {nombre}")
    
    print("\n" + "=" * 70)
    
    if exitosas == total:
        print("🎉 ¡ÉXITO! Todo está listo para usar Apache Arrow")
        print("\n🚀 Siguiente paso:")
        print("   1. Lee INICIO_RAPIDO.md")
        print("   2. Elige el script que necesites")
        print("   3. ¡Empieza a procesar datos!")
    else:
        print(f"⚠️  {total - exitosas} verificaciones fallaron")
        print("\n💡 Revisa los mensajes de error arriba")
        print("   Si las librerías faltan, instálalas con:")
        print("   pip install -r requirements.txt")
    
    print("=" * 70)
    
    return exitosas == total


if __name__ == "__main__":
    exito = main()
    sys.exit(0 if exito else 1)
