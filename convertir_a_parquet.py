#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Simplificado para Convertir CSV/JSON Grande a Parquet

Este script es para archivos únicos de gran tamaño (como 30 GB)
que no caben en la memoria RAM.

Uso:
    python convertir_a_parquet.py
"""

import polars as pl
import os
import time
from pathlib import Path


def convertir_csv_a_parquet(
    archivo_origen: str,
    archivo_destino: str = "dataset_final.parquet",
    separador: str = ",",
    batch_size: int = 100000
):
    """
    Convierte un archivo CSV grande a formato Parquet usando streaming.
    
    Args:
        archivo_origen: Ruta al archivo CSV original
        archivo_destino: Nombre del archivo Parquet de salida
        separador: Separador del CSV (default: coma)
        batch_size: Número de filas a procesar por lote
    """
    print("="*70)
    print("🔄 CONVERSIÓN CSV → PARQUET (APACHE ARROW)")
    print("="*70)
    
    # Verificar que el archivo existe
    if not os.path.exists(archivo_origen):
        print(f"❌ ERROR: No se encuentra el archivo '{archivo_origen}'")
        print("\n💡 Verifica:")
        print("   1. Que el archivo existe en la ruta especificada")
        print("   2. Que la ruta sea correcta (usa / o \\\\ en Windows)")
        return False
    
    # Obtener tamaño del archivo
    tamaño_gb = os.path.getsize(archivo_origen) / (1024**3)
    print(f"\n📁 Archivo origen: {archivo_origen}")
    print(f"📊 Tamaño: {tamaño_gb:.2f} GB")
    print(f"💾 Archivo destino: {archivo_destino}")
    
    inicio = time.time()
    
    try:
        print(f"\n🚀 Iniciando conversión...")
        print(f"   (Esto puede tardar varios minutos para archivos grandes)")
        
        # Usar scan_csv para lectura lazy (no carga todo en RAM)
        df = pl.scan_csv(
            archivo_origen,
            separator=separador,
            ignore_errors=True,  # Salta filas con errores
            try_parse_dates=True,  # Intenta detectar fechas automáticamente
            infer_schema_length=10000  # Analiza las primeras 10k filas para tipos de datos
        )
        
        # Escribir directamente a Parquet sin cargar todo en RAM
        df.sink_parquet(
            archivo_destino,
            compression="zstd",  # Máxima compresión
            row_group_size=batch_size
        )
        
        # Calcular estadísticas
        tiempo_total = time.time() - inicio
        tamaño_salida_gb = os.path.getsize(archivo_destino) / (1024**3)
        ratio_compresion = (1 - tamaño_salida_gb/tamaño_gb) * 100
        
        print("\n" + "="*70)
        print("✅ CONVERSIÓN COMPLETADA EXITOSAMENTE")
        print("="*70)
        print(f"📦 Archivo guardado: {archivo_destino}")
        print(f"📊 Tamaño original: {tamaño_gb:.2f} GB")
        print(f"📊 Tamaño final: {tamaño_salida_gb:.2f} GB")
        print(f"📉 Compresión: {ratio_compresion:.1f}% de reducción")
        print(f"⏱️  Tiempo total: {tiempo_total/60:.2f} minutos")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR durante la conversión: {e}")
        print("\n💡 Posibles soluciones:")
        print("   1. Verifica que el archivo no esté abierto en otro programa")
        print("   2. Asegúrate de tener espacio en disco suficiente")
        print("   3. Si el CSV usa punto y coma (;), cambia separador=';'")
        return False


def convertir_json_a_parquet(
    archivo_origen: str,
    archivo_destino: str = "dataset_final.parquet"
):
    """
    Convierte un archivo JSON/NDJSON a formato Parquet.
    
    Args:
        archivo_origen: Ruta al archivo JSON original
        archivo_destino: Nombre del archivo Parquet de salida
    """
    print("="*70)
    print("🔄 CONVERSIÓN JSON → PARQUET (APACHE ARROW)")
    print("="*70)
    
    if not os.path.exists(archivo_origen):
        print(f"❌ ERROR: No se encuentra el archivo '{archivo_origen}'")
        return False
    
    tamaño_gb = os.path.getsize(archivo_origen) / (1024**3)
    print(f"\n📁 Archivo origen: {archivo_origen}")
    print(f"📊 Tamaño: {tamaño_gb:.2f} GB")
    
    inicio = time.time()
    
    try:
        print(f"\n🚀 Iniciando conversión...")
        
        # Para JSON línea por línea (NDJSON)
        df = pl.scan_ndjson(archivo_origen)
        df.sink_parquet(archivo_destino, compression="zstd")
        
        tiempo_total = time.time() - inicio
        tamaño_salida_gb = os.path.getsize(archivo_destino) / (1024**3)
        
        print("\n✅ Conversión completada")
        print(f"📦 Tamaño final: {tamaño_salida_gb:.2f} GB")
        print(f"⏱️  Tiempo: {tiempo_total/60:.2f} minutos")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Si tu JSON es un array [ {...}, {...} ],")
        print("   considera usar pl.read_json() para archivos pequeños")
        return False


def verificar_parquet(archivo_parquet: str):
    """
    Verifica el contenido del archivo Parquet generado.
    
    Args:
        archivo_parquet: Ruta del archivo Parquet a verificar
    """
    print("\n" + "="*70)
    print("🔍 VERIFICACIÓN DEL DATASET")
    print("="*70)
    
    try:
        # Leer metadata sin cargar datos
        df = pl.scan_parquet(archivo_parquet)
        esquema = df.collect_schema()
        
        print(f"\n📋 Esquema del dataset ({len(esquema)} columnas):")
        for nombre, tipo in esquema.items():
            print(f"   • {nombre}: {tipo}")
        
        # Leer solo las primeras 5 filas para preview
        print(f"\n📊 Primeras 5 filas:")
        df_preview = pl.read_parquet(archivo_parquet, n_rows=5)
        print(df_preview)
        
        # Contar filas totales (sin cargar todo)
        total_filas = df.select(pl.count()).collect()[0, 0]
        print(f"\n📈 Total de registros: {total_filas:,}")
        
    except Exception as e:
        print(f"❌ Error al verificar: {e}")


def main():
    """
    Función principal - Configura aquí tu archivo a convertir
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          CONVERTIDOR A APACHE ARROW/PARQUET                     ║
    ║          Para archivos CSV/JSON de gran tamaño                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ==================== CONFIGURACIÓN ====================
    # 🔧 AJUSTA ESTAS VARIABLES SEGÚN TU ARCHIVO
    
    # Tipo de archivo: "csv" o "json"
    TIPO_ARCHIVO = "csv"
    
    # Ruta completa al archivo de origen
    # Ejemplos:
    #   Windows: "C:/MisDatos/archivo_grande.csv"
    #   Linux: "/home/usuario/datos/archivo.csv"
    #   Pendrive: "E:/datos.csv"
    ARCHIVO_ORIGEN = "datos_grandes.csv"
    
    # Nombre del archivo de salida
    ARCHIVO_DESTINO = "dataset_final.parquet"
    
    # Solo para CSV: separador de columnas
    SEPARADOR_CSV = ","  # Cambia a ";" si tu CSV usa punto y coma
    
    # ========================================================
    
    print(f"📝 Tipo de archivo: {TIPO_ARCHIVO.upper()}")
    print(f"📂 Archivo origen: {ARCHIVO_ORIGEN}")
    print(f"📦 Archivo destino: {ARCHIVO_DESTINO}\n")
    
    # Validar configuración
    if not os.path.exists(ARCHIVO_ORIGEN):
        print("❌ CONFIGURACIÓN INCORRECTA")
        print(f"\nEl archivo '{ARCHIVO_ORIGEN}' no existe.")
        print("\n💡 SOLUCIÓN:")
        print("   1. Abre este archivo (convertir_a_parquet.py)")
        print("   2. Modifica la variable ARCHIVO_ORIGEN con la ruta correcta")
        print("   3. Si el archivo está en un pendrive, usa la letra correcta (E:/, F:/, etc.)")
        print("   4. Guarda y vuelve a ejecutar\n")
        print("Ejemplo de ruta correcta:")
        print('   ARCHIVO_ORIGEN = "F:/mi_carpeta/archivo.csv"')
        return
    
    # Ejecutar conversión según el tipo
    if TIPO_ARCHIVO.lower() == "csv":
        exito = convertir_csv_a_parquet(
            ARCHIVO_ORIGEN,
            ARCHIVO_DESTINO,
            SEPARADOR_CSV
        )
    elif TIPO_ARCHIVO.lower() == "json":
        exito = convertir_json_a_parquet(ARCHIVO_ORIGEN, ARCHIVO_DESTINO)
    else:
        print(f"❌ Tipo de archivo '{TIPO_ARCHIVO}' no soportado")
        print("   Tipos válidos: 'csv' o 'json'")
        return
    
    # Si fue exitoso, verificar
    if exito:
        verificar_parquet(ARCHIVO_DESTINO)
        
        print("\n" + "="*70)
        print("🎯 SIGUIENTES PASOS")
        print("="*70)
        print("\n1. Para cargar tu dataset en Python:")
        print(f"""
    import polars as pl
    df = pl.read_parquet("{ARCHIVO_DESTINO}")
    print(df.head())
        """)
        
        print("\n2. Para entrenamiento de modelos de IA:")
        print("""
    # Seleccionar columnas específicas
    df_train = df.select(['columna1', 'columna2', 'objetivo'])
    
    # Convertir a numpy para PyTorch/TensorFlow
    X = df_train.drop('objetivo').to_numpy()
    y = df_train.select('objetivo').to_numpy()
        """)
        
        print("\n3. Para análisis exploratorio:")
        print("""
    # Ver estadísticas
    print(df.describe())
    
    # Filtrar datos
    df_filtrado = df.filter(pl.col('columna') > 100)
        """)
        
        print("\n" + "="*70)
        print("✅ ¡Conversión completada! Tu dataset está listo para IA.")
        print("="*70)


if __name__ == "__main__":
    main()
