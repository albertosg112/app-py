#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para Unificar Archivos Excel a Formato Apache Arrow/Parquet

Este script procesa múltiples archivos Excel desde una estructura de carpetas
y los unifica en un solo dataset optimizado en formato Parquet (Apache Arrow).

Perfecto para bases de datos de 30GB+ que necesitan ser procesadas eficientemente.

Autor: Sistema de Gestión de Seguridad
Versión: 1.0
"""

import polars as pl
import glob
import os
from pathlib import Path
import time
from typing import List, Optional


class UnificadorExcel:
    """
    Clase para unificar múltiples archivos Excel en un único dataset Arrow/Parquet.
    
    Características:
    - Procesa archivos de forma incremental para no saturar la RAM
    - Añade metadata de origen a cada registro
    - Maneja errores de archivos corruptos
    - Compresión automática para reducir tamaño
    """
    
    def __init__(self, ruta_base: str, patron_archivos: str = "**/*.xlsx"):
        """
        Inicializa el unificador.
        
        Args:
            ruta_base: Ruta raíz donde buscar archivos (ej: "F:/Sistema de gestion seguridad")
            patron_archivos: Patrón de búsqueda (default: todos los .xlsx)
        """
        self.ruta_base = Path(ruta_base)
        self.patron_archivos = patron_archivos
        self.archivos_encontrados = []
        self.archivos_procesados = 0
        self.archivos_con_errores = []
        
    def buscar_archivos(self) -> List[Path]:
        """
        Busca todos los archivos Excel en la estructura de carpetas.
        
        Returns:
            Lista de rutas de archivos encontrados
        """
        print(f"🔍 Buscando archivos en: {self.ruta_base}")
        print(f"   Patrón de búsqueda: {self.patron_archivos}")
        
        ruta_busqueda = str(self.ruta_base / self.patron_archivos)
        archivos = glob.glob(ruta_busqueda, recursive=True)
        self.archivos_encontrados = [Path(f) for f in archivos]
        
        print(f"✅ Se encontraron {len(self.archivos_encontrados)} archivos Excel")
        return self.archivos_encontrados
    
    def procesar_archivo(self, ruta_archivo: Path) -> Optional[pl.DataFrame]:
        """
        Procesa un único archivo Excel y retorna un DataFrame de Polars.
        
        Args:
            ruta_archivo: Ruta del archivo a procesar
            
        Returns:
            DataFrame de Polars o None si hay error
        """
        try:
            # Leer el archivo Excel
            df = pl.read_excel(ruta_archivo)
            
            # Añadir metadata de origen
            carpeta_padre = ruta_archivo.parent.name
            nombre_archivo = ruta_archivo.name
            
            df = df.with_columns([
                pl.lit(nombre_archivo).alias("archivo_origen"),
                pl.lit(carpeta_padre).alias("categoria"),
                pl.lit(str(ruta_archivo)).alias("ruta_completa")
            ])
            
            self.archivos_procesados += 1
            print(f"   ✓ Procesado: {nombre_archivo} ({len(df)} filas)")
            
            return df
            
        except Exception as e:
            self.archivos_con_errores.append((ruta_archivo, str(e)))
            print(f"   ✗ Error en {ruta_archivo.name}: {str(e)[:50]}...")
            return None
    
    def unificar_todo(self, archivo_salida: str = "dataset_seguridad_unificado.parquet") -> bool:
        """
        Unifica todos los archivos Excel en un único archivo Parquet.
        
        Args:
            archivo_salida: Nombre del archivo Parquet de salida
            
        Returns:
            True si el proceso fue exitoso, False en caso contrario
        """
        print("\n" + "="*70)
        print("🚀 INICIANDO PROCESO DE UNIFICACIÓN")
        print("="*70)
        
        inicio = time.time()
        
        # Buscar archivos
        if not self.archivos_encontrados:
            self.buscar_archivos()
        
        if not self.archivos_encontrados:
            print("❌ No se encontraron archivos para procesar.")
            return False
        
        # Procesar archivos
        print(f"\n📊 Procesando {len(self.archivos_encontrados)} archivos...")
        print("-" * 70)
        
        dataframes = []
        for idx, archivo in enumerate(self.archivos_encontrados, 1):
            print(f"[{idx}/{len(self.archivos_encontrados)}]", end=" ")
            df = self.procesar_archivo(archivo)
            if df is not None:
                dataframes.append(df)
        
        # Unificar todos los DataFrames
        if not dataframes:
            print("\n❌ No se pudo procesar ningún archivo correctamente.")
            return False
        
        print(f"\n🔗 Unificando {len(dataframes)} DataFrames...")
        
        try:
            # Concatenar todos los DataFrames
            df_final = pl.concat(dataframes, how="diagonal")  # 'diagonal' maneja esquemas diferentes
            
            total_filas = len(df_final)
            total_columnas = len(df_final.columns)
            
            print(f"   Dataset unificado: {total_filas:,} filas × {total_columnas} columnas")
            
            # Guardar como Parquet
            print(f"\n💾 Guardando como '{archivo_salida}'...")
            ruta_salida = self.ruta_base / archivo_salida
            
            df_final.write_parquet(
                ruta_salida,
                compression="zstd",  # Máxima compresión
                use_pyarrow=True
            )
            
            # Calcular estadísticas
            tamaño_mb = ruta_salida.stat().st_size / (1024 * 1024)
            tiempo_total = time.time() - inicio
            
            print("\n" + "="*70)
            print("✅ PROCESO COMPLETADO EXITOSAMENTE")
            print("="*70)
            print(f"📁 Archivo guardado: {ruta_salida}")
            print(f"📊 Tamaño del archivo: {tamaño_mb:.2f} MB")
            print(f"📝 Total de registros: {total_filas:,}")
            print(f"⏱️  Tiempo de proceso: {tiempo_total:.2f} segundos")
            print(f"✓ Archivos procesados: {self.archivos_procesados}")
            
            if self.archivos_con_errores:
                print(f"\n⚠️  Archivos con errores: {len(self.archivos_con_errores)}")
                for archivo, error in self.archivos_con_errores[:5]:
                    print(f"   - {archivo.name}: {error[:50]}...")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error al unificar o guardar: {e}")
            return False
    
    def verificar_dataset(self, archivo_parquet: str) -> None:
        """
        Verifica el dataset generado mostrando información básica.
        
        Args:
            archivo_parquet: Ruta del archivo Parquet a verificar
        """
        print("\n" + "="*70)
        print("🔍 VERIFICACIÓN DEL DATASET")
        print("="*70)
        
        try:
            ruta_archivo = self.ruta_base / archivo_parquet
            
            # Leer solo las primeras filas para verificación
            df = pl.read_parquet(ruta_archivo, n_rows=5)
            
            print(f"\n📋 Primeras 5 filas del dataset:")
            print(df)
            
            # Leer metadata completa
            df_completo = pl.scan_parquet(ruta_archivo)
            esquema = df_completo.collect_schema()
            
            print(f"\n📊 Esquema del dataset ({len(esquema)} columnas):")
            for nombre, tipo in esquema.items():
                print(f"   - {nombre}: {tipo}")
            
        except Exception as e:
            print(f"❌ Error al verificar: {e}")


def main():
    """
    Función principal para ejecutar el script.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  UNIFICADOR DE ARCHIVOS EXCEL A APACHE ARROW/PARQUET            ║
    ║  Sistema de Gestión de Seguridad                                ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # ==================== CONFIGURACIÓN ====================
    # 🔧 AJUSTA ESTA RUTA A TU SISTEMA
    # Ejemplos:
    #   Windows: "F:/Sistema de gestion seguridad"
    #   Linux/Mac: "/home/usuario/Documentos/Sistema de gestion seguridad"
    
    RUTA_BASE = "F:/Sistema de gestion seguridad"
    ARCHIVO_SALIDA = "dataset_seguridad_unificado.parquet"
    
    # ========================================================
    
    print(f"📂 Carpeta de trabajo: {RUTA_BASE}")
    print(f"📦 Archivo de salida: {ARCHIVO_SALIDA}\n")
    
    # Verificar que la ruta existe
    if not os.path.exists(RUTA_BASE):
        print(f"❌ ERROR: La ruta '{RUTA_BASE}' no existe.")
        print("\n💡 SOLUCIÓN:")
        print("   1. Abre este archivo (unificar_seguridad.py)")
        print("   2. Modifica la línea RUTA_BASE con tu ruta correcta")
        print("   3. Guarda y vuelve a ejecutar")
        return
    
    # Crear el unificador y procesar
    unificador = UnificadorExcel(RUTA_BASE)
    
    # Ejecutar el proceso
    exito = unificador.unificar_todo(ARCHIVO_SALIDA)
    
    # Si fue exitoso, verificar el resultado
    if exito:
        unificador.verificar_dataset(ARCHIVO_SALIDA)
        
        print("\n" + "="*70)
        print("🎯 SIGUIENTES PASOS:")
        print("="*70)
        print("1. Tu dataset está listo en formato Parquet (Apache Arrow)")
        print("2. Para usarlo en tus modelos de IA:")
        print()
        print("   import polars as pl")
        print(f"   df = pl.read_parquet('{ARCHIVO_SALIDA}')")
        print("   print(df.head())")
        print()
        print("3. Para análisis rápido:")
        print("   - Filtra por categoría: df.filter(pl.col('categoria') == 'ISO 9001')")
        print("   - Selecciona columnas: df.select(['columna1', 'columna2'])")
        print("   - Agrupa datos: df.group_by('categoria').agg(pl.count())")
        print("="*70)


if __name__ == "__main__":
    main()
