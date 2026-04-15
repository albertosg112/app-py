#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integración n8n → Apache Arrow/Parquet

Este script recibe datos desde flujos de n8n y los almacena
en formato Parquet para entrenamiento de modelos de IA.

Casos de uso:
- Recolección de logs de seguridad
- Datos de sensores IoT
- Análisis de eventos
- Dataset incremental para entrenamiento

Autor: Sistema de IA Soberana
"""

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
from datetime import datetime
import json


# ==================== CONFIGURACIÓN ====================
CARPETA_DATASETS = Path("datasets_n8n")
CARPETA_DATASETS.mkdir(exist_ok=True)

# Crear la aplicación FastAPI
app = FastAPI(
    title="n8n → Arrow Bridge",
    description="Recibe datos de n8n y los almacena en formato Parquet",
    version="1.0.0"
)


# ==================== MODELOS DE DATOS ====================
class DatosN8N(BaseModel):
    """Modelo para recibir datos desde n8n"""
    datos: List[Dict[str, Any]]
    nombre_dataset: Optional[str] = "dataset_n8n"
    categoria: Optional[str] = "general"


# ==================== ENDPOINTS ====================
@app.get("/")
async def raiz():
    """Endpoint de bienvenida"""
    return {
        "servicio": "n8n → Apache Arrow Bridge",
        "estado": "activo",
        "version": "1.0.0",
        "documentacion": "/docs",
        "endpoints": {
            "guardar_datos": "POST /guardar",
            "listar_datasets": "GET /datasets",
            "estadisticas": "GET /stats/{nombre_dataset}"
        }
    }


def sanitizar_nombre_archivo(nombre: str) -> str:
    """
    Sanitiza el nombre del archivo para prevenir path traversal.
    
    Args:
        nombre: Nombre propuesto para el archivo
        
    Returns:
        Nombre seguro sin caracteres peligrosos
    """
    import re
    # Remover caracteres peligrosos y path separators
    nombre_seguro = re.sub(r'[^\w\-_]', '_', nombre)
    # Limitar longitud
    return nombre_seguro[:100]


@app.post("/guardar")
async def guardar_datos(payload: DatosN8N):
    """
    Recibe datos desde n8n y los guarda en formato Parquet.
    
    Ejemplo de uso desde n8n:
    - Nodo HTTP Request configurado como POST
    - URL: http://localhost:8000/guardar
    - Body JSON:
      {
        "datos": [
          {"sensor": "temp_01", "valor": 23.5, "timestamp": "2024-01-01T10:00:00"},
          {"sensor": "temp_02", "valor": 24.1, "timestamp": "2024-01-01T10:00:00"}
        ],
        "nombre_dataset": "sensores_temperatura",
        "categoria": "iot"
      }
    """
    try:
        if not payload.datos:
            raise HTTPException(status_code=400, detail="No se recibieron datos")
        
        # Sanitizar nombre del dataset para prevenir path injection
        nombre_seguro = sanitizar_nombre_archivo(payload.nombre_dataset)
        
        # Convertir a DataFrame de Polars
        df = pl.DataFrame(payload.datos)
        
        # Añadir metadata
        df = df.with_columns([
            pl.lit(payload.categoria).alias("_categoria"),
            pl.lit(datetime.now().isoformat()).alias("_fecha_ingesta")
        ])
        
        # Definir ruta del archivo (ahora con nombre sanitizado)
        archivo_parquet = CARPETA_DATASETS / f"{nombre_seguro}.parquet"
        
        # Si el archivo existe, append; si no, crear nuevo
        if archivo_parquet.exists():
            # Leer dataset existente
            df_existente = pl.read_parquet(archivo_parquet)
            # Concatenar
            df_final = pl.concat([df_existente, df], how="diagonal")
            # Guardar
            df_final.write_parquet(archivo_parquet, compression="zstd")
            modo = "actualizado"
            total_registros = len(df_final)
        else:
            # Crear nuevo archivo
            df.write_parquet(archivo_parquet, compression="zstd")
            modo = "creado"
            total_registros = len(df)
        
        return {
            "estado": "éxito",
            "modo": modo,
            "archivo": str(archivo_parquet),
            "registros_nuevos": len(df),
            "total_registros": total_registros,
            "columnas": df.columns
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # No exponer stack traces completos
        raise HTTPException(status_code=500, detail="Error al guardar los datos")


@app.get("/datasets")
async def listar_datasets():
    """Lista todos los datasets disponibles"""
    archivos = list(CARPETA_DATASETS.glob("*.parquet"))
    
    datasets = []
    for archivo in archivos:
        try:
            # Leer metadata sin cargar datos
            df = pl.scan_parquet(archivo)
            total_filas = df.select(pl.count()).collect()[0, 0]
            esquema = df.collect_schema()
            
            tamaño_mb = archivo.stat().st_size / (1024 * 1024)
            
            datasets.append({
                "nombre": archivo.stem,
                "archivo": str(archivo),
                "registros": total_filas,
                "columnas": len(esquema),
                "tamaño_mb": round(tamaño_mb, 2),
                "ultima_modificacion": datetime.fromtimestamp(
                    archivo.stat().st_mtime
                ).isoformat()
            })
        except Exception as e:
            datasets.append({
                "nombre": archivo.stem,
                "error": str(e)
            })
    
    return {
        "total_datasets": len(datasets),
        "datasets": datasets
    }


@app.get("/stats/{nombre_dataset}")
async def estadisticas_dataset(nombre_dataset: str):
    """Obtiene estadísticas de un dataset específico"""
    # Sanitizar nombre para prevenir path injection
    nombre_seguro = sanitizar_nombre_archivo(nombre_dataset)
    archivo_parquet = CARPETA_DATASETS / f"{nombre_seguro}.parquet"
    
    if not archivo_parquet.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{nombre_dataset}' no encontrado")
    
    try:
        # Leer dataset
        df = pl.read_parquet(archivo_parquet)
        
        # Calcular estadísticas
        stats = {
            "nombre": nombre_dataset,
            "total_registros": len(df),
            "columnas": df.columns,
            "tipos_datos": {col: str(dtype) for col, dtype in df.schema.items()},
            "memoria_mb": round(df.estimated_size("mb"), 2),
            "primeras_filas": df.head(5).to_dicts(),
            "estadisticas_numericas": {}
        }
        
        # Estadísticas para columnas numéricas
        columnas_numericas = [col for col, dtype in df.schema.items() 
                             if dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
        
        if columnas_numericas:
            stats["estadisticas_numericas"] = df.select(columnas_numericas).describe().to_dict()
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer dataset: {str(e)}")


@app.delete("/dataset/{nombre_dataset}")
async def eliminar_dataset(nombre_dataset: str):
    """Elimina un dataset"""
    # Sanitizar nombre para prevenir path injection
    nombre_seguro = sanitizar_nombre_archivo(nombre_dataset)
    archivo_parquet = CARPETA_DATASETS / f"{nombre_seguro}.parquet"
    
    # Verificar que el archivo está dentro de CARPETA_DATASETS
    if not archivo_parquet.resolve().parent == CARPETA_DATASETS.resolve():
        raise HTTPException(status_code=400, detail="Ruta de archivo inválida")
    
    if not archivo_parquet.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{nombre_seguro}' no encontrado")
    
    try:
        archivo_parquet.unlink()
        return {
            "estado": "éxito",
            "mensaje": f"Dataset '{nombre_dataset}' eliminado correctamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al eliminar: {str(e)}")


@app.post("/entrenar/{nombre_dataset}")
async def preparar_para_entrenamiento(nombre_dataset: str, columnas_features: List[str], columna_objetivo: str):
    """
    Prepara un dataset para entrenamiento de modelo.
    
    Exporta el dataset en formato optimizado para scikit-learn, PyTorch, etc.
    """
    # Sanitizar nombre para prevenir path injection
    nombre_seguro = sanitizar_nombre_archivo(nombre_dataset)
    archivo_parquet = CARPETA_DATASETS / f"{nombre_seguro}.parquet"
    
    if not archivo_parquet.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{nombre_dataset}' no encontrado")
    
    try:
        df = pl.read_parquet(archivo_parquet)
        
        # Verificar que las columnas existen
        columnas_requeridas = columnas_features + [columna_objetivo]
        columnas_faltantes = set(columnas_requeridas) - set(df.columns)
        if columnas_faltantes:
            raise HTTPException(
                status_code=400,
                detail=f"Columnas no encontradas: {list(columnas_faltantes)}"
            )
        
        # Seleccionar columnas
        df_entrenamiento = df.select(columnas_requeridas)
        
        # Guardar versión para entrenamiento (usar nombre sanitizado)
        archivo_train = CARPETA_DATASETS / f"{nombre_seguro}_train.parquet"
        df_entrenamiento.write_parquet(archivo_train, compression="zstd")
        
        return {
            "estado": "éxito",
            "archivo_entrenamiento": str(archivo_train),
            "registros": len(df_entrenamiento),
            "features": columnas_features,
            "objetivo": columna_objetivo,
            "codigo_ejemplo": f"""
import polars as pl
from sklearn.model_selection import train_test_split

# Cargar datos
df = pl.read_parquet('{archivo_train}')

# Separar features y objetivo
X = df.drop('{columna_objetivo}').to_numpy()
y = df.select('{columna_objetivo}').to_numpy()

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Ahora puedes entrenar tu modelo
"""
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ==================== FUNCIÓN PRINCIPAL ====================
def iniciar_servidor(host: str = "127.0.0.1", port: int = 8000):
    """
    Inicia el servidor FastAPI para recibir datos de n8n.
    
    Args:
        host: Dirección IP (127.0.0.1 para localhost solamente, 0.0.0.0 para red local)
        port: Puerto del servidor
    
    ADVERTENCIA DE SEGURIDAD:
        Por defecto, el servidor solo acepta conexiones locales (127.0.0.1).
        Si necesitas acceso desde la red local, cambia host a "0.0.0.0",
        pero considera implementar autenticación antes de hacerlo.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           n8n → APACHE ARROW BRIDGE                             ║
    ║           Servidor de Recolección de Datos                      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🚀 Iniciando servidor en http://{host}:{port}")
    print(f"📁 Carpeta de datasets: {CARPETA_DATASETS.absolute()}")
    print(f"📖 Documentación interactiva: http://localhost:{port}/docs")
    print("\n" + "="*70)
    print("CONFIGURACIÓN EN n8n:")
    print("="*70)
    print(f"1. Añade un nodo 'HTTP Request' al final de tu flujo")
    print(f"2. Configura:")
    print(f"   - Method: POST")
    print(f"   - URL: http://localhost:{port}/guardar")
    print(f"   - Body:")
    print("""     {
       "datos": {{{{ $json.items }}}},
       "nombre_dataset": "mi_dataset",
       "categoria": "produccion"
     }""")
    print("="*70)
    print("\n✅ Servidor listo. Presiona Ctrl+C para detener.\n")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    # Iniciar el servidor
    iniciar_servidor()
