# 🚀 Guía Completa: Apache Arrow para Datasets Masivos

## 📋 Tabla de Contenidos
1. [¿Qué es Apache Arrow?](#qué-es-apache-arrow)
2. [Instalación](#instalación)
3. [Uso del Script](#uso-del-script)
4. [Ejemplos Prácticos](#ejemplos-prácticos)
5. [Preguntas Frecuentes](#preguntas-frecuentes)

---

## ¿Qué es Apache Arrow?

Apache Arrow es el "contenedor de alta velocidad" para tus datos. Piensa en él como el lenguaje universal que todas tus herramientas de IA entienden sin necesidad de traducción.

### ✅ Beneficios para tu proyecto:

- **Eficiencia de Memoria**: 30 GB de datos CSV → ~5-8 GB en Parquet
- **Velocidad**: Carga datos 100x más rápido que CSV
- **Compatibilidad**: Funciona con Pandas, Polars, PyTorch, TensorFlow, etc.
- **Sin Traducción**: Los datos van directamente de disco a tu modelo de IA

---

## 🔧 Instalación

### Paso 1: Instalar las Librerías Necesarias

Abre tu terminal (CMD en Windows) y ejecuta:

```bash
pip install pyarrow polars openpyxl
```

**¿Qué instalamos?**
- `pyarrow`: La implementación oficial de Apache Arrow para Python
- `polars`: Motor ultra-rápido que usa Arrow internamente
- `openpyxl`: Necesario para leer archivos Excel (.xlsx)

### Paso 2: Verificar la Instalación

```bash
python -c "import polars as pl; print('✅ Polars instalado correctamente')"
python -c "import pyarrow as pa; print('✅ PyArrow instalado correctamente')"
```

---

## 📖 Uso del Script

### Configuración Inicial

1. **Descarga el script** `unificar_seguridad.py` a tu computadora

2. **Abre el archivo con un editor** (Notepad, VS Code, etc.)

3. **Modifica la ruta base** en la línea que dice:
   ```python
   RUTA_BASE = "F:/Sistema de gestion seguridad"
   ```
   
   Cámbiala por la ruta donde tienes tus archivos Excel.

4. **Guarda el archivo**

### Ejecución

1. Abre la terminal (CMD)

2. Navega a la carpeta donde guardaste el script:
   ```bash
   cd C:\TuCarpeta\DondeEstaElScript
   ```

3. Ejecuta el script:
   ```bash
   python unificar_seguridad.py
   ```

### ¿Qué hace el script?

```
🔍 Busca todos los archivos .xlsx en tu carpeta
    ↓
📊 Lee cada archivo Excel
    ↓
🏷️  Añade metadata (nombre de archivo, categoría)
    ↓
🔗 Unifica todos en una sola tabla
    ↓
💾 Guarda como archivo .parquet optimizado
```

---

## 💡 Ejemplos Prácticos

### Ejemplo 1: Leer el Dataset Unificado

```python
import polars as pl

# Cargar el dataset completo
df = pl.read_parquet("dataset_seguridad_unificado.parquet")

# Ver las primeras filas
print(df.head())

# Ver información del dataset
print(f"Total de registros: {len(df):,}")
print(f"Columnas: {df.columns}")
```

### Ejemplo 2: Filtrar por Categoría

```python
# Ver todas las categorías disponibles
categorias = df.select("categoria").unique()
print(categorias)

# Filtrar solo datos de ISO 9001
df_iso9001 = df.filter(pl.col("categoria") == "ISO 9001")
print(f"Registros de ISO 9001: {len(df_iso9001):,}")
```

### Ejemplo 3: Análisis Rápido

```python
# Contar registros por categoría
resumen = df.group_by("categoria").agg([
    pl.count().alias("total_registros"),
    pl.col("archivo_origen").n_unique().alias("archivos")
])

print(resumen)
```

### Ejemplo 4: Exportar a Pandas (para análisis tradicional)

```python
# Convertir a Pandas si necesitas usar bibliotecas antiguas
df_pandas = df.to_pandas()

# Ahora puedes usar todas las funciones de Pandas
print(df_pandas.describe())
```

### Ejemplo 5: Preparar Datos para Entrenamiento de IA

```python
import polars as pl
from sklearn.model_selection import train_test_split

# Cargar el dataset
df = pl.read_parquet("dataset_seguridad_unificado.parquet")

# Seleccionar columnas relevantes para tu modelo
# (ajusta según las columnas que tenga tu dataset)
df_entrenamiento = df.select([
    "columna_caracteristica_1",
    "columna_caracteristica_2",
    "columna_objetivo"
])

# Convertir a numpy arrays para modelos de IA
X = df_entrenamiento.drop("columna_objetivo").to_numpy()
y = df_entrenamiento.select("columna_objetivo").to_numpy()

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"Datos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape}")
```

---

## 🔍 Preguntas Frecuentes

### ¿Por qué usar Parquet en lugar de CSV?

| Característica | CSV | Parquet (Arrow) |
|----------------|-----|-----------------|
| Tamaño (30 GB) | 30 GB | ~5-8 GB |
| Velocidad de lectura | Lenta | 100x más rápida |
| Memoria necesaria | 3-5x el tamaño | Eficiente |
| Lectura selectiva | ❌ Lee todo | ✅ Solo columnas necesarias |

### ¿Qué pasa si el script da error?

**Error común 1**: `FileNotFoundError`
- **Solución**: Verifica que la ruta en `RUTA_BASE` sea correcta

**Error común 2**: `MemoryError`
- **Solución**: El script procesa archivos de forma incremental, pero si tienes archivos Excel extremadamente grandes individuales, aumenta la RAM o procesa por lotes

**Error común 3**: Archivos Excel corruptos
- **Solución**: El script saltará automáticamente los archivos con errores y mostrará un reporte al final

### ¿Puedo usar esto con n8n?

¡Sí! Tienes dos opciones:

**Opción 1**: Exportar desde n8n a Excel/CSV y luego usar este script

**Opción 2**: Crear un endpoint en Python que reciba datos de n8n vía webhook:

```python
from fastapi import FastAPI
import polars as pl

app = FastAPI()

@app.post("/guardar_datos")
async def guardar_datos(datos: dict):
    # Convertir JSON de n8n a DataFrame de Polars
    df = pl.DataFrame(datos)
    
    # Guardar en Parquet
    df.write_parquet("datos_n8n.parquet", compression="zstd")
    
    return {"status": "guardado"}
```

### ¿Cómo verifico que mi dataset está bien?

Ejecuta este código:

```python
import polars as pl

# Cargar el dataset
df = pl.scan_parquet("dataset_seguridad_unificado.parquet")

# Ver estadísticas
print(f"Total de filas: {df.select(pl.count()).collect()[0, 0]:,}")
print(f"Columnas: {len(df.collect_schema())}")
print("\nEsquema:")
print(df.collect_schema())
```

### ¿Puedo procesar archivos de otras ubicaciones?

Sí, simplemente modifica la variable `RUTA_BASE` en el script:

```python
# Para procesar desde un pendrive
RUTA_BASE = "E:/MiPendrive/Datos"

# Para procesar desde red
RUTA_BASE = "//servidor/compartido/datos"

# Para procesar múltiples carpetas, ejecuta el script varias veces
# cambiando RUTA_BASE y ARCHIVO_SALIDA
```

---

## 🎯 Próximos Pasos

Una vez que tengas tu dataset en formato Parquet:

1. **Explora tus datos**:
   ```bash
   python -c "import polars as pl; df = pl.read_parquet('dataset_seguridad_unificado.parquet'); print(df.head())"
   ```

2. **Integra con tus proyectos de IA** (Jarvis, Nexus, EMET Prime)

3. **Crea visualizaciones**:
   ```python
   import plotly.express as px
   df_pandas = df.to_pandas()
   fig = px.bar(df_pandas.groupby('categoria').size())
   fig.show()
   ```

4. **Automatiza el proceso** con tareas programadas (cron, Task Scheduler)

---

## 📚 Recursos Adicionales

- [Documentación oficial de Polars](https://pola-rs.github.io/polars/)
- [Documentación de Apache Arrow](https://arrow.apache.org/)
- [Guía de PyArrow](https://arrow.apache.org/docs/python/)

---

## 🆘 Soporte

Si encuentras algún problema:

1. Verifica que todas las librerías estén instaladas
2. Revisa que la ruta sea correcta
3. Ejecuta el script en modo debug:
   ```bash
   python -u unificar_seguridad.py
   ```

---

**¡Éxito con tu proyecto de IA soberana! 🚀**
