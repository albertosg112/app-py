# 📚 Ejemplos Prácticos de Uso - Apache Arrow

Este archivo contiene ejemplos reales de cómo usar los scripts para diferentes casos de uso.

## 📋 Tabla de Contenidos
1. [Procesar Archivos Excel de Seguridad](#1-procesar-archivos-excel-de-seguridad)
2. [Convertir CSV Grande a Parquet](#2-convertir-csv-grande-a-parquet)
3. [Integrar con n8n](#3-integrar-con-n8n)
4. [Entrenar Modelos de IA](#4-entrenar-modelos-de-ia)

---

## 1. Procesar Archivos Excel de Seguridad

### Caso: Tienes múltiples archivos Excel en carpetas (ISO 9001, ISO 45001, etc.)

```python
# unificar_seguridad.py - Ya configurado

# PASO 1: Abre el archivo y modifica la ruta
RUTA_BASE = "F:/Sistema de gestion seguridad"

# PASO 2: Ejecuta desde terminal
# python unificar_seguridad.py

# PASO 3: Una vez generado el archivo .parquet, puedes usarlo así:
import polars as pl

# Cargar el dataset unificado
df = pl.read_parquet("dataset_seguridad_unificado.parquet")

# Ver resumen
print(f"Total de registros: {len(df):,}")
print(f"Categorías: {df['categoria'].unique().to_list()}")

# Filtrar por categoría específica
df_iso9001 = df.filter(pl.col("categoria") == "ISO 9001")
print(f"Registros ISO 9001: {len(df_iso9001):,}")

# Exportar una categoría a Excel (si necesitas)
df_iso9001.write_excel("analisis_iso9001.xlsx")
```

---

## 2. Convertir CSV Grande a Parquet

### Caso A: Archivo CSV de 30 GB en pendrive

```python
# convertir_a_parquet.py

# CONFIGURACIÓN
TIPO_ARCHIVO = "csv"
ARCHIVO_ORIGEN = "E:/datos_grandes.csv"  # Pendrive en E:
ARCHIVO_DESTINO = "dataset_final.parquet"
SEPARADOR_CSV = ","  # o ";" si usa punto y coma

# Ejecutar: python convertir_a_parquet.py
```

### Caso B: Procesar solo columnas específicas de un CSV grande

```python
import polars as pl

# Leer solo las columnas que necesitas (ahorra memoria)
df = pl.scan_csv(
    "E:/datos_grandes.csv",
    ignore_errors=True
).select([
    "columna_importante_1",
    "columna_importante_2",
    "fecha",
    "valor"
])

# Filtrar mientras procesas (más eficiente)
df = df.filter(pl.col("fecha") > "2024-01-01")

# Guardar resultado
df.sink_parquet("dataset_filtrado.parquet", compression="zstd")

print("✅ Procesado solo las columnas y filas necesarias")
```

### Caso C: CSV con formato europeo (punto y coma, comas como decimales)

```python
import polars as pl

df = pl.scan_csv(
    "datos_europeos.csv",
    separator=";",           # Separador punto y coma
    decimal_comma=True,      # Usa coma como separador decimal
    ignore_errors=True
)

df.sink_parquet("dataset_europa.parquet")
```

---

## 3. Integrar con n8n

### Caso: Recolectar logs de seguridad desde n8n

#### Paso 1: Iniciar el servidor

```bash
# Terminal
python n8n_arrow_bridge.py

# Verás:
# 🚀 Iniciando servidor en http://0.0.0.0:8000
# 📖 Documentación: http://localhost:8000/docs
```

#### Paso 2: Configurar n8n

En tu flujo de n8n, añade al final un nodo "HTTP Request":

- **Method**: POST
- **URL**: `http://localhost:8000/guardar`
- **Body** (JSON):
```json
{
  "datos": {{ $json.items }},
  "nombre_dataset": "logs_seguridad",
  "categoria": "seguridad"
}
```

#### Paso 3: Verificar que los datos llegan

```python
import polars as pl

# Leer el dataset que n8n está alimentando
df = pl.read_parquet("datasets_n8n/logs_seguridad.parquet")

print(df.head())
print(f"Total de eventos: {len(df)}")

# Ver categorías de eventos
if "tipo_evento" in df.columns:
    print(df["tipo_evento"].value_counts())
```

#### Ejemplo avanzado: Análisis en tiempo real

```python
import polars as pl
from pathlib import Path
import time

# Monitorear el dataset que n8n va llenando
archivo = Path("datasets_n8n/logs_seguridad.parquet")

print("🔍 Monitoreando dataset...")

registros_anterior = 0
while True:
    if archivo.exists():
        df = pl.read_parquet(archivo)
        registros_actual = len(df)
        
        if registros_actual > registros_anterior:
            nuevos = registros_actual - registros_anterior
            print(f"📊 +{nuevos} nuevos registros | Total: {registros_actual}")
            
            # Análisis de los últimos registros
            ultimos = df.tail(nuevos)
            print(f"   Última categoría: {ultimos['_categoria'][0]}")
            
            registros_anterior = registros_actual
    
    time.sleep(10)  # Revisar cada 10 segundos
```

---

## 4. Entrenar Modelos de IA

### Caso A: Clasificación con Scikit-learn

```python
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Cargar dataset
df = pl.read_parquet("dataset_seguridad_unificado.parquet")

# 2. Preparar datos (ejemplo: predecir categoría basado en otras columnas)
# Ajusta según tus columnas reales

# Supongamos que queremos predecir 'categoria' basado en otras características
# Primero, veamos qué columnas tenemos
print("Columnas disponibles:", df.columns)

# Ejemplo genérico - AJUSTA SEGÚN TUS DATOS
features = df.select([
    # Selecciona las columnas numéricas o de texto que tengas
    # "temperatura", "presion", "velocidad", etc.
]).to_numpy()

# Objetivo a predecir
target = df.select("categoria").to_series().to_list()

# Codificar categorías a números
le = LabelEncoder()
y = le.fit_transform(target)

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42
)

# 4. Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 5. Evaluar
accuracy = modelo.score(X_test, y_test)
print(f"✅ Precisión del modelo: {accuracy*100:.2f}%")

# 6. Guardar modelo
import joblib
joblib.dump(modelo, "modelo_clasificacion.pkl")
joblib.dump(le, "label_encoder.pkl")
print("💾 Modelo guardado")
```

### Caso B: Procesamiento por lotes para datasets muy grandes

```python
import polars as pl
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Para datasets que no caben en RAM, procesar por lotes
archivo = "dataset_grande.parquet"

# Leer metadata
df_lazy = pl.scan_parquet(archivo)
total_filas = df_lazy.select(pl.count()).collect()[0, 0]

# Configurar procesamiento por lotes
batch_size = 100000
num_batches = (total_filas // batch_size) + 1

print(f"📊 Procesando {total_filas:,} filas en {num_batches} lotes")

# Entrenar incrementalmente
modelo = RandomForestClassifier(warm_start=True, n_estimators=10)

for i in range(num_batches):
    # Leer solo un lote
    offset = i * batch_size
    df_batch = pl.read_parquet(
        archivo,
        n_rows=batch_size,
        row_count_offset=offset
    )
    
    # Preparar datos del lote
    X_batch = df_batch.select(["feature1", "feature2"]).to_numpy()
    y_batch = df_batch.select("target").to_numpy().ravel()
    
    # Entrenar con este lote
    modelo.fit(X_batch, y_batch)
    
    print(f"✓ Lote {i+1}/{num_batches} procesado")

print("✅ Entrenamiento incremental completado")
```

### Caso C: Deep Learning con PyTorch

```python
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Crear un Dataset personalizado
class ParquetDataset(Dataset):
    def __init__(self, archivo_parquet):
        # Cargar datos
        df = pl.read_parquet(archivo_parquet)
        
        # Separar features y target
        self.X = torch.tensor(
            df.select(["feature1", "feature2", "feature3"]).to_numpy(),
            dtype=torch.float32
        )
        self.y = torch.tensor(
            df.select("target").to_numpy(),
            dtype=torch.float32
        )
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Crear DataLoader
dataset = ParquetDataset("dataset_final.parquet")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Definir modelo
import torch.nn as nn

class ModeloSimple(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. Entrenar
modelo = ModeloSimple(input_size=3)  # 3 features en este ejemplo
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)

# Loop de entrenamiento
for epoch in range(10):
    for X_batch, y_batch in dataloader:
        # Forward pass
        predictions = modelo(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

print("✅ Entrenamiento completado")

# 5. Guardar modelo
torch.save(modelo.state_dict(), "modelo_pytorch.pth")
```

---

## 5. Análisis Exploratorio de Datos

### Resumen estadístico completo

```python
import polars as pl

df = pl.read_parquet("dataset_final.parquet")

# Estadísticas descriptivas
print("📊 RESUMEN ESTADÍSTICO")
print("="*70)
print(df.describe())

# Valores únicos por columna
print("\n🔍 VALORES ÚNICOS POR COLUMNA")
print("="*70)
for col in df.columns:
    n_unique = df[col].n_unique()
    print(f"{col:30} → {n_unique:,} valores únicos")

# Valores nulos
print("\n⚠️  VALORES NULOS")
print("="*70)
null_counts = df.null_count()
print(null_counts)

# Correlaciones (para columnas numéricas)
print("\n🔗 CORRELACIONES")
print("="*70)
numeric_cols = [col for col, dtype in df.schema.items() 
                if dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
if numeric_cols:
    print(df.select(numeric_cols).corr())
```

### Visualización con Plotly

```python
import polars as pl
import plotly.express as px

df = pl.read_parquet("dataset_final.parquet")

# Convertir a pandas para plotly
df_pandas = df.to_pandas()

# Gráfico de barras
fig = px.bar(
    df_pandas.groupby('categoria').size().reset_index(name='count'),
    x='categoria',
    y='count',
    title='Distribución por Categoría'
)
fig.show()

# Histograma (para columna numérica)
if 'valor' in df_pandas.columns:
    fig = px.histogram(df_pandas, x='valor', title='Distribución de Valores')
    fig.show()

# Scatter plot
if 'feature1' in df_pandas.columns and 'feature2' in df_pandas.columns:
    fig = px.scatter(
        df_pandas,
        x='feature1',
        y='feature2',
        color='categoria',
        title='Relación entre Features'
    )
    fig.show()
```

---

## 🎯 Tips y Trucos

### 1. Verificar tamaño antes de cargar

```python
import polars as pl

# Ver tamaño sin cargar
df = pl.scan_parquet("dataset_grande.parquet")
total_filas = df.select(pl.count()).collect()[0, 0]
print(f"El dataset tiene {total_filas:,} filas")

# Solo cargar si es manejable
if total_filas < 1_000_000:
    df_real = pl.read_parquet("dataset_grande.parquet")
else:
    print("Dataset muy grande, usar scan_parquet y filtros")
```

### 2. Limpiar datos antes de entrenar

```python
import polars as pl

df = pl.read_parquet("dataset_sucio.parquet")

# Eliminar duplicados
df = df.unique()

# Eliminar filas con nulos en columnas críticas
df = df.drop_nulls(subset=["columna_importante"])

# Rellenar nulos con la media (para columnas numéricas)
df = df.fill_nan(0)

# Guardar dataset limpio
df.write_parquet("dataset_limpio.parquet")
```

### 3. Optimizar espacio en disco

```python
import polars as pl

# Leer dataset
df = pl.read_parquet("dataset_grande.parquet")

# Optimizar tipos de datos
df_optimizado = df.select([
    # Convertir int64 a int32 si los valores lo permiten
    pl.col("id").cast(pl.Int32),
    # Usar categorías para strings repetitivos
    pl.col("categoria").cast(pl.Categorical),
    # Mantener otras columnas
    *[col for col in df.columns if col not in ["id", "categoria"]]
])

# Guardar optimizado
df_optimizado.write_parquet(
    "dataset_optimizado.parquet",
    compression="zstd"  # Máxima compresión
)
```

---

¿Necesitas ayuda con algún caso específico? Consulta la documentación completa en `README_ARROW.md`
