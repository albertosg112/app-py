# 🚀 Guía de Inicio Rápido - Apache Arrow

## ⚡ En 5 Minutos Empieza a Usar Arrow

### Paso 1: Instalar Todo (2 minutos)

Abre tu terminal (CMD en Windows) y ejecuta:

```bash
pip install polars pyarrow openpyxl fastapi uvicorn
```

Espera a que termine y verás:
```
✅ Successfully installed polars-... pyarrow-... openpyxl-...
```

---

### Paso 2: Elige Tu Caso de Uso (1 minuto)

Tienes **3 scripts listos para usar**. Elige el que necesites:

#### 📊 Opción A: Procesar Archivos Excel (Múltiples Archivos)
**¿Cuándo usar?** Tienes muchos archivos .xlsx en carpetas
- **Script**: `unificar_seguridad.py`
- **Resultado**: Un solo archivo .parquet con todo

#### 📄 Opción B: Convertir CSV/JSON Grande (Archivo Único)
**¿Cuándo usar?** Tienes un archivo grande (10-30 GB) en CSV o JSON
- **Script**: `convertir_a_parquet.py`
- **Resultado**: Archivo .parquet comprimido y rápido

#### 🔄 Opción C: Integrar con n8n (Flujo Continuo)
**¿Cuándo usar?** Recolectas datos con n8n y quieres almacenarlos
- **Script**: `n8n_arrow_bridge.py`
- **Resultado**: Servidor que recibe y almacena datos automáticamente

---

### Paso 3: Configurar y Ejecutar (2 minutos)

#### Si elegiste Opción A (Excel):

1. Abre `unificar_seguridad.py` con un editor
2. Busca la línea que dice:
   ```python
   RUTA_BASE = "F:/Sistema de gestion seguridad"
   ```
3. Cámbiala por tu ruta real (donde están tus Excel)
4. Guarda el archivo
5. Ejecuta en terminal:
   ```bash
   python unificar_seguridad.py
   ```

#### Si elegiste Opción B (CSV/JSON):

1. Abre `convertir_a_parquet.py` con un editor
2. Busca estas líneas:
   ```python
   TIPO_ARCHIVO = "csv"  # o "json"
   ARCHIVO_ORIGEN = "datos_grandes.csv"
   ```
3. Cambia `ARCHIVO_ORIGEN` por la ruta de tu archivo
4. Guarda
5. Ejecuta:
   ```bash
   python convertir_a_parquet.py
   ```

#### Si elegiste Opción C (n8n):

1. Solo ejecuta:
   ```bash
   python n8n_arrow_bridge.py
   ```
2. Verás:
   ```
   🚀 Iniciando servidor en http://127.0.0.1:8000
   📖 Documentación: http://localhost:8000/docs
   ```
3. En n8n, añade un nodo HTTP Request al final de tu flujo:
   - URL: `http://127.0.0.1:8000/guardar` (o `http://localhost:8000/guardar`)
   - Method: POST
   - Body:
     ```json
     {
       "datos": {{ $json.items }},
       "nombre_dataset": "mi_dataset"
     }
     ```

---

## 🎯 Uso Básico del Dataset Generado

Una vez que tengas tu archivo `.parquet` generado, úsalo así:

```python
import polars as pl

# Cargar el dataset
df = pl.read_parquet("dataset_final.parquet")  # o el nombre que hayas usado

# Ver primeras filas
print(df.head())

# Ver estadísticas
print(f"Total de registros: {len(df):,}")
print(f"Columnas: {df.columns}")

# Filtrar datos (ejemplo)
df_filtrado = df.filter(pl.col("columna") > 100)

# Exportar a Excel (si necesitas)
df_filtrado.write_excel("resultado.xlsx")
```

---

## 🆘 Solución Rápida a Problemas Comunes

### ❌ Error: "No such file or directory"
**Solución**: La ruta del archivo está mal escrita.
- En Windows usa `/` o `\\` (no `\`)
- Ejemplo correcto: `"F:/MiCarpeta/archivo.csv"`
- Ejemplo correcto: `"F:\\MiCarpeta\\archivo.csv"`

### ❌ Error: "ModuleNotFoundError: No module named 'polars'"
**Solución**: No instalaste las librerías.
```bash
pip install polars pyarrow openpyxl
```

### ❌ El script no hace nada
**Solución**: Verifica que ejecutaste:
```bash
python nombre_del_script.py
```
(No solo abrir el archivo)

### ❌ "Permission denied" o "Access denied"
**Solución**: 
- Cierra Excel si tienes el archivo abierto
- Ejecuta la terminal como Administrador (Windows)

---

## 📚 Siguientes Pasos

### 1. Explora tus datos

```python
import polars as pl

df = pl.read_parquet("dataset_final.parquet")

# Ver todas las columnas
print(df.columns)

# Contar valores únicos
print(df.select(pl.all().n_unique()))

# Buscar valores nulos
print(df.null_count())
```

### 2. Limpia tus datos

```python
# Eliminar duplicados
df = df.unique()

# Eliminar filas con valores nulos
df = df.drop_nulls()

# Guardar versión limpia
df.write_parquet("dataset_limpio.parquet")
```

### 3. Prepara para IA

```python
# Para scikit-learn, PyTorch, TensorFlow
X = df.drop("columna_objetivo").to_numpy()
y = df.select("columna_objetivo").to_numpy()

# Ya puedes entrenar
from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier()
modelo.fit(X, y)
```

---

## 🎓 Recursos de Aprendizaje

1. **Documentación completa**: Lee `README_ARROW.md`
2. **Ejemplos avanzados**: Lee `EJEMPLOS_USO.md`
3. **API interactiva** (si usas n8n): http://localhost:8000/docs

---

## 💡 Tips Pro

### Tip 1: Verifica antes de procesar
```python
import os
archivo = "F:/mi_archivo.csv"
if os.path.exists(archivo):
    print("✅ Archivo encontrado")
else:
    print("❌ Archivo NO existe en esa ruta")
```

### Tip 2: Procesa solo lo necesario
```python
# No cargues TODO si solo necesitas algunas columnas
df = pl.scan_parquet("dataset_grande.parquet").select([
    "columna1",
    "columna2"
]).collect()
```

### Tip 3: Compara tamaños
```bash
# En terminal, ver tamaño de archivos
# Windows:
dir *.csv
dir *.parquet

# Linux/Mac:
ls -lh *.csv
ls -lh *.parquet
```

Verás que el `.parquet` es mucho más pequeño (70-90% menos)

---

## 🎯 Checklist de Éxito

- [ ] Instalé polars, pyarrow, openpyxl
- [ ] Elegí el script correcto para mi caso
- [ ] Configuré la ruta del archivo correctamente
- [ ] Ejecuté el script y se generó el .parquet
- [ ] Verifiqué el resultado con `pl.read_parquet()`
- [ ] Exploré los datos básicos (head, columns, len)

---

## 🚀 ¡Listo!

Si llegaste aquí y completaste el checklist:

**✅ Ya estás usando Apache Arrow profesionalmente**

Tus datos ahora:
- ✅ Ocupan 70-90% menos espacio
- ✅ Se cargan 100x más rápido
- ✅ Están listos para IA/ML
- ✅ Son compatibles con cualquier herramienta moderna

---

**¿Dudas?** Consulta:
- `README_ARROW.md` - Guía completa
- `EJEMPLOS_USO.md` - Casos de uso detallados

**¿Problemas?** Verifica:
1. Ruta del archivo correcta
2. Librerías instaladas
3. Archivo no abierto en otro programa
