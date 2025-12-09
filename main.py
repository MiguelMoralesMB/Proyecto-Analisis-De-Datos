import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
ARCHIVO_CSV = "Earthquakes_USGS.csv"
ARCHIVO_PARQUET = "terremotos_limpios.parquet"

def optimizar_memoria(df):
    """
    Reduce el uso de memoria convirtiendo tipos de datos.
    float64 -> float32
    int64 -> int32
    object -> category (si hay pocos valores únicos)
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"   Memoria antes de optimizar: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            # Optimizar Números
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            # Optimizar Texto a Categoría si hay pocos valores únicos (menos del 50%)
            num_unique = len(df[col].unique())
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"   Memoria después de optimizar: {end_mem:.2f} MB")
    return df

# =============================================================================
# LÓGICA DE CARGA INTELIGENTE
# =============================================================================

if os.path.exists(ARCHIVO_PARQUET):
    # --- CAMINO RÁPIDO ---
    print(f"🚀 Archivo optimizado detectado: '{ARCHIVO_PARQUET}'")
    print("Cargando datos procesados (esto será rápido)...")
    terremotos = pd.read_parquet(ARCHIVO_PARQUET)
    print("✅ Carga completada.")

else:
    # --- CAMINO LENTO (Solo la primera vez) ---
    print(f"⚠️ Archivo optimizado no encontrado.")
    print(f"Cargando '{ARCHIVO_CSV}' (esto tomará tiempo, paciencia)...")
    
    # 1. CARGA
    # Usamos low_memory=False para evitar DtypeWarning y errores, aunque use más RAM
    try:
        terremotos = pd.read_csv(ARCHIVO_CSV, low_memory=False)
    except Exception as e:
        print(f"Error crítico al cargar: {e}")
        exit()

    print("✅ CSV Cargado. Iniciando limpieza y optimización...")

    # 2. OPTIMIZACIÓN DE MEMORIA PARA RAM 16gb
    terremotos = optimizar_memoria(terremotos)

    # 3. LIMPIEZA DE DUPLICADOS
    filas_antes = len(terremotos)
    terremotos.drop_duplicates(inplace=True)
    print(f"   Duplicados eliminados: {filas_antes - len(terremotos)}")

    # 4. TRATAMIENTO DE NULOS
    # Estrategia: 
    # - Eliminar filas si faltan datos críticos (Ej: Magnitud o Ubicación)
    # - Rellenar datos secundarios con la mediana o 'Desconocido'
    
    # A. Eliminar nulos críticos (Ajusta 'mag' según tus columnas reales)
    if 'mag' in terremotos.columns:
        terremotos.dropna(subset=['mag', 'latitude', 'longitude'], inplace=True)
    
    # B. Imputar nulos numéricos restantes con la Mediana
    cols_numericas = terremotos.select_dtypes(include=['float32', 'int32', 'float64']).columns
    for col in cols_numericas:
        terremotos[col] = terremotos[col].fillna(terremotos[col].median())

    # C. Imputar nulos de texto con 'Desconocido'
    cols_texto = terremotos.select_dtypes(include=['object', 'category']).columns
    for col in cols_texto:
        if terremotos[col].dtype.name == 'category':
            # Añadir la categoría antes de llenar
            if 'Desconocido' not in terremotos[col].cat.categories:
                terremotos[col] = terremotos[col].cat.add_categories('Desconocido')
            terremotos[col] = terremotos[col].fillna('Desconocido')
        else:
            terremotos[col] = terremotos[col].fillna('Desconocido')

    # 5. FORMATOS
    # Convertir columna de tiempo a datetime
    if 'time' in terremotos.columns:
        terremotos['time'] = pd.to_datetime(terremotos['time'], errors='coerce')

    # Normalizar texto (Minúsculas y sin espacios extra)
    if 'place' in terremotos.columns:
        terremotos['place'] = terremotos['place'].astype(str).str.lower().str.strip()

    # 6. NORMALIZACIÓN (Min-Max Scaling)
    # Solo normalizamos columnas útiles para modelos (Ej: profundidad y magnitud)
    cols_a_normalizar = ['depth', 'mag'] # Asegúrate que existan
    cols_existentes = [c for c in cols_a_normalizar if c in terremotos.columns]
    
    if cols_existentes:
        scaler = MinMaxScaler()
        # Creamos nuevas columnas normalizadas para no perder las originales
        nombres_nuevos = [f"{c}_norm" for c in cols_existentes]
        terremotos[nombres_nuevos] = scaler.fit_transform(terremotos[cols_existentes])
        print("   Normalización aplicada a: ", cols_existentes)

    # 7. GUARDAR PROGRESO
    print("💾 Guardando resultado en formato Parquet...")
    terremotos.to_parquet(ARCHIVO_PARQUET, index=False)
    print("✅ ¡Proceso completado y guardado! La próxima ejecución será instantánea.")

# =============================================================================
# ZONA DE TRABAJO Y ANÁLISIS
# =============================================================================

print("\n" + "="*40)
print("INFORMACIÓN DEL DATASET LISTO")
print("="*40)
print(terremotos.info())
print("\nPrimeras 5 filas:")
print(terremotos.head())

# --- TU CÓDIGO DE MINERÍA VA AQUÍ ABAJO ---