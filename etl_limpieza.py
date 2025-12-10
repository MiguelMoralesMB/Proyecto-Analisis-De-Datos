import pandas as pd
import numpy as np
import os
import json # Necesario para guardar el resumen estadístico
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1. CONFIGURACIÓN Y RUTAS
# =============================================================================
CARPETA_DATA = "data"
ARCHIVO_CSV = os.path.join(CARPETA_DATA, "Earthquakes_USGS.csv")
ARCHIVO_PARQUET = os.path.join(CARPETA_DATA, "terremotos_limpios.parquet")
ARCHIVO_RESUMEN = os.path.join(CARPETA_DATA, "resumen_estadistico.json") # Nuevo archivo para el resumen

if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)
    
# =============================================================================
# 2. FUNCIONES DE LIMPIEZA Y ANÁLISIS DESCRIPTIVO
# =============================================================================

def optimizar_memoria(df):
    """
    Reduce el uso de memoria convirtiendo tipos de datos.
    float64 -> float32, int64 -> int32, object -> category.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"   Memoria antes de optimizar: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
             # Lógica de optimización numérica (mantenida de tu script)
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            # Lógica de optimización de texto a Categoría
            num_unique = len(df[col].unique())
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"   Memoria después de optimizar: {end_mem:.2f} MB")
    return df

def crear_resumen_estadistico(df):
    """
    Crea un resumen estadístico clave usando describe() y groupby()  
    y lo guarda como un archivo JSON para que Flask lo pueda leer.
    """
    print("   Generando Resumen Estadístico para Flask...")
    
    # 1. Descripción General (describe())
    # Usamos solo los principales tipos de datos para evitar errores en JSON
    resumen_general = df.describe(include=[np.number, 'category']).transpose()
    # Convertimos a string y luego a dict, ya que .to_dict() directo puede fallar con categorías.
    resumen_dict = resumen_general.astype(str).to_dict()

    # 2. Conteo por Año/Década (groupby())
    if 'time' in df.columns:
        # Añadir columna de año y década para el análisis temporal
        df['year'] = df['time'].dt.year
        df['decade'] = (df['year'] // 10 * 10).astype('category')
        
        # Conteo de sismos por década
        sismos_por_decada = df.groupby('decade')['mag'].count().to_dict()
    else:
        sismos_por_decada = {"Error": "Columna 'time' no encontrada o no convertida."}

    # 3. Conteo de países/regiones más afectadas (groupby())
    if 'place' in df.columns:
        # Encontramos la región más prominente de los 10 primeros
        conteo_regiones = df['place'].value_counts().nlargest(10).to_dict()
    else:
        conteo_regiones = {"Error": "Columna 'place' no encontrada"}

    resumen_final = {
        "general": resumen_dict,
        "sismos_por_decada": sismos_por_decada,
        "top_regiones": conteo_regiones
    }
    
    # Guardar el JSON
    with open(ARCHIVO_RESUMEN, 'w') as f:
        json.dump(resumen_final, f, indent=4) 
    
    print(f"✅ Resumen estadístico guardado en: {ARCHIVO_RESUMEN}")


# =============================================================================
# 3. LÓGICA PRINCIPAL DEL ETL
# =============================================================================

if os.path.exists(ARCHIVO_PARQUET):
    # --- CAMINO RÁPIDO ---
    print(f"🚀 Archivo optimizado detectado: '{ARCHIVO_PARQUET}'.")
    terremotos = pd.read_parquet(ARCHIVO_PARQUET)
    crear_resumen_estadistico(terremotos.copy()) #Crea un json
    print("✅ Carga completada.")

else:
    # --- CAMINO LENTO (Solo la primera vez) ---
    print(f"⚠️ Archivo optimizado no encontrado.")
    print(f"Cargando '{ARCHIVO_CSV}' (esto tomará tiempo, paciencia)...")
    
    # 1. CARGA
    try:
        terremotos = pd.read_csv(ARCHIVO_CSV, low_memory=False)
    except FileNotFoundError:
        print(f"Error crítico: El archivo {ARCHIVO_CSV} no se encuentra.")
        exit()

    print("✅ CSV Cargado. Iniciando limpieza y optimización...")

    # 2. OPTIMIZACIÓN DE MEMORIA
    terremotos = optimizar_memoria(terremotos)

    # 3. LIMPIEZA DE DUPLICADOS
    filas_antes = len(terremotos)
    terremotos.drop_duplicates(inplace=True)
    print(f"   Duplicados eliminados: {filas_antes - len(terremotos)}")

    # 4. TRATAMIENTO DE NULOS
    # A. Eliminar nulos críticos (magnitud y ubicación)
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
            # Manejar la adición de categoría 'Desconocido'
            if 'Desconocido' not in terremotos[col].cat.categories:
                 terremotos[col] = terremotos[col].cat.add_categories('Desconocido')
            terremotos[col] = terremotos[col].fillna('Desconocido')
        else:
            terremotos[col] = terremotos[col].fillna('Desconocido')

    # 5. FORMATOS y Consistencia
    # Convertir columna de tiempo a datetime
    if 'time' in terremotos.columns:
        terremotos['time'] = pd.to_datetime(terremotos['time'], errors='coerce')
        
    # Normalizar texto (Minúsculas y sin espacios extra)
    if 'place' in terremotos.columns:
        terremotos['place'] = terremotos['place'].astype(str).str.lower().str.strip()

    # 6. NORMALIZACIÓN (Min-Max Scaling)
    cols_a_normalizar = ['depth', 'mag'] 
    cols_existentes = [c for c in cols_a_normalizar if c in terremotos.columns]
    
    if cols_existentes:
        scaler = MinMaxScaler()
        nombres_nuevos = [f"{c}_norm" for c in cols_existentes]
        # Creamos nuevas columnas normalizadas
        terremotos[nombres_nuevos] = scaler.fit_transform(terremotos[cols_existentes])
        print("   Normalización aplicada a: ", cols_existentes)

    # 7. GUARDAR PROGRESO (Parquet y Resumen Estadístico)
    crear_resumen_estadistico(terremotos.copy()) # Creamos el resumen del dataset limpio
    
    print("💾 Guardando resultado en formato Parquet...")
    terremotos.to_parquet(ARCHIVO_PARQUET, index=False)
    print("✅ ¡Proceso completado y guardado! La próxima ejecución será instantánea.")

# =============================================================================
# FIN DEL PROCESO ETL Y PIE PARA ANÁLISIS POSTERIORES
# =============================================================================

print("\n" + "="*50)
print("ETL COMPLETADO. Los datos limpios están en la carpeta 'data'.")
print("Puedes continuar con el análisis exploratorio (Etapa 2) desde app.py.")
print("="*50)