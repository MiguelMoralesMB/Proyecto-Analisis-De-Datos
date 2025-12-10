import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Función auxiliar de carga (Reutilizable)
def cargar_datos(ruta="data/terremotos_limpios.parquet"):
    if pd.io.common.file_exists(ruta):
        return pd.read_parquet(ruta)
    return None

# --- ETAPA 2: ANÁLISIS EXPLORATORIO ---

def obtener_top_regiones(n=5):
    """Identifica los lugares con más actividad sísmica."""
    df = cargar_datos()
    if df is None: return {}
    # Filtramos por columna 'place' si existe
    if 'place' in df.columns:
        return df['place'].value_counts().head(n).to_dict()
    return {}

def frecuencia_temporal():
    """Analiza frecuencia por año para tendencias."""
    df = cargar_datos()
    if df is None or 'time' not in df.columns: return {}
    
    # Agrupamos por año
    conteo_anual = df['time'].dt.year.value_counts().sort_index().to_dict()
    return conteo_anual

# --- ETAPA 4: MODELOS ANALÍTICOS ---

def modelo_regresion_lineal():
    """
    Aplica Regresión Lineal:
    Intenta predecir la Magnitud (Y) basándose en la Profundidad (X).
    """
    df = cargar_datos()
    if df is None: return None

    # Preparamos variables (X debe ser 2D)
    X = df[['depth']].values
    y = df['mag'].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    # Retornamos los coeficientes para mostrarlos en la web
    return {
        "coeficiente": modelo.coef_[0],
        "intercepto": modelo.intercept_,
        "r2_score": modelo.score(X, y) # Qué tan bueno es el modelo (0 a 1)
    }

def modelo_clustering_kmeans(n_clusters=3):
    """
    Aplica Clustering (K-Means) para agrupar sismos por ubicación geográfica.
    Devuelve los centros de los clusters (lat, lon).
    """
    df = cargar_datos()
    if df is None: return None

    # Usamos Latitud y Longitud para agrupar
    X = df[['latitude', 'longitude']].dropna()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    return kmeans.cluster_centers_.tolist()