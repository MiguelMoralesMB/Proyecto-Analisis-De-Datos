import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Configuración de ruta (Robustez)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_PARQUET = os.path.join(BASE_DIR, "data", "terremotos_limpios.parquet")

def cargar_datos():
    """Carga los datos limpios desde el Parquet."""
    if os.path.exists(RUTA_PARQUET):
        return pd.read_parquet(RUTA_PARQUET)
    return None

# =============================================================================
# LÓGICA DE CONTINENTES (Requisito: Comparación Regional) [cite: 68]
# =============================================================================
def asignar_continente(texto_lugar):
    """Asigna continente basado en palabras clave del lugar."""
    texto = str(texto_lugar).lower()
    
    if any(x in texto for x in ['chile', 'peru', 'mexico', 'usa', 'california', 'alaska', 'argentina', 'colombia', 'ecuador', 'panama', 'bolivia', 'venezuela']):
        return 'América'
    elif any(x in texto for x in ['japan', 'indonesia', 'china', 'philippines', 'india', 'taiwan', 'iran', 'turkey', 'nepal', 'afghanistan']):
        return 'Asia'
    elif any(x in texto for x in ['italy', 'greece', 'iceland', 'portugal', 'spain', 'romania', 'albania']):
        return 'Europa'
    elif any(x in texto for x in ['fiji', 'tonga', 'new zealand', 'vanuatu', 'solomon', 'papua', 'australia']):
        return 'Oceanía'
    elif any(x in texto for x in ['africa', 'congo', 'south africa', 'morocco', 'algeria']):
        return 'África'
    else:
        return 'Otros/Océano'

def obtener_datos_con_continentes():
    """Devuelve el DF con una nueva columna 'continente'."""
    df = cargar_datos()
    if df is not None and 'place' in df.columns:
        df['continente'] = df['place'].apply(asignar_continente)
        # Filtramos 'Otros' para limpiar la gráfica
        return df[df['continente'] != 'Otros/Océano']
    return df

# =============================================================================
# MODELOS ANALÍTICOS (Etapa 4) [cite: 75]
# =============================================================================
def modelo_regresion_lineal():
    """Predice Magnitud basada en Profundidad."""
    df = cargar_datos()
    if df is None: return None
    
    # Preparamos datos sin nulos
    datos = df[['depth', 'mag']].dropna()
    X = datos[['depth']].values
    y = datos['mag'].values
    
    if len(X) < 10: return None # Validación mínima

    modelo = LinearRegression()
    modelo.fit(X, y)
    
    return {
        "coeficiente": round(modelo.coef_[0], 5),
        "intercepto": round(modelo.intercept_, 4),
        "r2_score": round(modelo.score(X, y), 4)
    }

def modelo_clustering_kmeans(n_clusters=5):
    """Agrupa sismos por ubicación geográfica."""
    df = cargar_datos()
    if df is None: return []

    X = df[['latitude', 'longitude']].dropna()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    return kmeans.cluster_centers_.tolist()