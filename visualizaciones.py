import matplotlib
matplotlib.use('Agg') # Modo "sin ventana" para servidores webs
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import folium
from analisis import cargar_datos

# Configuración de estilo
sns.set_theme(style="whitegrid")

def codificar_grafico(fig):
    """Convierte una figura de Matplotlib a string Base64 para HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) # Cierra la figura para liberar memoria RAM
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{data}"

# --- GRÁFICOS ESTÁTICOS ---

def plot_histograma_magnitud():
    df = cargar_datos()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['mag'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribución de Magnitudes")
    return codificar_grafico(fig)

def plot_scatter_profundidad_mag():
    df = cargar_datos()
    # Tomamos una muestra para que el gráfico no sea una mancha ilegible
    sample = df.sample(min(len(df), 2000)) 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sample, x='depth', y='mag', alpha=0.5, color='coral', ax=ax)
    ax.set_title("Correlación: Profundidad vs Magnitud")
    
    # Añadimos línea de tendencia simple
    sns.regplot(data=sample, x='depth', y='mag', scatter=False, color='red', ax=ax)
    return codificar_grafico(fig)

def plot_boxplot_comparativo():
    """Compara magnitudes entre las top 5 regiones más activas."""
    df = cargar_datos()
    top_places = df['place'].value_counts().head(5).index
    df_filtered = df[df['place'].isin(top_places)]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='place', y='mag', data=df_filtered, palette="Set3", ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Comparación de Magnitudes: Top 5 Regiones")
    return codificar_grafico(fig)

def plot_matriz_correlacion():
    df = cargar_datos()
    cols = ['mag', 'depth', 'latitude', 'longitude']
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Matriz de Correlación")
    return codificar_grafico(fig)

# --- MAPAS INTERACTIVOS (FOLIUM) ---

def mapa_interactivo():
    df = cargar_datos()
    # Centrado en el pacífico
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Agregamos solo una muestra de 500 sismos significativos para no colgar el navegador
    sismos_fuertes = df[df['mag'] > 5.5].sample(min(len(df), 500))

    for _, row in sismos_fuertes.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['mag'],
            popup=f"Lugar: {row['place']}<br>Mag: {row['mag']}",
            color="crimson",
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
        
    return m._repr_html_()