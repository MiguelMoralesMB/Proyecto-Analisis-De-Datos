import matplotlib
matplotlib.use('Agg') # CRÍTICO: Backend no interactivo para Flask
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import folium
import plotly.express as px
import plotly.io as pio
import analisis # Importamos para pedir los datos

sns.set_theme(style="whitegrid")

def codificar_grafico(fig):
    """Helper para convertir Matplotlib a Base64."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{data}"

# --- GRÁFICOS ESTÁTICOS ---
def plot_histograma_magnitud():
    df = analisis.cargar_datos()
    if df is None: return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['mag'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribución de Magnitudes")
    return codificar_grafico(fig)

def plot_scatter_profundidad_mag():
    df = analisis.cargar_datos()
    if df is None: return None
    sample = df.sample(min(len(df), 2000)) # Muestra para rendimiento
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sample, x='depth', y='mag', alpha=0.5, color='coral', ax=ax)
    ax.set_title("Correlación: Profundidad vs Magnitud")
    return codificar_grafico(fig)

def plot_matriz_correlacion():
    df = analisis.cargar_datos()
    if df is None: return None
    cols = ['mag', 'depth', 'latitude', 'longitude']
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Matriz de Correlación")
    return codificar_grafico(fig)

def plot_comparacion_continentes():
    """Boxplot para comparar regiones[cite: 68]."""
    df = analisis.obtener_datos_con_continentes()
    if df is None or df.empty: return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='continente', y='mag', data=df, palette="Set3", ax=ax)
    ax.set_title("Distribución de Magnitud por Continente")
    return codificar_grafico(fig)

# --- MAPAS INTERACTIVOS ---
def mapa_folium():
    """Genera HTML de mapa Folium[cite: 65]."""
    df = analisis.cargar_datos()
    if df is None: return "Sin datos"
    
    m = folium.Map(location=[0, 0], zoom_start=2)
    sample = df[df['mag'] > 5.0].sample(min(len(df), 300)) # Solo sismos fuertes
    
    for _, row in sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['mag'],
            color="red", fill=True, fill_opacity=0.6,
            popup=f"Mag: {row['mag']}"
        ).add_to(m)
    return m._repr_html_()

def mapa_plotly():
    """Genera div HTML de mapa Plotly[cite: 66]."""
    df = analisis.cargar_datos()
    if df is None: return "<div>Sin datos</div>"
    
    sample = df[df['mag'] > 4.5].sample(min(len(df), 500))
    fig = px.scatter_geo(sample, lat='latitude', lon='longitude', 
                         color='mag', size='mag', projection="natural earth",
                         title="Mapa Global (Plotly)", color_continuous_scale="Reds")
    return pio.to_html(fig, full_html=False)