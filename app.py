from flask import Flask, render_template, jsonify
import os
import json
import analisis
import visualizaciones

app = Flask(__name__)

# Ruta al JSON generado por el ETL
RUTA_RESUMEN = os.path.join("data", "resumen_estadistico.json")

def cargar_resumen():
    if not os.path.exists(RUTA_RESUMEN): return None
    try:
        with open(RUTA_RESUMEN, 'r') as f: return json.load(f)
    except: return None

# --- HOME ---
@app.route('/')
def index():
    resumen = cargar_resumen()
    # Valores por defecto para evitar errores si no hay JSON
    if not resumen:
        return "<h1>⚠️ Error: Ejecuta 'python etl_limpieza.py' primero.</h1>"
    
    general = resumen.get('general', {}).get('mag', {})
    return render_template('index.html',
                           total=int(float(general.get('count', 0))),
                           promedio=round(float(general.get('mean', 0)), 2),
                           maximo=round(float(general.get('max', 0)), 2),
                           regiones=resumen.get('top_regiones', {}))

# --- GRÁFICOS (Etapa 3) ---
@app.route('/grafico/magnitudes')
def grafico_magnitudes():
    return render_template('grafico_simple.html', titulo="Histograma de Magnitudes", 
                           imagen=visualizaciones.plot_histograma_magnitud())

@app.route('/grafico/correlacion')
def grafico_correlacion():
    return render_template('grafico_doble.html', titulo="Correlación Magnitud/Profundidad",
                           img1=visualizaciones.plot_scatter_profundidad_mag(),
                           img2=visualizaciones.plot_matriz_correlacion())

@app.route('/comparacion/continentes') # [cite: 72]
def comparacion_continentes():
    return render_template('grafico_simple.html', titulo="Comparación por Continentes", 
                           imagen=visualizaciones.plot_comparacion_continentes())

# --- MAPAS (Folium, Plotly, Leaflet) ---
@app.route('/mapa/mundial') # Folium [cite: 71]
def mapa_mundial():
    return render_template('mapa.html', mapa=visualizaciones.mapa_folium())

@app.route('/mapa/plotly') # Plotly
def mapa_plotly_route():
    return render_template('mapa_plotly.html', mapa_div=visualizaciones.mapa_plotly())

@app.route('/mapa/leaflet') # Leaflet (consume API)
def mapa_leaflet_route():
    return render_template('mapa_leaflet.html')

# --- API JSON (Para Leaflet) [cite: 67] ---
@app.route('/api/datos_sismos')
def api_sismos():
    df = analisis.cargar_datos()
    if df is None: return jsonify([])
    # Retornamos los 100 sismos más fuertes para no saturar JS
    top = df.nlargest(100, 'mag')[['latitude', 'longitude', 'mag', 'place']]
    return jsonify(top.to_dict(orient='records'))

# --- MODELOS (Etapa 4) [cite: 76] ---
@app.route('/modelos')
def modelos():
    return render_template('modelos.html', 
                           regresion=analisis.modelo_regresion_lineal(),
                           centroides=analisis.modelo_clustering_kmeans())

if __name__ == '__main__':
    app.run(debug=True, port=5000)