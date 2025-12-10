from flask import Flask, render_template
import os
import json
import analisis          # Tu archivo analisis.py
import visualizaciones   # Tu archivo visualizaciones.py

app = Flask(__name__)

# --- CONFIGURACIÓN ---
# Ruta al archivo JSON que genera el ETL
ARCHIVO_RESUMEN = os.path.join("data", "resumen_estadistico.json")

def cargar_resumen_json():
    """
    Función auxiliar para leer el resumen estadístico generado por etl_limpieza.py.
    Incluye diagnósticos para saber si falla.
    """
    if not os.path.exists(ARCHIVO_RESUMEN):
        print(f"ERROR CRÍTICO: No encuentro el archivo '{ARCHIVO_RESUMEN}'.")
        print("SOLUCIÓN: Ejecuta 'python etl_limpieza.py' primero.")
        return None
    
    try:
        with open(ARCHIVO_RESUMEN, 'r') as f:
            data = json.load(f)
            print("✅ JSON cargado correctamente.")
            return data
    except Exception as e:
        print(f"ERROR al leer el JSON: {e}")
        return None

# =============================================================================
# RUTA 1: PÁGINA DE INICIO (DASHBOARD)
# =============================================================================
@app.route('/')
def index():
    # 1. Cargamos datos del JSON
    resumen = cargar_resumen_json()

    # 2. Si falla la carga, mostramos error en pantalla
    if resumen is None:
        return """
        <div style='color: red; text-align: center; margin-top: 50px;'>
            <h1>⚠️ Error: Datos no encontrados</h1>
            <p>El archivo <code>data/resumen_estadistico.json</code> no existe.</p>
            <p><strong>Solución:</strong> Ve a tu terminal y ejecuta: <code>python etl_limpieza.py</code></p>
        </div>
        """

    # 3. Extraemos las variables con seguridad (usando .get para que no rompa si falta algo)
    # Accedemos a ['general']['mag'] porque así lo guarda el pandas describe()
    try:
        stats_mag = resumen.get('general', {}).get('mag', {})
        total_sismos = int(stats_mag.get('count', 0))
        promedio_mag = round(float(stats_mag.get('mean', 0)), 2)
        max_mag = round(float(stats_mag.get('max', 0)), 2)
        
        # Datos para gráficos simples en el home (si los usas)
        top_regiones = resumen.get('top_regiones', {})
    except Exception as e:
        print(f"Error procesando datos del JSON: {e}")
        total_sismos = "Error"
        promedio_mag = "Error"
        max_mag = "Error"
        top_regiones = {}

    # 4. Renderizamos el HTML pasando las variables
    return render_template('index.html', 
                           total=total_sismos, 
                           promedio=promedio_mag,
                           maximo=max_mag,
                           regiones=top_regiones)

# =============================================================================
# ETAPA 3: VISUALIZACIONES (Rutas del PDF)
# =============================================================================

@app.route('/grafico/magnitudes')
def grafico_magnitudes():
    # Llama a la función en visualizaciones.py
    imagen_b64 = visualizaciones.plot_histograma_magnitud()
    
    return render_template('grafico_simple.html', 
                           titulo="Histograma de Magnitudes", 
                           descripcion="Distribución de la frecuencia de los sismos según su magnitud.",
                           imagen=imagen_b64)

@app.route('/grafico/correlacion')
def grafico_correlacion():
    # Generamos dos gráficos para esta vista
    scatter_b64 = visualizaciones.plot_scatter_profundidad_mag()
    heatmap_b64 = visualizaciones.plot_matriz_correlacion()
    
    # Podrías crear un template 'grafico_doble.html' o reutilizar el simple mostrando dos imgs
    # Aquí asumo que usas grafico_doble.html que te pasé antes
    return render_template('grafico_doble.html', 
                           titulo="Análisis de Correlación", 
                           img1=scatter_b64, 
                           img2=heatmap_b64)

@app.route('/comparacion/continentes')
def comparacion_continentes():
    boxplot_b64 = visualizaciones.plot_boxplot_comparativo()
    
    return render_template('grafico_simple.html', 
                           titulo="Comparación Regional", 
                           descripcion="Variabilidad de la magnitud en las regiones más activas.",
                           imagen=boxplot_b64)

@app.route('/mapa/mundial')
def mapa_mundial():
    # Esto devuelve el HTML crudo del mapa
    mapa_html = visualizaciones.mapa_interactivo()
    
    return render_template('mapa.html', mapa=mapa_html)

# =============================================================================
# ETAPA 4: MODELOS ANALÍTICOS
# =============================================================================

@app.route('/modelos')
def modelos():
    # Llamamos a la lógica matemática en analisis.py
    datos_regresion = analisis.modelo_regresion_lineal()
    datos_clustering = analisis.modelo_clustering_kmeans()
    
    return render_template('modelos.html', 
                           regresion=datos_regresion, 
                           centroides=datos_clustering)

# =============================================================================
# INICIO DE LA APP
# =============================================================================
if __name__ == '__main__':
    # debug=True permite que los cambios se vean sin reiniciar el servidor
    app.run(debug=True, port=5000)