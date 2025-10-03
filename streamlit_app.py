# streamlit_app.py
import streamlit as st
import pandas as pd
from io import StringIO
import folium
from streamlit_folium import st_folium
import json
import streamlit.components.v1 as components

# Importaciones de tus módulos
from solver import ACO_CVRP_Solver
from visualization import get_folium_results_map, to_excel, generate_html_report, format_duration

# --- Configuración de la Página ---
st.set_page_config(page_title="Optimizador de Rutas (CVRP-ACO)", page_icon="🚚", layout="wide")

# --- Estado de la Sesión ---
if 'solution' not in st.session_state: st.session_state.solution = None
# CORRECCIÓN DE SINTAXIS APLICADA EN LA LÍNEA 19
if 'evaluation' not in st.session_state: st.session_state.evaluation = None
if 'solver' not in st.session_state: st.session_state.solver = None
if 'customer_data' not in st.session_state: st.session_state.customer_data = None
if 'depot_lat' not in st.session_state: st.session_state.depot_lat = 3.90089
if 'depot_lon' not in st.session_state: st.session_state.depot_lon = -76.29783

# --- Funciones de Utilidad ---
@st.cache_data
def get_example_data():
    csv_data = """name,lat,lon,demand
Cliente 1,3.87103,-76.3117,8
Cliente 2,3.87163,-76.3168,14
Cliente 3,3.87227,-76.3201,3
Cliente 4,3.86949,-76.3129,20
Cliente 5,3.86733,-76.3148,15
Cliente 6,3.86906,-76.3023,12
"""
    return pd.read_csv(StringIO(csv_data))

# --- FUNCIÓN load_data CORREGIDA (LÍNEA 79) ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None: return None
    try:
        try:
            df = pd.read_csv(uploaded_file, delimiter=';', encoding='latin1')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, delimiter=',', encoding='utf-8')

        df.columns = [col.strip().lower() for col in df.columns]
        
        # Mapeo de posibles nombres de columnas a nombres estándar
        COLUMN_MAP = {
            'name': ['nombre', 'cliente'],
            'lat': ['latitud_aproximada', 'latitud', 'lat'],
            'lon': ['longitud_aproximada', 'longitud', 'lon', 'lng'],
            'demand': ['ctd paquetes', 'demanda', 'pasajeros', 'demand']
        }
        
        df_renamed = pd.DataFrame()
        
        for standard_name, possible_names in COLUMN_MAP.items():
            for name in possible_names:
                if name in df.columns:
                    df_renamed[standard_name] = df[name]
                    break
        
        if not all(k in df_renamed.columns for k in ['lat', 'lon', 'demand']):
            st.error("Error: Archivo debe contener columnas de latitud, longitud y demanda.")
            return None
        
        if 'name' not in df_renamed.columns:
            df_renamed['name'] = [f"Cliente {i+1}" for i in range(len(df_renamed))]

        df_renamed[['lat', 'lon', 'demand']] = df_renamed[['lat', 'lon', 'demand']].astype(float)
        
        # CORRECCIÓN APLICADA: Eliminar filas con NaN en las columnas críticas
        df_renamed.dropna(subset=['lat', 'lon', 'demand'], inplace=True) 
        
        return df_renamed
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

# --- Interfaz Principal ---
st.title("🚚 Optimizador de Rutas Vehiculares (CVRP) (ACO)")
st.write("Una herramienta inteligente para encontrar las rutas más eficientes y económicas para tu flota.")

tab_config, tab_results, tab_about = st.tabs(["⚙️ 1. Configuración", "📊 2. Resultados", "👨‍💻 Acerca de"])

with tab_config:
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.subheader("1. Cargar Datos de Clientes")
        uploaded_file = st.file_uploader("Sube tu archivo (delimitado por ';' o ',')", type="csv", label_visibility="collapsed")
        if st.button("📍 Usar Datos de Ejemplo"):
            st.session_state.customer_data = get_example_data()
            st.success(f"Datos de ejemplo cargados: {len(st.session_state.customer_data)} ubicaciones.")
        if uploaded_file:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.customer_data = df
                st.success(f"Archivo cargado: {len(st.session_state.customer_data)} ubicaciones.")

        st.subheader("2. Seleccionar Ubicación del Depósito")
        st.info("Haz clic en el mapa para establecer el punto de partida.")
        
        map_center = [st.session_state.depot_lat, st.session_state.depot_lon]
        if st.session_state.customer_data is not None and not st.session_state.customer_data.empty:
            map_center = [st.session_state.customer_data['lat'].mean(), st.session_state.customer_data['lon'].mean()]

        m_config = folium.Map(location=map_center, zoom_start=13, tiles="cartodbpositron")
        folium.Marker([st.session_state.depot_lat, st.session_state.depot_lon], popup="Depósito", icon=folium.Icon(color="red", icon="truck", prefix='fa')).add_to(m_config)
        if st.session_state.customer_data is not None:
            for _, row in st.session_state.customer_data.iterrows():
                folium.Marker([row['lat'], row['lon']], tooltip=row['name']).add_to(m_config)
        
        map_data = st_folium(m_config, width='100%', height=400, key="depot_map")
        if map_data and map_data["last_clicked"]:
            st.session_state.depot_lat, st.session_state.depot_lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            st.rerun()

    with col2:
        st.subheader("3. Parámetros de Flota")
        st.write(f"**Depósito:** `Lat: {st.session_state.depot_lat:.5f}, Lon: {st.session_state.depot_lon:.5f}`")
        n_vehicles = st.number_input("Número de Vehículos", 1, 100, 10)
        vehicle_capacity = st.number_input("Capacidad por Vehículo", 1, 1000, 150)
        
        st.subheader("4. Parámetros de Simulación")
        cost_per_km = st.number_input("Costo por KM ($)", 0.0, 100000.0, 1500.0, 50.0, format="%.2f")
        speed_kmh = st.number_input("Velocidad Promedio (km/h)", 1.0, 120.0, 40.0, 1.0)
        service_time_min = st.number_input("Tiempo de Servicio por Parada (min)", 0, 120, 10)
        
        with st.expander("5. Ajustes Avanzados del Algoritmo (ACO)"):
            n_ants = st.slider("Hormigas", 5, 100, 30)
            n_iterations = st.slider("Iteraciones", 10, 1000, 200)
            alpha = st.slider("Alpha (α)", 0.1, 5.0, 1.0, 0.1)
            beta = st.slider("Beta (β)", 0.1, 5.0, 2.0, 0.1)
            rho = st.slider("Rho (ρ)", 0.01, 1.0, 0.5, 0.01)
            q_val = st.number_input("Q", value=100)
            use_2_opt = st.toggle("Búsqueda Local (2-opt)", True)
            elitism_weight = st.slider("Peso Elitista", 1.0, 10.0, 3.0, 0.5)

    st.divider()
    if st.button("🚀 Iniciar Optimización", type="primary", use_container_width=True):
        if st.session_state.customer_data is None or st.session_state.customer_data.empty:
            st.error("Por favor, carga datos de clientes o usa el ejemplo.")
        else:
            with st.spinner("Optimizando rutas..."):
                progress_bar = st.progress(0, text="Iniciando...")
                def progress_callback(i, total, cost):
                    progress = i / total
                    cost_str = f"{cost:,.1f}" if cost != float('inf') else "Calculando..."
                    progress_bar.progress(progress, text=f"Iteración {i}/{total} - Mejor Distancia: {cost_str} km")

                solver = ACO_CVRP_Solver(
                    depot_coord=(st.session_state.depot_lon, st.session_state.depot_lat),
                    customer_coords=st.session_state.customer_data[['lon', 'lat']].values.tolist(),
                    customer_demands=st.session_state.customer_data['demand'].values.tolist(),
                    n_vehicles=n_vehicles, vehicle_capacity=vehicle_capacity,
                    params={'alpha': alpha, 'beta': beta, 'rho': rho, 'Q': q_val},
                    use_2_opt=use_2_opt, elitism_weight=elitism_weight
                )
                best_routes, _ = solver.solve(n_ants, n_iterations, progress_callback)
                
                if not best_routes:
                    st.error("No se encontró solución. Prueba ajustar los parámetros.")
                    st.session_state.solution = None
                    st.session_state.evaluation = None
                else:
                    evaluation = solver.evaluate_solution({'routes': best_routes}, cost_per_km, speed_kmh, service_time_min)
                    st.session_state.solution = {'routes': best_routes}
                    st.session_state.evaluation = evaluation
                    st.session_state.solver = solver
                    st.success("¡Optimización completada! Ve a la pestaña 'Resultados'.")
                progress_bar.empty()

with tab_results:
    st.header("Análisis de la Solución Optimizada")
    if st.session_state.evaluation:
        eval_data = st.session_state.evaluation
        
        st.subheader("📈 Resumen Ejecutivo")
        m1, m2, m3 = st.columns(3)
        m1.metric("Costo Total de Rutas", f"${eval_data.get('total_cost', 0):,.0f}")
        m2.metric("Distancia Total", f"{eval_data.get('total_distance_km', 0):,.1f} km")
        m3.metric("Duración Total (Horas-Hombre)", f"{format_duration(eval_data.get('total_duration_hours', 0))}")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Vehículos Usados", f"{eval_data.get('num_vehicles_used', 0)} / {getattr(st.session_state.solver, 'n_vehicles', 'N/A')}")
        m5.metric("Clientes Visitados", f"{eval_data.get('customers_visited_count', 0)} / {getattr(st.session_state.solver, 'n_customers', 'N/A')}")
        m6.metric("Uso Promedio de Capacidad", f"{eval_data.get('avg_vehicle_utilization_percent', 0):.1f}%")

        st.subheader("🗺️ Visualización de Rutas Optimizadas")
        results_map = get_folium_results_map(st.session_state.solution['routes'], st.session_state.solver, st.session_state.customer_data, eval_data)
        st_folium(results_map, width='100%', height=500, returned_objects=[])
        
        st.subheader("📋 Detalles por Ruta")
        for i, detail in enumerate(eval_data.get('route_details', [])):
            cost, duration_h = detail.get('cost', 0), detail.get('duration_hours', 0)
            load, utilization = detail.get('load', 0), detail.get('utilization', 0)
            expander_title = (f"**Ruta {i+1}** | Costo: `${cost:,.0f}` | Duración: `{format_duration(duration_h)}` | Carga: `{load:.0f} ({utilization:.1f}%)`")
            with st.expander(expander_title):
                route_indices = [idx - 1 for idx in detail.get('sequence', [])]
                if not route_indices: continue
                route_df = st.session_state.customer_data.iloc[route_indices]
                st.dataframe(route_df[['name', 'demand', 'lat', 'lon']], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("📥 Descargar Informes")
        
        resumen_df = pd.DataFrame(eval_data.get('route_details', []))
        if not resumen_df.empty:
            resumen_df['vehiculo'] = [f"Ruta {i+1}" for i in range(len(resumen_df))]
            resumen_df = resumen_df[['vehiculo', 'stops', 'load', 'utilization', 'distance', 'cost', 'duration_hours']]
        
        informe_sheets = {"Resumen": resumen_df}
        for i, detail in enumerate(eval_data.get('route_details', [])):
            sheet_name = f"Ruta {i+1}"
            route_indices = [idx - 1 for idx in detail.get('sequence', [])]
            ruta_df = st.session_state.customer_data.iloc[route_indices]
            informe_sheets[sheet_name] = ruta_df[['name', 'lat', 'lon', 'demand']]

        col_excel, col_pdf = st.columns(2)
        with col_excel:
            excel_data = to_excel(informe_sheets)
            st.download_button(label="📥 Descargar (Excel)", data=excel_data,
                               file_name="informe_rutas_aco.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet",
                               use_container_width=True)
        with col_pdf:
            html_report = generate_html_report(eval_data, st.session_state.customer_data)
            html_escaped = json.dumps(html_report)
            components.html(f"""
                <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
                <button id="descargar_pdf" onclick="descargarPDF()">📄 Descargar (PDF)</button>
                <script>
                    const informeHtml = {html_escaped};
                    function descargarPDF() {{
                        const opt = {{ margin: 0.5, filename: 'informe_rutas_aco.pdf', image: {{ type: 'jpeg', quality: 0.98 }},
                                       html2canvas: {{ scale: 2 }}, jsPDF: {{ unit: 'in', format: 'letter', orientation: 'portrait' }} }};
                        html2pdf().from(informeHtml).set(opt).save();
                    }}
                </script>
                <style>
                    #descargar_pdf {{ width: 100%; padding: 0.25rem 0.75rem; border-radius: 0.5rem;
                                      border: 1px solid rgba(49, 51, 63, 0.2); background-color: #FFFFFF;
                                      color: #31333F; cursor: pointer; line-height: 2.5; }}
                    #descargar_pdf:hover {{ border-color: #FF4B4B; color: #FF4B4B; }}
                </style>
            """, height=50)
    else:
        st.info("Completa y ejecuta la configuración para ver los resultados.")

with tab_about:
    st.header("Acerca del Proyecto")
    st.markdown("""
    Esta aplicación fue desarrollada como una herramienta avanzada para la optimización logística, aplicando metaheurísticas para resolver problemas complejos de ruteo de vehículos.
    ### Tecnología Utilizada
    - **Framework:** Streamlit, Folium
    - **Algoritmo:** Optimización por Colonia de Hormigas (ACO) con Elitismo y Búsqueda Local (2-opt).
    - **Visualización:** Folium
    - **Lenguaje:** Python
    """)
