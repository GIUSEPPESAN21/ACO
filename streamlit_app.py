# streamlit_app.py
# Versión final corregida para manejar KeyError de forma robusta.

import streamlit as st
import pandas as pd
import numpy as np
from solver import ACO_CVRP_Solver
from streamlit_folium import st_folium
import folium
from io import StringIO

# --- Configuración de la Página ---
st.set_page_config(page_title="Optimizador de Rutas (CVRP-ACO)", page_icon="🚚", layout="wide")

# --- Estado de la Sesión ---
if 'solution' not in st.session_state: st.session_state.solution = None
if 'evaluation' not in st.session_state: st.session_state.evaluation = None
if 'solver' not in st.session_state: st.session_state.solver = None
if 'customer_data' not in st.session_state: st.session_state.customer_data = None
if 'depot_lat' not in st.session_state: st.session_state.depot_lat = 3.90089
if 'depot_lon' not in st.session_state: st.session_state.depot_lon = -76.29783

# --- Funciones de Utilidad ---
@st.cache_data
def get_example_data():
    csv_data = """nombre,lat,lon,pasajeros
    Cliente 1,3.87103,-76.3117,8
    Cliente 2,3.87163,-76.3168,14
    Cliente 3,3.87227,-76.3201,3
    Cliente 4,3.86949,-76.3129,20
    Cliente 5,3.86733,-76.3148,15
    Cliente 6,3.86906,-76.3023,12
    """
    return pd.read_csv(StringIO(csv_data))

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None: return None
    try:
        df = pd.read_csv(uploaded_file, delimiter=';', encoding='latin1')
        df.columns = [col.strip().lower() for col in df.columns]
        column_map = {'nombre': 'name', 'lat': 'lat', 'lon': 'lon', 'pasajeros': 'demand'}
        final_map = {k: v for k, v in column_map.items() if k in df.columns}
        if not all(k in final_map.values() for k in ['lat', 'lon', 'demand']):
            st.error("Error: Archivo debe contener columnas de latitud, longitud y demanda ('pasajeros').")
            return None
        df = df.rename(columns=final_map)[list(final_map.values())]
        df[['lat', 'lon', 'demand']] = df[['lat', 'lon', 'demand']].astype(float)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

def format_duration(hours):
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h} h {m} min"

def get_folium_results_map(solution_routes, solver, customer_data):
    depot_coord = solver.depot_coord
    map_center = [customer_data['lat'].mean(), customer_data['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=14, tiles="cartodbpositron")
    route_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    for i, route in enumerate(solution_routes):
        route_coords = [(solver.cities_coords[c][1], solver.cities_coords[c][0]) for c in route]
        detail = st.session_state.evaluation['route_details'][i]
        # CORREGIDO: Usar .get() para evitar errores si las claves no existen
        cost = detail.get('cost', 0)
        duration_h = detail.get('duration_hours', 0)
        tooltip = f"Ruta {i+1} | Costo: ${cost:,.0f} | Duración: {format_duration(duration_h)}"
        folium.PolyLine(locations=route_coords, color=route_colors[i % len(route_colors)], weight=4, opacity=0.8, tooltip=tooltip).add_to(m)

    for _, row in customer_data.iterrows():
        folium.Marker(location=[row['lat'], row['lon']], popup=f"<b>{row['name']}</b><br>Demanda: {row['demand']}", tooltip=row['name'], icon=folium.Icon(color="blue", icon="user", prefix='fa')).add_to(m)
    
    folium.Marker(location=[depot_coord[1], depot_coord[0]], popup="Depósito", tooltip="Depósito", icon=folium.Icon(color="red", icon="truck", prefix='fa')).add_to(m)
    return m

# --- Interfaz Principal ---
st.title("🚚 Optimizador de Rutas Vehiculares (CVRP)")
st.write("Una herramienta inteligente para encontrar las rutas más eficientes y económicas para tu flota.")

tab_config, tab_results, tab_about = st.tabs(["⚙️ 1. Configuración", "📊 2. Resultados", "👨‍💻 Acerca de"])

with tab_config:
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.subheader("1. Cargar Datos de Clientes")
        uploaded_file = st.file_uploader("Sube tu archivo (delimitado por ';')", type="csv", label_visibility="collapsed")
        if st.button("📍 Usar Datos de Ejemplo"):
            st.session_state.customer_data = get_example_data()
            st.success(f"Datos de ejemplo cargados: {len(st.session_state.customer_data)} ubicaciones.")
        if uploaded_file:
            st.session_state.customer_data = load_data(uploaded_file)
            if st.session_state.customer_data is not None:
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
        
        map_data = st_folium(m_config, width=700, height=400, key="depot_map")
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
                    st.session_state.evaluation = None # Limpiar evaluación en caso de error
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
        # CORREGIDO: Usar .get() con un valor por defecto (0) para evitar KeyError
        m1.metric("Costo Total de Rutas", f"${eval_data.get('total_cost', 0):,.0f}")
        m2.metric("Distancia Total", f"{eval_data.get('total_distance_km', 0):,.1f} km")
        m3.metric("Duración Total (Horas-Hombre)", f"{format_duration(eval_data.get('total_duration_hours', 0))}")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Vehículos Usados", f"{eval_data.get('num_vehicles_used', 0)} / {getattr(st.session_state.solver, 'n_vehicles', 'N/A')}")
        m5.metric("Clientes Visitados", f"{eval_data.get('customers_visited_count', 0)} / {getattr(st.session_state.solver, 'n_customers', 'N/A')}")
        m6.metric("Uso Promedio de Capacidad", f"{eval_data.get('avg_vehicle_utilization_percent', 0):.1f}%")

        st.subheader("🗺️ Visualización de Rutas Optimizadas")
        results_map = get_folium_results_map(st.session_state.solution['routes'], st.session_state.solver, st.session_state.customer_data)
        st_folium(results_map, width=1000, height=500, returned_objects=[])
        
        st.subheader("📋 Detalles por Ruta")
        for i, detail in enumerate(eval_data.get('route_details', [])):
            # CORREGIDO: Usar .get() también aquí
            cost = detail.get('cost', 0)
            duration_h = detail.get('duration_hours', 0)
            load = detail.get('load', 0)
            utilization = detail.get('utilization', 0)
            
            expander_title = (f"**Ruta {i+1}** | "
                              f"Costo: `${cost:,.0f}` | "
                              f"Duración: `{format_duration(duration_h)}` | "
                              f"Carga: `{load:.0f} ({utilization:.1f}%)`")
            with st.expander(expander_title):
                route_indices = detail.get('sequence', [])
                if not route_indices: continue
                route_df = st.session_state.customer_data.iloc[[idx - 1 for idx in route_indices]]
                st.dataframe(route_df[['name', 'demand', 'lat', 'lon']])
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
