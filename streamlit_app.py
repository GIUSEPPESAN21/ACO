# streamlit_app.py
# Interfaz de usuario principal para el Optimizador CVRP.

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from solver import ACO_CVRP_Solver

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Optimizador de Rutas (CVRP)",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funciones de la Aplicación ---

@st.cache_data
def load_data(file):
    """Carga y procesa el archivo CSV de clientes."""
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        # Detección flexible de columnas
        column_map = {
            'lat': ['latitud_aproximada', 'latitud', 'lat'],
            'lon': ['longitud_aproximada', 'longitud', 'lon', 'lng'],
            'demand': ['ctd paquetes', 'demanda', 'demand', 'volumen']
        }
        
        rename_dict = {}
        for generic_name, possible_names in column_map.items():
            found = False
            for name in possible_names:
                if name in df.columns:
                    rename_dict[name] = generic_name
                    found = True
                    break
            if not found:
                st.error(f"Error: No se encontró una columna para '{generic_name}'. Opciones válidas: {possible_names}")
                return None
        
        df = df.rename(columns=rename_dict)
        df = df[['lat', 'lon', 'demand']]
        df = df.astype(float)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

def get_pydeck_chart(df_customers, depot_coord, solution_routes, solver_coords):
    """Crea y devuelve un gráfico de Pydeck con las rutas y puntos."""
    if df_customers is None:
        return None

    # Colores para las rutas
    route_colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], 
        [255, 0, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0]
    ]

    # Capa de Clientes
    customer_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_customers,
        get_position=["lon", "lat"],
        get_color=[0, 0, 200, 160],
        get_radius=100,
        pickable=True,
        tooltip={"text": "Cliente\nLat: {lat}, Lon: {lon}\nDemanda: {demand}"}
    )
    
    # Capa de Depósito
    depot_df = pd.DataFrame([{'lat': depot_coord[1], 'lon': depot_coord[0]}])
    depot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=depot_df,
        get_position=["lon", "lat"],
        get_color=[200, 0, 0, 255],
        get_radius=250,
        pickable=True,
        tooltip={"text": "Depósito Central"}
    )
    
    layers = [customer_layer, depot_layer]

    # Capas de Rutas
    if solution_routes:
        for i, route in enumerate(solution_routes):
            path_data = []
            for city_idx in route:
                lon, lat = solver_coords[city_idx]
                path_data.append([lon, lat])
            
            route_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": path_data, "name": f"Ruta {i+1}", "color": route_colors[i % len(route_colors)]}],
                get_path="path",
                get_width=5,
                width_min_pixels=3,
                get_color="color",
                pickable=True,
                tooltip={"text": "{name}"}
            )
            layers.append(route_layer)
    
    view_state = pdk.ViewState(
        latitude=depot_coord[1],
        longitude=depot_coord[0],
        zoom=11,
        pitch=45,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={"style": {"backgroundColor": "steelblue", "color": "white"}}
    )
    
# --- Interfaz de Usuario ---

st.title("🚚 Optimizador de Rutas Vehiculares (CVRP)")
st.write("Esta herramienta utiliza un **Algoritmo de Colonia de Hormigas (ACO)** para encontrar las rutas más eficientes para una flota de vehículos, minimizando la distancia total recorrida.")

# --- Barra Lateral de Configuración ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("1. Cargar Datos de Clientes")
    uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")
    use_example = st.checkbox("Usar datos de ejemplo", value=True)

    if use_example:
        uploaded_file = "data/sample_data.csv"

    df_customers = load_data(uploaded_file)
    
    if df_customers is not None:
        st.success(f"Cargados {len(df_customers)} clientes.")
    else:
        st.warning("Por favor, sube un archivo CSV o usa los datos de ejemplo.")
        st.stop()
    
    st.subheader("2. Parámetros del Problema")
    depot_lat = st.number_input("Latitud Depósito", value=3.90089, format="%.5f")
    depot_lon = st.number_input("Longitud Depósito", value=-76.29783, format="%.5f")
    n_vehicles = st.number_input("Número de Vehículos", min_value=1, value=10)
    vehicle_capacity = st.number_input("Capacidad por Vehículo", min_value=1, value=150)

    st.subheader("3. Parámetros del Algoritmo (ACO)")
    with st.expander("Ajustes Avanzados"):
        n_ants = st.slider("Número de Hormigas", 5, 100, 30)
        n_iterations = st.slider("Número de Iteraciones", 10, 1000, 200)
        alpha = st.slider("Alpha (α)", 0.1, 5.0, 1.0, 0.1, help="Influencia de la feromona.")
        beta = st.slider("Beta (β)", 0.1, 5.0, 2.0, 0.1, help="Influencia de la visibilidad (distancia).")
        rho = st.slider("Rho (ρ)", 0.01, 1.0, 0.1, 0.01, help="Tasa de evaporación de la feromona.")
        q_val = st.number_input("Q", value=100, help="Constante de depósito de feromona.")

    start_button = st.button("🚀 Iniciar Optimización", type="primary", use_container_width=True)

# --- Área Principal de Resultados ---

# Inicializar el estado de la sesión para guardar los resultados
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'solver' not in st.session_state:
    st.session_state.solver = None

# Lógica de Optimización
if start_button:
    with st.spinner("Optimizando rutas... El algoritmo está trabajando."):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        def progress_callback(iteration, total_iterations, best_cost):
            """Función para actualizar la UI durante la optimización."""
            progress = iteration / total_iterations
            progress_bar.progress(progress)
            cost_str = f"{best_cost:.2f}" if best_cost != float('inf') else "N/A"
            progress_text.text(f"Iteración {iteration}/{total_iterations} - Mejor costo actual: {cost_str} km")

        params_aco = {'alpha': alpha, 'beta': beta, 'rho': rho, 'Q': q_val}
        depot_coord = (depot_lon, depot_lat)
        
        solver = ACO_CVRP_Solver(
            depot_coord=depot_coord,
            customer_coords=df_customers[['lon', 'lat']].values.tolist(),
            customer_demands=df_customers['demand'].values.tolist(),
            n_vehicles=n_vehicles,
            vehicle_capacity=vehicle_capacity,
            params=params_aco
        )

        best_routes, best_cost = solver.solve(
            n_ants=n_ants, 
            n_iterations=n_iterations,
            progress_callback=progress_callback
        )
        
        progress_text.text("Optimización completada. Generando resultados...")

        if not best_routes:
            st.error("No se pudo encontrar una solución válida. Intenta ajustar los parámetros (ej. más vehículos, mayor capacidad o más iteraciones).")
            st.session_state.solution = None
            st.session_state.evaluation = None
            st.session_state.solver = None
        else:
            evaluation = solver.evaluate_solution({'routes': best_routes, 'cost': best_cost})
            st.session_state.solution = {'routes': best_routes, 'cost': best_cost}
            st.session_state.evaluation = evaluation
            st.session_state.solver = solver # Guardar el objeto solver completo
        
        progress_bar.empty()
        progress_text.empty()

# Mostrar Resultados si existen
if st.session_state.solution:
    st.header("📊 Resultados de la Optimización")
    
    eval_data = st.session_state.evaluation
    
    # 1. Métricas Clave
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Costo Total", f"{eval_data['total_distance_km']:.2f} km")
    col2.metric("Vehículos Usados", f"{eval_data['num_vehicles_used']} / {n_vehicles}")
    col3.metric("Clientes Visitados", f"{eval_data['customers_visited_count']} / {len(df_customers)}")
    col4.metric("Uso Promedio Capacidad", f"{eval_data['avg_vehicle_utilization_percent']:.1f}%")

    # 2. Mapa
    st.subheader("🗺️ Visualización de Rutas en el Mapa")
    pydeck_chart = get_pydeck_chart(
        df_customers, 
        (depot_lon, depot_lat),
        st.session_state.solution['routes'],
        st.session_state.solver.cities_coords
    )
    if pydeck_chart:
        st.pydeck_chart(pydeck_chart)
    else:
        st.warning("No se pudo generar el mapa.")

    # 3. Detalles por Ruta
    st.subheader("📋 Detalles por Ruta")
    for i, route_detail in enumerate(eval_data['route_details']):
        color_index = i % 10 # Para que coincida con los colores del mapa
        with st.expander(f"**Ruta {i+1}** (Distancia: {route_detail['distance']:.2f} km, Carga: {route_detail['load']:.0f} / {vehicle_capacity})"):
            st.write(f"**Paradas:** {route_detail['stops']}")
            st.write(f"**Utilización de capacidad:** {route_detail['utilization']:.1f}%")
            
            # Mapear índices a coordenadas para mostrar en tabla
            route_customer_indices = route_detail['sequence']
            route_df = df_customers.iloc[[idx - 1 for idx in route_customer_indices]].copy()
            route_df.insert(0, "Orden", range(1, len(route_df) + 1))
            st.dataframe(route_df)

else:
    st.info("Configura los parámetros en la barra lateral y haz clic en 'Iniciar Optimización' para ver los resultados.")
    st.subheader("🗺️ Vista Previa del Mapa de Clientes")
    pydeck_chart = get_pydeck_chart(df_customers, (depot_lon, depot_lat), None, None)
    if pydeck_chart:
        st.pydeck_chart(pydeck_chart)
