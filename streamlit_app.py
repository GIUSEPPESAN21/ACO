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
    layout="wide"
)

# --- Estado de la Sesión ---
# Inicializar el estado para guardar los resultados entre ejecuciones
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'solver' not in st.session_state:
    st.session_state.solver = None
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None


# --- Funciones de la Aplicación ---
@st.cache_data
def load_data(uploaded_file):
    """
    Carga y procesa el archivo CSV de clientes.
    Versión robusta que normaliza columnas y maneja la codificación 'latin1'.
    """
    if uploaded_file is None:
        return None
    try:
        # CORRECCIÓN: Se añade encoding='latin1' para leer tildes y caracteres especiales.
        df = pd.read_csv(uploaded_file, delimiter=';', encoding='latin1')

        # Mapeo de columnas requeridas (origen -> destino)
        column_map = {
            'nombre': 'name',
            'lat': 'lat',
            'lon': 'lon',
            'pasajeros': 'demand'
        }

        # Normalizar las columnas del DataFrame (minúsculas, sin espacios)
        df.columns = [col.strip().lower() for col in df.columns]

        # Verificar si todas las columnas de origen requeridas existen
        for required_col in column_map.keys():
            if required_col not in df.columns:
                st.error(f"Error: La columna requerida '{required_col}' no se encontró en el archivo CSV. Columnas encontradas: {df.columns.tolist()}")
                return None
        
        # Renombrar y seleccionar las columnas necesarias
        df = df.rename(columns=column_map)
        df = df[list(column_map.values())] # Selecciona ['name', 'lat', 'lon', 'demand']
        
        # Convertir tipos de datos
        df[['lat', 'lon', 'demand']] = df[['lat', 'lon', 'demand']].astype(float)
        return df

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None


def get_pydeck_chart(df_customers, depot_coord, solution_routes, solver_coords):
    """Crea y devuelve un gráfico de Pydeck con las rutas y puntos."""
    if df_customers is None:
        return None

    route_colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], 
        [255, 0, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
        [128, 0, 128], [0, 128, 128], [255, 165, 0], [255, 20, 147], [60, 179, 113]
    ]

    customer_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_customers,
        get_position=["lon", "lat"],
        get_fill_color=[0, 0, 200, 160],
        get_radius=2000,
        pickable=True,
        auto_highlight=True,
        tooltip={"html": "<b>Cliente:</b> {name}<br/><b>Demanda:</b> {demand}"}
    )
    
    depot_df = pd.DataFrame([{'lat': depot_coord[1], 'lon': depot_coord[0]}])
    depot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=depot_df,
        get_position=["lon", "lat"],
        get_fill_color=[255, 0, 0, 255],
        get_radius=3000,
        pickable=True,
        tooltip={"text": "Depósito Central"}
    )
    
    layers = [customer_layer, depot_layer]

    if solution_routes:
        for i, route in enumerate(solution_routes):
            path_data = [[solver_coords[city_idx][0], solver_coords[city_idx][1]] for city_idx in route]
            
            route_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": path_data, "name": f"Ruta {i+1}", "color": route_colors[i % len(route_colors)]}],
                get_path="path",
                get_width=5,
                width_min_pixels=3,
                get_color="color",
                pickable=True,
                auto_highlight=True,
                tooltip={"html": "<b>{name}</b>"}
            )
            layers.append(route_layer)
    
    view_state = pdk.ViewState(
        latitude=df_customers['lat'].mean(),
        longitude=df_customers['lon'].mean(),
        zoom=6,
        pitch=45,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={"style": {"backgroundColor": "steelblue", "color": "white"}}
    )

# --- Interfaz Principal ---

st.title("🚚 Optimizador de Rutas Vehiculares (CVRP)")
st.write("Esta herramienta utiliza un **Algoritmo de Colonia de Hormigas (ACO)** para encontrar las rutas más eficientes, minimizando la distancia total recorrida.")

# --- Definición de Pestañas ---
tab_config, tab_results, tab_about = st.tabs(["⚙️ Configuración y Ejecución", "📊 Resultados", "👨‍💻 Acerca de"])

# --- Pestaña 1: Configuración ---
with tab_config:
    st.header("Parámetros de Entrada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Datos del Problema")
        uploaded_file = st.file_uploader("Sube tu archivo de clientes (delimitado por ';')", type="csv")

        depot_lat = st.number_input("Latitud Depósito", value=4.685, format="%.5f")
        depot_lon = st.number_input("Longitud Depósito", value=-74.140, format="%.5f")
        n_vehicles = st.number_input("Número de Vehículos", min_value=1, value=10)
        vehicle_capacity = st.number_input("Capacidad por Vehículo", min_value=1, value=150)

    with col2:
        st.subheader("2. Parámetros del Algoritmo (ACO)")
        n_ants = st.slider("Número de Hormigas", 5, 100, 30)
        n_iterations = st.slider("Número de Iteraciones", 10, 1000, 200)
        alpha = st.slider("Alpha (α)", 0.1, 5.0, 1.0, 0.1, help="Influencia de la feromona.")
        beta = st.slider("Beta (β)", 0.1, 5.0, 2.0, 0.1, help="Influencia de la visibilidad (distancia).")
        rho = st.slider("Rho (ρ)", 0.01, 1.0, 0.5, 0.01, help="Tasa de evaporación de la feromona.")
        q_val = st.number_input("Q", value=100, help="Constante de depósito de feromona.")

    st.divider()
    start_button = st.button("🚀 Iniciar Optimización", type="primary", use_container_width=True)
    
    # Lógica de carga de datos
    if uploaded_file:
        st.session_state.customer_data = load_data(uploaded_file)
        if st.session_state.customer_data is not None:
             st.success(f"Archivo cargado correctamente: {len(st.session_state.customer_data)} clientes encontrados.")

    # Lógica de Optimización al presionar el botón
    if start_button:
        if st.session_state.customer_data is None:
            st.error("Por favor, carga un archivo de datos de clientes válido antes de iniciar la optimización.")
        else:
            with st.spinner("Optimizando rutas... El algoritmo está trabajando, por favor espera."):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                def progress_callback(iteration, total_iterations, best_cost):
                    progress = iteration / total_iterations
                    progress_bar.progress(progress)
                    cost_str = f"{best_cost:,.2f}" if best_cost != float('inf') else "N/A"
                    progress_text.text(f"Iteración {iteration}/{total_iterations} - Mejor costo: {cost_str} km")

                params_aco = {'alpha': alpha, 'beta': beta, 'rho': rho, 'Q': q_val}
                depot_coord = (depot_lon, depot_lat)
                
                solver = ACO_CVRP_Solver(
                    depot_coord=depot_coord,
                    customer_coords=st.session_state.customer_data[['lon', 'lat']].values.tolist(),
                    customer_demands=st.session_state.customer_data['demand'].values.tolist(),
                    n_vehicles=n_vehicles,
                    vehicle_capacity=vehicle_capacity,
                    params=params_aco
                )

                best_routes, best_cost = solver.solve(n_ants, n_iterations, progress_callback)
                
                if not best_routes:
                    st.error("No se encontró una solución válida. Prueba ajustar los parámetros (ej. más vehículos, mayor capacidad o más iteraciones).")
                    st.session_state.solution = None
                else:
                    evaluation = solver.evaluate_solution({'routes': best_routes, 'cost': best_cost})
                    st.session_state.solution = {'routes': best_routes, 'cost': best_cost}
                    st.session_state.evaluation = evaluation
                    st.session_state.solver = solver
                    st.success("¡Optimización completada! Ve a la pestaña 'Resultados' para ver la solución.")
                
                progress_bar.empty()
                progress_text.empty()

# --- Pestaña 2: Resultados ---
with tab_results:
    st.header("Visualización de Resultados")
    
    if st.session_state.solution:
        eval_data = st.session_state.evaluation
        
        st.subheader("📈 Métricas Clave de la Solución")
        cols = st.columns(4)
        cols[0].metric("Costo Total", f"{eval_data['total_distance_km']:,.2f} km")
        cols[1].metric("Vehículos Usados", f"{eval_data['num_vehicles_used']} / {st.session_state.solver.n_vehicles}")
        cols[2].metric("Clientes Visitados", f"{eval_data['customers_visited_count']} / {st.session_state.solver.n_customers}")
        cols[3].metric("Uso Promedio Capacidad", f"{eval_data['avg_vehicle_utilization_percent']:.1f}%")

        st.subheader("🗺️ Visualización de Rutas en el Mapa")
        pydeck_chart = get_pydeck_chart(
            st.session_state.customer_data, 
            st.session_state.solver.depot_coord,
            st.session_state.solution['routes'],
            st.session_state.solver.cities_coords
        )
        if pydeck_chart:
            st.pydeck_chart(pydeck_chart)
        
        st.subheader("📋 Detalles por Ruta")
        for i, route_detail in enumerate(eval_data['route_details']):
            with st.expander(f"**Ruta {i+1}** | Distancia: {route_detail['distance']:.2f} km | Carga: {route_detail['load']:.0f} ({route_detail['utilization']:.1f}%)"):
                route_customer_indices = route_detail['sequence']
                route_df = st.session_state.customer_data.iloc[[idx - 1 for idx in route_customer_indices]].copy()
                route_df.insert(0, "Orden", range(1, len(route_df) + 1))
                st.dataframe(route_df[['Orden', 'name', 'demand', 'lat', 'lon']])

    else:
        st.info("Completa la configuración en la pestaña 'Configuración y Ejecución' y haz clic en 'Iniciar Optimización' para ver los resultados aquí.")
        if st.session_state.customer_data is not None:
             st.subheader("🗺️ Vista Previa de Ubicaciones de Clientes")
             # Se definen depot_lon y depot_lat aquí por si el usuario no ha tocado los inputs
             depot_lon_preview = -74.140 
             depot_lat_preview = 4.685
             pydeck_chart = get_pydeck_chart(st.session_state.customer_data, (depot_lon_preview, depot_lat_preview), None, None)
             st.pydeck_chart(pydeck_chart)


# --- Pestaña 3: Acerca de ---
with tab_about:
    st.header("Acerca del Proyecto y del Autor")
    st.image("https://i.imgur.com/8bf3k8u.png")
    st.markdown("""
    Esta aplicación fue desarrollada como una herramienta avanzada para la optimización logística, aplicando metaheurísticas para resolver problemas complejos de ruteo de vehículos.
    
    ### Autor
    - **Nombre:** (Aquí va tu nombre)
    - **Contacto:** (Tu email o red de contacto)
    - **LinkedIn:** [Tu Perfil](https://www.linkedin.com/in/tu-usuario/)
    
    ### Tecnología Utilizada
    - **Framework:** Streamlit
    - **Algoritmo:** Optimización por Colonia de Hormigas (ACO)
    - **Visualización:** Pydeck (deck.gl)
    - **Lenguaje:** Python
    
    *El código de esta aplicación ha sido analizado y potenciado con la asistencia de IA para mejorar su estructura, eficiencia y experiencia de usuario.*
    """)
