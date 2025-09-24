# streamlit_app.py
# Interfaz de usuario mejorada para el Optimizador CVRP.
# Incluye selección de depósito en mapa, UI simplificada y datos de ejemplo.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import ACO_CVRP_Solver
from streamlit_folium import st_folium # NUEVA LIBRERÍA
import folium # NUEVA LIBRERÍA
from io import StringIO # NUEVA LIBRERÍA

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Optimizador de Rutas (CVRP)",
    page_icon="🚚",
    layout="wide"
)

# --- Estado de la Sesión (MODIFICADO) ---
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'solver' not in st.session_state:
    st.session_state.solver = None
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
# NUEVO: Estado para las coordenadas del depósito
if 'depot_lat' not in st.session_state:
    st.session_state.depot_lat = 3.90089 # Valor por defecto (Buga, Colombia)
if 'depot_lon' not in st.session_state:
    st.session_state.depot_lon = -76.29783 # Valor por defecto

# --- Datos de Ejemplo (NUEVO) ---
@st.cache_data
def get_example_data():
    csv_data = """nombre,lat,lon,pasajeros
    Cliente 1,3.87103,-76.3117,8
    Cliente 2,3.87163,-76.3168,14
    Cliente 3,3.87227,-76.3201,3
    Cliente 4,3.86949,-76.3129,20
    Cliente 5,3.86733,-76.3148,15
    Cliente 6,3.86906,-76.3023,12
    Cliente 7,3.87413,-76.2952,16
    Cliente 8,3.87971,-76.2942,9
    Cliente 9,3.88373,-76.2941,18
    Cliente 10,3.88909,-76.2917,11
    """
    return pd.read_csv(StringIO(csv_data))

# --- Funciones de la Aplicación (MODIFICADO) ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file, delimiter=';', encoding='latin1')
        column_map = {'nombre': 'name', 'lat': 'lat', 'lon': 'lon', 'pasajeros': 'demand'}
        df.columns = [col.strip().lower() for col in df.columns]

        # Mapeo flexible de columnas
        final_map = {}
        for key, value in column_map.items():
            if key in df.columns:
                final_map[key] = value
        
        if 'lat' not in final_map.values() or 'lon' not in final_map.values() or 'demand' not in final_map.values():
             st.error("Error: El archivo debe contener columnas de latitud, longitud y demanda ('pasajeros').")
             return None

        df = df.rename(columns=final_map)
        df = df[list(final_map.values())]
        df[['lat', 'lon', 'demand']] = df[['lat', 'lon', 'demand']].astype(float)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

def get_plotly_chart(df_customers, depot_coord, solution_routes, solver):
    # (Esta función no necesita cambios significativos, se mantiene igual)
    if df_customers is None:
        return go.Figure()

    fig = go.Figure()

    if solution_routes and solver:
        all_coords = solver.cities_coords
        route_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
        for i, route in enumerate(solution_routes):
            route_coords = [all_coords[city_idx] for city_idx in route]
            route_lon = [coord[0] for coord in route_coords]
            route_lat = [coord[1] for coord in route_coords]
            route_detail = st.session_state.evaluation['route_details'][i]
            fig.add_trace(go.Scattermapbox(
                lon=route_lon, lat=route_lat, mode='lines',
                line=dict(color=route_colors[i % len(route_colors)], width=3),
                name=f"Ruta {i+1} ({route_detail['distance']:.1f} km, Carga: {route_detail['load']:.0f})",
                hoverinfo='name'
            ))

    fig.add_trace(go.Scattermapbox(
        lon=df_customers['lon'], lat=df_customers['lat'], mode='markers+text',
        marker=dict(color='blue', size=14),
        text=[f"<b>{i+1}</b>" for i in df_customers.index],
        textfont=dict(color='white', size=8), textposition='middle center',
        hovertext=df_customers.apply(lambda row: f"{row['name']}<br>Demanda: {row['demand']}", axis=1),
        hoverinfo='text', name='Clientes'
    ))

    fig.add_trace(go.Scattermapbox(
        lon=[depot_coord[0]], lat=[depot_coord[1]], mode='markers',
        marker=dict(color='#D62728', size=25, symbol='star'),
        hovertext=f"<b>Depósito</b><br>Lon: {depot_coord[0]:.5f}<br>Lat: {depot_coord[1]:.5f}",
        hoverinfo='text', name='Depósito'
    ))

    all_lons = df_customers['lon'].tolist() + [depot_coord[0]]
    all_lats = df_customers['lat'].tolist() + [depot_coord[1]]
    center_lon, center_lat = np.mean(all_lons), np.mean(all_lats)
    
    lon_range = np.abs(np.max(all_lons) - np.min(all_lons))
    lat_range = np.abs(np.max(all_lats) - np.min(all_lats))
    max_range = max(lon_range, lat_range)
    
    zoom_levels = {0.001: 15, 0.01: 13, 0.1: 11, 1: 9, 5: 7}
    zoom = 5
    for r, z in zoom_levels.items():
        if max_range < r:
            zoom = z
            break

    fig.update_layout(
        title='<b>Mejor Solución de Ruteo Encontrada</b>', showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="right", x=1.1, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=10, r=10, b=10, t=50), hovermode='closest',
        mapbox=dict(style="open-street-map", center=dict(lon=center_lon, lat=center_lat), zoom=zoom)
    )
    return fig

# --- Interfaz Principal (REDISEÑADA) ---
st.title("🚚 Optimizador de Rutas Vehiculares (CVRP)")
st.write("Esta herramienta utiliza un **Algoritmo de Colonia de Hormigas (ACO)** para encontrar las rutas más eficientes. Carga tus datos, selecciona el depósito en el mapa y configura los parámetros para comenzar.")

tab_config, tab_results, tab_about = st.tabs(["⚙️ 1. Configuración", "📊 2. Resultados", "👨‍💻 Acerca de"])

with tab_config:
    st.header("Parámetros de Entrada")
    
    col1, col2 = st.columns([0.6, 0.4]) # Dar más espacio a la configuración de datos
    
    with col1:
        st.subheader("1. Cargar Datos de Clientes")
        
        # Lógica de carga de archivo y datos de ejemplo
        uploaded_file = st.file_uploader("Sube tu archivo de clientes (delimitado por ';')", type="csv")
        if st.button("📍 Usar Datos de Ejemplo"):
            st.session_state.customer_data = get_example_data()
            st.success(f"Datos de ejemplo cargados: {len(st.session_state.customer_data)} ubicaciones.")
        
        if uploaded_file:
            st.session_state.customer_data = load_data(uploaded_file)
            if st.session_state.customer_data is not None:
                 st.success(f"Archivo cargado: {len(st.session_state.customer_data)} ubicaciones encontradas.")

        st.subheader("2. Seleccionar Ubicación del Depósito")
        st.info("Haz clic en el mapa para establecer el punto de partida de los vehículos.")
        
        # Mapa interactivo para seleccionar depósito
        map_center = [st.session_state.depot_lat, st.session_state.depot_lon]
        if st.session_state.customer_data is not None and not st.session_state.customer_data.empty:
            map_center = [st.session_state.customer_data['lat'].mean(), st.session_state.customer_data['lon'].mean()]

        m = folium.Map(location=map_center, zoom_start=13)
        folium.Marker(
            [st.session_state.depot_lat, st.session_state.depot_lon], 
            popup="Depósito Actual", 
            tooltip="Depósito",
            icon=folium.Icon(color="red", icon="truck", prefix='fa')
        ).add_to(m)

        if st.session_state.customer_data is not None:
             for idx, row in st.session_state.customer_data.iterrows():
                 folium.Marker([row['lat'], row['lon']], popup=row['name'], tooltip=row['name']).add_to(m)

        map_data = st_folium(m, width=700, height=500, key="depot_map")
        
        if map_data and map_data["last_clicked"]:
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            st.session_state.depot_lat = clicked_lat
            st.session_state.depot_lon = clicked_lon
            st.rerun() # Recargar para actualizar el marcador en el mapa

    with col2:
        st.subheader("3. Parámetros de la Flota")
        
        st.write(f"**Coordenadas del Depósito Seleccionado:**")
        st.code(f"Lat: {st.session_state.depot_lat:.5f}, Lon: {st.session_state.depot_lon:.5f}")

        n_vehicles = st.number_input("Número de Vehículos", min_value=1, value=10)
        vehicle_capacity = st.number_input("Capacidad por Vehículo", min_value=1, value=150)

        st.divider()

        st.subheader("4. Parámetros del Algoritmo")
        # Parámetros avanzados ocultos
        with st.expander("Ajustes Avanzados (ACO)"):
            n_ants = st.slider("Número de Hormigas", 5, 100, 30)
            n_iterations = st.slider("Número de Iteraciones", 10, 1000, 200)
            alpha = st.slider("Alpha (α)", 0.1, 5.0, 1.0, 0.1, help="Influencia de la feromona.")
            beta = st.slider("Beta (β)", 0.1, 5.0, 2.0, 0.1, help="Influencia de la distancia.")
            rho = st.slider("Rho (ρ)", 0.01, 1.0, 0.5, 0.01, help="Tasa de evaporación de feromona.")
            q_val = st.number_input("Q", value=100, help="Constante de depósito de feromona.")
            use_2_opt = st.toggle("Búsqueda Local (2-opt)", value=True, help="Mejora las rutas para evitar cruces.")
            elitism_weight = st.slider("Peso de Elitismo", 1.0, 10.0, 3.0, 0.5, help="Refuerza la mejor ruta encontrada.")

    st.divider()
    if st.button("🚀 Iniciar Optimización", type="primary", use_container_width=True):
        if st.session_state.customer_data is None or st.session_state.customer_data.empty:
            st.error("Por favor, carga datos de clientes o usa el ejemplo para comenzar.")
        else:
            with st.spinner("Optimizando rutas... Por favor espera."):
                progress_bar = st.progress(0, text="Iniciando optimización...")
                
                def progress_callback(iteration, total_iterations, best_cost):
                    progress = iteration / total_iterations
                    cost_str = f"{best_cost:,.2f}" if best_cost != float('inf') else "Calculando..."
                    progress_bar.progress(progress, text=f"Iteración {iteration}/{total_iterations} - Mejor costo: {cost_str} km")

                params_aco = {'alpha': alpha, 'beta': beta, 'rho': rho, 'Q': q_val}
                depot_coord = (st.session_state.depot_lon, st.session_state.depot_lat)
                
                solver = ACO_CVRP_Solver(
                    depot_coord=depot_coord,
                    customer_coords=st.session_state.customer_data[['lon', 'lat']].values.tolist(),
                    customer_demands=st.session_state.customer_data['demand'].values.tolist(),
                    n_vehicles=n_vehicles,
                    vehicle_capacity=vehicle_capacity,
                    params=params_aco,
                    use_2_opt=use_2_opt,
                    elitism_weight=elitism_weight
                )
                best_routes, best_cost = solver.solve(n_ants, n_iterations, progress_callback)
                
                if not best_routes:
                    st.error("No se encontró una solución válida. Prueba ajustar los parámetros (más vehículos, mayor capacidad o más iteraciones).")
                    st.session_state.solution = None
                else:
                    evaluation = solver.evaluate_solution({'routes': best_routes})
                    st.session_state.solution = {'routes': best_routes, 'cost': evaluation['total_distance_km']}
                    st.session_state.evaluation = evaluation
                    st.session_state.solver = solver
                    st.success("¡Optimización completada! Ve a la pestaña 'Resultados' para ver la solución.")
                
                progress_bar.empty()

with tab_results:
    st.header("Visualización de Resultados")
    
    if st.session_state.solution:
        eval_data = st.session_state.evaluation
        st.subheader("📈 Métricas Clave de la Solución")
        cols = st.columns(4)
        cols[0].metric("Costo Total", f"{st.session_state.solution['cost']:,.2f} km")
        cols[1].metric("Vehículos Usados", f"{eval_data['num_vehicles_used']} / {st.session_state.solver.n_vehicles}")
        cols[2].metric("Clientes Visitados", f"{eval_data['customers_visited_count']} / {st.session_state.solver.n_customers}")
        cols[3].metric("Uso Promedio Capacidad", f"{eval_data['avg_vehicle_utilization_percent']:.1f}%")

        st.subheader("🗺️ Visualización de Rutas")
        plotly_chart = get_plotly_chart(
            st.session_state.customer_data, 
            st.session_state.solver.depot_coord,
            st.session_state.solution['routes'],
            st.session_state.solver
        )
        st.plotly_chart(plotly_chart, use_container_width=True)
        
        st.subheader("📋 Detalles por Ruta")
        for i, route_detail in enumerate(eval_data['route_details']):
            with st.expander(f"**Ruta {i+1}** | Distancia: {route_detail['distance']:.2f} km | Carga: {route_detail['load']:.0f} ({route_detail['utilization']:.1f}%)"):
                route_customer_indices = route_detail['sequence']
                if not route_customer_indices: continue
                
                route_df = st.session_state.customer_data.iloc[[idx - 1 for idx in route_customer_indices]].copy()
                
                depot_lon, depot_lat = st.session_state.solver.depot_coord
                depot_start_df = pd.DataFrame([{'name': 'Depósito (Salida)', 'demand': '-', 'lat': depot_lat, 'lon': depot_lon}])
                depot_end_df = pd.DataFrame([{'name': 'Depósito (Llegada)', 'demand': '-', 'lat': depot_lat, 'lon': depot_lon}])
                
                full_route_display_df = pd.concat([depot_start_df, route_df.rename(columns={'name': 'name', 'demand': 'demand'}), depot_end_df], ignore_index=True)
                full_route_display_df.insert(0, "Orden", range(1, len(full_route_display_df) + 1))
                
                st.dataframe(full_route_display_df[['Orden', 'name', 'demand', 'lat', 'lon']])
    else:
        st.info("Completa y ejecuta la configuración para ver los resultados aquí.")

with tab_about:
    st.header("Acerca del Proyecto y del Autor")
    st.image("https://i.imgur.com/8bf3k8u.png") # Puedes cambiar esta imagen
    st.markdown("""
    Esta aplicación fue desarrollada como una herramienta avanzada para la optimización logística, aplicando metaheurísticas para resolver problemas complejos de ruteo de vehículos.
    ### Autor
    - **Nombre:** (Tu nombre aquí)
    - **Contacto:** (Tu email o red de contacto)
    - **LinkedIn:** [Tu Perfil](https://www.linkedin.com/in/tu-usuario/)
    ### Tecnología Utilizada
    - **Framework:** Streamlit, Folium
    - **Algoritmo:** Optimización por Colonia de Hormigas (ACO) con Elitismo y Búsqueda Local (2-opt).
    - **Visualización:** Plotly
    - **Lenguaje:** Python
    """)
