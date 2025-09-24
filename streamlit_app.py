# streamlit_app.py
# Interfaz de usuario principal para el Optimizador CVRP.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from solver import ACO_CVRP_Solver

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Optimizador de Rutas (CVRP)",
    page_icon="🚚",
    layout="wide"
)

# --- Estado de la Sesión ---
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
    """
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file, delimiter=';', encoding='latin1')
        column_map = {'nombre': 'name', 'lat': 'lat', 'lon': 'lon', 'pasajeros': 'demand'}
        df.columns = [col.strip().lower() for col in df.columns]

        for required_col in column_map.keys():
            if required_col not in df.columns:
                st.error(f"Error: La columna requerida '{required_col}' no se encontró en el archivo CSV.")
                return None
        
        df = df.rename(columns=column_map)
        df = df[list(column_map.values())]
        df[['lat', 'lon', 'demand']] = df[['lat', 'lon', 'demand']].astype(float)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None


def get_plotly_chart(df_customers, depot_coord, solution_routes, solver):
    """
    Crea y devuelve un gráfico de Plotly con las rutas y puntos,
    similar a la versión original.
    """
    if df_customers is None:
        return go.Figure()

    fig = go.Figure()

    # 1. Añadir clientes
    fig.add_trace(go.Scatter(
        x=df_customers['lon'],
        y=df_customers['lat'],
        mode='markers',
        marker=dict(color='blue', size=8),
        name='Clientes',
        text=df_customers.apply(lambda row: f"{row['name']}<br>Demanda: {row['demand']}", axis=1),
        hoverinfo='text'
    ))

    # 2. Añadir depósito
    fig.add_trace(go.Scatter(
        x=[depot_coord[0]],
        y=[depot_coord[1]],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star'),
        name='Depósito'
    ))

    # 3. Añadir rutas si existen
    if solution_routes and solver:
        all_coords = solver.cities_coords
        route_colors = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
            '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]
        
        for i, route in enumerate(solution_routes):
            route_coords = [all_coords[city_idx] for city_idx in route]
            route_x = [coord[0] for coord in route_coords]
            route_y = [coord[1] for coord in route_coords]
            
            # Obtener detalles de la ruta para la leyenda
            route_detail = st.session_state.evaluation['route_details'][i]
            
            fig.add_trace(go.Scatter(
                x=route_x,
                y=route_y,
                mode='lines+markers',
                line=dict(color=route_colors[i % len(route_colors)], width=2),
                marker=dict(size=5),
                name=f"Ruta {i+1} ({route_detail['distance']:.1f} km, Carga: {route_detail['load']:.0f})"
            ))

    fig.update_layout(
        title='<b>Mejor Solución de Ruteo Encontrada</b>',
        xaxis_title='Longitud',
        yaxis_title='Latitud',
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="right", x=1.2),
        margin=dict(l=60, r=200, b=50, t=50),
        hovermode='closest',
        paper_bgcolor='#f9fafb',
        plot_bgcolor='#ffffff'
    )
    return fig

# --- Interfaz Principal ---
st.title("🚚 Optimizador de Rutas Vehiculares (CVRP)")
st.write("Esta herramienta utiliza un **Algoritmo de Colonia de Hormigas (ACO)** potenciado para encontrar las rutas más eficientes.")

tab_config, tab_results, tab_about = st.tabs(["⚙️ Configuración y Ejecución", "📊 Resultados", "👨‍💻 Acerca de"])

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
        
        st.subheader("3. Mejoras de Potencia")
        use_2_opt = st.toggle("Búsqueda Local (2-opt)", value=True, help="Mejora las rutas para evitar cruces.")
        elitism_weight = st.slider("Peso de Elitismo", 1.0, 10.0, 3.0, 0.5, help="Refuerza la mejor ruta encontrada.")

    st.divider()
    start_button = st.button("🚀 Iniciar Optimización", type="primary", use_container_width=True)
    
    if uploaded_file:
        st.session_state.customer_data = load_data(uploaded_file)
        if st.session_state.customer_data is not None:
             st.success(f"Archivo cargado correctamente: {len(st.session_state.customer_data)} clientes encontrados.")

    if start_button:
        if st.session_state.customer_data is None:
            st.error("Por favor, carga un archivo de datos de clientes válido.")
        else:
            with st.spinner("Optimizando rutas... Por favor espera."):
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
                    params=params_aco,
                    use_2_opt=use_2_opt,
                    elitism_weight=elitism_weight
                )
                best_routes, best_cost = solver.solve(n_ants, n_iterations, progress_callback)
                
                if not best_routes:
                    st.error("No se encontró una solución válida. Prueba ajustar los parámetros.")
                    st.session_state.solution = None
                else:
                    final_cost = sum(solver.evaluate_solution({'routes': [route]})['total_distance_km'] for route in best_routes)
                    evaluation = solver.evaluate_solution({'routes': best_routes})
                    st.session_state.solution = {'routes': best_routes, 'cost': final_cost}
                    st.session_state.evaluation = evaluation
                    st.session_state.solver = solver
                    st.success("¡Optimización completada! Ve a la pestaña 'Resultados'.")
                
                progress_bar.empty()
                progress_text.empty()

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
                route_df = st.session_state.customer_data.iloc[[idx - 1 for idx in route_customer_indices]].copy()
                route_df.insert(0, "Orden", range(1, len(route_df) + 1))
                st.dataframe(route_df[['Orden', 'name', 'demand', 'lat', 'lon']])
    else:
        st.info("Completa y ejecuta la configuración para ver los resultados.")
        if st.session_state.customer_data is not None:
             st.subheader("🗺️ Vista Previa de Ubicaciones")
             depot_coord = (st.session_state.get('depot_lon', -74.140), st.session_state.get('depot_lat', 4.685))
             plotly_chart = get_plotly_chart(st.session_state.customer_data, depot_coord, None, None)
             st.plotly_chart(plotly_chart, use_container_width=True)

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
    - **Algoritmo:** Optimización por Colonia de Hormigas (ACO) con Elitismo y Búsqueda Local (2-opt).
    - **Visualización:** Plotly
    - **Lenguaje:** Python
    """)

