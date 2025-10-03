# streamlit_app.py
import streamlit as st
import pandas as pd
from solver import ACO_CVRP_Solver
from streamlit_folium import st_folium
import folium
from io import StringIO
# ¡NUEVA IMPORTACIÓN!
from visualization import get_folium_results_map, to_excel, generate_html_report, format_duration
import json
import streamlit.components.v1 as components


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
    csv_data = """name,lat,lon,demand
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
        # Se intenta leer con ';' y si falla, con ','
        try:
            df = pd.read_csv(uploaded_file, delimiter=';', encoding='latin1')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, delimiter=',', encoding='utf-8')

        df.columns = [col.strip().lower() for col in df.columns]
        column_map = {
            'nombre': 'name', 'cliente': 'name',
            'latitud_aproximada': 'lat', 'latitud': 'lat', 'lat': 'lat',
            'longitud_aproximada': 'lon', 'longitud': 'lon', 'lon': 'lon', 'lng': 'lon',
            'ctd paquetes': 'demand', 'demanda': 'demand', 'pasajeros': 'demand'
        }
        
        # Mapeo robusto
        df_renamed = pd.DataFrame()
        for col_stand, col_name in {'name', 'lat', 'lon', 'demand'}:
            for key in column_map:
                if column_map[key] == col_name and key in df.columns:
                    df_renamed[col_name] = df[key]
                    break
        
        if not all(k in df_renamed.columns for k in ['lat', 'lon', 'demand']):
            st.error("Error: Archivo debe contener columnas de latitud, longitud y demanda.")
            return None
        
        if 'name' not in df_renamed.columns:
            df_renamed['name'] = [f"Cliente {i+1}" for i in range(len(df_renamed))]

        df_renamed[['lat', 'lon', 'demand']] = df_renamed[['lat', 'lon', 'demand']].astype(float)
        return df_renamed
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

# --- Interfaz Principal (sin cambios en Configuración) ---
st.title("🚚 Optimizador de Rutas Vehiculares (CVRP) (ACO)")
st.write("Una herramienta inteligente para encontrar las rutas más eficientes y económicas para tu flota.")

tab_config, tab_results, tab_about = st.tabs(["⚙️ 1. Configuración", "📊 2. Resultados", "👨‍💻 Acerca de"])

# ... (El código de la pestaña 'tab_config' y 'tab_about' se mantiene igual) ...
# ... PEGA AQUÍ TU CÓDIGO EXISTENTE PARA LAS PESTAÑAS DE CONFIGURACIÓN Y ACERCA DE ...

# --- PESTAÑA DE RESULTADOS (ACTUALIZADA) ---
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
        st_folium(results_map, width=1000, height=500, returned_objects=[])
        
        st.subheader("📋 Detalles por Ruta")
        for i, detail in enumerate(eval_data.get('route_details', [])):
            cost, duration_h = detail.get('cost', 0), detail.get('duration_hours', 0)
            load, utilization = detail.get('load', 0), detail.get('utilization', 0)
            expander_title = (f"**Ruta {i+1}** | Costo: `${cost:,.0f}` | Duración: `{format_duration(duration_h)}` | Carga: `{load:.0f} ({utilization:.1f}%)`")
            with st.expander(expander_title):
                route_indices = [idx - 1 for idx in detail.get('sequence', [])] # -1 para ajustar a índice de pandas
                if not route_indices: continue
                route_df = st.session_state.customer_data.iloc[route_indices]
                st.dataframe(route_df[['name', 'demand', 'lat', 'lon']], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        # --- SECCIÓN DE DESCARGA (NUEVA) ---
        st.subheader("📥 Descargar Informes")
        
        # Preparar datos para los informes
        resumen_df = pd.DataFrame(eval_data.get('route_details', []))
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
