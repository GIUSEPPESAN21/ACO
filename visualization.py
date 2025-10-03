import streamlit as st
import pandas as pd
import folium
from io import BytesIO
import json
import streamlit.components.v1 as components

# --- Funciones para Generación de Informes (Adaptadas de Rout-2) ---

def to_excel(df_dict):
    """Crea un archivo Excel en memoria a partir de un diccionario de DataFrames."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

def generate_html_report(evaluation_data, customer_data):
    """Genera un informe HTML a partir de los datos de la solución."""
    resumen_df = pd.DataFrame(evaluation_data.get('route_details', []))
    customer_data_dict = customer_data.set_index('name').to_dict('index')

    html = """
    <html>
    <head><style>
        body { font-family: sans-serif; margin: 20px; }
        h1, h2, h3 { color: #1E3A8A; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; font-size: 12px; }
        th, td { border: 1px solid #ddd; padding: 6px; text-align: left; }
        th { background-color: #f2f2f2; }
        .page-break { page-break-before: always; }
        .no-break { page-break-inside: avoid; }
    </style></head>
    <body><h1>Informe de Rutas - ACO Optimizer</h1>
    """
    if not resumen_df.empty:
        df_reporte = resumen_df.copy()
        df_reporte['vehiculo'] = [f"Ruta {i+1}" for i in range(len(df_reporte))]
        df_reporte.rename(columns={
            'load': 'Demanda Total', 'utilization': '% Capacidad',
            'distance': 'Distancia (km)', 'cost': 'Costo ($)'
        }, inplace=True)
        html += "<div class='no-break'><h2>Resumen General</h2>"
        html += df_reporte[['vehiculo', 'Demanda Total', '% Capacidad', 'Distancia (km)', 'Costo ($)']].to_html(index=False, justify='center')
        html += "</div><h2 class='page-break'>Detalle por Ruta</h2>"

        for i, ruta in enumerate(resumen_df.to_dict('records')):
            html += f"<div class='no-break'><h3>Ruta {i+1}</h3>"
            # -1 porque los índices en la secuencia del solver empiezan en 1 para clientes
            route_indices = [idx - 1 for idx in ruta.get('sequence', [])]
            ruta_df = customer_data.iloc[route_indices]
            html += ruta_df[['name', 'lat', 'lon', 'demand']].to_html(index=False, justify='center')
            html += "</div>"
    html += "</body></html>"
    return html

# --- Función de Mapa (Movida desde streamlit_app.py) ---

def format_duration(hours):
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h} h {m} min"

def get_folium_results_map(solution_routes, solver, customer_data, evaluation_data):
    depot_coord = solver.depot_coord
    map_center = [customer_data['lat'].mean(), customer_data['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=14, tiles="cartodbpositron")
    route_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']

    for i, route in enumerate(solution_routes):
        route_coords = [(solver.cities_coords[c][1], solver.cities_coords[c][0]) for c in route]
        detail = evaluation_data['route_details'][i]
        cost = detail.get('cost', 0)
        duration_h = detail.get('duration_hours', 0)
        tooltip = f"Ruta {i+1} | Costo: ${cost:,.0f} | Duración: {format_duration(duration_h)}"
        folium.PolyLine(locations=route_coords, color=route_colors[i % len(route_colors)], weight=4, opacity=0.8, tooltip=tooltip).add_to(m)

    for _, row in customer_data.iterrows():
        folium.Marker(location=[row['lat'], row['lon']], popup=f"<b>{row['name']}</b><br>Demanda: {row['demand']}", tooltip=row['name'], icon=folium.Icon(color="blue", icon="user", prefix='fa')).add_to(m)

    folium.Marker(location=[depot_coord[1], depot_coord[0]], popup="Depósito", tooltip="Depósito", icon=folium.Icon(color="red", icon="truck", prefix='fa')).add_to(m)
    return m
