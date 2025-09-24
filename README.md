Optimizador de Rutas Vehiculares (CVRP) con Streamlit y ACO
Este proyecto es una aplicación web interactiva para resolver el Problema de Ruteo de Vehículos con Capacidad (CVRP) utilizando un algoritmo de Optimización por Colonia de Hormigas (ACO).

La aplicación, construida con Streamlit, permite a los usuarios cargar datos de clientes, configurar parámetros del problema y del algoritmo, y visualizar las rutas óptimas resultantes sobre un mapa interactivo.

Características Principales
Interfaz Intuitiva: Todos los parámetros se configuran fácilmente desde una barra lateral.

Carga de Datos Flexible: Sube tus propios datos de clientes en formato CSV o utiliza un conjunto de datos de ejemplo incluido.

Visualización en Mapa Interactivo: Las rutas generadas se muestran sobre un mapa real, proporcionando un contexto geográfico claro.

Feedback en Tiempo Real: Monitorea el progreso del algoritmo durante la ejecución.

Resultados Detallados: Obtén un resumen de métricas clave (costo total, vehículos usados) y un desglose detallado de cada ruta.

Código Modular: La lógica del algoritmo ACO está separada de la interfaz de usuario para mayor claridad y mantenibilidad.

Estructura del Repositorio
.
├── data/
│   └── sample_data.csv       # Archivo de datos de ejemplo
├── solver.py                 # Módulo con la clase ACO_CVRP_Solver
├── streamlit_app.py          # Script principal de la aplicación Streamlit
├── requirements.txt          # Dependencias de Python
└── README.md                 # Este archivo

Cómo Ejecutar la Aplicación
Clonar el Repositorio (Opcional):

git clone <url-del-repositorio>
cd <nombre-del-repositorio>

Crear un Entorno Virtual (Recomendado):

python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

Instalar las Dependencias:
Asegúrate de tener todas las librerías necesarias instaladas.

pip install -r requirements.txt

Iniciar la Aplicación:
Ejecuta el siguiente comando en tu terminal:

streamlit run streamlit_app.py

Se abrirá una nueva pestaña en tu navegador con la aplicación en funcionamiento.

Formato del Archivo CSV
El archivo CSV de entrada debe contener al menos las siguientes columnas. Los nombres pueden variar, pero deben incluir alguna de estas opciones:

Latitud: latitud_aproximada, latitud, lat

Longitud: longitud_aproximada, longitud, lon, lng

Demanda: ctd paquetes, demanda, demand, volumen
