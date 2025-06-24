# app.py - Backend para el Optimizador CVRP (Versión Mejorada)
# Ahora sirve el frontend y maneja la lógica de la API.
#
# Para ejecutar:
# 1. Instalar dependencias: pip install Flask Flask-Cors pandas numpy
# 2. Correr el servidor: python app.py
# 3. Abrir en el navegador: http://127.0.0.1:5000

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import random
import os
from math import radians, sin, cos, sqrt, asin

# --- Clase del Solucionador ACO (sin cambios) ---
class ACO_CVRP_Solver:
    def __init__(self, depot_coord, customer_coords, customer_demands, n_vehicles, vehicle_capacity, params):
        self.depot_coord = depot_coord
        self.customer_coords = customer_coords
        self.customer_demands = [float(d) for d in customer_demands]
        
        self.cities_coords = [self.depot_coord] + self.customer_coords
        self.demands = [0.0] + self.customer_demands
        
        self.n_cities = len(self.cities_coords)
        self.n_customers = len(self.customer_coords)
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = float(vehicle_capacity)
        self.effective_vehicle_capacity = self.vehicle_capacity * 1.05

        self.depot_index = 0
        self.distances = self._calculate_distance_matrix_haversine()
        
        self.alpha = float(params.get('alpha', 1.0))
        self.beta = float(params.get('beta', 2.0))
        self.rho = float(params.get('rho', 0.1))
        self.Q = float(params.get('Q', 100))
        self.pheromones = np.ones((self.n_cities, self.n_cities))

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        return R * c

    def _calculate_distance_matrix_haversine(self):
        n = self.n_cities
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    lon1, lat1 = self.cities_coords[i]
                    lon2, lat2 = self.cities_coords[j]
                    distances[i, j] = self._haversine_distance(lat1, lon1, lat2, lon2)
        return distances.tolist()

    def _select_next_city(self, current_city_idx, unvisited_customers_global_indices, current_vehicle_load):
        feasible_customers = [
            cust_idx for cust_idx in unvisited_customers_global_indices
            if current_vehicle_load + self.demands[cust_idx] <= self.effective_vehicle_capacity
        ]
        if not feasible_customers:
            return None

        probabilities = np.zeros(len(feasible_customers))
        for i, customer_idx in enumerate(feasible_customers):
            distance = self.distances[current_city_idx][customer_idx]
            visibility = 1.0 / distance if distance != 0 else 1e9
            probabilities[i] = (self.pheromones[current_city_idx, customer_idx] ** self.alpha) * (visibility ** self.beta)
        
        sum_probabilities = np.sum(probabilities)
        if sum_probabilities == 0:
            return random.choice(feasible_customers) if feasible_customers else None

        probabilities /= sum_probabilities
        selected_local_idx = np.random.choice(len(feasible_customers), p=probabilities)
        return feasible_customers[selected_local_idx]

    def _construct_solution_for_ant(self):
        solution_routes = []
        solution_total_cost = 0.0
        unvisited_global_customer_indices = list(range(1, self.n_cities))
        random.shuffle(unvisited_global_customer_indices)
        num_vehicles_used = 0

        while unvisited_global_customer_indices and num_vehicles_used < self.n_vehicles:
            num_vehicles_used += 1
            current_route = [self.depot_index]
            current_city_idx = self.depot_index
            current_vehicle_load = 0.0
            route_cost = 0.0

            while True:
                next_customer_idx = self._select_next_city(current_city_idx, unvisited_global_customer_indices, current_vehicle_load)
                if next_customer_idx is None:
                    break
                
                current_vehicle_load += self.demands[next_customer_idx]
                route_cost += self.distances[current_city_idx][next_customer_idx]
                current_route.append(next_customer_idx)
                unvisited_global_customer_indices.remove(next_customer_idx)
                current_city_idx = next_customer_idx

            route_cost += self.distances[current_city_idx][self.depot_index]
            current_route.append(self.depot_index)
            
            if len(current_route) > 2:
                solution_routes.append(current_route)
                solution_total_cost += route_cost
        
        return solution_routes, solution_total_cost, unvisited_global_customer_indices

    def solve(self, n_ants=10, n_iterations=100):
        best_overall_solution_routes = None
        best_overall_cost = float('inf')

        for iteration in range(n_iterations):
            iteration_valid_solutions = []
            for _ in range(n_ants):
                routes, cost, unvisited = self._construct_solution_for_ant()
                if not unvisited:
                    iteration_valid_solutions.append({'routes': routes, 'cost': cost})
                    if cost < best_overall_cost:
                        best_overall_cost = cost
                        best_overall_solution_routes = routes

            self.pheromones *= (1.0 - self.rho)
            if iteration_valid_solutions:
                for sol_info in iteration_valid_solutions:
                    if sol_info['cost'] > 0:
                        deposit_value = self.Q / sol_info['cost']
                        for route in sol_info['routes']:
                            for i in range(len(route) - 1):
                                city1_idx, city2_idx = route[i], route[i+1]
                                self.pheromones[city1_idx, city2_idx] += deposit_value
                                self.pheromones[city2_idx, city1_idx] += deposit_value
            print(f"Iteración {iteration + 1}/{n_iterations} - Mejor costo: {best_overall_cost if best_overall_cost != float('inf') else 'N/A'}")
            
        return best_overall_solution_routes, best_overall_cost

    def evaluate_solution(self, solution_routes, solution_cost):
        if not solution_routes: return {}
        
        metrics = {}
        metrics['num_vehicles_used'] = len(solution_routes)
        metrics['total_distance_km'] = solution_cost
        metrics['route_details'] = []
        all_visited_customers = set()
        total_demand_serviced = 0

        for route in solution_routes:
            dist = sum(self.distances[route[i]][route[i+1]] for i in range(len(route)-1))
            route_customers = route[1:-1]
            load = sum(self.demands[city_idx] for city_idx in route_customers)
            util_percent_nominal = (load / self.vehicle_capacity) * 100 if self.vehicle_capacity > 0 else 0
            
            metrics['route_details'].append({
                'sequence': route_customers, 'distance': dist, 'load': load,
                'stops': len(route_customers), 'utilization': util_percent_nominal
            })
            
            for c in route_customers: all_visited_customers.add(c)
            total_demand_serviced += load
        
        metrics['customers_visited_count'] = len(all_visited_customers)
        metrics['all_customers_serviced'] = (len(all_visited_customers) == self.n_customers)
        num_used = metrics['num_vehicles_used']
        metrics['avg_vehicle_utilization_percent'] = (total_demand_serviced / (num_used * self.vehicle_capacity)) * 100 if num_used > 0 and self.vehicle_capacity > 0 else 0
        return metrics

# --- Configuración de la API con Flask ---
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- NUEVA RUTA ---
# Sirve el archivo index.html cuando alguien visita la raíz del sitio (ej. http://127.0.0.1:5000)
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# --- Ruta de la API (sin cambios, pero ahora funciona en conjunto con la de arriba) ---
@app.route('/optimize', methods=['POST'])
def optimize_route():
    print("Recibida solicitud de optimización...")
    data = request.json
    try:
        params = data['params']
        solver = ACO_CVRP_Solver(
            depot_coord=params['depotCoord'],
            customer_coords=data['customerData']['coords'],
            customer_demands=data['customerData']['demands'],
            n_vehicles=int(params['nVehicles']),
            vehicle_capacity=float(params['vehicleCapacity']),
            params=params['aco']
        )
        best_routes, best_cost = solver.solve(
            n_ants=int(params['aco']['nAnts']),
            n_iterations=int(params['aco']['nIterations'])
        )
        if not best_routes:
            return jsonify({'error': 'No se encontró una solución válida.'}), 400

        evaluation = solver.evaluate_solution(best_routes, best_cost)
        response_data = {
            'solution': {'routes': best_routes, 'cost': best_cost},
            'evaluation': evaluation,
            'solver_config': {
                'cities_coords': solver.cities_coords, 'depot_coord': solver.depot_coord,
                'customer_coords': solver.customer_coords, 'demands': solver.demands,
                'n_customers': solver.n_customers, 'n_vehicles': solver.n_vehicles
            }
        }
        print("Optimización completada. Enviando resultados.")
        return jsonify(response_data)
    except Exception as e:
        print(f"Error durante la optimización: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=====================================================")
    print("Iniciando servidor de Optimización de Rutas...")
    print("Para usar la aplicación, abre tu navegador y ve a:")
    print(f"http://127.0.0.1:5000")
    print("=====================================================")
    app.run(debug=True, port=5000)
