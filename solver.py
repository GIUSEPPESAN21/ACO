# solver.py
# Contiene la clase ACO_CVRP_Solver con la lógica del algoritmo.
# (Sin cambios respecto a la versión anterior)

import numpy as np
import random
from math import radians, sin, cos, sqrt, asin

class ACO_CVRP_Solver:
    """
    Resuelve el Problema de Ruteo de Vehículos con Capacidad (CVRP)
    utilizando un algoritmo de Optimización por Colonia de Hormigas (ACO).
    """
    def __init__(self, depot_coord, customer_coords, customer_demands, n_vehicles, vehicle_capacity, params):
        self.depot_coord = depot_coord
        self.customer_coords = customer_coords
        self.customer_demands = [float(d) for d in customer_demands]
        
        # El índice 0 es el depósito
        self.cities_coords = [self.depot_coord] + self.customer_coords
        self.demands = [0.0] + self.customer_demands
        
        self.n_cities = len(self.cities_coords)
        self.n_customers = len(self.customer_coords)
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = float(vehicle_capacity)
        # Permitir una ligera sobrecapacidad para encontrar soluciones más fácilmente
        self.effective_vehicle_capacity = self.vehicle_capacity * 1.05

        self.depot_index = 0
        self.distances = self._calculate_distance_matrix_haversine()
        
        # Parámetros del algoritmo ACO
        self.alpha = float(params.get('alpha', 1.0))
        self.beta = float(params.get('beta', 2.0))
        self.rho = float(params.get('rho', 0.1))
        self.Q = float(params.get('Q', 100))
        
        # Inicialización de la matriz de feromonas
        self.pheromones = np.ones((self.n_cities, self.n_cities))

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calcula la distancia Haversine entre dos puntos geográficos en km."""
        R = 6371  # Radio de la Tierra en km
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        return R * c

    def _calculate_distance_matrix_haversine(self):
        """Pre-calcula la matriz de distancias entre todas las ubicaciones."""
        n = self.n_cities
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                lon1, lat1 = self.cities_coords[i]
                lon2, lat2 = self.cities_coords[j]
                dist = self._haversine_distance(lat1, lon1, lat2, lon2)
                distances[i, j] = distances[j, i] = dist
        return distances.tolist()

    def _select_next_city(self, current_city_idx, unvisited_customers_global_indices, current_vehicle_load):
        """
        Selecciona la siguiente ciudad para una hormiga, considerando la capacidad del vehículo,
        la distancia (visibilidad) y el rastro de feromonas.
        """
        feasible_customers = [
            cust_idx for cust_idx in unvisited_customers_global_indices
            if current_vehicle_load + self.demands[cust_idx] <= self.effective_vehicle_capacity
        ]
        
        if not feasible_customers:
            return None

        probabilities = np.zeros(len(feasible_customers))
        for i, customer_idx in enumerate(feasible_customers):
            distance = self.distances[current_city_idx][customer_idx]
            # La visibilidad es el inverso de la distancia
            visibility = 1.0 / distance if distance > 0 else 1e9
            
            pheromone_level = self.pheromones[current_city_idx, customer_idx]
            probabilities[i] = (pheromone_level ** self.alpha) * (visibility ** self.beta)
        
        sum_probabilities = np.sum(probabilities)
        if sum_probabilities == 0:
            # Si todas las probabilidades son cero, elige uno al azar
            return random.choice(feasible_customers)

        probabilities /= sum_probabilities
        selected_local_idx = np.random.choice(len(feasible_customers), p=probabilities)
        return feasible_customers[selected_local_idx]

    def _construct_solution_for_ant(self):
        """Construye una solución completa (conjunto de rutas) para una hormiga."""
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
                next_customer_idx = self._select_next_city(
                    current_city_idx, 
                    unvisited_global_customer_indices, 
                    current_vehicle_load
                )
                
                if next_customer_idx is None:
                    # No más clientes factibles para esta ruta
                    break
                
                current_vehicle_load += self.demands[next_customer_idx]
                route_cost += self.distances[current_city_idx][next_customer_idx]
                current_route.append(next_customer_idx)
                unvisited_global_customer_indices.remove(next_customer_idx)
                current_city_idx = next_customer_idx

            # Regresar al depósito
            route_cost += self.distances[current_city_idx][self.depot_index]
            current_route.append(self.depot_index)
            
            if len(current_route) > 2:  # Ruta válida (más que solo Depósito -> Depósito)
                solution_routes.append(current_route)
                solution_total_cost += route_cost
        
        return solution_routes, solution_total_cost, unvisited_global_customer_indices

    def solve(self, n_ants=10, n_iterations=100, progress_callback=None):
        """
        Ejecuta el bucle principal del algoritmo ACO.
        """
        best_overall_solution_routes = None
        best_overall_cost = float('inf')

        for iteration in range(n_iterations):
            iteration_valid_solutions = []
            for _ in range(n_ants):
                routes, cost, unvisited = self._construct_solution_for_ant()
                # Una solución es válida solo si todos los clientes fueron visitados
                if not unvisited:
                    iteration_valid_solutions.append({'routes': routes, 'cost': cost})
                    if cost < best_overall_cost:
                        best_overall_cost = cost
                        best_overall_solution_routes = routes

            # Evaporación de feromonas
            self.pheromones *= (1.0 - self.rho)

            # Depósito de feromonas por las hormigas que encontraron soluciones válidas
            if iteration_valid_solutions:
                for sol_info in iteration_valid_solutions:
                    if sol_info['cost'] > 0:
                        deposit_value = self.Q / sol_info['cost']
                        for route in sol_info['routes']:
                            for i in range(len(route) - 1):
                                city1_idx, city2_idx = route[i], route[i+1]
                                self.pheromones[city1_idx, city2_idx] += deposit_value
                                self.pheromones[city2_idx, city1_idx] += deposit_value # Matriz simétrica

            if progress_callback:
                progress_callback(iteration + 1, n_iterations, best_overall_cost)
            
        return best_overall_solution_routes, best_overall_cost

    def evaluate_solution(self, solution):
        """Calcula métricas detalladas para una solución dada."""
        solution_routes = solution['routes']
        solution_cost = solution['cost']

        if not solution_routes: 
            return {}
        
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
            
            for c in route_customers: 
                all_visited_customers.add(c)
            total_demand_serviced += load
        
        metrics['customers_visited_count'] = len(all_visited_customers)
        metrics['all_customers_serviced'] = (len(all_visited_customers) == self.n_customers)
        num_used = metrics['num_vehicles_used']
        metrics['avg_vehicle_utilization_percent'] = (total_demand_serviced / (num_used * self.vehicle_capacity)) * 100 if num_used > 0 and self.vehicle_capacity > 0 else 0
        return metrics

