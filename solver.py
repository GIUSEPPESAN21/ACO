# solver.py
# Versión final con restricción de capacidad estricta y evaluación de costos/tiempos.

import numpy as np
import random
from math import radians, sin, cos, sqrt, asin

class ACO_CVRP_Solver:
    def __init__(self, depot_coord, customer_coords, customer_demands, n_vehicles, vehicle_capacity, params, use_2_opt=True, elitism_weight=1.0):
        self.depot_coord = depot_coord
        self.customer_coords = customer_coords
        self.customer_demands = [float(d) for d in customer_demands]
        
        self.cities_coords = [self.depot_coord] + self.customer_coords
        self.demands = [0.0] + self.customer_demands
        
        self.n_cities = len(self.cities_coords)
        self.n_customers = len(self.customer_coords)
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = float(vehicle_capacity)
        # CORREGIDO: Se elimina la tolerancia para evitar sobrecupo. La capacidad efectiva es la nominal.
        self.effective_vehicle_capacity = self.vehicle_capacity

        self.depot_index = 0
        self.distances = self._calculate_distance_matrix_haversine()
        
        self.alpha = float(params.get('alpha', 1.0))
        self.beta = float(params.get('beta', 2.0))
        self.rho = float(params.get('rho', 0.1))
        self.Q = float(params.get('Q', 100))
        
        self.use_2_opt = use_2_opt
        self.elitism_weight = float(elitism_weight)

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
            for j in range(i + 1, n):
                lon1, lat1 = self.cities_coords[i]
                lon2, lat2 = self.cities_coords[j]
                dist = self._haversine_distance(lat1, lon1, lat2, lon2)
                distances[i, j] = distances[j, i] = dist
        return distances

    def _select_next_city(self, current_city_idx, unvisited_customers, current_vehicle_load):
        feasible_customers = [
            cust_idx for cust_idx in unvisited_customers
            if current_vehicle_load + self.demands[cust_idx] <= self.effective_vehicle_capacity
        ]
        if not feasible_customers:
            return None

        probabilities = []
        for customer_idx in feasible_customers:
            distance = self.distances[current_city_idx, customer_idx]
            visibility = (1.0 / distance) if distance > 0 else 1e9
            pheromone = self.pheromones[current_city_idx, customer_idx]
            probabilities.append((pheromone ** self.alpha) * (visibility ** self.beta))
        
        sum_probabilities = sum(probabilities)
        if sum_probabilities == 0:
            return random.choice(feasible_customers)

        probabilities = [p / sum_probabilities for p in probabilities]
        return random.choices(feasible_customers, weights=probabilities, k=1)[0]
    
    def _two_opt(self, route):
        if len(route) <= 4: return route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j == i + 1: continue
                    current_dist = self.distances[route[i-1], route[i]] + self.distances[route[j-1], route[j]]
                    new_dist = self.distances[route[i-1], route[j-1]] + self.distances[route[i], route[j]]
                    if new_dist < current_dist:
                        route[i:j] = route[j-1:i-1:-1]
                        improved = True
        return route

    def _construct_solution_for_ant(self):
        solution_routes, unvisited = [], list(range(1, self.n_cities))
        random.shuffle(unvisited)
        for _ in range(self.n_vehicles):
            if not unvisited: break
            current_route, current_city, current_load = [self.depot_index], self.depot_index, 0.0
            while unvisited:
                next_city = self._select_next_city(current_city, unvisited, current_load)
                if next_city is None: break
                current_load += self.demands[next_city]
                current_route.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city
            current_route.append(self.depot_index)
            if self.use_2_opt: current_route = self._two_opt(current_route)
            if len(current_route) > 2: solution_routes.append(current_route)
        cost = sum(self.distances[route[i], route[i+1]] for route in solution_routes for i in range(len(route)-1))
        return solution_routes, cost, unvisited

    def solve(self, n_ants, n_iterations, progress_callback=None):
        best_overall_routes, best_overall_cost = None, float('inf')
        for iteration in range(n_iterations):
            iteration_solutions = []
            for _ in range(n_ants):
                routes, cost, unvisited = self._construct_solution_for_ant()
                if not unvisited:
                    iteration_solutions.append({'routes': routes, 'cost': cost})
                    if cost < best_overall_cost:
                        best_overall_cost, best_overall_routes = cost, routes
            self.pheromones *= (1.0 - self.rho)
            for sol in iteration_solutions:
                if sol['cost'] > 0:
                    deposit_value = self.Q / sol['cost']
                    for route in sol['routes']:
                        for i in range(len(route) - 1):
                            u, v = route[i], route[i+1]
                            self.pheromones[u, v] += deposit_value
                            self.pheromones[v, u] += deposit_value
            if best_overall_routes:
                elitist_deposit = self.elitism_weight * (self.Q / best_overall_cost)
                for route in best_overall_routes:
                    for i in range(len(route) - 1):
                        u, v = route[i], route[i+1]
                        self.pheromones[u, v] += elitist_deposit
                        self.pheromones[v, u] += elitist_deposit
            if progress_callback: progress_callback(iteration + 1, n_iterations, best_overall_cost)
        return best_overall_routes, best_overall_cost

    # MODIFICADO: Ahora acepta parámetros de simulación para calcular costo y duración.
    def evaluate_solution(self, solution, cost_per_km, speed_kmh, service_time_min):
        solution_routes = solution.get('routes', [])
        if not solution_routes: 
            return {
                'total_distance_km': 0, 'num_vehicles_used': 0, 'route_details': [],
                'customers_visited_count': 0, 'avg_vehicle_utilization_percent': 0,
                'total_cost': 0, 'total_duration_hours': 0
            }

        metrics = {'route_details': []}
        all_visited_customers, total_demand_serviced = set(), 0

        for route in solution_routes:
            dist = sum(self.distances[route[i], route[i+1]] for i in range(len(route)-1))
            route_customers = route[1:-1]
            load = sum(self.demands[city_idx] for city_idx in route_customers)
            stops = len(route_customers)
            
            # NUEVOS CÁLCULOS
            cost = dist * cost_per_km
            travel_time_h = dist / speed_kmh if speed_kmh > 0 else 0
            service_time_h = stops * (service_time_min / 60.0)
            duration_h = travel_time_h + service_time_h
            
            metrics['route_details'].append({
                'sequence': route_customers, 'distance': dist, 'load': load,
                'stops': stops, 'utilization': (load / self.vehicle_capacity) * 100 if self.vehicle_capacity > 0 else 0,
                'cost': cost, 'duration_hours': duration_h # NUEVO
            })
            for c in route_customers: all_visited_customers.add(c)
            total_demand_serviced += load
        
        metrics['total_distance_km'] = sum(r['distance'] for r in metrics['route_details'])
        metrics['total_cost'] = sum(r['cost'] for r in metrics['route_details']) # NUEVO
        metrics['total_duration_hours'] = sum(r['duration_hours'] for r in metrics['route_details']) # NUEVO
        
        metrics['num_vehicles_used'] = len(solution_routes)
        metrics['customers_visited_count'] = len(all_visited_customers)
        num_used = metrics['num_vehicles_used']
        metrics['avg_vehicle_utilization_percent'] = (total_demand_serviced / (num_used * self.vehicle_capacity)) * 100 if num_used > 0 and self.vehicle_capacity > 0 else 0
        return metrics
