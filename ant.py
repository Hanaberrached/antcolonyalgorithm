import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generer_temps_trajet(nb_villes, temps_min, temps_max):
    temps_trajet = np.zeros((nb_villes, nb_villes), dtype=int)
    for i in range(nb_villes):
        for j in range(i + 1, nb_villes):
            temps = random.randint(temps_min, temps_max)
            temps_trajet[i][j] = temps
            temps_trajet[j][i] = temps
    return temps_trajet

def extract_submatrix(matrix, indices):
    indices = [x - 1 for x in indices]  
    return matrix[np.ix_(indices, indices)]

class Graph:
    def __init__(self, num_points, temps_min, temps_max):
        self.num_points = num_points
        self.coordinates = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(1, num_points + 1)}
        self.travel_time = generer_temps_trajet(num_points, temps_min, temps_max)
        self.pheromone = np.ones((num_points, num_points)) * 0.1
        self.start_city = random.randint(1, num_points)

    def afficher_temps_trajet(self):
        print("Matrice des temps de trajet :")
        print(self.travel_time)

    def afficher_ville_depart(self):
        print(f"Ville de départ/arrivée : {self.start_city}")

class Ant:
    def __init__(self, id):
        self.id = id
        self.route = []
        self.total_time = 0

class Generation_data:
    def __init__(self, nb_objets, nbres_camions):
        self.nb_objets = nb_objets
        self.nbres_camions = nbres_camions
        self.grand_tableau = self.generer_tableau()
        self.tableau_camion = self.gen_tab_camion()

    def generer_tableau(self):
        nb_villes = self.nb_objets * 2
        tab_ville = list(range(1, nb_villes + 1))
        grand_tableau = []

        for _ in range(nb_villes // 2):
            ville_collecte = tab_ville.pop(random.randint(0, len(tab_ville) - 1))
            ville_livraison = tab_ville.pop(random.randint(0, len(tab_ville) - 1))
            camion = random.randint(1, self.nbres_camions)
            grand_tableau.append([ville_collecte, ville_livraison, camion])

        return grand_tableau

    def gen_tab_camion(self):
        tab_camion = [[] for _ in range(self.nbres_camions)]
        for route in self.grand_tableau:
            camion_index = route[2] - 1
            tab_camion[camion_index].append((route[0], route[1]))
        return tab_camion

    def affichage_camions(self):
        for i, camions in enumerate(self.tableau_camion):
            print(f"Camion {i + 1} : {camions}")

def simulate(graph, logistics, ants, iterations=100):
    alpha = 1.0
    beta = 2.0
    evaporation = 0.5
    pheromone_boost = 1.0
    start_city = graph.start_city  # Ville de départ/arrivée

    best_routes = [[] for _ in range(logistics.nbres_camions)]
    best_times = [float('inf')] * logistics.nbres_camions

    for _ in tqdm(range(iterations), desc="Simulation Progress"):
        for ant in ants:
            ant.route = [start_city]  # Commencer à la ville de départ
            ant.total_time = 0

        for camion_index, routes in enumerate(logistics.tableau_camion):
            for ant_index in range(len(ants)):
                ant = ants[ant_index]
                current_city = start_city

                full_route = [start_city]  # Start at the start city

                for collection_city, delivery_city in routes:
                    full_route.append(collection_city)
                    full_route.append(delivery_city)
                full_route.append(start_city)  # Return to the start city

                ant.route = full_route[:]
                ant.total_time = 0

                for i in range(len(full_route) - 1):
                    ant.total_time += graph.travel_time[full_route[i] - 1][full_route[i + 1] - 1]

                # Check if this is the best route for this truck
                if ant.total_time < best_times[camion_index]:
                    best_times[camion_index] = ant.total_time
                    best_routes[camion_index] = ant.route[:]

                # Update pheromones
                for i in range(len(full_route) - 1):
                    graph.pheromone[full_route[i] - 1][full_route[i + 1] - 1] *= (1 - evaporation)
                    graph.pheromone[full_route[i] - 1][full_route[i + 1] - 1] += pheromone_boost / graph.travel_time[full_route[i] - 1][full_route[i + 1] - 1]

    # Print best routes and their times
    for i, (route, time) in enumerate(zip(best_routes, best_times)):
        print(f"Best route for truck {i + 1}: {route} with total time {time}")

    return best_routes

def draw_ants_routes(graph, ants):
    plt.figure(figsize=(10, 8))

    # Plot the routes of all ants
    for ant in ants:
        route_coords = [graph.coordinates[city] for city in ant.route]
        plt.plot([coord[0] for coord in route_coords], [coord[1] for coord in route_coords], 'o-', alpha=0.5, label=f"Ant {ant.id}")

    # Annotate the cities
    for city, coords in graph.coordinates.items():
        plt.text(coords[0], coords[1], city, fontsize=12, ha='right')
    plt.legend()
    plt.title("Routes for All Ants")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

def draw_trucks_routes(graph, best_routes):
    plt.figure(figsize=(10, 8))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    drawn_routes = {}  # To keep track of drawn routes to superimpose them

    for i, route in enumerate(best_routes):
        route_key = tuple(route[:2])  # Use the first two cities as the key
        color = colors[i % len(colors)]
        marker_style = 'o' if route_key not in drawn_routes else 's'

        route_coords = [graph.coordinates[city] for city in route]
        plt.plot([coord[0] for coord in route_coords], [coord[1] for coord in route_coords], marker_style+'-', color=color, label=f"Truck {i + 1} Best Route")

        # Update drawn routes dictionary
        if route_key not in drawn_routes:
            drawn_routes[route_key] = color

    # Annotate the cities
    for city, coords in graph.coordinates.items():
        plt.text(coords[0], coords[1], city, fontsize=12, ha='right')
    plt.legend()
    plt.title("Best Routes for All Trucks")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

def print_truck_routes(best_routes):
    for i, route in enumerate(best_routes):
        route_str = " -> ".join(map(str, route))
        print(f"Truck {i + 1} route: {route_str}")

def print_object_assignments(logistics):
    for i, camions in enumerate(logistics.tableau_camion):
        obj_str = ", ".join([f"({collect}->{deliver})" for collect, deliver in camions])
        print(f"Truck {i + 1} objects: {obj_str}")

def main():
    random.seed(42)
    np.random.seed(42)

    nombres_objets = 10
    nombres_camions = 3
    nombres_fourmis = 10
    temps_min = 10
    temps_max = 100

    print(f"Number of objects: {nombres_objets}")
    logistics = Generation_data(nombres_objets, nombres_camions)
    logistics.affichage_camions()

    num_points = nombres_objets * 2
    graph = Graph(num_points, temps_min, temps_max)
    graph.afficher_temps_trajet()
    graph.afficher_ville_depart()
    ants = [Ant(i + 1) for i in range(nombres_fourmis)]

    best_routes = simulate(graph, logistics, ants)
    print_truck_routes(best_routes)
    print_object_assignments(logistics)
    draw_ants_routes(graph, ants)
    draw_trucks_routes(graph, best_routes)

main()
