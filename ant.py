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
            tab_camion[camion_index].extend([route[0], route[1]])
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

                for i in range(0, len(routes), 2):
                    collection_city = routes[i]
                    delivery_city = routes[i + 1]

                    # Ensure current_city is included in sub_matrix_indices
                    sub_matrix_indices = [start_city, collection_city, delivery_city]
                    if current_city not in sub_matrix_indices:
                        sub_matrix_indices.append(current_city)
                    
                    sub_matrix = extract_submatrix(graph.travel_time, sub_matrix_indices)
                    sub_pheromone = extract_submatrix(graph.pheromone, sub_matrix_indices)

                    current_index = sub_matrix_indices.index(current_city)

                    # Calculate probabilities for moving to the next city
                    probabilities = np.array([
                        (sub_pheromone[current_index][j] ** alpha) * ((1.0 / sub_matrix[current_index][j]) ** beta)
                        if sub_matrix[current_index][j] > 0 else 0 for j in range(len(sub_matrix))
                    ])
                    
                    # Normalize probabilities safely
                    if probabilities.sum() == 0:
                        probabilities = np.ones_like(probabilities) / len(probabilities)  # Equal probability if sum is 0
                    else:
                        probabilities /= probabilities.sum()

                    # Choose the next city based on probabilities
                    next_city_index = np.random.choice(range(len(sub_matrix)), p=probabilities)
                    next_city = sub_matrix_indices[next_city_index]
                    ant.route.append(next_city)
                    ant.total_time += sub_matrix[current_index][next_city_index]
                    
                    # Update pheromones
                    try:
                        travel_time = sub_matrix[current_index][next_city_index]
                        if travel_time > 0:
                            graph.pheromone[current_city-1, next_city-1] *= (1 - evaporation)
                            graph.pheromone[current_city-1, next_city-1] += pheromone_boost / travel_time
                    except ZeroDivisionError:
                        print(f"Error: Division by zero when updating pheromones between {current_city} and {next_city}")

                    current_city = next_city

                # Return to the start city at the end of the route
                if current_city != start_city:
                    ant.route.append(start_city)
                    ant.total_time += graph.travel_time[current_city-1, start_city-1]

                # Check if this is the best route for this truck
                if ant.total_time < best_times[camion_index]:
                    best_times[camion_index] = ant.total_time
                    best_routes[camion_index] = ant.route[:]

    # Print best routes and their times
    for i, (route, time) in enumerate(zip(best_routes, best_times)):
        print(f"Best route for truck {i + 1}: {route} with total time {time}")

def draw_routes(graph, ants):
    plt.figure(figsize=(10, 8))

    for ant in ants:
        route_coords = [graph.coordinates[city] for city in ant.route]
        plt.plot([coord[0] for coord in route_coords], [coord[1] for coord in route_coords], 'o-', label=f"Ant {ant.id}")

    for city, coords in graph.coordinates.items():
        plt.text(coords[0], coords[1], city, fontsize=12, ha='right')
    plt.legend()
    plt.title("Routes for all ants")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

def main():
    random.seed(42)
    np.random.seed(42)

    nombres_objets = 10
    nombres_camions = 3
    nombres_fourmis = 10
    temps_min = 10
    temps_max = 100
    
    logistics = Generation_data(nombres_objets, nombres_camions)
    logistics.affichage_camions()

    num_points = nombres_objets * 2
    graph = Graph(num_points, temps_min, temps_max)
    graph.afficher_temps_trajet()
    graph.afficher_ville_depart()
    ants = [Ant(i + 1) for i in range(nombres_fourmis)]

    simulate(graph, logistics, ants)
    draw_routes(graph, ants)

main()
