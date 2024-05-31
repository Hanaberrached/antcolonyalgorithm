import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Graph:
    def __init__(self, num_points, temps_min, temps_max):
        self.num_points = num_points
        self.coordinates = {i: (random.uniform(0, 100), random.uniform(0, 100)) for i in range(1, num_points + 1)}
        self.travel_time = np.array(generer_temps_trajet(num_points, temps_min, temps_max))
        self.pheromone = np.ones((num_points, num_points)) * 0.1
        self.start_city = random.randint(1, num_points)  # Ville de départ/arrivée aléatoire

    def afficher_temps_trajet(self):
        print("Matrice des temps de trajet :")
        for row in self.travel_time:
            print(row)

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
            grand_tableau.append([ville_collecte, ville_livraison, [camion]])

        return grand_tableau

    def gen_tab_camion(self):
        tab_camion = [[] for _ in range(self.nbres_camions)]

        for route in self.grand_tableau:
            camion_index = route[2][0] - 1
            tab_camion[camion_index].extend([route[0], route[1]])

        return tab_camion

    def affichage_camions(self):
        for i, camions in enumerate(self.tableau_camion):
            print(f"Camion {i + 1} : {camions}")

def generer_temps_trajet(nb_villes, temps_min, temps_max):
    temps_trajet = []
    for i in range(nb_villes):
        ligne = []
        for j in range(nb_villes):
            if i == j:
                ligne.append(0)
            elif i < j:
                temps = random.randint(temps_min, temps_max)
                ligne.append(temps)
            else:
                ligne.append(temps_trajet[j][i])
        temps_trajet.append(ligne)
    return temps_trajet

def simulate(graph, logistics, ants, iterations=100):
    alpha = 1.0
    beta = 2.0
    evaporation = 0.5
    pheromone_boost = 1.0
    start_city = graph.start_city  # Ville de départ/arrivée

    for _ in tqdm(range(iterations), desc="Simulation Progress"):
        for ant in ants:
            ant.route = [start_city]  # Commencer à la ville de départ
            ant.total_time = 0

        for route in logistics.grand_tableau:
            camion_index = route[2][0] - 1
            ant = ants[camion_index]
            current_city = start_city
            destination_city = route[1]
            ant.route.append(route[0])  # Ajouter la ville de collecte

            while current_city != destination_city:
                probabilities = np.array([
                    (graph.pheromone[current_city-1, i] ** alpha) * ((1.0 / graph.travel_time[current_city-1, i]) ** beta)
                    if graph.travel_time[current_city-1, i] > 0 and i != current_city-1 else 0 for i in range(graph.num_points)
                ])

                if probabilities.sum() == 0:
                    break

                probabilities /= probabilities.sum()
                next_city = np.random.choice(range(1, graph.num_points + 1), p=probabilities)
                ant.route.append(next_city)
                ant.total_time += graph.travel_time[current_city-1, next_city-1]
                graph.pheromone[current_city-1, next_city-1] *= (1 - evaporation)
                graph.pheromone[current_city-1, next_city-1] += pheromone_boost / graph.travel_time[current_city-1, next_city-1]
                current_city = next_city
                print(f"Ant {ant.id} at city {current_city} moving to {next_city}, total time now: {ant.total_time}")

            ant.route.append(start_city)  # Retourner à la ville de départ
            ant.total_time += graph.travel_time[current_city-1, start_city-1]

def draw_routes(graph, ants):
    best_time = float('inf')
    best_route = None

    plt.figure(figsize=(10, 8))

    for ant in ants:
        route_coords = [graph.coordinates[city] for city in ant.route]
        plt.plot([coord[0] for coord in route_coords], [coord[1] for coord in route_coords], 'o-', label=f"Ant {ant.id}")
        if ant.total_time < best_time:
            best_time = ant.total_time
            best_route = ant.route
        print(f"Evaluating best time: Current best time is {best_time}")

    for city, coords in graph.coordinates.items():
        plt.text(coords[0], coords[1], city, fontsize=12, ha='right')
    plt.legend()
    plt.title("Routes for all ants")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

    if best_route is not None:
        draw_best_route(graph, best_route, best_time)

def draw_best_route(graph, best_route, best_time):
    plt.figure(figsize=(10, 8))
    route_coords = [graph.coordinates[city] for city in best_route]
    plt.plot([coord[0] for coord in route_coords], [coord[1] for coord in route_coords], 'o-', color='red', label="Best Route")
    for city, coords in graph.coordinates.items():
        plt.text(coords[0], coords[1], city, fontsize=12, ha='right')
    plt.legend()
    plt.title(f"Best Route (Total Time: {best_time})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

    print(f"Best route: {' -> '.join(map(str, best_route))}")
    print(f"Total travel time: {best_time}")

def main():
    random.seed(42)
    np.random.seed(42)

    nombres_objets = 10
    nombres_camions = 3
    temps_min = 10
    temps_max = 100
    
    logistics = Generation_data(nombres_objets, nombres_camions)
    logistics.affichage_camions()
    print(logistics.grand_tableau, logistics.tableau_camion)

    num_points = nombres_objets * 2
    graph = Graph(num_points, temps_min, temps_max)
    graph.afficher_temps_trajet()  # Afficher les temps de trajet
    graph.afficher_ville_depart()  # Afficher la ville de départ/arrivée
    ants = [Ant(i+1) for i in range(nombres_camions)]

    simulate(graph, logistics, ants)
    draw_routes(graph, ants)

main()
