import math, sys, time
import random
from typing import List

from dataclasses import dataclass, field
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import pandas as pd

from src.algorithms import *
from src.constants import *


@dataclass
class Chromosome:
    stops: List[int] = field(default_factory=list)
    vehicles: List[int] = field(default_factory=list)
    fitness: int = 0


# generates a single chromosome
def gen_chromosome(orders):
    # generates a random sequence of customers and a corresponding array which vehicle stops at this customer
    random.shuffle(orders)
    _vehicles = []
    for _ in range(0, len(orders)):
        _vehicles.append(random.randint(0, NO_VEHICLES))

    return Chromosome(orders.copy(), _vehicles.copy())


# generates the initial population
def gen_population():
    _chromosomes = []
    for _ in range(0, POPULATION_SIZE):
        _chromosomes.append(gen_chromosome())
    return _chromosomes


# returns the best chromosome in a population
def get_best_chromosome(population):
    max_fitness = - sys.maxsize
    best_chrom = Chromosome([], [])
    for c in population:
        if c.fitness > max_fitness:
            max_fitness = c.fitness
            best_chrom = c
    return best_chrom


# creates the distance matrix from order's addresses
def calculate_map_context(rows: list[tuple]):
    _dist_matrix = []
    for i in range(0, len(rows)):
        row = [0] * len(rows)
        _dist_matrix.append(row)

    for i in range(0, len(rows)):
        for j in range(i, len(rows)):
            start_x = rows[i][2]
            start_y = rows[i][3]
            end_x = rows[j][2]
            end_y = rows[j][3]
            dist = calc_dist(start_x, start_y, end_x, end_y)
            _dist_matrix[i][j] = dist
            _dist_matrix[j][i] = dist
    return _dist_matrix, rows


def calculate_distance_matrix_geopy(coords: list[tuple]):
    distance_matrix = [[geodesic(from_coord, to_coord).km for to_coord in coords] 
                         for from_coord in coords]
    
    return distance_matrix, coords


#  Choose one
distance_matrix, data_matrix = calculate_map_context(coords)
# distance_matrix, data_matrix = calculate_distance_matrix_geopy(coords)

# calculate the path_costs
def calculate_path_costs(c: Chromosome):
    path_costs = [0] * NO_VEHICLES
    prev_stop = [0] * NO_VEHICLES

    for i in range(0, len(c.vehicles)):
        stop = c.stops[i]  # the current stop
        vehicle_no = c.vehicles[i]  # the current driver that stops for this customer
        dist = distance_matrix[prev_stop[vehicle_no]][stop]  # distance driver makes for this customer
        path_costs[vehicle_no] += dist
        prev_stop[vehicle_no] = stop

    # calculate costs for return to depot
    for i in range(0, len(prev_stop)):
        return_dist = distance_matrix[prev_stop[i]][0]
        path_costs[i] += return_dist

    return path_costs


# calculates the distance based on euclidean metric measurement
def calc_dist(x1, y1, x2, y2):
    return math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


# print costs of a single iteration of the best chromosome
def print_cost(costs, iteration, runtime):
    print("Iteration: ", iteration, " runtime: ", "{:.4f}".format(runtime), ", costs: ", "{:.2f}".format(sum(costs)), sep="")
    return sum(costs)


# shows the phenotype of a chromosome
def print_phenotype(c: Chromosome):
    path_costs = calculate_path_costs(c)
    print("The total costs of the paths are:", "{:.2f}".format(sum(path_costs)))

    for i in range(0, NO_VEHICLES):
        print("Route #", i + 1, ":", sep="", end=" ")
        for j in range(0, len(c.vehicles)):
            if c.vehicles[j] == i:
                print(c.stops[j], end=" ")
        print("")


# plot the routes of the vehicles of a chromosome as a map
def plot_map(c: Chromosome, data):
    x_data = [d[0] for d in data]
    y_data = [d[1] for d in data]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    routes = []

    for i in range(0, NO_VEHICLES):
        route = [0]
        for j in range(0, len(c.vehicles)):
            if c.vehicles[j] == i:
                route.append(c.stops[j])
        route.append(0)
        routes.append(route)

    for i in range(0, len(routes)):
        x_points = []
        y_points = []
        for j in routes[i]:
            x_points.append(x_data[j])
            y_points.append(y_data[j])
        plt.plot(x_points[1:-1], y_points[1:-1], label="Route" + str(i + 1), marker='o', color=colors[i])
        plt.plot(x_points[:2], y_points[:2], color=colors[i], linestyle="--")
        plt.plot(x_points[-2:], y_points[-2:], color=colors[i], linestyle="--")

    plt.plot(x_data[0], y_data[0], marker='o', color='black')

    plt.legend()
    plt.savefig("GA_VRP.png")
    plt.show()


__all__ = [
    "Chromosome", 
    "gen_chromosome", 
    "gen_population", 
    "get_best_chromosome",
    "calculate_map_context",
    "distance_matrix", 
    "data_matrix",
    "calculate_path_costs",
    "calc_dist",
    "print_cost",
    "print_phenotype",
    "plot_map",
]