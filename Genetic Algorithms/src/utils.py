import math, sys, time
import random
from typing import List

from dataclasses import dataclass, field
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.algorithms import *
from src.constants import *
from src.test_data import *


@dataclass
class Chromosome:
    stops: List[int] = field(default_factory=list)
    vehicles: List[int] = field(default_factory=list)
    fitness: int = 0


# generates the initial population
def gen_population(data_generation_method, *args, **kwargs):
    data = data_generation_method(*args, **kwargs)
    _chromosomes = []
    for _ in range(0, POPULATION_SIZE):
        _chromosomes.append(_gen_chromosome(data))
    return _chromosomes


# generates a single chromosome
def _gen_chromosome(data):
    # generates a random sequence of customers and a corresponding array which vehicle stops at this customer
    random.shuffle(data)
    _vehicles = []
    for _ in range(0, len(data)):
        _vehicles.append(random.randint(1, NO_VEHICLES))

    return Chromosome(data.copy(), _vehicles.copy())


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
def calculate_distance_matrix_dataframe_points(chrom: Chromosome):
    rows = chrom.stops
    _dist_matrix = np.zeros((len(rows),len(rows)))

    for i in range(0, len(rows)):
        for j in range(i, len(rows)):
            start_x = rows[i][0]
            start_y = rows[i][1]
            end_x = rows[j][0]
            end_y = rows[j][1]
            dist = _calc_dist(start_x, start_y, end_x, end_y)
            _dist_matrix[i][j] = dist
            _dist_matrix[j][i] = dist
    return _dist_matrix, rows


def calculate_distance_matrix_geopy(chrom: Chromosome):
    coords = chrom.stops
    distance_matrix = [[geodesic(from_coord, to_coord).km for to_coord in coords] 
                         for from_coord in coords]
    
    return distance_matrix, coords

#
# dodać funkcje która pokaże koszt dla każdego samochodu
#

def calculate_path_costs(c: Chromosome, calculate_distance_method):
    path_costs = [0] * (NO_VEHICLES + 1)
    prev_stop = [0] * (NO_VEHICLES + 1)
    distance_matrix, data_matrix = calculate_distance_method(c) 

    for i in range(0, len(c.vehicles)):
        stop = c.stops.index(c.stops[i])  # the current stop index
        vehicle_no = c.vehicles[i]  # the current driver that stops for this customer
        dist = distance_matrix[prev_stop[vehicle_no]][stop]  # distance driver makes for this customer
        path_costs[vehicle_no] += dist
        prev_stop[vehicle_no] = stop

    # calculate costs for return to depot
    for i in range(0, len(prev_stop)):
        return_dist = distance_matrix[prev_stop[i]][0]
        path_costs[i] += return_dist

    return path_costs, data_matrix


# calculates the distance based on euclidean metric measurement
def _calc_dist(x1, y1, x2, y2):
    return math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


# print costs of a single iteration of the best chromosome
def print_cost(costs, iteration, runtime):
    print("Iteration: ", iteration, " runtime: ", "{:.4f}".format(runtime), ", costs: ", "{:.2f}".format(sum(costs)), sep="")
    return sum(costs)


# shows the phenotype of a chromosome
def print_phenotype(c: Chromosome, calculate_distance, *args, **kwargs):
    path_costs = calculate_path_costs(c, calculate_distance, *args, **kwargs)[0]
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
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for _ in range(NO_VEHICLES)]
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
    "gen_population", 
    "get_best_chromosome",
    "calculate_distance_matrix_dataframe_points", 
    "calculate_distance_matrix_geopy",
    "calculate_path_costs",
    "print_cost",
    "print_phenotype",
    "plot_map",
]