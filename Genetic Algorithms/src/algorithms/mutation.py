import random

from .fitness import *
from ..constants import *
from ..utils import *

# does the mutation by swapping to random elements
def do_mutation(c: Chromosome, calculate_distance_method, depot_address):

    old_chrom = Chromosome(c.stops.copy(), c.vehicles.copy(), c.fitness)
    distance_matrix = calculate_distance_method(c)[0]

    for j in range(0, NO_OF_MUTATIONS):
        if random.uniform(0, 1) < MUTATION_RATE:
            rand = random.uniform(0, 1)
            if rand < 0.3:
                swap_gene_stops(c)
            if rand < 0.6:
                swap_gene_vehicles(c)
            else:
                two_opt_one_path(c, distance_matrix, depot_address)

        evaluate_fitness(c, calculate_distance_method)
        if c.fitness < old_chrom.fitness:
            c.stops = old_chrom.stops.copy()
            c.vehicles = old_chrom.vehicles.copy()
        else:
            old_chrom.stops = c.stops.copy()
            old_chrom.vehicles = c.vehicles.copy()
            old_chrom.fitness = c.fitness


# swaps two genes in the stops array
def swap_gene_stops(c: Chromosome):
    swapping_index_1 = random.randint(1, len(c.stops) - 1)
    swapping_index_2 = random.randint(1, len(c.stops) - 1)

    temp_stop = c.stops[swapping_index_1]
    c.stops[swapping_index_1] = c.stops[swapping_index_2]
    c.stops[swapping_index_2] = temp_stop


# swaps two genes in the vehicles array
def swap_gene_vehicles(c: Chromosome):
    swapping_index_1 = random.randint(0, len(c.stops) - 1)
    swapping_index_2 = random.randint(0, len(c.stops) - 1)

    temp_vehicle = c.vehicles[swapping_index_1]
    c.vehicles[swapping_index_1] = c.vehicles[swapping_index_2]
    c.vehicles[swapping_index_2] = temp_vehicle


# removes twists in the path of one vehicle in a chromosome
def two_opt_one_path(c: Chromosome, distance_matrix, depot_address):
    vehicle_to_check = random.randint(1, NO_VEHICLES)
    route = [depot_address]
    for i in range(0, len(c.vehicles)):
        if c.vehicles[i] == vehicle_to_check:
            route.append(c.stops[i])

    route.append(depot_address)
    route = two_opt_route(c, distance_matrix, route)
    route.pop(0)

    for i in range(0, len(c.vehicles)):
        if c.vehicles[i] == vehicle_to_check:
            c.stops[i] = route.pop(0)


# implements part of the 2-opt route algorithm to avoid twists in a route
def two_opt_route(c, distance_matrix, route):
    path_size = len(route)
    for i in range(1, path_size - 2):
        for j in range(i + 1, path_size):
            if j - i == 1:
                continue
            if _cost_route_change(c, distance_matrix, route[i - 1], route[i], route[j - 1], route[j]) < 0:
                route[i:j] = route[j - 1:i - 1:-1]
    return route


# calculates the costs if the route is changed with these four stops
def _cost_route_change(c, distance_matrix, stop1, stop2, stop3, stop4):
    return distance_matrix[c.stops.index(stop1)][c.stops.index(stop3)] + distance_matrix[c.stops.index(stop2)][c.stops.index(stop4)] - distance_matrix[c.stops.index(stop1)][c.stops.index(stop2)] - distance_matrix[c.stops.index(stop3)][c.stops.index(stop4)]


__all__ = [
    "do_mutation",
    "swap_gene_stops", 
    "swap_gene_vehicles",
    "two_opt_one_path",
    "two_opt_route"
]