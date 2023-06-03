import random

from ..utils import *
from .fitness import *

# do the crossover, implemented according to the order crossover
def do_crossover(parent1: Chromosome, parent2: Chromosome, calculate_distance_method):
    crossover_point_1 = random.randint(0, len(parent1.stops) - 1)
    crossover_point_2 = random.randint(0, len(parent1.stops) - 1)
    child_stops = [-1] * len(parent1.stops)
    used_values = []

    for i in range(min(crossover_point_1, crossover_point_2), max(crossover_point_1, crossover_point_2) + 1):
        child_stops[i] = parent1.stops[i]
        used_values.append(parent1.stops[i])

    available_values = [ele for ele in parent2.stops if ele not in used_values]

    for i in range(0, len(parent1.stops)):
        if child_stops[i] == -1:
            child_stops[i] = available_values.pop(0)

    child_1 = Chromosome(child_stops, parent1.vehicles.copy())
    child_2 = Chromosome(child_stops, parent2.vehicles.copy())

    evaluate_fitness(child_1, calculate_distance_method)
    evaluate_fitness(child_2, calculate_distance_method)

    if child_1.fitness > child_2.fitness:
        return child_1
    else:
        return child_2


__all__ = [
    "do_crossover"
]