import random

from .fitness import *
from ..constants import *
from ..utils import *

# does the mutation by swapping to random elements
def do_mutation(c: Chromosome):

    old_chrom = Chromosome(c.stops.copy(), c.vehicles.copy(), c.fitness)

    for j in range(0, NO_OF_MUTATIONS):
        if random.uniform(0, 1) < MUTATION_RATE:
            rand = random.uniform(0, 1)
            if rand < 0.5:
                swap_gene_stops(c)
            else:
                swap_gene_vehicles(c)

        evaluate_fitness(c)
        if c.fitness < old_chrom.fitness:
            c.stops = old_chrom.stops.copy()
            c.vehicles = old_chrom.vehicles.copy()
        else:
            old_chrom.stops = c.stops.copy()
            old_chrom.vehicles = c.vehicles.copy()
            old_chrom.fitness = c.fitness


# swaps two genes in the stops array
def swap_gene_stops(c: Chromosome):
    swapping_index_1 = random.randint(0, len(c.stops) - 1)
    swapping_index_2 = random.randint(0, len(c.stops) - 1)

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


__all__ = [
    "do_mutation",
    "swap_gene_stops", 
    "swap_gene_vehicles",
]