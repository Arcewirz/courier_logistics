from ..utils import *
from ..constants import *

# calculates the fitness of a chromosome, the higher the fitness the better is the chromosome
def evaluate_fitness(c: Chromosome):
    path_costs = calculate_path_costs(c)

    total_fitness = 0
    for i in range(0, len(path_costs)):
        f = path_costs[i]
        total_fitness += f

    c.fitness = 1 / total_fitness


__all__ = [
    "evaluate_fitness",
]