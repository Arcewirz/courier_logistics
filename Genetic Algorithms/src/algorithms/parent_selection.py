import random

# select the parent using the roulette wheel selection
def select_parent(chromosomes):
    total_fitness = 0
    chroms_fitness = []
    for chrom in chromosomes:
        total_fitness += chrom.fitness
        chroms_fitness.append(chrom.fitness)

    # create the selection probabilities from the scaled fitness
    selection_probabilities = [f_s / total_fitness for f_s in chroms_fitness]

    selected_chrom = random.choices(chromosomes, weights=selection_probabilities)[0]

    return selected_chrom


__all__ = [
    "select_parent"
]