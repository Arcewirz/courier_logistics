import random, time

from src import *


# implements the Genetic Algorithm
def ga_solve(data_generation_method, *args, **kwargs):
    curr_population = gen_population(data_generation_method, *args, **kwargs)
    for chrom in curr_population:
        evaluate_fitness(chrom)

    for i in range(0, NO_GENERATIONS):
        new_population = []
        for j in range(0, POPULATION_SIZE):
            parent1 = select_parent(curr_population)

            if random.uniform(0, 1) < CROSSOVER_RATE:
                parent2 = select_parent(curr_population)
                child = do_crossover(parent1, parent2)
            else:
                child = parent1

            do_mutation(child)
            evaluate_fitness(child)
            new_population.append(child)

        if KEEP_BEST:
            best = get_best_chromosome(curr_population)
            new_population[0] = best

        curr_population = new_population

    return get_best_chromosome(curr_population)


if __name__ == '__main__':

    best_chrom_runtime = 0
    best_chrom_total_cost = 0
    best_chromosome = Chromosome([], [])
    total_cpu_time = 0

    for i in range(0, NO_EXPERIMENT_ITERATIONS):
        start_time = time.time()
        chromosome = ga_solve(create_dataframe_points)
        end_time = time.time()

        costs_i, data_matrix = calculate_path_costs(chromosome)
        print_cost(costs_i, i + 1, end_time - start_time)
        print_phenotype(chromosome)
        total_cpu_time += end_time - start_time

        if chromosome.fitness > best_chromosome.fitness:
            best_chromosome = chromosome
            best_chrom_total_cost = sum(costs_i)
            best_chrom_runtime = end_time - start_time

    print("\nBest result in detail\n")
    print("Total CPU Time: ", "{:.2f}".format(total_cpu_time), "s", sep="")
    print("Total number of runs:", NO_EXPERIMENT_ITERATIONS)
    print("Runtime of the algorithm for the best solution: ", "{:.2f}".format(best_chrom_runtime), "s", sep="")

    print_phenotype(best_chromosome)
    plot_map(best_chromosome, data_matrix)