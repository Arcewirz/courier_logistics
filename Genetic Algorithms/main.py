import random, time

# import sys, os

# # ~/.venv/Lib/site-packages
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.venv/Lib/site-packages'))
# print(path)
# if not path in sys.path:
#     sys.path.append(os.path.join(path,'src'))
# del path

# print(sys.path)

import src


# implements the Genetic Algorithm
def ga_solve():
    for chrom in curr_population:
        src.evaluate_fitness(chrom)

    for i in range(0, src.NO_GENERATIONS):
        new_population = []
        for j in range(0, src.POPULATION_SIZE):
            parent1 = src.select_parent(curr_population)

            if random.uniform(0, 1) < src.CROSSOVER_RATE:
                parent2 = src.select_parent(curr_population)
                child = src.do_crossover(parent1, parent2)
            else:
                child = parent1

            src.do_mutation(child)
            src.evaluate_fitness(child)
            new_population.append(child)

        if src.KEEP_BEST:
            best = src.get_best_chromosome(curr_population)
            new_population[0] = best

        curr_population = new_population

    return src.get_best_chromosome(curr_population)


if __name__ == '__main__':

    best_chrom_runtime = 0
    best_chrom_total_cost = 0
    best_chromosome = src.Chromosome([], [])
    total_cpu_time = 0

    for i in range(0, src.NO_EXPERIMENT_ITERATIONS):
        
        # punkty w 2D
        curr_population = src.create_dataframe_points()

        ## adresy w geopy
        # curr_population = src.create_dataframe_points()

        start_time = time.time()
        chromosome = ga_solve()
        end_time = time.time()

        # punkty w 2D
        costs_i, data_matrix = src.calculate_path_costs(chromosome, src.calculate_map_context, curr_population)
        # # adresy w geopy
        # costs_i, data_matrix = src.calculate_path_costs(chromosome, src.calculate_distance_matrix_geopy, curr_population)
        
        src.print_cost(costs_i, i + 1, end_time - start_time)
        src.print_phenotype(chromosome, src.calculate_map_context, curr_population)
        total_cpu_time += end_time - start_time

        if chromosome.fitness > best_chromosome.fitness:
            best_chromosome = chromosome
            best_chrom_total_cost = sum(costs_i)
            best_chrom_runtime = end_time - start_time

    print("\nBest result in detail\n")
    print("Total CPU Time: ", "{:.2f}".format(total_cpu_time), "s", sep="")
    print("Total number of runs:", src.NO_EXPERIMENT_ITERATIONS)
    print("Runtime of the algorithm for the best solution: ", "{:.2f}".format(best_chrom_runtime), "s", sep="")

    src.print_phenotype(best_chromosome)
    src.plot_map(best_chromosome, data_matrix)