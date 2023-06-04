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
# calculate_distance_method:   calculate_distance_matrix_dataframe_points/calculate_distance_matrix_geopy
def ga_solve(calculate_distance_method, curr_population):
    for chrom in curr_population:  
        src.evaluate_fitness(chrom, calculate_distance_method)

    for _ in range(0, src.NO_GENERATIONS):
        for _ in range(0, src.POPULATION_SIZE):
            new_population = []
            parent1 = src.select_parent(curr_population)

            if random.uniform(0, 1) < src.CROSSOVER_RATE:
                parent2 = src.select_parent(curr_population)
                child = src.do_crossover(parent1, parent2, calculate_distance_method)
            else:
                child = parent1

            src.do_mutation(child, calculate_distance_method)
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
        curr_population = src.gen_population(src.create_dataframe_points)

        # punkty w 2D z wagami
        # curr_population = src.create_dataframe_weighted_points()

        ## adresy w geopy
        # curr_population = src.create_random_addresses()

        start_time = time.time()
        chromosome = ga_solve(src.calculate_distance_matrix_dataframe_points, curr_population=curr_population)
        end_time = time.time()

        # punkty w 2D
        costs_i, data_matrix = src.calculate_path_costs(chromosome, src.calculate_distance_matrix_dataframe_points)
        # # adresy w geopy
        # costs_i, data_matrix = src.calculate_path_costs(chromosome, src.calculate_distance_matrix_geopy, curr_population)
        
        # use when multiple iterations
        # src.print_cost(costs_i, i + 1, end_time - start_time)
        src.print_phenotype(chromosome, src.calculate_distance_matrix_dataframe_points)
        total_cpu_time += end_time - start_time

        if chromosome.fitness > best_chromosome.fitness:
            best_chromosome = chromosome
            best_chrom_total_cost = sum(costs_i)
            best_chrom_runtime = end_time - start_time

    print("\nBest result in detail\n")
    print("Total CPU Time: ", "{:.2f}".format(total_cpu_time), "s", sep="")
    print("Total number of runs:", src.NO_EXPERIMENT_ITERATIONS)
    print("Runtime of the algorithm for the best solution: ", "{:.2f}".format(best_chrom_runtime), "s", sep="")

    src.print_phenotype(best_chromosome, src.calculate_distance_matrix_dataframe_points)
    src.print_single_vehicle_cost(costs_i)
    src.plot_map(best_chromosome, costs_i, data_matrix)