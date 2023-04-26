import csv, math, random, time, sys
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from typing import List

NO_GENERATIONS = 100
POPULATION_SIZE = 45
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.35
NO_OF_MUTATIONS = 5
KEEP_BEST = True

NO_VEHICLES = 3
NO_EXPERIMENT_ITERATIONS = 1000


@dataclass
class Chromosome:
    stops: List[int] = field(default_factory=list)
    vehicles: List[int] = field(default_factory=list)
    fitness: int = 0


# generates a single chromosome
def gen_chromosome():
    # generates a random sequence of customers and a corresponding array which vehicle stops at this customer
    _stops = list(range(0, 45))
    random.shuffle(_stops)
    _vehicles = []
    for i in range(0, len(_stops)):
        _vehicles.append(random.randint(0, 2))

    return Chromosome(_stops.copy(), _vehicles.copy())


# generates the initial population
def gen_population():
    _chromosomes = []
    for i in range(0, POPULATION_SIZE):
        _chromosomes.append(gen_chromosome())
    return _chromosomes


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


# calculates the fitness of a chromosome, the higher the fitness the better is the chromosome
def evaluate_fitness(c: Chromosome):
    path_costs = calculate_path_costs(c)

    total_fitness = 0
    for i in range(0, len(path_costs)):
        f = path_costs[i]
        total_fitness += f

    c.fitness = 1 / total_fitness


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


# do the crossover, implemented according to the order crossover
def do_crossover(parent1: Chromosome, parent2: Chromosome):
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

    evaluate_fitness(child_1)
    evaluate_fitness(child_2)

    if child_1.fitness > child_2.fitness:
        return child_1
    else:
        return child_2


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


# returns the best chromosome in a population
def get_best_chromosome(population):
    max_fitness = - sys.maxsize
    best_chrom = Chromosome([], [])
    for c in population:
        if c.fitness > max_fitness:
            max_fitness = c.fitness
            best_chrom = c
    return best_chrom


# implements the Genetic Algorithm
def ga_solve():
    curr_population = gen_population()
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


# calculates the distance based on euclidean metric measurement
def calc_dist(x1, y1, x2, y2):
    return math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))


# creates the distance matrix
def calculate_map_context():
    file = open("data.csv")
    csvreader = csv.reader(file)
    next(csvreader)  # skip header
    rows = []
    for row in csvreader:
        rows.append(list(map(int, row)))

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


# plot the routes of the vehicles of a chromosome as a map
def plot_map(c: Chromosome, data):
    x_data = [d[2] for d in data]
    y_data = [d[3] for d in data]
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


if __name__ == '__main__':

    distance_matrix, data_matrix = calculate_map_context()

    best_chrom_runtime = 0
    best_chrom_total_cost = 0
    best_chromosome = Chromosome([], [])
    total_cpu_time = 0

    for i in range(0, NO_EXPERIMENT_ITERATIONS):
        start_time = time.time()
        chromosome = ga_solve()
        end_time = time.time()

        costs_i = calculate_path_costs(chromosome)
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