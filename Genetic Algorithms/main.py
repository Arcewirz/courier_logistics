import random, time
import pandas as pd
import folium

# import sys, os

# # ~/.venv/Lib/site-packages
# path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.venv/Lib/site-packages'))
# print(path)
# if not path in sys.path:
#     sys.path.append(os.path.join(path,'src'))
# del path

# print(sys.path)

import src

# nieużywane
def ga_alg(liczba_kurierow: int, 
            wsp_punktu_startowego: tuple[float], 
            wsp_punktow_do_odwiedzenia: tuple[tuple[float]]) -> list[list[tuple]]:
    
    src.POPULATION_SIZE = len(wsp_punktow_do_odwiedzenia)
    src.NO_VEHICLES = liczba_kurierow

    best_chromosome = src.Chromosome([], [])
    wsp_punktow_do_odwiedzenia.insert(0, wsp_punktu_startowego)

    curr_population = src.gen_population_from_data(no_couriers=liczba_kurierow, addresses_to_visit=wsp_punktow_do_odwiedzenia)
    chromosome = ga_solve(src.calculate_distance_matrix_dataframe_points, curr_population=curr_population, depot_address=wsp_punktu_startowego)
    
    if chromosome.fitness > best_chromosome.fitness:
        best_chromosome = chromosome

    routes = []
    for i in range(1, src.NO_VEHICLES+1):
        route = [wsp_punktu_startowego]
        for j in range(0, len(chromosome.vehicles)):
            if chromosome.vehicles[j] == i:
                route.append(chromosome.stops[j])
        route.append(wsp_punktu_startowego)
        routes.append(route)
    
    return routes


# implements the Genetic Algorithm
# calculate_distance_method:   calculate_distance_matrix_dataframe_points/calculate_distance_matrix_geopy
def ga_solve(calculate_distance_method, curr_population, depot_address):
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

            src.do_mutation(child, calculate_distance_method, depot_address)
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

        # # punkty z Wrocławia

        # # depot_address = (51.11235095963508, 17.027098599384292)
        # depot_address = (51.1277617, 17.1070973)
        # addresses = pd.read_csv("addresses.csv")
        # addresses['tuples'] = list(zip(addresses['51.1277617'], addresses['17.1070973']))
        # addresses_to_visit = addresses['tuples'].to_list()
        # addresses_to_visit.insert(0, depot_address)
        # src.POPULATION_SIZE = len(addresses_to_visit)

        # curr_population = src.gen_population_from_data(no_couriers=src.NO_VEHICLES, addresses_to_visit=addresses_to_visit)

        # start_time = time.time()
        # chromosome = ga_solve(src.calculate_distance_matrix_geopy, curr_population=curr_population, depot_address=depot_address)
        # end_time = time.time()

        # costs_i, data_matrix = src.calculate_path_costs(chromosome, src.calculate_distance_matrix_geopy)

        # src.print_phenotype(best_chromosome, src.calculate_distance_matrix_geopy)
        # src.print_single_vehicle_cost(costs_i)

        # punkty 2D

        lista_adresow = pd.read_csv("stala_lista.csv")["punkty"].to_list()
        addresses_to_visit  = [eval(ele) for ele in lista_adresow]
        depot_address = (25.0, 25.0)
        src.POPULATION_SIZE = len(addresses_to_visit)

        curr_population = src.gen_population_from_data(no_couriers=src.NO_VEHICLES, addresses_to_visit=addresses_to_visit)

        start_time = time.time()
        chromosome = ga_solve(src.calculate_distance_matrix_dataframe_points, curr_population=curr_population, depot_address=depot_address)
        end_time = time.time()

        costs_i, data_matrix = src.calculate_path_costs(chromosome, src.calculate_distance_matrix_dataframe_points)
        
        # src.print_cost(costs_i, i + 1, end_time - start_time)
        # src.print_phenotype(chromosome, src.calculate_distance_matrix_dataframe_points)
        total_cpu_time += end_time - start_time

        if chromosome.fitness > best_chromosome.fitness:
            best_chromosome = chromosome
            best_chrom_total_cost = sum(costs_i)
            best_chrom_runtime = end_time - start_time

    print("\nBest result in detail\n")
    print("Total CPU Time: ", "{:.2f}".format(total_cpu_time), "s", sep="")

    # print("Total number of runs:", src.NO_EXPERIMENT_ITERATIONS)
    # print("Runtime of the algorithm for the best solution: ", "{:.2f}".format(best_chrom_runtime), "s", sep="")

    src.print_phenotype(best_chromosome, src.calculate_distance_matrix_dataframe_points)
    src.print_single_vehicle_cost(costs_i)



    # # Tworzenie mapy
    # m = folium.Map(location=depot_address, zoom_start=5) #im większy zoom start tym bliżej widać, location to bedzie nasz magazyn

    # rgb_color = ['blue', 'gray', 'red', 'green', 'pink', 'orange', 'darkpurple', 'white', 'lightblue', 'beige', 'darkgreen']

    # # Tworzenie liniowej trasy na podstawie koordynatów
    # for i in range(len(best_chromosome.stops)):
    #     folium.PolyLine(locations=final_df.iloc[i]["queue"], color=(rgb_color[final_df.iloc[i]["driver"] - 1])).add_to(m)

    #     for coord in final_df.iloc[i]["queue"][1:-1]:
    #         lat, lon = coord
    #         folium.Marker(location=[lat, lon], icon=folium.Icon(icon= "car", color=rgb_color[final_df.iloc[i]["driver"] - 1])).add_to(m)

    # lat, lon = final_df.iloc[0]["queue"][0]
    # folium.Marker(location=[lat, lon], icon=folium.Icon(icon = "home", color="black")).add_to(m)

    # m.save('mapa.html')

    src.plot_map(best_chromosome, costs_i, depot_address=depot_address)