from vrp_viz import VRPPlot
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def algorytm_or_tools(liczba_kurierow: int, 
                      wsp_punktu_startowego: tuple[float], 
                      wsp_punktow_do_odwiedzenia: tuple[tuple[float]]) -> list[list[int]]:
    
    
    locations = [(wsp_punktu_startowego[0], wsp_punktu_startowego[1])] + wsp_punktow_do_odwiedzenia
    
    distances = [[distance_between_two_nodes(a, b) for a in locations]
                              for b in locations]

    data = _create_data_model(distances=distances, num_vehicles=liczba_kurierow)
    
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), 
                                           data['num_vehicles'], 
                                           data['depot'])
    
    routing = pywrapcp.RoutingModel(manager)

    solution = perform_TSP(data=data, manager=manager, routing=routing)
    
    return return_solution(data, manager, routing, solution, locations)


def perform_TSP(data, manager, routing):
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Return solution or print solution on console
    if solution:
        return solution

    
def distance_between_two_nodes(start: tuple, end: tuple):
    return int(((end[1]-start[1])**2+(end[0]-start[0])**2)**(1/2))


def _create_data_model(distances, num_vehicles=None):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = distances
    data['num_vehicles'] = 1 if num_vehicles == None else num_vehicles
    data['depot'] = 0
    return data


def return_solution(data, manager, routing, solution, locations):
    index = routing.Start(0)
    output = []
    #distances = []
    #total_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        single_output = []
        index = routing.Start(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            single_output.append(locations[manager.IndexToNode(index)])
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            
        single_output.append(locations[data['depot']])
        output.append(single_output)
        #distances.append(int(route_distance))
        #total_distance += int(route_distance)
     
    return output
    #return output, distances, total_distance
    