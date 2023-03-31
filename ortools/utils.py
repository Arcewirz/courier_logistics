from geopy.distance import geodesic

def create_data_model(coords: list[tuple],
                      num_vehicles: int = 1,
                      depot: int = 0) -> dict:
    """Generates data for the problem.

    Args:
        coords (list[tuple]): List containing coordinates of locations on map
        num_vehicles (int): Number of vehicles in the fleet
        depot (int): Index of location where all vehicles start and end routes.

    Returns:
        dict: Data for the problem.
    """
    
    distance_matrix = [[geodesic(from_coord, to_coord).km for to_coord in coords] 
                         for from_coord in coords]
    
    data = {}
    data['distance_matrix'] = distance_matrix
    data['num_locations'] = len(coords)
    data['num_vehicles'] = num_vehicles
    data['depot'] = depot
    return data