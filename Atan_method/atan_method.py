#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import copy
from operator import itemgetter

def find_angle(point1, point2):
    '''
    Find the angle between two points.
    '''
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    angle_rad = np.math.atan2(delta_y, delta_x)
    angle_deg = np.math.degrees(angle_rad)

    return angle_deg


def df_rnn(courier, depot, points):
    '''
    Create a data frame with point coordinates, 
    assign a cluster
    and calculates the distance to the center.
    '''
    x = list(map(itemgetter(0), points))
    y = list(map(itemgetter(1), points))

    linspace = np.linspace(-180, 180, courier+1)
    degrees = [find_angle(x, depot) for x in points]
    which_cluster = [np.searchsorted(linspace, x) for x in degrees]
    
    dist_to_depot = [np.math.dist(points[i], depot) for i in range(len(points))]

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "dist_to_depot": dist_to_depot,
        "cluster": which_cluster
    })

    return(df)


  
## TSP algorithm

def cities_to_dict(tup):
    '''
    Create dictionary.
    '''
    cities = {}
    for i in range(len(tup)):
        cities[str(i)] = tup[i]
    return(cities)


def find_distances(tup):
    '''
    Calculate distances between points.
    '''
    cities = cities_to_dict(tup)
    distance_dict = {}

    for i, city in enumerate(cities):
        geo1 = cities[city]
        dist_dict_city = {}
        for j, dest in enumerate(cities):
            geo2 = cities[dest]
            if city != dest :
                dist_dict_city[dest] = np.math.dist(geo1, geo2)
        distance_dict[city] = dist_dict_city
    return distance_dict


def nearest_neighbour(node, dist_cities):
    '''
    Perform the nearest neighbors algorithm.
    '''
    path = [node]
    total_dist = 0
    c = copy.deepcopy(dist_cities)

    for i in range(0, (len(dist_cities)-1)):
        for x in path[:-1]: del c[path[-1]][x]  
        total_dist = total_dist + min(c[path[-1]].values())
        path.append(min(c[path[-1]], key = c[path[-1]].get))
        
    path.append(path[0])
    total_dist = total_dist + dist_cities[path[-1]][path[-2]]
    return([path, total_dist])


def repeat_neighbour(node, dist_cities):
    possible_routes = []
    total_dist = []

    for i in dist_cities.keys():
        trip, dist = nearest_neighbour(i, dist_cities)
        possible_routes.append(trip)
        total_dist.append(dist)
    route = min(total_dist)
    index = total_dist.index(min(total_dist))
    rnn = possible_routes[index]

    return(route, rnn)


def main(courier, points, depot):
    df = df_rnn(courier, depot, points)
    result_df = pd.DataFrame(columns=["driver", "queue", "distance"])
    
    for i in range(1, courier+1):
        # iteruje po wszystkich tych "dużych" grupach
        tuple_list = list(df.loc[df["cluster"] == i][['x', 'y']].to_records(index=False))
        tuple_list.append(depot)
        distance_points0 = find_distances(tuple_list)

        if len(tuple_list) == 1:
            result_df.loc[len(result_df)] = [i, np.array(depot), 0]

        else:
            len1, r1 = repeat_neighbour(str(len(tuple_list) - 1), distance_points0)
            idx = np.array(r1, dtype = int)
            queue = np.array(tuple_list, dtype=object)[idx]
            queue_from_depot = np.concatenate((queue[idx.argmax():], queue[:idx.argmax()+1] ))

            result_df.loc[len(result_df)] = [i, queue_from_depot, len1]

    return(result_df)


# Example
if __name__ == '__main__':
    points = np.random.uniform(0, 50, size=(100,2))
    courier = 5
    depot = (25,25)
    dff = main(courier, points, depot)
    print(dff.head())