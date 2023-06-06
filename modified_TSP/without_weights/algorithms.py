#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import copy
from operator import itemgetter
from geopy import distance


def find_angle(point1, point2):
    '''
    Find the angle between two points.
    '''
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    angle_rad = np.math.atan2(delta_y, delta_x)
    angle_deg = np.math.degrees(angle_rad)

    return angle_deg



  
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
                dist_dict_city[dest] = distance.distance(geo1, geo2).kilometers
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
