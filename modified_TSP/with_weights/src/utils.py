import numpy as np
import pandas as pd
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


def too_heavy(df, cluster_number, capacity):
    '''
    Checks if the cluster capacity is smaller than the vehicle's capacity.
    '''
    if df.loc[(df.cluster == cluster_number)].weight.sum() <= capacity:
        return(False)
    else:
        return(True)
    

def reduce(df, capacity):
    '''
    Reduce cluster capacity.
    '''
    j = 0
    max_cluster = df.cluster.unique().max()
    for i in range(int(df["cluster"].unique().max())):
        if too_heavy(df, i, capacity):
            j = j + 1
            index = []
            while df.loc[(df.cluster == i)]["weight"].sum() > capacity:
                index_min = df.loc[(df.cluster == i)]["dist_to_depot"].idxmin()
                df.loc[index_min, "cluster"] = max_cluster + j
    if j == 0:
        return(True)
    

def reduce_all(df, capacity):
    '''
    Reduce all clusters.
    '''
    W = reduce(df, capacity)
    while W != True:
        W = reduce(df, capacity) 


def df_rnn(courier, depot, points, weights):
    '''
    Create a data frame with point coordinates, weights
    assign a cluster
    and calculates the distance to the center.
    '''
    x = list(map(itemgetter(0), points))
    y = list(map(itemgetter(1), points))

    linspace = np.linspace(-180, 180, courier+1)
    degrees = [find_angle(x, depot) for x in points]
    which_cluster = [np.searchsorted(linspace, x) for x in degrees]
    
    dist_to_depot = [distance.distance(points[i], depot).kilometers for i in range(len(points))]

    df = pd.DataFrame({
        "x" : x,
        "y" : y,
        "weight" : weights,
        "dist_to_depot" : dist_to_depot,
        "cluster" : which_cluster
    })
    return(df)