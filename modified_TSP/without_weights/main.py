#!/usr/bin/env python
# coding: utf-8
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from algorithms import *


def df_rnn_sklearn(courier, depot, points):
    '''
    Create a data frame with point coordinates, 
    assign a cluster
    and calculates the distance to the center.
    '''
    x = list(map(itemgetter(0), points))
    y = list(map(itemgetter(1), points))
    
    kmeans = KMeans(n_clusters=courier, n_init='auto').fit(points)
    which_cluster = kmeans.labels_
            
    dist_to_depot = [distance.distance(points[i], depot).kilometers for i in range(len(points))]


    df = pd.DataFrame({
        "x": x,
        "y": y,
        "dist_to_depot": dist_to_depot,
        "cluster": which_cluster
    })

    return(df)



def df_rnn_atan(courier, depot, points):
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


def main_sklearn(courier, points, depot):
    df = df_rnn_sklearn(courier, depot, points) #!!!!!
    result_df = pd.DataFrame(columns=["driver", "queue", "distance"])
    
    for i in range(1, courier): #!!!!!!!!
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

def main_atan(courier, points, depot):
    df = df_rnn_atan(courier, depot, points)
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
    points = np.random.uniform(0, 10, size=(100,2))
    courier = 5
    depot = (5,5)
    df_sklearn = main_sklearn(courier, points, depot)
    df_atan = main_atan(courier, points, depot)
    print("sklearn: ",df_sklearn["distance"].sum(), "\natan: ", df_atan["distance"].sum())