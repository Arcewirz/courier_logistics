import json, os, random
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim


def create_dataframe_points(xy_from=0, xy_to = 50, num_points = 175):
    """ Function for generation sample data on 2D space.

        Returns:
            list: two lists of tuples.
            First constains only x and y, second contains x, y and weight of parcel.
    """
    data = np.random.uniform(xy_from, xy_to, size=(num_points,2))
    depot = np.array([xy_to/2, xy_to/2])

    weights = [1]*int(0.5429*num_points) + [2]*int(0.2571*num_points) + [3]*int(0.1429*num_points) + [4]*int(0.0571*num_points)
    if len(weights) < num_points:
        weights.extend([1]*(num_points - len(weights)))
    np.random.shuffle(weights)
    weights = np.array([weights])

    X = np.concatenate((data, weights.T), axis=1)
    df = pd.DataFrame({
            "x": X[:][:,0],
            "y": X[:][:,1],
            "weight": X[:][:,2]     
            })
    df.iloc[0] = [depot[0], depot[1], 1.0]

    return [[(float(df.loc[index, 'x']), float(df.loc[index, 'y'])) for index, rowb in df.iterrows()],
            [(float(df.loc[index, 'x']), float(df.loc[index, 'y']), float(df.loc[index, 'weight'])) for index, rowb in df.iterrows()]]


def create_random_addresses(num_points = 175):
    """ Generates data for the problem. 
    
    Args:
        num_points (int): Number of orders
    
    Returns:
        list: Exact addresses of orders
    """
    os.chdir('..')
    os.chdir('Data')
    f = open('data.json', encoding="utf8")
    # returns JSON object as a dictionary
    data = json.load(f)

    street = random.choices(list(data.items()), k=num_points)
    return [i[0] + ' ' + random.choice(i[1]) for i in street]


def read_coords_geopy(address_list):
    """Find latitude and longitude of order.

    Args:
        address_list (list): Exact addresses of orders

    Returns:
        list: list of tuples that contains coordinates of order
    """
    app = Nominatim(user_agent="tmp")

    df = pd.DataFrame({"lat": [],
                       "lon": []})
    
    locations = [app.geocode(address) for address in address_list]

    for i in range(len(locations)):
        df.loc[len(df.index)] = [locations[i][1][0], locations[i][1][1]]

    return [(float(df.loc[index, 'lat']), float(df.loc[index, 'lon']))
            for index, rowb in df.iterrows()]


__all__ = [
    "create_dataframe_points",
    "create_random_addresses",
    "read_coords_geopy",
]

