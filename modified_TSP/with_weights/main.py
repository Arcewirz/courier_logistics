from src.tsp import *
from src.utils import *
import pandas as pd
import numpy as np

def get_result_df(df, cars, capacity, depot):
    reduce_all(df, capacity)

    result_df = pd.DataFrame(columns=["driver", "queue", "distance"])
    
    for i in range(1, cars+1):
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


    route, length, tuple_seq = [], [], []
    for i in range(cars+1, int(df["cluster"].unique().max())+1):
        tuple_list = list(df.loc[df["cluster"] == i][['x', 'y']].to_records(index=False))
        tuple_list.append(tuple(depot))

        distance_points0 = find_distances(tuple_list)

        len1, r1 = repeat_neighbour(str(len(tuple_list) - 1), distance_points0)
        route.append(r1)
        length.append(len1)
        tuple_seq.append(tuple_list)

    for i in range(len(length)):
        driver_index = result_df.groupby('driver')['distance'].sum().idxmin()
        inde = length.index(max(length))
        max_len_idx = length.index(max(length))
        idx = np.array(route[inde], dtype = int)
        queue = np.array(tuple_seq[inde], dtype=object)[idx]
        queue_from_depot = np.concatenate((queue[idx.argmax():], queue[:idx.argmax()+1] ))

        result_df.loc[len(result_df)] = [driver_index, queue_from_depot, length[inde]]
        del length[max_len_idx]
        del tuple_seq[max_len_idx]
        del route[max_len_idx]

    return(result_df)


# Example
if __name__ == '__main__':
    r_choice = [1]*95 + [2]*45 + [3]*25 + [4]*10
    np.random.shuffle(r_choice)
    points = np.random.uniform(0, 10, size=(175,2))
    courier = 5
    depot = (5,5)
    capacity = 60
    df = df_rnn(courier, depot, points, r_choice)
    final_df = get_result_df(df, courier, capacity, depot)
    print("175 węzłów z planszy 10x10 \ncałkowita długość trasy modyfikacja TSP: ",final_df["distance"].sum())
