import numpy as np
import statistics

def std_filter(list, idx):  # [[number,number,number], [number, number,number]...]
    idx_pool = []
    float_idx_pool = np.array(idx_pool, dtype=np.float32)

    for element in list:
        item = element[idx]
        if isinstance(item, str) == True:
            print("item in element must be an int or float")
            print(str(item))
            break
        idx_pool.append(item)
    float_idx_pool = np.append(float_idx_pool, idx_pool)

    standard_deviation = statistics.stdev(idx_pool)
    element_mean = [np.mean(float_idx_pool, axis=0)]

    final_list = [x for x in idx_pool if (x > np.subtract(element_mean, 2 * standard_deviation))]
    final_list = [x for x in final_list if (x < np.add(element_mean, 2 * standard_deviation))]

    return final_list
