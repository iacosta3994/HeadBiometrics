import numpy as np
import statistics

def std_filter(list, idx): #[[number,number,number], [number, number,number]...]
    idx_pool = []
    for element in list:
        for item in element[idx]:
            if isinstance(item,str) == True:
                print("item in element must be an int or float")
                break
            idx_pool_variance.append(item)
    standard_deviation = statistics.stdev(idx_pool)
    element_mean = [np.mean(list[idx], axis=0)]

    final_list = [x for x in idx_pool if (x > element_mean - 2 * standard_deviation)]
    final_list = [x for x in final_list if (x < element_mean + 2 * standard_deviation)]

    return final_list
