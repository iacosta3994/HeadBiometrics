import numpy as np
import statistics

def std_filter(list, idx, deviations_count=2):  # [[number,number,number], [number, number,number]...]
    idx_pool = []
    float_idx_pool = np.array(idx_pool, dtype=np.float32)

    for element in list:
        item = element[idx]
        if isinstance(item, float) == True:
            idx_pool.append(item)
        elif isinstance(item,int) == True:
            idx_pool.append(item)
        else:
            print("Must be float or int to perform standard_deviation")
    float_idx_pool = np.append(float_idx_pool, idx_pool)
    if len(float_idx_pool) > 2:
        standard_deviation = statistics.stdev(float_idx_pool)
        element_mean = [np.mean(float_idx_pool, axis=0)]

        final_list = [x for x in idx_pool if (x > np.subtract(element_mean, deviations_count * standard_deviation))]
        final_list = [x for x in final_list if (x < np.add(element_mean, deviations_count * standard_deviation))]
        return final_list
    else:
        return [0]
