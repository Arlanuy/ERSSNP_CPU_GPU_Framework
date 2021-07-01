import numpy as np
import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.autoinit
import random
from src.abstracts.getrandomnums import getrandom
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

def getrandommax(parents, len_orig_parents):
    
    np_list = np.zeros(len_orig_parents, dtype=np.int32) 

    for z in range(len_orig_parents):
        if z < len(parents):
            maxi = len(parents[z]['system'].rule) - 1
            #print("maxi is ", maxi)
            np_list[z] = maxi
        else:
            np_list[z] = -1    
    return np_list


def crossover_gpu_defined(parents_size, len_orig_parents, num_crosses, prev_offspring, random_rule_parents_limit):

    random_init_list = np.zeros(num_crosses,dtype=np.int32)

    random_fin_list = getrandom(parents_size, num_crosses, np.random.randint(123456789), random_rule_parents_limit)

    random_fin_list.get(random_init_list)

    return random_init_list











