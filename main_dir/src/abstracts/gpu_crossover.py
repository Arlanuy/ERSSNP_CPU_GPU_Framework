#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.autoinit
import random
from src.abstracts.getrandomnums import getrandom
from gpu_struct import GPUStruct
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

def getrandommax(parents, len_orig_parents):
    
    np_list = np.zeros(len_orig_parents, dtype=np.int32) 
    #np_list = numpy.random.randn(n,2).astype('int32')
    for z in range(len_orig_parents):
        if z < len(parents):
            maxi = len(parents[z]['system'].rule) - 1
            #print("maxi is ", maxi)
            np_list[z] = maxi
        else:
            np_list[z] = -1    
    return np_list
    #a_gpu = gpuarray.to_gpu(np_list)
    #a_gpu = cuda.mem_alloc(np_list.size * np_list.dtype.itemsize)
    #cuda.memcpy_htod(a_gpu, np_list)
    #return a_gpu

def crossover_gpu_defined(parents_size, len_orig_parents, num_crosses, prev_offspring, random_rule_parents_limit):
    #print("num parents is ", parents_size,"while len original parents is ", len_orig_parents)
    random_init_list = np.zeros(num_crosses,dtype=np.int32)
    #random_gpu = gpuarray.to_gpu(random_init_list) 
    

    #res = np.zeros((parents_size * parents_size, max_rules * max_rules),dtype=np.int32)
    #random_gpu = cuda.mem_alloc(random_init_list.size * random_init_list.dtype.itemsize)
    #cuda.memcpy_htod(random_gpu, random_init_list)
    random_fin_list = getrandom(parents_size, num_crosses, np.random.randint(123456789), random_rule_parents_limit)
    #cross = mod.get_function("get_every_poss_of_ruleswap")
    #cross(random_rule_parents_limit, random_gpu, block=(max_rules, max_rules,1), grid=(parents_size, parents_size,1))
    
    #cuda.memcpy_dtoh(res, res_gpu)
    random_fin_list.get(random_init_list)
    #print("res in crossover is ", random_init_list)
    #print("random limit is ", random_rule_parents_limit)
    #print("random_fin is ", random_init_list)
    #print("real random_fin is ", random_fin_list)
    return random_init_list











