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
            np_list[z] = maxi
        else:
            np_list[z] = -1    
    return np_list
    #a_gpu = gpuarray.to_gpu(np_list)
    #a_gpu = cuda.mem_alloc(np_list.size * np_list.dtype.itemsize)
    #cuda.memcpy_htod(a_gpu, np_list)
    return a_gpu

def getrandom2(parents):
    n = len(parents)
    np_list = [[0 for x in range(n)] for y in range(2)] 
    #np_list = numpy.random.randn(n,2).astype('int32')
    for z in range(n):
        mini = 0
        maxi = len(parents[z].rule)
        np_list[z][0] = numpy.random.randint(mini, maxi)
        np_list[z][1] = numpy.random.randint(mini, maxi)
    a_gpu = gpuarray.to_gpu(np_list)
    return a_gpu


mod = SourceModule("""
    struct Rule{
        int *source;
        int *sink;
        int *prod;
        int *con;
        int *delay;
        int *size_rssnp;
    };

    __global__ void print_struct(Rule *a){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        printf("source %d\\n", a->source[idx]);
        printf("sink %d\\n", a->sink[idx]);
        printf("prod %d\\n", a->prod[idx]);
        printf("con %d\\n", a->con[idx]);
        printf("delay %d\\n", a->delay[idx]);
        printf("size %d\\n", a->size_rssnp[0]);
    }

    __global__ void swap_struct(Rule *t, int *r1, int *r2, int *p1, int *p2){
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;
        printf("source %d %d %d %d\\n", r1[tidx],r2[tidx],p1[tidx],p2[tidx]);


        int idx = t->size_rssnp[p1[tidx]]+r1[tidx];
        int idy = t->size_rssnp[p2[tidx]]+r2[tidx];

        printf("idx: %d idy: %d\\n",idx,idy);

        int temp_source = t->source[idx];
        int temp_sink   = t->sink[idx];
        int temp_prod   = t->prod[idx];
        int temp_con    = t->con[idx];
        int temp_delay  = t->delay[idx];

        t->source[idx] = t->source[idy];
        t->sink[idx]   = t->sink[idy];
        t->prod[idx]   = t->prod[idy];
        t->con[idx]    = t->con[idy];
        t->delay[idx]  = t->delay[idy];

        t->source[idy] = temp_source;
        t->sink[idy]   = temp_sink;
        t->prod[idy]   = temp_prod;
        t->con[idy]    = temp_con;
        t->delay[idy]  = temp_delay;
    }
    //1d grid of 2d blocks of size 4 x 4 blocks in a grid and 20 threads in both x and y dimension 
    __global__ void get_every_poss_of_ruleswap2(int* res, int* random_rules) {
        int tidx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
        int size_tuple = 4;

        if ((blockIdx.x != blockIdx.y) && (threadIdx.x < random_rules[blockIdx.x]) && (threadIdx.y < random_rules[blockDim.x + blockIdx.y])) {
            printf("dimensions are %d %d %d %d", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
            res[tidx * size_tuple] = blockIdx.x;
            res[tidx * size_tuple + 1] = blockIdx.y;
            res[tidx * size_tuple + 2] = threadIdx.x;
            res[tidx * size_tuple + 3] = threadIdx.y;
            printf("random rules are %d and %d", random_rules[res[tidx * size_tuple + 2]],  random_rules[res[tidx * size_tuple + 3]]);
            //printf("passed here with parents %d %d", res[tidx * size_tuple], res[tidx * size_tuple + 1]);            
            //printf("passed here with rules %d %d", res[tidx * size_tuple + 2], res[tidx * size_tuple + 3]);
     
        }
       

    }

    //1d grid of 2d blocks of size 4 x 4 blocks in a grid and 20 threads in both x and y dimension  int tidz = threadIdx.z + blockIdx.z * blockDim.z;
    __global__ void get_every_poss_of_ruleswap(int* random_rules_limit, int* random_gpu) {
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;;
        int tidy = threadIdx.y + blockIdx.y * blockDim.y;
        
        //int id = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


        //printf("gpu: x = %d, y = %d", tidx, tidy);

        if (blockIdx.x != blockIdx.y) {
            //printf("block: x = %d, block y = %d", blockIdx.x, blockIdx.y); 
            if (threadIdx.x < random_rules_limit[blockIdx.x]) {
                random_gpu[blockIdx.x] = threadIdx.x;
            }

            if (threadIdx.y < random_rules_limit[blockIdx.y]) {
                random_gpu[blockDim.x + blockIdx.y] = threadIdx.y;
            }

        }
       

    }


    """)

def crossover_gpu_defined(parents_size, len_orig_parents, num_crosses, prev_offspring):
    max_rules = 20
    print("num parents is ", parents_size, " while max rules is ", max_rules, " while len original parents is ", len_orig_parents)
    random_init_list = np.zeros(num_crosses,dtype=np.int32)
    #random_gpu = gpuarray.to_gpu(random_init_list) 
    random_rule_parents_limit = getrandommax(prev_offspring, len_orig_parents)

    #res = np.zeros((parents_size * parents_size, max_rules * max_rules),dtype=np.int32)
    #random_gpu = cuda.mem_alloc(random_init_list.size * random_init_list.dtype.itemsize)
    #cuda.memcpy_htod(random_gpu, random_init_list)
    random_fin_list = getrandom(parents_size, num_crosses, np.random.randint(1), random_rule_parents_limit)
    #cross = mod.get_function("get_every_poss_of_ruleswap")
    #cross(random_rule_parents_limit, random_gpu, block=(max_rules, max_rules,1), grid=(parents_size, parents_size,1))
    
    #cuda.memcpy_dtoh(res, res_gpu)
    random_fin_list.get(random_init_list)
    #print("res in crossover is ", random_init_list)
    print("random limit is ", random_rule_parents_limit)
    print("random_fin is ", random_init_list)
    return random_init_list

def get_every_poss(size_rssnp):
    list_poss = []
    for i in range(0,len(size_rssnp)):
        for j in range (0,len(size_rssnp)):
            if(i != j):
                list_poss.append([i,j])
    
    return list_poss

def get_every_rule(size_rssnp, rssnp_id): #size = the number of rules #rssnp_id = the rssnp id
    list_poss = []
    for i in rssnp_id:
        for j in range (0,size_rssnp[i[0]]):
            for k in range (0,size_rssnp[i[1]]):
                list_poss.append([i[0],i[1],j,k])
    
    return list_poss



class rule:
    def __init__(self,source,sink,prod,con,delay,size_rssnp):
        self.source, self.sink, self.prod, self.con, self.delay, self.size_rssnp = source, sink, prod, con, delay , size_rssnp

        self.source_array = np.array(self.source,dtype=np.int32) 
        self.sink_array   = np.array(self.sink,dtype=np.int32) 
        self.prod_array   = np.array(self.prod,dtype=np.int32) 
        self.con_array    = np.array(self.con,dtype=np.int32) 
        self.delay_array  = np.array(self.delay,dtype=np.int32)
        self.size_rssnp_array  = np.array(self.size_rssnp,dtype=np.int32)  

        self.ftmp_gpu = GPUStruct([
            (np.int32,'*source',self.source_array),
            (np.int32,'*sink',self.sink_array),
            (np.int32,'*prod',self.prod_array),
            (np.int32,'*con',self.con_array),
            (np.int32,'*delay',self.delay_array),
            (np.int32,'*size_rssnp',self.size_rssnp_array)
            ])

    def print_rule(self):
        func = mod.get_function('print_struct')

        self.ftmp_gpu.copy_to_gpu()

        func(self.ftmp_gpu.get_ptr(),block= (26,1,1),grid =(1,1,1))

    def change(self,r1,r2,p1,p2):
        func = mod.get_function('swap_struct')

        self.ftmp_gpu.copy_to_gpu()

        r1_gpu = cuda.mem_alloc(r1.size * r1.dtype.itemsize)
        r2_gpu = cuda.mem_alloc(r2.size * r2.dtype.itemsize)
        p1_gpu = cuda.mem_alloc(p1.size * p1.dtype.itemsize)
        p2_gpu = cuda.mem_alloc(p2.size * p2.dtype.itemsize)

        cuda.memcpy_htod(r1_gpu, r1)
        cuda.memcpy_htod(r2_gpu, r2)
        cuda.memcpy_htod(p1_gpu, p1)
        cuda.memcpy_htod(p2_gpu, p2)

        func(self.ftmp_gpu.get_ptr(),r1_gpu,r2_gpu,p1_gpu,p2_gpu,block= (4,1,1),grid =(1,1,1))
        
        self.ftmp_gpu.copy_from_gpu()
        print(self.ftmp_gpu)

# class rssnp:
#     def __init__(self,rules):
#         self.rules = rules

#     def print_rule(self):
#         self.rules.source_array

#     def run(self,r,r1):
#         self.rules[r].run(r1)

#     def change(self,r):
#         self.rules[r].change()

def create_size_array(size_rssnp):
    array = [0]
    for i in range(0,len(size_rssnp)-1):
        array.append(array[i]+size_rssnp[i])

    return array

def swap (rssnp_1, rssnp_2, rule_1, rule_2,t):
    print()
    idx = t.size_rssnp[rssnp_1]+rule_1
    idy = t.size_rssnp[rssnp_2]+rule_2

    print(idx,idy)
    print(t.source)
    print(t.sink)
    print(t.prod)
    print(t.con)
    print(t.delay)

    temp_source = t.source[idx]
    temp_sink   = t.sink[idx]
    temp_prod   = t.prod[idx]
    temp_con    = t.con[idx]
    temp_delay  = t.delay[idx]

    t.source[idx] = t.source[idy]
    t.sink[idx]   = t.sink[idy]
    t.prod[idx]   = t.prod[idy]
    t.con[idx]    = t.con[idy]
    t.delay[idx]  = t.delay[idy]

    t.source[idy] = temp_source
    t.sink[idy]   = temp_sink
    t.prod[idy]   = temp_prod
    t.con[idy]    = temp_con
    t.delay[idy]  = temp_delay

    print("new rssnp")
    print(t.source)
    print(t.sink)
    print(t.prod)
    print(t.con)
    print(t.delay)





