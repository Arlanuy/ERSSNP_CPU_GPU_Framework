#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.autoinit
import random

from gpu_struct import GPUStruct
from pycuda.compiler import SourceModule


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

    """)

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

    def run(self):
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


def get_every_rule(size_rssnp, rssnp_id): #size = the number of rules #rssnp_id = the rssnp id
    list_poss = []
    for i in rssnp_id:
        for j in range (0,size_rssnp[i[0]]):
            for k in range (0,size_rssnp[i[1]]):
                list_poss.append([i[0],i[1],j,k])
    
    return list_poss


def get_param(array,limit):             #limit - number of params to get array-the list of all possible params
    param = []
    # print(len(array),array)
    while(len(param)<limit):
        r = []
        j = 0
        temp = random.choice(array)
        # print("temp",temp)
        # f.write("temp "+ str(temp)+ '\n')
        for i in array:
            if((i[0] == temp[0] and i[1] == temp[1]) or (i[0] == temp[1] and i[1] == temp[0])):
                r.append(i)
                # f.write("remove " +str(i)+ '\n')
        
        for i in r:
            array.remove(i)
        print(len(array))
        # f.write(str(len(array)) + str(array) + '\n')
        param.append(temp)
        print(param)
        # f.write(str(param)+ '\n')

    return param


if __name__ == "__main__":
    # test = rule([1,1],[2,2],[1,1],[1,2],[0,0])
    # test.change()
    # test.run()
    test = []
    # f= open("text.txt","w+")


    source = []
    sink = []
    prod = []
    con = []
    delay = []
    size_rssnp = [20,20,20,20]

    for i in range(0,len(size_rssnp)):
        for i in range(0,size_rssnp[i]):
            source.append(1)
            sink.append(1)
            prod.append(1)
            con.append(1)
            delay.append(0)


    print(source)
    print(sink)
    print(prod)
    print(con)
    print(delay)

    # print(source[])

    array_size = create_size_array(size_rssnp)
    print(array_size)

    for i in array_size:
        print(source[i])
    test = rule(source,sink,prod,con,delay,array_size)


    test_list = get_every_poss(size_rssnp)
    test_list_rule = get_every_rule(size_rssnp,test_list)

    #print(test_list)

    # print(test_list_rule)

    param = get_param(test_list_rule,4)
    print(param)
    # print(param)
    # for x in param:
    #     print(x[0])

    # rssnp = np.array([0,1,2,3],dtype=np.int32)
    # rssnp2 = np.array([1,2,3,0],dtype=np.int32)
    # rule1 = np.array([1,1,1,1],dtype=np.int32)
    # rule2 = np.array([2,2,2,2],dtype=np.int32)
    # test.change(rule1,rule2,rssnp,rssnp2)
    #test.run()

    # test.append(rule([1,3],[2,2],[1,1],[1,2],[0,0]))
    #test.append(rule([1,0],[2,2],[1,1],[1,2],[0,0]))
    # print(test[0].source)

    # rssnp_test = []
    # rssnp_test.append(rssnp(test))
    # rssnp_test[0].run(0,1)
    # rssnp_test[0].run(0,0)
    # rssnp_test[0].change(0)
    # rssnp_test[0].run(0,1)
    # rssnp_test[0].run(0,0)
    #rssnp_test[0].run(1,0)
    # rssnp_test[0].run(0,1)


    # test = []
    # test.append(rule([1,1],[2,2],[1,1],[1,2],[0,0]))
    # test.append(rule([1,0],[2,2],[1,1],[1,2],[0,0]))
    # # test[0].run(0)
    # # test[1].run(1)
    # rssnp_test = rssnp(test)
    # rssnp_test.run(0)
    # rssnp_test.run(1)
