#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.autoinit

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

    __global__ void print_struct(Rule *a, int i){
        int idx = i;
        printf("source %d\\n", a->source[idx]);
        printf("sink %d\\n", a->sink[idx]);
        printf("prod %d\\n", a->prod[idx]);
        printf("con %d\\n", a->con[idx]);
        printf("delay %d\\n", a->delay[idx]);
    }

    __global__ void change_struct(Rule *a){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        a->source[idx] = a->source[idx] + 2;
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

    def run(self,i):
        func = mod.get_function('print_struct')

        self.ftmp_gpu.copy_to_gpu()

        func(self.ftmp_gpu.get_ptr(),np.int32(i),block= (1,1,1),grid =(1,1,1))

    def change(self):
        func = mod.get_function('change_struct')

        self.ftmp_gpu.copy_to_gpu()

        func(self.ftmp_gpu.get_ptr(),block= (2,1,1),grid =(1,1,1))
        
        self.ftmp_gpu.copy_from_gpu()

class rssnp:
    def __init__(self,rules):
        self.rules = rules

    def print_rule(self):
        self.rules.source_array

    def run(self,r,r1):
        self.rules[r].run(r1)

    def change(self,r):
        self.rules[r].change()

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


if __name__ == "__main__":
    # test = rule([1,1],[2,2],[1,1],[1,2],[0,0])
    # test.change()
    # test.run()
    test = []

    source = []
    sink = []
    prod = []
    con = []
    delay = []
    size_rssnp = [5,6,7,8]


    for i in range(0,len(size_rssnp)):
        for i in range(0,size_rssnp[i]):
            source.append(i)
            sink.append(i)
            prod.append(i)
            con.append(i)
            delay.append(0)


    print(source)
    print(sink)
    print(prod)
    print(con)
    print(delay)

    print(source[5])

    array_size = create_size_array(size_rssnp)
    print(array_size)

    for i in array_size:
        print(source[i])
    test = rule(source,sink,prod,con,delay,array_size)
    swap(1,2,0,3,test)
    swap(0,1,0,0,test)
    swap(2,3,0,3,test)
    swap(1,2,0,3,test)


    rssnp = [0,1,2,3]
    rssnp2 = [1,2,3,0]
    rssnp.random()
    print(rssnp)
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
