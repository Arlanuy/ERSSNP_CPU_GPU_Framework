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
    def __init__(self,source,sink,prod,con,delay):
        self.source, self.sink, self.prod, self.con, self.delay = source, sink, prod, con, delay

        self.source_array = np.array(self.source,dtype=np.int32) 
        self.sink_array   = np.array(self.sink,dtype=np.int32) 
        self.prod_array   = np.array(self.prod,dtype=np.int32) 
        self.con_array    = np.array(self.con,dtype=np.int32) 
        self.delay_array  = np.array(self.delay,dtype=np.int32) 

        self.ftmp_gpu = GPUStruct([
            (np.int32,'*source',self.source_array),
            (np.int32,'*sink',self.sink_array),
            (np.int32,'*prod',self.prod_array),
            (np.int32,'*con',self.con_array),
            (np.int32,'*delay',self.delay_array)
            ])

        self.ftmp = np.zeros_like(self.ftmp_gpu)

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

if __name__ == "__main__":
    # test = rule([1,1],[2,2],[1,1],[1,2],[0,0])
    # test.change()
    # test.run()
    test = []
    test.append(rule([1,3],[2,2],[1,1],[1,2],[0,0]))
    test.append(rule([1,0],[2,2],[1,1],[1,2],[0,0]))

    rssnp_test = []
    rssnp_test.append(rssnp(test))
    rssnp_test[0].run(0,1)
    rssnp_test[0].run(0,0)
    rssnp_test[0].change(0)
    rssnp_test[0].run(0,1)
    rssnp_test[0].run(0,0)
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
