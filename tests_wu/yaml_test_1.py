import yaml
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import numpy as np


mod = SourceModule("""
#include <stdlib.h>
__device__ int max1(int a,int b){
    if(a>b){
        return a;
    }
    else{
        return b;
    }
}

__global__ void lc_substring(int *X,int *Y,int *res, int *LCSuff,int row_width,int col_width){
   int j = blockIdx.x * blockDim.x + threadIdx.x;
   if (j < col_width){
        for (int i = 1; i < row_width; i++) {
            if (i == 0 || j == 0){
                LCSuff[i*col_width+j] = 0;
                __syncthreads();
            }
            else if (X[i-1] == Y[j-1]) {  
                LCSuff[i*col_width+j] = LCSuff[((i-1)*col_width)+j-1] + 1;
                __syncthreads();
                //printf("compare %d %d\\n",res[0],LCSuff[i*col_width+j]);
                //printf("res %d %d %d\\n",res[0],i,j);
                
            } 
            else{
                LCSuff[i*col_width+j] = max1(LCSuff[(i-1)*col_width+j],LCSuff[i*col_width+j-1]);
                __syncthreads();
            }
            
        }

    } 
}


""")


def conf_load(filename):
	with open(filename, 'r') as stream:
		try:
			ga_params = yaml.safe_load(stream)

			#getting generation zero from run zero
			generation_zero = ga_params['runs'][0]['generations'][0]
			# print(generation_zero)

			#getting rssnp index 0 in generation zero
			rssnp_zero = generation_zero['rssnp_chromosomes'][0]
			# print(rssnp_zero) 

			#getting out_pairs indexed 0 at rssnp indexed 0 in generation zero
			out_pairs_zero = rssnp_zero['out_pairs'][0]
			# print(out_pairs_zero)
			return generation_zero, rssnp_zero, out_pairs_zero
		except yaml.YAMLError as exc:
			print(exc)

filename = 'ga_conf_out.yaml'
generation ,rssnp , out_pairs = conf_load(filename)

print(generation)
print('\nrssnp')
print(rssnp)
print("\nout_pairs")
print(out_pairs)

for a in rssnp:
	print(a)

output1 = []
output2 = []
for a in out_pairs:
	output1.append(a[0])
for a in out_pairs:
	output2.append(a[1])

print(output1)
print(output2)


output1arr = np.asarray(output1)

LCS = mod.get_function("lc_substring")

a = np.array(output1[0],dtype=np.int32) #row the width
b = np.array([0,0,0,1],dtype=np.int32) #col

print(a)
# res = np.array([0],dtype=np.int32)
# LCSuff = np.zeros((a.size+1,b.size+1),dtype=np.int32)

# print(LCSuff[0])

# a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
# b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
# LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
# res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

# for a in out_pairs:

