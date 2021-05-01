#source came from talonmies in https://stackoverflow.com/questions/46169633/how-to-generate-random-number-inside-pycuda-kernel

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

def getrandom(N, num_crosses, random_seed, max_limit):

    code = """
        #include <curand_kernel.h>

        const int nstates = %(NGENERATORS)s;
        __device__ curandState_t* states[nstates];

        extern "C" {

        __global__ void initkernel(int seed)
        {
            int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < nstates) {
                curandState_t* s = new curandState_t;
                if (s != 0) {
                    curand_init(seed, tidx, 0, s);
                }

                states[tidx] = s;
            }
        }

        __global__ void randfillkernel(float *values, int *values_int, int N, int* max_lim, int* min_lim)
        {
            int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < nstates) {
                curandState_t s = *states[tidx];
                for(int i=tidx; i < N; i += blockDim.x * gridDim.x) {
                    values[i] = curand_uniform(&s);
                    
                    values[i] *= (max_lim[tidx] - min_lim[tidx] + 0.999999);
                    values[i] += min_lim[tidx];
                    values_int[i] = __float2int_rz(values[i]); 
                }
                *states[tidx] = s;
            }
        }
        }
    """

    mod = SourceModule(code % { "NGENERATORS" : num_crosses }, no_extern_c=True)
    init_func = mod.get_function("initkernel")
    fill_func = mod.get_function("randfillkernel")

    
    
    
    init_func(np.int32(random_seed), block=(num_crosses,1,1), grid=(1,1,1))
    gdata_int = gpuarray.zeros(num_crosses, dtype=np.int32)
    gdata = gpuarray.zeros(num_crosses, dtype=np.float32)
    min_lim = gdata_int #same minimum limit of all zeros
    max_lim = gpuarray.to_gpu(max_limit)
    #max_lim = max_limit
    fill_func(gdata, gdata_int, np.int32(N), max_lim, min_lim, block=(num_crosses,1,1), grid=(1,1,1))
    return gdata_int

print(getrandom(10, 20, 1234, np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])))
