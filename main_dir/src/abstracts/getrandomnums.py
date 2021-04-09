#source came from talonmies in https://stackoverflow.com/questions/46169633/how-to-generate-random-number-inside-pycuda-kernel

import numpy as np
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray

def getrandom(N):

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

        __global__ void randfillkernel(float *values, int N)
        {
            int tidx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tidx < nstates) {
                curandState_t s = *states[tidx];
                for(int i=tidx; i < N; i += blockDim.x * gridDim.x) {
                    values[i] = curand_uniform(&s);
                }
                *states[tidx] = s;
            }
        }
        }
    """

    mod = SourceModule(code % { "NGENERATORS" : N }, no_extern_c=True)
    init_func = mod.get_function("initkernel")
    fill_func = mod.get_function("randfillkernel")

    seed = np.int32(123456789)
    nvalues = 10 * N
    init_func(seed, block=(N,1,1), grid=(1,1,1))
    gdata = gpuarray.zeros(nvalues, dtype=np.float32)
    fill_func(gdata, np.int32(nvalues), block=(N,1,1), grid=(1,1,1))
    return gdata