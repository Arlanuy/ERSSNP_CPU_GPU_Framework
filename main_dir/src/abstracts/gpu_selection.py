import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy, math, os
from src.abstracts.gpu_timer import *


MAXTHREADSIZE = 32

def assurepowtwo(divider):
    if divider & (divider - 1) == 0:

        return divider
    else:
        powertwo = math.ceil(math.exp(math.log(2) * math.ceil(math.log(divider, 2))))

        return powertwo


def adder(total_fitness_list, len_tf_gpu_list, blockSize):

    mod = SourceModule("""
        __device__ void warpReduce(volatile int *sdata, unsigned int tid, unsigned int blockSize) {
            if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
                if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
                if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
                if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
                if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
                if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; 
        }
        
        __global__ void sum(int *g_idata, int *g_odata, unsigned int blockSize)
        {
            extern __shared__ int sdata[];
            unsigned int tid = int(threadIdx.x);
            unsigned int i = blockIdx.x*(blockSize*2) + tid;
            unsigned int gridSize = blockSize*2*gridDim.x;
            sdata[tid] = 0;
            

            while (i < blockSize) 
            {
                sdata[tid] += g_idata[i] + g_idata[i+blockSize];  
                i += gridSize; 
            }


            __syncthreads();

            if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
            if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
            if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

            if (tid < 32) 
            {
                warpReduce(sdata, tid, blockSize);
            }
            __syncthreads();
            
            if (tid == 0) {
                g_odata[blockIdx.x] = sdata[0]; 
            }

            
        }
    """)

    
    sum_func = mod.get_function("sum")
    
    tf_gpu_list = drv.mem_alloc(total_fitness_list.size * total_fitness_list.dtype.itemsize)
    drv.memcpy_htod(tf_gpu_list , total_fitness_list)
    block_num = int(numpy.ceil(len_tf_gpu_list/blockSize))

    tf_gpu_list_output = numpy.zeros((block_num + 1), dtype=numpy.int32)
    tf_gpu_list_out = drv.mem_alloc(tf_gpu_list_output.size * tf_gpu_list_output.dtype.itemsize)
    timer_gpu = GpuTimer()
    timer_gpu.tic()

    sum_func(tf_gpu_list, tf_gpu_list_out, numpy.int32(blockSize), block=(blockSize,1,1),grid=(block_num,1,1), shared = blockSize * block_num * total_fitness_list.dtype.itemsize)
    timer_gpu.toc()
    timer_write("Selection", timer_gpu.time())
    drv.memcpy_dtoh(tf_gpu_list_output, tf_gpu_list_out)
    return tf_gpu_list_output

def init_tf_adder(tf_gpu_list, len_result):
    
    divider = int(numpy.ceil(len(tf_gpu_list)/2)) % MAXTHREADSIZE
    blockSize = assurepowtwo(divider)
    result = numpy.zeros(blockSize * 2, dtype=int)
    result [:len_result] = tf_gpu_list

    
    tf_sum = adder(result, len_result, blockSize)


    return tf_sum[0]




