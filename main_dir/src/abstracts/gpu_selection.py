import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy
BLOCKSIZE = 8


def adder(total_fitness_list, len_tf_gpu_list):

    mod = SourceModule("""
        
        
        __global__ void sum(int *g_idata, int *g_odata, unsigned int n)
        {
            unsigned int blockSize = blockDim.x;
            extern __shared__ int sdata[];
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*(blockSize*2) + tid;
            unsigned int gridSize = blockSize*2*gridDim.x;
            sdata[tid] = 0;

            while (i < n) 
            {
                printf("i is %d", i); 
                sdata[tid] += g_idata[i] + g_idata[i+blockSize]; 
                i += gridSize; 
            }
            printf("n is %d", n);

            __syncthreads();

            if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
            if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
            if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

            if (tid < 32) 
            {
                if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
                if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
                if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
                if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
                if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
                if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
            }

            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }
    """)

    sum_func = mod.get_function("sum")
    tf_gpu_list = drv.mem_alloc(total_fitness_list.size * total_fitness_list.dtype.itemsize)
    drv.memcpy_htod(tf_gpu_list , total_fitness_list)

    thread_num = int(numpy.ceil(len_tf_gpu_list/BLOCKSIZE))
    print("thred num is ", thread_num)
    tf_gpu_list_output = numpy.zeros((thread_num + 1), dtype=numpy.int32)
    tf_gpu_list_out = drv.mem_alloc(tf_gpu_list_output.size * tf_gpu_list_output.dtype.itemsize)

    print("lengths are ", len(total_fitness_list), " and out is ", len(tf_gpu_list_output))
    sum_func(tf_gpu_list, tf_gpu_list_out, numpy.int32(thread_num), block=(BLOCKSIZE,1,1),grid=(thread_num,1,1))
    print("output is ", tf_gpu_list_output, " while out is  ", tf_gpu_list_out)
    drv.memcpy_dtoh(tf_gpu_list_output, tf_gpu_list_out)
    return tf_gpu_list_output

tf_gpu_list = numpy.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
tf_sum = adder(tf_gpu_list, 16)
print("tf sum is ", tf_sum)






