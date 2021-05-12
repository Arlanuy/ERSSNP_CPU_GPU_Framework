import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy
BLOCKSIZE = 8

#extern __shared__ int sdata[];
#volatile int* sdata;
#blockDim.x
def adder(total_fitness_list, len_tf_gpu_list):

    mod = SourceModule("""
        __device__ void warpReduce(volatile int *sdata, unsigned int tid, unsigned int blockSize) {
            if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
                if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
                if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
                if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
                if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
                if (blockSize >= 2) {sdata[tid] += sdata[tid + 1]; printf("new sdata is %d while adjacent is %d at tid %d", sdata[tid], sdata[tid + 1], tid);}
        }
        
        __global__ void sum(int *g_idata, int *g_odata, unsigned int n)
        {
            unsigned int blockSize = 2;
            extern __shared__ int sdata[];
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*(blockSize*2) + tid;
            unsigned int gridSize = blockSize*2*gridDim.x;
            sdata[tid] = 0;

            while (i < n) 
            {
                printf("i is %d at %d\\n", i, tid); 
                sdata[tid] += g_idata[i] + g_idata[i+blockSize];
                printf("sdata:tid is %d with addition of %d and %d\\n", sdata[tid], g_idata[i], g_idata[i+blockSize]);  
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
            printf("sdata tid value is %d at tid %d\\n", sdata[tid], tid);
            if (tid == 0) {
                if (blockIdx.x == 1) {
                    printf("blockidx is 1\\n");
                    printf("\\n g_odata are %d and sdata are %d\\n", g_odata[blockIdx.x], sdata[0]);
                }
                g_odata[blockIdx.x] = sdata[0]; 
                printf("godata is %d from sdata %d at block %d\\n", g_odata[blockIdx.x], sdata[0], blockIdx.x); 
            }
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
    sum_func(tf_gpu_list, tf_gpu_list_out, numpy.int32(BLOCKSIZE), block=(BLOCKSIZE,1,1),grid=(thread_num,1,1))
    print("output is ", tf_gpu_list_output, " while out is  ", tf_gpu_list_out)
    drv.memcpy_dtoh(tf_gpu_list_output, tf_gpu_list_out)
    return tf_gpu_list_output

tf_gpu_list = numpy.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
tf_sum = adder(tf_gpu_list, 16)
print("tf sum is ", tf_sum)






