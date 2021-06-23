import pycuda.autoinit
import pycuda.driver as drv
import numpy, math, os

from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray as pg

from pycuda import gpuarray
from src.abstracts.gpu_timer import *

#Source in Stackoverflow with question how to properly copy a gpuarray (longstanding bug)
def gpuarray_copy(array: gpuarray.GPUArray):
    array_copy = array.copy()
    array_copy.strides = array.strides
    array_copy.flags.f_contiguous = array.flags.f_contiguous
    array_copy.flags.c_contiguous = array.flags.c_contiguous
    array_copy.flags.forc = array.flags.forc



def GPUlcs(output_dataset, output_spike_train, len_dataset):
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

    __global__ void lc_subsequence(int *X,int *Y, int *LCSuff,int row_width,int col_width){
       int j = blockIdx.x * blockDim.x + threadIdx.x;
       if (j < col_width){
            for (int i = 0; i < row_width; i++) {
                if (i == 0 || j == 0){
                    LCSuff[i*col_width+j] = 0;
                    __syncthreads();
                }
                else if (X[i-1] == Y[j-1]) {  
                    LCSuff[i*col_width+j] = LCSuff[((i-1)*col_width)+j-1] + 1;
                    __syncthreads();
                    
                } 
                else{
                    LCSuff[i*col_width+j] = max1(LCSuff[(i-1)*col_width+j],LCSuff[i*col_width+j-1]);
                    __syncthreads();
                }
                
            }

        } 
    }


    """)
    LCSQ = mod.get_function("lc_subsequence")

    #a = numpy.array(test,dtype=numpy.int32) #row the width
    #b = numpy.array([0,0,0,1],dtype=numpy.int32) #col
    a = numpy.array(output_spike_train,dtype=numpy.int32)
    b = numpy.array(output_dataset,dtype=numpy.int32)
    LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)


    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)


    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)


    root_num = math.ceil(math.sqrt(len_dataset))
    thread_num = root_num % 1024
    grid_num = math.ceil(root_num / 1024)
    timer_gpu = GpuTimer()
    timer_gpu.tic()

    LCSQ(a_gpu,b_gpu,LCSuff_gpu, numpy.int32(a.size+1),numpy.int32(b.size+1), block=(thread_num,1,1),grid=(thread_num,1,1))
    timer_gpu.toc()
    timer_write("Evaluate", timer_gpu.time())
    
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)

    return LCSuff[a.size][b.size]

def GPULCSubStr(output_dataset, output_spike_train, len_dataset): 

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
            for (int i = 0; i < row_width; i++) {
                if (X[i-1] == Y[j-1] && !(i == 0 || j == 0)) {  
                    LCSuff[i*col_width+j] = LCSuff[(i-1)*col_width+j-1] + 1;
                    __syncthreads();
                    //printf("compare %d %d\\n",res[0],LCSuff[i*col_width+j]);
                    res[0] = max1(res[0], LCSuff[i*col_width+j]);
                    __syncthreads();
                    //printf("res %d %d %d\\n",res[0],i,j);
                    
                } 
                else{
                    LCSuff[i*col_width+j] = 0;
                    __syncthreads();
                }
                
            }

        } 
    }


    """)
    LCS = mod.get_function("lc_substring")
    #no = [1,1,1,0,1,0,1]
    #a = numpy.array(no,dtype=numpy.int32) #row the width
    #b = numpy.array([0,0,0,0,0,1],dtype=numpy.int32) #col
    a = numpy.array(output_spike_train,dtype=numpy.int32)
    b = numpy.array(output_dataset,dtype=numpy.int32)
    res = numpy.array([0],dtype=numpy.int32)
    LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)

    #print(LCSuff[0])
    

    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
    res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)
    drv.memcpy_htod(res_gpu, res)

    root_num = math.ceil(math.sqrt(len_dataset))
    thread_num = root_num % 1024
    grid_num = math.ceil(root_num / 1024)
    timer_gpu = GpuTimer()
    timer_gpu.tic()
    LCS(a_gpu,b_gpu,res_gpu,LCSuff_gpu, numpy.int32(a.size+1),numpy.int32(b.size+1), block=(thread_num,1,1),grid=(thread_num,1,1))
    timer_gpu.toc()
    timer_write("Evaluate", timer_gpu.time())
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    drv.memcpy_dtoh(res, res_gpu)


    #print ("input 1 ", a)
    #print ("input 2 ", b)

    #print(LCSuff)
    
    #print("cpu res substr ", LCS(a, b, len(a), len(b)))
    #print("res substr", res)
    return res[0] 

def GPUeditDistDP0(output_dataset, output_spike_train, max_row_width, max_col_width, len_dataset, output_dataset_lengths, output_rssnp_lengths):
    mod = SourceModule("""
    #include <stdlib.h>
    __device__ int min1(int a,int b){
        if(a<b){
            return a;
        }
        else{
            return b;
        }
    }
    __global__ void edit_distDP(int* result_mat_gpu, int *dataset_gpu, int *output_gpu, int row_width, int col_width, int len_dataset, int *LCSuff, int *output_dataset_lengths, int *output_rssnp_lengths){
       const int z = threadIdx.x + blockDim.x * blockIdx.x;
       if (z < len_dataset) {
           //int max_val = 0;
           //printf("on thread %d i constrained by %d j constrained by %d", z, output_rssnp_lengths[z], output_dataset_lengths[z]);
           //printf("with content %d", result_mat_gpu[z]);
           //printf("row width is %d col width is %d", row_width, col_width);
           int i_constraint = output_dataset_lengths[z];
           int j_constraint = output_rssnp_lengths[z];
           int delete_point = 1;
           //printf("with j constraint as %d and i cons as %d delete point is %f", j_constraint, i_constraint, delete_point);
           for (int j = 0; j < j_constraint + 1; j++) {
                
                for (int i = 0; i <  i_constraint + 1; i++){
                    //printf("computed value is %d", (z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + i*row_width + i));
                    int* LCSuff_base = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + i*row_width + i)];
                    __syncthreads();
                    if (i == 0){
                        *LCSuff_base = j;
                        //printf("A %d B %d LC %d\\n", i,j,*LCSuff_base);          
                        __syncthreads();
                    }
                    else if (j == 0){
                        *LCSuff_base = i;
                        //printf("A %d B %d LC %d\\n", i,j,*LCSuff_base);
                        __syncthreads();
                    }
                    else{
                        int delt = 0;
                        if (dataset_gpu[z * row_width + (i-1)] != output_gpu[z * col_width + (j-1)]) {  
                            delt = delete_point;
                        }
                        int* LCSuff_row_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + ((j-1)*len_dataset*row_width + i*row_width + i)];
                        int* LCSuff_col_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + (i-1)*row_width + i-1)];
                        int* LCSuff_both_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + ((j-1)*len_dataset*row_width + (i-1)*row_width + i-1)];
                        //printf(" gpu %d %d %d with %d and %d compared\\n", * LCSuff_col_decrem, *LCSuff_row_decrem, * LCSuff_both_decrem, dataset_gpu[z * row_width + (i-1)], output_gpu[z * col_width + (j-1)]);
                        *LCSuff_base = min1(min1(*LCSuff_col_decrem + delete_point, *LCSuff_row_decrem + delete_point), *LCSuff_both_decrem + delt);

                        __syncthreads();
                    }
                       
                }
            }
            result_mat_gpu[z] = LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j_constraint*len_dataset*row_width + i_constraint*row_width + i_constraint)];
            
        } 
    }
    """)
    ED = mod.get_function("edit_distDP")
    #print("legendary shape is ", output_spike_train.shape, " with content ", output_dataset)
    a = numpy.ndarray(shape= output_dataset.shape, buffer =  output_dataset, dtype=numpy.int32) 
    b = numpy.ndarray(shape= output_spike_train.shape, buffer =  output_spike_train, dtype=numpy.int32) 
    #a = numpy.array(output_dataset, dtype=numpy.int32)
    #b = numpy.array(output_spike_train, dtype=numpy.int32)
    #print("a strides and at index 0", a.strides, a[0].strides)
    #print("b strides", b.strides)

    #print("a size is ", a.size, " while b size is ", b.size)


    LCSuff = numpy.zeros((len_dataset, a.size+1,b.size+1),dtype=numpy.int32)
    #print("LCSuff shape is ", LCSuff.shape)
    row_width = numpy.int32(max_row_width)
    col_width = numpy.int32(max_col_width)
    result_mat = numpy.zeros((len_dataset),dtype=numpy.int32)
    #LCSuff = LCSuff.flatten()

        #print("LCSuff line is ", LCSuff[z][0])
    #print("LCSuff orig is ", LCSuff)
    #print("output dataset lengths is ", output_dataset_lengths)
    #print("output rssnp lengths is ", output_rssnp_lengths)
    #c = numpy.ndarray(shape= output_dataset_lengths.shape, buffer =  output_dataset_lengths, dtype=numpy.int32) 
    #d = numpy.ndarray(shape= output_rssnp_lengths.shape, buffer =  output_rssnp_lengths, dtype=numpy.int32) 
    #c = pg.get(output_dataset_lengths)
    #d = pg.get(output_rssnp_lengths)
    #print("c and d are magically ", c, " and ",  d, " with type ", type(c))
    #inout_pairs_view_gpu = drv.mem_alloc(inout_pairs_view.size * inout_pairs_view.dtype.itemsize)
    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    result_mat_gpu = drv.mem_alloc(result_mat.size * result_mat.dtype.itemsize)
    #print("LCSuff size is ", LCSuff.size)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
  
    #func(drv.In(a), drv.InOut(e), np.int32(N), block=block_size, grid=grid_size)
    #c_gpu = drv.mem_alloc(c.size * c.dtype.itemsize)
    #d_gpu = drv.mem_alloc(d.size * d.dtype.itemsize)
    #c_gpu = drv.mem_alloc(c.nbytes)
    #d_gpu = drv.mem_alloc(d.nbytes)
    c_gpu = gpuarray.to_gpu(output_dataset_lengths.astype(numpy.int32))
    d_gpu = gpuarray.to_gpu(output_rssnp_lengths.astype(numpy.int32))

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)
    #drv.memcpy_htod(c_gpu, c)
    #drv.memcpy_htod(d_gpu, d)
    #print("FINALLY entered with values ", c, " and ", d)
    root_num = math.ceil(math.sqrt(len_dataset))
    thread_num = root_num % 1024
    grid_num = math.ceil(root_num / 1024)
    timer_gpu = GpuTimer()
    timer_gpu.tic()
    ED(result_mat_gpu, a_gpu,b_gpu, numpy.int32(row_width), numpy.int32(col_width), numpy.int32(len_dataset), LCSuff_gpu, c_gpu, d_gpu, block=(thread_num,1,1),grid=(thread_num,1,1))
    timer_gpu.toc()
    timer_write("Evaluate", timer_gpu.time())
    #drv.memcpy_dtoh(d, d_gpu)
    #drv.memcpy_dtoh(c, c_gpu)
    drv.memcpy_dtoh(result_mat, result_mat_gpu)
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    #print("result mat is ", result_mat)
    #print("FINALLY 2 entered with values ", c, " and ", d)
    sum_result = 0
    for index in range(len(result_mat)):
        maxlen = output_rssnp_lengths[index]
        minlen = output_dataset_lengths[index]
        #print("maxlen is ", maxlen)  
        #print("result mat is ", result_mat[index])
        sum_result += (maxlen - result_mat[index])/minlen#(maxlen * len_dataset * (minlen/maxlen))
    #print("sum is ", sum_result)
    return sum_result


def GPUeditDistDP(output_dataset, output_spike_train, max_row_width, max_col_width, len_dataset, output_dataset_lengths, output_rssnp_lengths):
    mod = SourceModule("""
    #include <stdlib.h>
    __device__ int min1(float a,float b){
        if(a<b){
            return a;
        }
        else{
            return b;
        }
    }
    __global__ void edit_distDP(float* result_mat_gpu, int *dataset_gpu, int *output_gpu, int row_width, int col_width, int len_dataset, float *LCSuff, int *output_dataset_lengths, int *output_rssnp_lengths, float* float_holder){
       int z = threadIdx.x + blockDim.x * blockIdx.x;
       if (z < len_dataset) {
           //int max_val = 0;
           //printf("on thread %d i constrained by %d j constrained by %d", z, output_rssnp_lengths[z], output_dataset_lengths[z]);
           //printf("with content %d", result_mat_gpu[z]);
           //printf("row width is %d col width is %d", row_width, col_width);
           int i_constraint = output_dataset_lengths[z];
           int j_constraint = output_rssnp_lengths[z];
           float delete_point = i_constraint * float(1.0f/j_constraint);
           //printf("with j constraint as %d and i cons as %d delete point is %f", j_constraint, i_constraint, delete_point);
           for (int j = 0; j < j_constraint + 1; j++) {
                
                for (int i = 0; i <  i_constraint + 1; i++){
                    //printf("computed value is %d", (z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + i*row_width + i));
                    float* LCSuff_base = (float*)&LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + i*row_width + i)];
                    __syncthreads();
                    if (i == 0){
                        atomicAdd(LCSuff_base, float(1.0f*j));
                        __syncthreads();
                        //printf("A %d B %d LC %f\\n", i,j,*LCSuff_base);          
                        
                    }
                    else if (j == 0){
                        atomicAdd(LCSuff_base, float(1.0f*i));
                        __syncthreads();
                        //printf("A %d B %d LC %f\\n", i,j,*LCSuff_base);
                        
                    }
                    else{
                        float delt = 0;
                        if (dataset_gpu[z * row_width + (i-1)] != output_gpu[z * col_width + (j-1)]) {  
                            delt = delete_point;
                        }
                        __syncthreads();
                        float* LCSuff_row_decrem = (float*)&LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + ((j-1)*len_dataset*row_width + i*row_width + i)];
                        __syncthreads();
                        float* LCSuff_col_decrem = (float*)&LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + (i-1)*row_width + i-1)];
                        __syncthreads();
                        float* LCSuff_both_decrem = (float*)&LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + ((j-1)*len_dataset*row_width + (i-1)*row_width + i-1)];
                        __syncthreads();
                        printf(" gpu %f %f %f with %d and %d compared\\n", * LCSuff_col_decrem, *LCSuff_row_decrem, * LCSuff_both_decrem, dataset_gpu[z * row_width + (i-1)], output_gpu[z * col_width + (j-1)]);
                        float_holder[0] = 0;
                        __syncthreads();
                        atomicAdd((float*)&float_holder[0], *LCSuff_col_decrem);
                        atomicAdd((float*)&float_holder[0], delete_point);
                        __syncthreads();
                        float_holder[1] = 0;
                        __syncthreads();
                        atomicAdd((float*)&float_holder[1], *LCSuff_row_decrem);
                        atomicAdd((float*)&float_holder[1], delete_point);
                        __syncthreads();
                        float_holder[2] = 0;
                        __syncthreads();
                        atomicAdd((float*)&float_holder[2], *LCSuff_both_decrem);
                        atomicAdd((float*)&float_holder[2], delt);
                        __syncthreads();
                        *LCSuff_base = min1(min1(float_holder[0], float_holder[1]), float_holder[2]);
                        __syncthreads();
                    }
                       
                }
            }
            result_mat_gpu[z] = LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j_constraint*len_dataset*row_width + i_constraint*row_width + i_constraint)];
            
        } 
    }
    """)
    ED = mod.get_function("edit_distDP")
    #print("legendary shape is ", output_spike_train.shape, " with content ", output_dataset)
    a = numpy.ndarray(shape= output_dataset.shape, buffer =  output_dataset, dtype=numpy.int32) 
    b = numpy.ndarray(shape= output_spike_train.shape, buffer =  output_spike_train, dtype=numpy.int32) 
    #a = numpy.array(output_dataset, dtype=numpy.int32)
    #b = numpy.array(output_spike_train, dtype=numpy.int32)
    #print("a strides and at index 0", a.strides, a[0].strides)
    #print("b strides", b.strides)

    #print("a size is ", a.size, " while b size is ", b.size)


    LCSuff = numpy.zeros((len_dataset, a.size+1,b.size+1),dtype=numpy.float32)
    #print("LCSuff shape is ", LCSuff.shape)
    row_width = numpy.int32(max_row_width)
    col_width = numpy.int32(max_col_width)
    result_mat = numpy.zeros((len_dataset),dtype=numpy.float32)
    #LCSuff = LCSuff.flatten()

        #print("LCSuff line is ", LCSuff[z][0])
    #print("LCSuff orig is ", LCSuff)
    #print("output dataset lengths is ", output_dataset_lengths)
    #print("output rssnp lengths is ", output_rssnp_lengths)
    #c = numpy.ndarray(shape= output_dataset_lengths.shape, buffer =  output_dataset_lengths, dtype=numpy.int32) 
    #d = numpy.ndarray(shape= output_rssnp_lengths.shape, buffer =  output_rssnp_lengths, dtype=numpy.int32) 
    #c = pg.get(output_dataset_lengths)
    #d = pg.get(output_rssnp_lengths)
    #print("c and d are magically ", c, " and ",  d, " with type ", type(c))
    #inout_pairs_view_gpu = drv.mem_alloc(inout_pairs_view.size * inout_pairs_view.dtype.itemsize)
    float_holder = numpy.zeros((3), dtype=numpy.float32)
    #float_holder_b = numpy.zeros((1), dtype=numpy.float32)
    #float_holder_c = numpy.zeros((1), dtype=numpy.float32)
    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)

    #result_mat_gpu = drv.mem_alloc(result_mat.size * result_mat.dtype.itemsize)
    #print("LCSuff size is ", LCSuff.size)
    #LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
  
    #func(drv.In(a), drv.InOut(e), np.int32(N), block=block_size, grid=grid_size)
    #c_gpu = drv.mem_alloc(c.size * c.dtype.itemsize)
    #d_gpu = drv.mem_alloc(d.size * d.dtype.itemsize)
    #c_gpu = drv.mem_alloc(c.nbytes)
    #d_gpu = drv.mem_alloc(d.nbytes)
    float_holder_gpu = gpuarray.to_gpu(float_holder.astype(numpy.float32))
    #float_holder_b_gpu = gpuarray.to_gpu(float_holder_b.astype(numpy.float32))
    #float_holder_c_gpu = gpuarray.to_gpu(float_holder_c.astype(numpy.float32))
    c_gpu = gpuarray.to_gpu(output_dataset_lengths.astype(numpy.int32))
    d_gpu = gpuarray.to_gpu(output_rssnp_lengths.astype(numpy.int32))
    LCSuff_gpu = gpuarray.to_gpu(LCSuff.astype(numpy.float32))
    result_mat_gpu = gpuarray.to_gpu(result_mat.astype(numpy.float32))
    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    #drv.memcpy_htod(LCSuff_gpu, LCSuff)
    #drv.memcpy_htod(c_gpu, c)
    #drv.memcpy_htod(d_gpu, d)
    #print("FINALLY entered with values ", c, " and ", d)
    root_num = math.ceil(math.sqrt(len_dataset))
    thread_num = root_num % 1024
    grid_num = math.ceil(root_num / 1024)
    #timer_gpu = GpuTimer()
    #timer_gpu.tic()
    ED(result_mat_gpu, a_gpu,b_gpu, numpy.int32(row_width), numpy.int32(col_width), numpy.int32(len_dataset), LCSuff_gpu, c_gpu, d_gpu, float_holder_gpu,  block=(thread_num,1,1),grid=(thread_num,1,1))
    #timer_gpu.toc()
    #timer_write("Evaluate", timer_gpu.time())
    #drv.memcpy_dtoh(d, d_gpu)
    #drv.memcpy_dtoh(c, c_gpu)
    #drv.memcpy_dtoh(result_mat, result_mat_gpu)
    result_mat_gpu.get(result_mat)
    #drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    print("result mat is ", result_mat)
    #print("FINALLY 2 entered with values ", c, " and ", d)
    sum_result = 0
    for index in range(len(result_mat)):
        maxlen = output_rssnp_lengths[index]
        print("maxlen is ", maxlen)  
        print("result mat is ", result_mat[index])
        sum_result += (maxlen - result_mat[index])/maxlen * 100
    print("sum is ", sum_result)
    return sum_result