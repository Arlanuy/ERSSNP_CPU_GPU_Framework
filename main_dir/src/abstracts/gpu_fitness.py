import pycuda.autoinit
import pycuda.driver as drv
import numpy
import math

from pycuda.compiler import SourceModule
from pycuda.gpuarray import GPUArray as pg

from pycuda import gpuarray

#Source whn in Stackoverflow with question how to properly copy a gpuarray (longstanding bug)
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

    __global__ void lc_subsequence(int *X,int *Y,int *res, int *LCSuff,int row_width,int col_width){
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
    LCSQ = mod.get_function("lc_subsequence")
    
    #a = numpy.array(test,dtype=numpy.int32) #row the width
    #b = numpy.array([0,0,0,1],dtype=numpy.int32) #col
    a = numpy.array(output_spike_train,dtype=numpy.int32)
    b = numpy.array(output_dataset,dtype=numpy.int32)
    res = numpy.array([0],dtype=numpy.int32)
    LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)


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

    LCSQ(a_gpu,b_gpu,res_gpu,LCSuff_gpu, numpy.int32(a.size+1),numpy.int32(b.size+1), block=(thread_num,1,1),grid=(thread_num,1,1))
    drv.memcpy_dtoh(res, res_gpu)
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    #print(LCSuff)
    #print(LCSuff[a[1].size][b.size])
    return LCSuff[a.size][b.size]
    #print("res lcs", res)

    #print (lcs(a[1], b)) 

    #return res[0]  

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

    LCS(a_gpu,b_gpu,res_gpu,LCSuff_gpu, numpy.int32(a.size+1),numpy.int32(b.size+1), block=(thread_num,1,1),grid=(thread_num,1,1))
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    drv.memcpy_dtoh(res, res_gpu)


    print ("input 1 ", a)
    print ("input 2 ", b)

    #print(LCSuff)
    
    #print("cpu res substr ", LCS(a, b, len(a), len(b)))
    print("res substr", res)
    return res[0] 


def GPUeditDistDP(output_dataset, output_spike_train, max_row_width, max_col_width, len_dataset, output_dataset_lengths, output_rssnp_lengths):
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
           int j_constraint = 8;//output_dataset_lengths[z];
           int i_constraint = 9;//output_rssnp_lengths[z];
           int* max_val = 0;
           for (int j = 0; j < j_constraint; j++) {
                
                for (int i = 0; i <  i_constraint; i++){
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
                        int delt = 1;
                        if (dataset_gpu[z * row_width + (i-1)] == output_gpu[z * col_width + (j-1)]) {  
                            delt = 0;
                        }
                        int* LCSuff_col_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + ((j-1)*len_dataset*row_width + i*row_width + i)];
                        int* LCSuff_row_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j*len_dataset*row_width + (i-1)*row_width + i-1)];
                        int* LCSuff_both_decrem = &LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + ((j-1)*len_dataset*row_width + (i-1)*row_width + i-1)];
                        //printf(" gpu %d %d %d with %d and %d compared\\n", * LCSuff_col_decrem, *LCSuff_row_decrem, * LCSuff_both_decrem, dataset_gpu[z * row_width + (i-1)], output_gpu[z * col_width + (j-1)]);
                        *LCSuff_base = min1(min1(*LCSuff_col_decrem + 1, *LCSuff_row_decrem), *LCSuff_both_decrem + delt);
                        max_val = LCSuff_base;
                        __syncthreads();
                    }
                       
                }
            }
            if (max_val != 0) {
                //result_mat_gpu[z] = LCSuff[(z*len_dataset*col_width*len_dataset*row_width) + (j_constraint*len_dataset*row_width + i_constraint*row_width + i_constraint)];
                result_mat_gpu[z] = *max_val;
            }
        } 
    }


    """)
    ED = mod.get_function("edit_distDP")
    print("legendary shape is ", output_spike_train.shape, " with content ", output_dataset)
    a = numpy.ndarray(shape= output_dataset.shape, buffer =  output_dataset, dtype=numpy.int32) 
    b = numpy.ndarray(shape= output_spike_train.shape, buffer =  output_spike_train, dtype=numpy.int32) 
    #a = numpy.array(output_dataset, dtype=numpy.int32)
    #b = numpy.array(output_spike_train, dtype=numpy.int32)
    print("a strides and at index 0", a.strides, a[0].strides)
    print("b strides", b.strides)

    print("a size is ", a.size, " while b size is ", b.size)


    LCSuff = numpy.zeros((len_dataset, a.size+1,b.size+1),dtype=numpy.int32)
    print("LCSuff shape is ", LCSuff.shape)
    row_width = numpy.int32(max_row_width)
    col_width = numpy.int32(max_col_width)
    result_mat = numpy.zeros((len_dataset),dtype=numpy.int32)
    #LCSuff = LCSuff.flatten()

        #print("LCSuff line is ", LCSuff[z][0])
    #print("LCSuff orig is ", LCSuff)

    #c = numpy.ndarray(shape= output_dataset_lengths.shape, buffer =  output_dataset_lengths, dtype=numpy.int32) 
    #d = numpy.ndarray(shape= output_rssnp_lengths.shape, buffer =  output_rssnp_lengths, dtype=numpy.int32) 
    c = pg.get(output_dataset_lengths)
    d = pg.get(output_rssnp_lengths)

    #print("c and d are magically ", c, " and ",  d, " with type ", type(c))
    #inout_pairs_view_gpu = drv.mem_alloc(inout_pairs_view.size * inout_pairs_view.dtype.itemsize)
    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    result_mat_gpu = drv.mem_alloc(result_mat.size * result_mat.dtype.itemsize)
    print("LCSuff size is ", LCSuff.size)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
  
    #func(drv.In(a), drv.InOut(e), np.int32(N), block=block_size, grid=grid_size)
    #c_gpu = drv.mem_alloc(c.size * c.dtype.itemsize)
    #d_gpu = drv.mem_alloc(d.size * d.dtype.itemsize)
    c_gpu = drv.mem_alloc(c.nbytes)
    d_gpu = drv.mem_alloc(d.nbytes)
    #c_gpu = gpuarray.to_gpu(output_dataset_lengths)
    #d_gpu = gpuarray.to_gpu(output_rssnp_lengths)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)
    drv.memcpy_htod(c_gpu, c)
    drv.memcpy_htod(d_gpu, d)
    #print("FINALLY entered with values ", c, " and ", d)
    root_num = math.ceil(math.sqrt(len_dataset))
    thread_num = root_num % 1024
    grid_num = math.ceil(root_num / 1024)

    ED(result_mat_gpu, a_gpu,b_gpu, numpy.int32(row_width), numpy.int32(col_width), numpy.int32(len_dataset), LCSuff_gpu, c_gpu, d_gpu, block=(thread_num,1,1),grid=(thread_num,1,1))

  
    drv.memcpy_dtoh(d, d_gpu)
    drv.memcpy_dtoh(c, c_gpu)
    drv.memcpy_dtoh(result_mat, result_mat_gpu)
    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    print("result mat is ", result_mat)
    print("FINALLY 2 entered with values ", c, " and ", d)
   
    sum = 0
    for index in range(len(result_mat)):
        sum += (result_mat[index]/output_dataset_lengths[index]) * 100
    return sum

def GPUeditDistDP2(output_dataset, output_spike_train):
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

    __global__ void edit_distDP(int j,int *X,int *Y, int *LCSuff,int row_width,int col_width){
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < row_width){
                if (i == 0){
                    LCSuff[i*col_width+j] = j;
                    __syncthreads();
                }
                else if (j == 0){

                    LCSuff[i*col_width+j] = i;
                    //printf("A %d B %d LC %d\\n", i,j,LCSuff[i*col_width+j]);
                    __syncthreads();
                }
                else if (X[i-1] == Y[j-1]) {  
                    LCSuff[i*col_width+j] = LCSuff[((i-1)*col_width)+j-1];
                    //printf("true");
                    __syncthreads();
                    //printf("compare %d %d\\n",res[0],LCSuff[i*col_width+j]);
                    //printf("res %d %d %d\\n",res[0],i,j);
                    
                } 
                else{
                    //printf("%d %d %d\\n",LCSuff[((i-1)*col_width)+j],LCSuff[i*col_width+j-1],LCSuff[((i-1)*col_width)+j-1]);
                    LCSuff[i*col_width+j] = 1+min1(min1(LCSuff[((i-1)*col_width)+j],LCSuff[i*col_width+j-1]),LCSuff[((i-1)*col_width)+j-1]);
                    __syncthreads();
                }

        } 
    }


    """)
    ED = mod.get_function("edit_distDP")

    #a = numpy.array([1,1,1,1,1],dtype=numpy.int32) #row the width
    #b = numpy.array([0,0,0,0,0],dtype=numpy.int32) #col
    a = numpy.array(output_spike_train,dtype=numpy.int32)
    b = numpy.array(output_dataset,dtype=numpy.int32)
    res = numpy.array([0],dtype=numpy.int32)
    LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)


    for i in range(b.size+1):
        LCSuff[0][i]=i
    for i in range(a.size+1):
        LCSuff[i][0]=i
    # print(LCSuff)

    a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
    LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
    res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(LCSuff_gpu, LCSuff)
    drv.memcpy_htod(res_gpu, res)

    for j in range(b.size+1):
        print("at index ", j)
        ED(numpy.int32(j),a_gpu,b_gpu,LCSuff_gpu,numpy.int32(a.size+1),numpy.int32(b.size+1) , block=(10,10,1),grid=(1,1,1))


    drv.memcpy_dtoh(LCSuff, LCSuff_gpu)
    drv.memcpy_dtoh(res, res_gpu)
    #print(LCSuff)
    print(LCSuff[a.size][b.size])
    #print(res)
    #print("res editdist", res)
    #print("CPU", editDistDP(a, b, len(a), len(b))) 

    return (LCSuff[a.size][b.size])