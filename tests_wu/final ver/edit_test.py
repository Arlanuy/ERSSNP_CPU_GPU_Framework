import pycuda.autoinit
import pycuda.driver as drv
import numpy
import time

from pycuda.compiler import SourceModule

def lcs(X , Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in xrange(m+1)] 
  
    """Following steps build L[m+1][n+1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    for a in L:
        print(a) 
    return "CPU",L[m][n] 

mod = SourceModule("""
#include <stdlib.h>
__device__ int min1(int a,int b){
    if(a<b && a != 0){
        return a;
    }
    else{
        return b;
    }
}

__global__ void lc_subsequence(int *X,int *Y,int *res, int *LCSuff,int row_width,int col_width){
   int j = blockIdx.x * blockDim.x + threadIdx.x+1;
   int temp;
   if (j < col_width){
        for (int i = 0; i < row_width; i++) {
            if (i == 0){
                temp = j;
                __syncthreads();
                LCSuff[i*col_width+j] = temp; 
            }
            else if (j == 0){
                temp = i;
                __syncthreads();
                LCSuff[i*col_width+j] = temp;
            }
            else if (X[i-1] == Y[j-1]) {  
                temp = LCSuff[((i-1)*col_width)+j-1];
                __syncthreads();
                LCSuff[i*col_width+j] = temp;
            } 
            else{
                temp = 1+min1(min1(LCSuff[((i-1)*col_width)+j],LCSuff[i*col_width+j-1]),LCSuff[((i-1)*col_width)+j-1]);
                __syncthreads();
                LCSuff[i*col_width+j] = temp;
                
            } 
            
        }

    } 
}


""")
LCS = mod.get_function("lc_subsequence")

a = numpy.array([1,1,1,1],dtype=numpy.int32) #row the width
b = numpy.array([0,1,0,1,0,1,0,1],dtype=numpy.int32) #col
res = numpy.array([0],dtype=numpy.int32)
LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)

print(LCSuff[0])

a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

drv.memcpy_htod(a_gpu, a)
drv.memcpy_htod(b_gpu, b)
drv.memcpy_htod(LCSuff_gpu, LCSuff)
drv.memcpy_htod(res_gpu, res)

start_time = time.time()
LCS(a_gpu,b_gpu,res_gpu,LCSuff_gpu, numpy.int32(a.size+1),numpy.int32(b.size+1), block=(10,1,1),grid=(1,1,1))
print("--- %s seconds ---" % (time.time() - start_time))
drv.memcpy_dtoh(res, res_gpu)
drv.memcpy_dtoh(LCSuff, LCSuff_gpu)

print(LCSuff)
print(LCSuff[a.size][b.size])
print(res)

print lcs(a, b) 

