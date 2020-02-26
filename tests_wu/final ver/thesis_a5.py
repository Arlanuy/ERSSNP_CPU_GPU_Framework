import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule

def LCSubStr(X, Y, m, n): 
    
    # Create a table to store lengths of 
    # longest common suffixes of substrings. 
    # Note that LCSuff[i][j] contains the 
    # length of longest common suffix of 
    # X[0...i-1] and Y[0...j-1]. The first 
    # row and first column entries have no 
    # logical meaning, they are used only 
    # for simplicity of the program. 
    
    # LCSuff is the table with zero 
    # value initially in each cell 
    LCSuff = [[0 for k in range(n+1)] for l in range(m+1)] 
    
    # To store the length of 
    # longest common substring 
    result = 0

    # Following steps to build 
    # LCSuff[m+1][n+1] in bottom up fashion 
    for i in range(m + 1): 
        for j in range(n + 1): 
            if (i == 0 or j == 0): 
                LCSuff[i][j] = 0
            elif (X[i-1] == Y[j-1]): 
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                result = max(result, LCSuff[i][j]) 
            else: 
                LCSuff[i][j] = 0
    for a in LCSuff:
        print(a)
    return result 


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
__global__ void lc_substring(int *X,int *Y, int *LCSuff,int row_width,int col_width){
   int j = blockIdx.x * blockDim.x + threadIdx.x;
   int temp;
   if (j < col_width){
        for (int i = 0; i < row_width; i++) {
            if (X[i-1] == Y[j-1] && !(i == 0 || j == 0)) {
                temp = LCSuff[(i-1)*col_width+j-1] + 1;
                __syncthreads();
                LCSuff[i*col_width+j] = temp;
            } 
            else{
                temp = 0;
                __syncthreads();
                LCSuff[i*col_width+j] = temp;
            }
            
        }

    } 
}


""")
LCS = mod.get_function("lc_substring")
no = [1,1,1,1]
a = numpy.array(no,dtype=numpy.int32) #row the width
b = numpy.array([0,1,1,1,1,1,0,1],dtype=numpy.int32) #col
# res = numpy.array([0],dtype=numpy.int32)
LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)

a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
# res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

drv.memcpy_htod(a_gpu, a)
drv.memcpy_htod(b_gpu, b)
drv.memcpy_htod(LCSuff_gpu, LCSuff)
# drv.memcpy_htod(res_gpu, res)

print "a",a.size+2
print "b",b.size+2
LCS(a_gpu,b_gpu,LCSuff_gpu, numpy.int32(.size+1),numpy.int32(b.size+1), block=(b.size+1,1,1),grid=(1,1,1))
# drv.memcpy_dtoh(res, res_gpu)
drv.memcpy_dtoh(LCSuff, LCSuff_gpu)

print "input 1 ",a
print "input 2 ",b
print(LCSuff)
res = list(map(max, LCSuff))
res = max(res)
print(res)
print("CPU", LCSubStr(a, b, len(a), len(b))) 


#need same length to be consistent
#make the first param the shorter one
