import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
def editDistDP(str1, str2, m, n): 
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 

    # Fill d[][] in bottom up manner 
    for i in range(m + 1): 
        for j in range(n + 1): 

            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j # Min. operations = j 

            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i # Min. operations = i 

            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 

            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],   # Insert 
                                dp[i-1][j],  # Remove 
                                dp[i-1][j-1]) # Replace 
    for a in dp:
        print(a)

    return dp[m][n] 

mod = SourceModule("""
#include <stdlib.h>
__device__ int min1(int a,int b){

    if(a<b && != 0){
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
                __syncthreads();
                //printf("compare %d %d\\n",res[0],LCSuff[i*col_width+j]);
                //printf("res %d %d %d\\n",res[0],i,j);
                
            } 
            else{
            	
                LCSuff[i*col_width+j] = 1+min1(min1(LCSuff[((i-1)*col_width)+j],LCSuff[i*col_width+j-1]),LCSuff[((i-1)*col_width)+j-1]);
                __syncthreads();
            }
            
        }

    } 
}


""")
LCS = mod.get_function("lc_substring")

a = numpy.array([1,1,1,1],dtype=numpy.int32) #row the width
b = numpy.array([0,0,0,0,0],dtype=numpy.int32) #col
res = numpy.array([0],dtype=numpy.int32)
LCSuff = numpy.zeros((a.size+1,b.size+1),dtype=numpy.int32)


# for i in range(b.size+1):
# 	LCSuff[0][i]=i
# for i in range(a.size+1):
# 	LCSuff[i][0]=i
# print(LCSuff)

a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = drv.mem_alloc(b.size * b.dtype.itemsize)
LCSuff_gpu = drv.mem_alloc(LCSuff.size * LCSuff.dtype.itemsize)
res_gpu = drv.mem_alloc(res.size * res.dtype.itemsize)

drv.memcpy_htod(a_gpu, a)
drv.memcpy_htod(b_gpu, b)
drv.memcpy_htod(LCSuff_gpu, LCSuff)
drv.memcpy_htod(res_gpu, res)


LCS(a_gpu,b_gpu,res_gpu,LCSuff_gpu, numpy.int32(a.size+1),numpy.int32(b.size+1), block=(10,10,1),grid=(10,10,1))
drv.memcpy_dtoh(res, res_gpu)
drv.memcpy_dtoh(LCSuff, LCSuff_gpu)

print(LCSuff)
print(LCSuff[a.size][b.size])
print(res)

print("CPU", editDistDP(a, b, len(a), len(b))) 