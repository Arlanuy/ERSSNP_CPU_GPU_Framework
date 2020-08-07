#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

struct Rule{
        int *source;
        int *sink;
        int *prod;
        int *con;
        int *delay;
        int *size_rssnp;
};

__host__ void create_rule(Rule rule, int* source, int* sink, int* prod, int* con, int* delay, int* size_rssnp) {
    rule.source = source;
    rule.sink = sink;
    rule.prod = prod;
    rule.con = con;
    rule.delay = delay;
    rule.size_rssnp = size_rssnp;
}

'''
def get_every_poss(size_rssnp):
    list_poss = []
    for i in range(0,len(size_rssnp)):
        for j in range (0,len(size_rssnp)):
            if(i != j):
                list_poss.append([i,j])
    
    return list_poss

def get_every_rule(size_rssnp, rssnp_id): #size = the number of rules #rssnp_id = the rssnp id
    list_poss = []
    for i in rssnp_id:
        for j in range (0,size_rssnp[i[0]]):
            for k in range (0,size_rssnp[i[1]]):
                list_poss.append([i[0],i[1],j,k])
    
    return list_poss
'''


__global__ void get_every_poss(int* list_poss_rssnp, int max_rssnp) {
    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    if (Row != Col && Row < max_rssnp && Col < max_Rssnp) {
        list_poss_rssnp[Row * max_rssnp] = Row;
        list_poss_rssnp[Row * max_rssnp + 1] = Col;
    }    
}
 
__global__ void get_every_rule(int* list_poss_rule, int max_rules, int* crossover_pair) {
    int Row = blockDim.y*blockIdx.y + threadIdx.y;
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    if (Row < max_rules && Col < max_rules) {

        list_poss_rule[Row * max_rules] = crossover_pair[0];
        list_poss_rule[Row * max_rules + 1] = crossover_pair[1];    
        list_poss_rule[Row * max_rules + 2] = Row;
        list_poss_rule[Row * max_rules + 3] = Col;
    }
}


__global__ void print_struct(Rule *a){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("source %d\\n", a->source[idx]);
    printf("sink %d\\n", a->sink[idx]);
    printf("prod %d\\n", a->prod[idx]);
    printf("con %d\\n", a->con[idx]);
    printf("delay %d\\n", a->delay[idx]);
    printf("size %d\\n", a->size_rssnp[0]);
}
__global__ void swap_struct(Rule *t, int *r1, int *r2, int *p1, int *p2){
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("source %d %d %d %d\\n", r1[tidx],r2[tidx],p1[tidx],p2[tidx]);
    int idx = t->size_rssnp[p1[tidx]]+r1[tidx];
    int idy = t->size_rssnp[p2[tidx]]+r2[tidx];
    printf("idx: %d idy: %d\\n",idx,idy);
    int temp_source = t->source[idx];
    int temp_sink   = t->sink[idx];
    int temp_prod   = t->prod[idx];
    int temp_con    = t->con[idx];
    int temp_delay  = t->delay[idx];
    t->source[idx] = t->source[idy];
    t->sink[idx]   = t->sink[idy];
    t->prod[idx]   = t->prod[idy];
    t->con[idx]    = t->con[idy];
    t->delay[idx]  = t->delay[idy];
    t->source[idy] = temp_source;
    t->sink[idy]   = temp_sink;
    t->prod[idy]   = temp_prod;
    t->con[idy]    = temp_con;
    t->delay[idy]  = temp_delay;
}



void main() {
    int max_rules = 4;
    int max_rssnp = 4;
    Rule* h_rules = malloc(max_rules * sizeof(Rule));
    int* h_rssnp_indexes_pair1 = malloc(max_rssnp * sizeof(int))
    int* h_rssnp_indexes_pair2 = malloc(max_rssnp * sizeof(int))
    

    //initializing the content of rssnp_indexes_pair1and2

    for (int i = 0; i < max_rssnp; i++) {
        h_rssnp_indexes_pair1[i] = i;
        h_rssnp_indexes_pair2[i] = i + 1;
    }

    Rule* d_rules;
    int* d_rssnp_indexes_pair1;
    int* d_rssnp_indexes_pair2;

    cudaMalloc((void **)&d_rules, sizeof(Rules)*max_rules);
    cudaMalloc((void **)&d_rssnp_indexes_pair1, sizeof(int)*max_rssnp);
    cudaMalloc((void  **)&d_rssnp_indexes_pair2, sizeof(int)*max_rssnp);


    '''
    self.ftmp_gpu.copy_to_gpu()

    r1_gpu = cuda.mem_alloc(r1.size * r1.dtype.itemsize)
    r2_gpu = cuda.mem_alloc(r2.size * r2.dtype.itemsize)
    p1_gpu = cuda.mem_alloc(p1.size * p1.dtype.itemsize)
    p2_gpu = cuda.mem_alloc(p2.size * p2.dtype.itemsize)

    cuda.memcpy_htod(r1_gpu, r1)
    cuda.memcpy_htod(r2_gpu, r2)
    cuda.memcpy_htod(p1_gpu, p1)
    cuda.memcpy_htod(p2_gpu, p2)

    func(self.ftmp_gpu.get_ptr(),r1_gpu,r2_gpu,p1_gpu,p2_gpu,block= (4,1,1),grid =(1,1,1))
    
    self.ftmp_gpu.copy_from_gpu()
    '''
    //creating the rules
    Rule rule_mat = malloc(max_rssnp * sizeof(Rule));
    
    int *source = malloc(max_rules * sizeof(int));
    int *sink = malloc(max_rules * sizeof(int));
    int *prod = malloc(max_rules * sizeof(int));
    int *con = malloc(max_rules * sizeof(int));
    int *delay = malloc(max_rules * sizeof(int));
    int *size_rssnp = malloc(max_rules * sizeof(int));
    int* h_list_poss_rssnp = malloc(max_rssnp * sizeof(int));
    int* h_list_poss_rules = malloc(max_rules * sizeof(int));
    int* h_size_rssnp = malloc(max_rssnp * sizeof(int));
    for (int i = 0; i < max_rssnp; i++) {
        source[i] = i + 1;
        sink[i] = i + + 1;
        prod[i] = i + 1;
        con[i] = i + 1;
        delay[i] = 0;
        h_size_rssnp[i] = 20;
                

    }
    create_rule(rule_mat[i], source, sink, prod, con, delay);
    int* list_poss_rssnp;
    int* list_poss_rules;
    int* size_rssnp;

    cudaMalloc((void **)&list_poss_rssnp, max_rssnp * sizeof(int);
    cudaMalloc((void **)&list_poss_rules, max_rules * sizeof(int));
    cudaMalloc((void  **)&size_rssnp, max_rssnp * sizeof(int));

    cudaMemcpy(size_rssnp, h_size_rssnp, sizeof(int)*max_rssnp, cudaMemcpyHostToDevice)
    int population = sizeof(size_rssnp)/sizeof(int);
    dim3 dimBlock(population, population, 1);    
    dim3 dimGrid(1, 1, 1);
    get_every_poss<<<dimGrid,  dimBlock>>>(list_poss_rssnp, max_rssnp);
    cudaMemcpy(h_list_poss_rssnp, list_poss_rssnp, sizeof(int)*max_rssnp, cudaMemcpyDeviceToHost);
    int crossover_pairs = sizeof(list_poss_rssnp)/sizeof(int*);
    for (int i = 0; i < crossover_pairs; i++) {
        dim3 dimBlock2(size_rssnp[h_list_poss_rssnp[i][0]], size_rssnp[h_list_poss_rssnp[i][1]], 1);    
        dim3 dimGrid2(1, 1, 1);
        get_every_rule<<<dimGrid2, dimBlock2>>>(list_poss_rules, max_rules, h_list_poss_rssnp[i]);    
    }

    cudaMemcpy(h_list_poss_rssnp, list_poss_rssnp, sizeof(int)*max_rssnp, cudaMemcpyDeviceToHost)
    int* tobechosen = malloc((max_rssnp/2) * sizeof(int));
    int* tobechosen2 2= malloc((max_rssnp/2) * sizeof(int));
    //choose which of the index pairs in h_list_poss_rssnp are chosen
    for (int i = 0; i < max_rssnp/2; i++) {
        tobechosen[i] = rand(max_rssnp);
        tobechosen2[i] = rand(max_rssnp);
    }


    cudaMemcpy(h_list_poss_rules, list_poss_rules, sizeof(int)*max_rssnp, cudaMemcpyDeviceToHost);
    //randomize the swapping of the rules
    int* tobereplaced = malloc((max_rssnp/2) * sizeof(int));
    int* tobereplaced 2= malloc((max_rssnp/2) * sizeof(int));
    for (int i = 0; i < max_rssnp/2; i++) {
        tobereplaced[i] = rand(max_rules);
        tobereplaced2[i] = rand(max_rules);

    }

    dim3 dimBlock3(population/2, population/2, 1);    
    dim3 dimGrid3(1, 1, 1);
    swap_struct<<<dimGrid3,  dimBlock3>>>(rule_mat, tobereplaced, tobereplaced2, tobechosen, tobechosen2);
}