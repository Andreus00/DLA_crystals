#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>

#ifndef UTILS
    #include "../../../utils/utils.c"
#endif

////////////////////////
// pseudo random function
__device__ float random_float_cuda(int* rng) {
    int n  = (*rng << 13U) ^ *rng;
    (*rng) *= 6364136223846793005ULL;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return (float)((n & 0x7fffffffU)/((float)(0x7fffffff)));
}

__device__ void get_random_position_cuda(struct coords voxel_size, int seed, struct particle *p) {
    p->rng = seed;
    p->coord.x = (int) (random_float_cuda(&p->rng) * voxel_size.x);
    p->coord.y = (int) (random_float_cuda(&p->rng) * voxel_size.y);
    p->coord.z = (int) (random_float_cuda(&p->rng) * voxel_size.z);

    int data[] = {
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
        voxel_size.x - 1, 0, 0,
        0, voxel_size.y - 1, 0,
        0, 0, voxel_size.z - 1
    };
    int r = (int) (random_float_cuda(&p->rng) * 6);
    if(r < 3) {
        p->coord.x *= data[r * 3];
        p->coord.y *= data[r * 3 + 1];
        p->coord.z *= data[r * 3 + 2];
    }
    else {
        p->coord.x = data[r * 3] + data[r * 3 - 9] * p->coord.x;
        p->coord.y = data[r * 3 + 1] + data[r * 3 - 9 + 1] * p->coord.y;
        p->coord.z = data[r * 3 + 2] + data[r * 3 - 9 + 2] * p->coord.z;
    }


}

__device__ void move_particle_cuda(struct coords c1, struct coords *out, int* rng, struct coords voxel_size){
    out->x = c1.x;
    out->y = c1.y;
    out->z = c1.z;
    int rx = ((int) (random_float_cuda(rng) * 3)) - 1;
    int ry = ((int) (random_float_cuda(rng) * 3)) - 1;
    int rz = ((int) (random_float_cuda(rng) * 3)) - 1;
    if((rx + out->x ) > 0 &&  (rx + out->x ) < voxel_size.x)
        out->x += rx;
    if((ry + out->y ) > 0 &&  (ry + out->y ) < voxel_size.y)
        out->y += ry;
    if((rz + out->z ) > 0 &&  (rz + out->z ) < voxel_size.z)
        out->z += rz;
    return;
}

__device__ int get_tid(){
    return ( gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * blockDim.y * blockDim.z \
     + blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
}
__device__ int get_num_threads(){
    return gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
}

__global__ void step(int* voxel_d, struct coords voxel_size, struct particle** list_d, struct particle** freezed_d, int particle_number) {
    int tid = get_tid();
    int num_threads = get_num_threads();

    for(int i = tid; i < particle_number ; i += num_threads)  {
        struct particle *part = list_d[i];
        // get value
        int cell_value = voxel_d[part->coord.x + part->coord.y * voxel_size.x + part->coord.z * voxel_size.x * voxel_size.y];
        if (cell_value != -1){
            struct coords new_pos;
            move_particle_cuda(part->coord, &new_pos, &part->rng, voxel_size);
            cell_value = voxel_d[new_pos.x + new_pos.y * voxel_size.x + new_pos.z * voxel_size.x * voxel_size.y];

            // passare il puntatore alla particella da list_d a freezed se (cell_value == -1)
            // altrimenti, fare che list_d = new_position
            
            if((cell_value != -1)){
                part->coord = new_pos;
            }
            else {
                freezed_d[i] = part;
                list_d[i] = NULL;
            }
        }
        else{
                free(list_d[i]);        // free sul device è supportato da CC 2.0+
                list_d[i] = NULL;
            }
    }
    return;
}

__global__ void init_particles_cuda(struct particle **list_d, int num, struct coords voxel_size ) {
    // recupera il suo id
     int tid =   get_tid();
     int num_threads = get_num_threads();
    for(int i = tid; i < num; i += num_threads)  {
        list_d[i] = (particle *) malloc(sizeof(struct particle));       // malloc sul device è supportato da CC 2.0+
        //      genera la particella a index i
        get_random_position_cuda(voxel_size, i + 1,list_d[i]);
    }
}
__global__ void freeze(int* voxel_d, struct coords voxel_size, struct particle** list_d, struct particle** freezed_d, int particle_number_d){
    // freezare particelle
   // __shared__ int deleted_particles_d;
    int tid =   get_tid();
    int num_threads = get_num_threads();
    for(int i = tid; i < particle_number_d; i += num_threads)  {
       if( freezed_d[i] != NULL ){
           voxel_d[freezed_d[i]->coord.x + freezed_d[i]->coord.y * voxel_size.x + freezed_d[i]->coord.z * voxel_size.x * voxel_size.y] = -1;
           free(freezed_d[i]);          // free sul device è supportato da CC 2.0+
           //deleted_particles_d++;
           freezed_d[i]=NULL;
       }
    }  
}

__global__ void pippo_franco(struct particle** particle_list_d, int particle_number_d, int* v_array_d, int level) {
    int tid =   get_tid();
    
    int num_threads = get_num_threads();
    if(particle_number_d == 1 && particle_list_d[0] == NULL) {
        v_array_d[0] = 0;
    }
    if(level == 0) {    // inizializzo la lizta dei v
        // for(int jj = 0; jj < 100000; jj++) printf("l: %d  - i: %d\n",level, tid);
        for(int i = tid * 2; i < (particle_number_d); i += (num_threads * 2)) {
            v_array_d[i] = (int) (particle_list_d[i] != NULL); // inizializzo a coppie e swappo.
            if((i + 1) < particle_number_d) {
                //v_array_d[i + 1] = (int) (particle_list_d[i + 1] != NULL);
                v_array_d[i] += (int) (particle_list_d[i + 1] != NULL);
                if(particle_list_d[i] == NULL && v_array_d[i] == 1) {
                    // printf("Swap cella: %d\n", i);
                    particle_list_d[i] = particle_list_d[i + 1];
                    particle_list_d[i + 1] = NULL;
                }
                // if(v_array_d[i] == 1 && v_array_d[i + 1] == 1) {
                //     particle_list_d[i] = particle_list_d[i + 1];
                //     particle_list_d[i + 1] = NULL;
                //     v_array_d[i + 1] = 0;
                //     // v_array_d[i] = 1;
                // }
            }
            // if (tid==0)
            //printf("l: %d  - i: %d - i/blk: %d - v1: %d - v2: %d\n",level, i, i/2,v_array_d[i],v_array_d[i + 1]);
        }
        // if (tid==0){
        // for (int i = 0;  i< 10; i++) {
        //     printf("%d - %p\n", i, particle_list_d[i]);
        // }
        // }
    }
    
    else {
        // for (int j=tid; j<particle_number_h; j+=num_threads)
        //     printf("\n");
        int block_len = (1 << level);
        for(int i = tid; i < (particle_number_d); i += num_threads) {
            // devo trovare l'indice iniziale del blocco in cui si trova l'elemento, 
            // e andare a vedere nell'array dei v, il valore che si trova in quel punto
            //for(int jj = 0; jj < 2; jj++) printf("l: %d  - i: %d - i/bl_len: %d\n",level, i, block_len);
            if(((int)(i / block_len)) % 2 == 0) {
                int m = block_len * ((int) (i / block_len) + 1);  // La metà equivale al primo indice del blocco dispari
                //if (level == 13) for(int jj = 0; jj < 10000; jj++) printf("l: %d  - i: %d - m: %d\n",level, i, m);
                if(m < particle_number_d) {
                    int casella = m - block_len + ((int)(level % 2 == 0));
                    int v1 = v_array_d[casella];
                    int v2 = v_array_d[m + ((int)(level % 2 == 0))];
                    //printf("l: %d  - i: %d - v1: %d - v2: %d  - casella: %d  - m: %d\n",level, i, v1, v2, casella, m);
                    if(particle_list_d[i] == NULL) {
                        int xx = m + v2 + (m - block_len) + v1 - i - 1;
                        //printf("swap: %d  ->  %d\n", i, xx);
                        particle_list_d[i] = particle_list_d[xx];
                        particle_list_d[xx] = NULL;
                        // if(xx > 10000 && level <= 10) printf("%d - lvl: %d\n", xx, level);
                        
                    }
                    if(i % block_len == 0) {
                        v_array_d[m - block_len + ((int)(level % 2 == 1))] = v1 + v2;
                    }
                    // if (i == 9983) {
                    //     printf("l: %d  - i: %d - i/blk: %d - v1: %d - v2: %d\n",level, i, i/block_len, v1, v2);
                    // }
                }
                else {
                    int xx = m - block_len + ((int)(level % 2 == 0));
                    int v1 = v_array_d[xx];
                    v_array_d[m - block_len + ((int)(level % 2 == 1))] = v1;
                    // printf("l: %d  - i: %d - v1: %d - casella: %d  - m: %d  - block_len: %d\n",level, i, v1, xx, m, block_len);
                }
                // if (i == 9983) {
                //         printf("l: %d  - i: %d - i/blk: %d - m: %d  - particle_number: %d  - v1: %d - v2: %d\n",level, i, i/block_len, m,\
                //          particle_number_d, v_array_d[m - block_len + level % 2 == 0], v_array_d[m + level % 2 == 0]);
                //     }
            }
            
        }
        if (tid==0){
        for (int i = 0;  i< 10; i++) {
            // printf("%d - %p\n", i, particle_list_d[i]);
        }
        }
    }
}

__global__ void print_the_fucking_array(struct particle** particle_list_d, int max) {
    if(get_tid() == 0)
        for(int i = 0; i < max; i++) 
            printf(" %d  -  %p\n", i, particle_list_d[i]);        
}


int *parallel_dla_cuda(struct coords space_size, int chunk_size, int particle_number_h) {
    // alloco lo spazio per il voxel
    struct coords voxel_size = {space_size.x * chunk_size, space_size.y * chunk_size, space_size.z * chunk_size};
    size_t space_number = voxel_size.x * voxel_size.y * voxel_size.z;

    int *space_h = (int *) calloc(space_number, sizeof(int));
    
    space_h[voxel_size.x / 2 + (voxel_size.y / 2) * voxel_size.x + (voxel_size.z / 2) * voxel_size.x * voxel_size.y] = -1 ;

    int *voxel_d;
    cudaError_t  err;
    err = cudaMalloc((void **) &voxel_d, space_number * sizeof(int));
    
    err = cudaMemcpy(voxel_d, space_h, space_number * sizeof(int), cudaMemcpyHostToDevice);

    // Alloco lo spazio per le particelle
    struct particle **particles_d;
    err = cudaMalloc(&particles_d, sizeof(struct particle *) * particle_number_h);

    dim3 GridDim = {8, 8, 8};
    dim3 BlockDim = {4, 4, 4};
    init_particles_cuda<<<GridDim, BlockDim>>>(particles_d, particle_number_h, voxel_size);

    struct particle **freezed_d;
    err = cudaMalloc(&freezed_d, sizeof(struct particle *) * particle_number_h);

    // int *particle_number_d;
    // err = cudaMalloc((void **) &particle_number_d, sizeof(int));

    // err = cudaMemcpy(particle_number_d, &particle_number_h, sizeof(int), cudaMemcpyHostToDevice);

    // Alloco lo spazio per i v
    int *v_array_d;
    err = cudaMalloc(&v_array_d, sizeof(int) * particle_number_h);

    // int *freezed_number_d;
    // err = cudaMalloc((void **) &freezed_number_d, sizeof(int));

    // err = cudaMemcpy(freezed_number_d, &particle_number_h, sizeof(int), cudaMemcpyHostToDevice);

    while(particle_number_h > 0) {
        step<<<GridDim, BlockDim>>>(voxel_d, voxel_size, particles_d, freezed_d, particle_number_h);
        cudaDeviceSynchronize();
        freeze<<<GridDim, BlockDim>>>(voxel_d, voxel_size, particles_d, freezed_d, particle_number_h);
        cudaDeviceSynchronize();
        int logarithm =  (int) ceil(log2(particle_number_h)) - 1 + ((int)particle_number_h == 1);
        for(int level = 0; level <= logarithm; level++) {
            pippo_franco<<<GridDim, BlockDim>>>(particles_d, particle_number_h, v_array_d, level);
        }
        cudaMemcpy(&particle_number_h, v_array_d + ((int)(logarithm % 2 == 1)), sizeof(int), cudaMemcpyDeviceToHost);

        if(particle_number_h % 10 == 1) printf("Particle_number: %d\n", particle_number_h);
    }
    // sto a pranzo
    cudaFree(freezed_d);
    cudaFree(voxel_d);
    cudaFree(v_array_d);
    cudaFree(particles_d);
    
    cudaMemcpy(space_h,  voxel_d, space_number *sizeof(int), cudaMemcpyDeviceToHost);

    return space_h;
}
