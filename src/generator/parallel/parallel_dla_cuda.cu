#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ 

__device__ struct particle *get_random_position_atomic(struct coords voxel_size, int seed) {
    struct particle *ret = (struct particle *) malloc(sizeof(struct particle));
    ret->rng = seed;
    ret->coord.x = (int) (atomic_random_float(&ret->rng) * voxel_size.x);
    ret->coord.y = (int) (atomic_random_float(&ret->rng) * voxel_size.y);
    ret->coord.z = (int) (atomic_random_float(&ret->rng) * voxel_size.z);
    switch((int) atomic_random_float(&ret->rng) * 6) {
        case 0:
            ret->coord.x = 0;
            break;
        case 1:
            ret->coord.x = voxel_size.x - 1;
            break;
        case 2:
            ret->coord.y = 0;
            break;
        case 3:
            ret->coord.y = voxel_size.y - 1;
            break;
        case 4:
            ret->coord.z = 0;
            break;
        default: 
            ret->coord.z = voxel_size.z - 1;
            break;
    }
    return ret;
}

__global__ void hello() {
    printf("hello world!\n");
}

int main() {
    hello<<<1,10>>>();
    cudaDeviceSynchronize();
    return 1;
}