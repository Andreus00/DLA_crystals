#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../../../utils/utils.c"

////////////////////////
// pseudo random function
__device__ float random_float_cuda(int* rng) {
    int n  = (*rng << 13U) ^ *rng;
    (*rng) *= 6364136223846793005ULL;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return (float)((n & 0x7fffffffU)/((float)(0x7fffffff)));
}

__device__ struct particle get_random_position_cuda(struct coords voxel_size, int* seed, int *x, int *y, int *z) {
    
    *x = (int) (random_float_cuda(seed) * voxel_size.x);
    *y = (int) (random_float_cuda(seed) * voxel_size.y);
    *z = (int) (random_float_cuda(seed) * voxel_size.z);
    switch((int) random_float_cuda(seed) * 6) {
        case 0:
            x = 0;
            break;
        case 1:
            x = voxel_size.x - 1;
            break;
        case 2:
            y = 0;
            break;
        case 3:
            y = voxel_size.y - 1;
            break;
        case 4:
            z = 0;
            break;
        default: 
            z = voxel_size.z - 1;
            break;
    }
    return ret;
}

__global__ void generate_crystal(struct particle_list* list) {
    // Allocazione della memoria della GPU

    // Trasferimento dato alla GPU

    // Esecuzione kernel CUDA
    // hello<<<1,10>>>();
    // cudaDeviceSynchronize();

    // Copia dei risultati dalla memoria della GPU

    // Reinizializzazione del dispositivo
    return 1;
}

__global__ generate_particles_cuda(int *particles_x_d, int *particles_y_d, int *particles_z_d, ) {

}

void parallel_dla_cuda(const coords space_size, const int chunk_size, const int particle_number) {
    // alloco lo spazio per il voxel
    struct voxel space;
    init_voxel(&space, space_size);

    size_t space_byte_number = sizeof(int) * (space_size.x * space_size.y * space_size.z) * chunk_size * chunk_size * chunk_size;

    struct coords initial_crystal = {space_size.x / 2 * 16, space_size.y / 2 * 16, space_size.z / 2 * 16};
    setValue(&space, initial_crystal, -1);

    void *voxel_d;

    cudaMalloc(&voxel_d, space_byte_number);
    cudaMemcpy(voxel_d, &space, space_byte_number, cudaMemcpyHostToDevice);

    // Alloco lo spazio per le particelle
    void *particles_x_d;
    cudaMalloc(&particles_x_d, sizeof(int) * particle_number);

    void *particles_y_d;
    cudaMalloc(&particles_y_d, sizeof(int) * particle_number);

    void *particles_z_d;
    cudaMalloc(&particles_z_d, sizeof(int) * particle_number);

    void *particles_rng_d;
    cudaMalloc(&particles_rng_d, sizeof(int) * particle_number);




}
