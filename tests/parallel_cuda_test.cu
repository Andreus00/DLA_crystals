#include "../src/generator/parallel/parallel_dla_cuda.cu"
#include <sys/time.h>
#define CHUNK_SIZE 16

int main(int argc, char const *argv[])
{
    // inizializzazione del volxel
    int pn_0 = 90000;
    int particle_number_h = pn_0;
    int const W = 8;
    struct coords space_size = {W, W, W};
    struct coords voxel_size = {space_size.x * CHUNK_SIZE, space_size.y * CHUNK_SIZE, space_size.z * CHUNK_SIZE};
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    int *space = parallel_dla_cuda(space_size, 16, particle_number_h);
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("Time elapsed: %ld.%06ld num: %d\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec, particle_number_h);
    // for(int i = 0; i < voxel_size.x; i++) {    // Ho W * W * W chunk
    //     for(int j = 0; j < voxel_size.y; j++) {
    //         for(int k = 0; k < voxel_size.z; k++) {
    //             int value = space[voxel_size.x * voxel_size.y * k + voxel_size.x * j + i];
    //             printf("x: %d - y: %d - z: %d - val: %d\n", i , j, k, value);
    //         }
    //     }
    // }

    //printf("Finito");
    // dim3 GridDim = {8, 8, 8};
    // dim3 BlockDim = {4, 4, 4};
    
    // int *v_array_d;
    // cudaMalloc(&v_array_d, sizeof(int) * particle_number_h);
    // struct particle **particles_d;
    // struct particle **particles_h = (struct particle **) malloc( sizeof(struct particle *) * particle_number_h);
    // cudaMalloc(&particles_d, sizeof(struct particle *) * particle_number_h);
    // init_particles_cuda<<<GridDim, BlockDim>>>(particles_d, particle_number_h, space_size);
    // cudaMemcpy(particles_h,particles_d,sizeof(struct particle *) * particle_number_h,cudaMemcpyDeviceToHost);

    // for (int i=0; i<particle_number_h; i++){
    //     if (i%3==0){
    //         particles_h[i] = NULL;
    //     }
    // }
    // struct timeval tval_before, tval_after, tval_result;
    // cudaMemcpy(particles_d,particles_h,sizeof(struct particle *) * particle_number_h,cudaMemcpyHostToDevice);
    // gettimeofday(&tval_before, NULL);
    // if (SERIAL){
    //     int last = -1;
    //     for(int i = 0; i <= particle_number_h; i++){
    //         if (particles_h[i] != NULL)
    //              particles_h[++last] = particles_h[i];
    //     }
    //     particle_number_h = last;
    //     gettimeofday(&tval_after, NULL);
    //     timersub(&tval_after, &tval_before, &tval_result);
    //     printf("Time elapsed: %ld.%06ld num: %d\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec, particle_number_h);
    // }
    // else{
    //     int logarithm =  (int) ceil(log2(particle_number_h)) - 1;
    //     for(int level = 0; level <= logarithm; level++) {
    //         //printf("\n\nLevel: %d\n", level);
    //         pippo_franco<<<GridDim, BlockDim>>>(particles_d, particle_number_h, v_array_d, level);
    //         //cudaDeviceSynchronize();
            
    //         // cudaMemcpy(particles_h,particles_d,sizeof(struct particle *) * particle_number_h,cudaMemcpyDeviceToHost);

    //         //printf("Particle_number: %d\n", particle_number_h);
    //         // for (int i = 0;  i< 10; i++) {
    //         //     printf("ext - %d - %p\n", i, particles_h[i]); 
    //         // }
    //         // printf("-----------------------------\n");
    //     }
    //     gettimeofday(&tval_after, NULL);
    //     timersub(&tval_after, &tval_before, &tval_result);
    //     cudaMemcpy(&particle_number_h, v_array_d + ((int)(logarithm % 2 == 1)), sizeof(int), cudaMemcpyDeviceToHost);
    //     print_the_fucking_array<<<GridDim, BlockDim>>>(particles_d, pn_0);
    //     cudaDeviceSynchronize();
    //     printf(" Time elapsed: %ld.%06ld num: %d\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec, particle_number_h);
        
    //     // cudaMemcpy(particles_h,particles_d,sizeof(struct particle *) * particle_number_h,cudaMemcpyDeviceToHost);
    //     // for (int i = 0;  i< pn_0; i++) {
    //     //     printf("%p\n", particles_h[i]);
    //     // }
    //     // printf("Particle_number: %d\n", particle_number_h);
    // }
    
    return 0;
}
